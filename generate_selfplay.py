"""Self-play game generator using MCTS for training data"""
import torch
import chess
import numpy as np
import random
import math
from pathlib import Path
from src.training.trainer import ChessNeuralNetwork, board_to_encoding

class MCTSNode:
    def __init__(self, board, parent=None, move=None):
        self.board = board.copy() if hasattr(board, 'copy') else board
        self.parent = parent
        self.move = move
        self.children = {}
        self.visits = 0
        self.value_sum = 0.0
        self.prior = 1.0
    
    def is_leaf(self):
        return len(self.children) == 0
    
    def value(self):
        if self.visits == 0:
            return 0
        return self.value_sum / self.visits
    
    def uct_score(self, parent_visits, c_puct=1.5):
        if self.visits == 0:
            return float('inf')
        exploit = self.value()
        explore = c_puct * self.prior * math.sqrt(parent_visits) / (1 + self.visits)
        return exploit + explore

class SelfPlayMCTS:
    def __init__(self, model, device, simulations=100, temperature=1.0):
        self.model = model
        self.device = device
        self.simulations = simulations
        self.temperature = temperature  # Leela-style move selection
        self.model.eval()
    
    def search(self, board):
        root = MCTSNode(board)
        
        for _ in range(self.simulations):
            node = root
            search_path = [node]
            
            # Selection
            while not node.is_leaf() and not node.board.is_game_over():
                node = max(node.children.values(), 
                          key=lambda n: n.uct_score(node.visits))
                search_path.append(node)
            
            # Expansion & Evaluation
            if not node.board.is_game_over():
                self._expand(node)
                if node.children:
                    node = random.choice(list(node.children.values()))
                    search_path.append(node)
            
            value = self._evaluate(node.board)
            
            # Backpropagation
            for n in reversed(search_path):
                n.visits += 1
                n.value_sum += value
                value = -value  # Flip perspective
        
        # Leela Zero: Sample move by visit counts with temperature
        if not root.children:
            return None
        
        children = list(root.children.values())
        visits = np.array([c.visits for c in children])
        
        if self.temperature == 0:
            # Deterministic: pick most visited
            best_idx = np.argmax(visits)
        else:
            # Stochastic: sample proportional to visits^(1/temp)
            probs = visits ** (1.0 / self.temperature)
            probs = probs / probs.sum()
            best_idx = np.random.choice(len(children), p=probs)
        
        return children[best_idx].move
    
    def _expand(self, node):
        if node.board.is_game_over():
            return
        
        legal_moves = list(node.board.legal_moves)
        if not legal_moves:
            return
        
        # Get policy from neural network
        encoding = board_to_encoding(node.board)
        x = torch.FloatTensor([encoding]).to(self.device)
        
        with torch.no_grad():
            policy_logits, _ = self.model(x)
            policy = torch.softmax(policy_logits[0], dim=0).cpu().numpy()
        
        for move in legal_moves:
            try:
                child_board = node.board.copy()
                child_board.push(move)
                child = MCTSNode(child_board, parent=node, move=move)
            except:
                continue
            
            # Set prior from policy network
            move_idx = self._move_to_index(move)
            child.prior = policy[move_idx] + 0.001  # Smoothing
            
            node.children[move] = child
    
    def _evaluate(self, board):
        if board.is_checkmate():
            return -1.0  # Loss for current player
        if board.is_game_over():
            return 0.0  # Draw
        
        encoding = board_to_encoding(board)
        x = torch.FloatTensor([encoding]).to(self.device)
        
        with torch.no_grad():
            _, value = self.model(x)
            return value.item()
    
    def _move_to_index(self, move):
        from_sq = move.from_square
        to_sq = move.to_square
        return min(from_sq * 64 + to_sq, 4671)

def generate_self_play_game(model, device, max_moves=200):
    """Generate one self-play game and return (positions, moves, outcome)"""
    # Leela Zero: High temperature early, low temperature late
    board = chess.Board()
    positions = []
    moves = []
    
    move_count = 0
    while not board.is_game_over() and move_count < max_moves:
        positions.append(board.fen())
        
        # Temperature schedule: explore early, exploit late
        temp = 1.0 if move_count < 30 else 0.1
        mcts = SelfPlayMCTS(model, device, simulations=50, temperature=temp)
        
        move = mcts.search(board)
        if move is None:
            break
        
        moves.append(move.uci())
        board.push(move)
        move_count += 1
    
    # Determine outcome
    if board.is_checkmate():
        outcome = -1.0 if board.turn == chess.WHITE else 1.0
    else:
        outcome = 0.0
    
    return positions, moves, outcome

def generate_training_data(num_games=300):
    """Generate self-play games for training (Leela Zero approach)"""
    print("="*60)
    print(f"LEELA-STYLE SELF-PLAY GENERATION ({num_games} games)")
    print("="*60)
    
    # Load bootstrap model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    model_path = Path("src/models/chess_model.pth")
    encoding_size = len(board_to_encoding(chess.Board()))
    model = ChessNeuralNetwork(encoding_size)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    print(f"Loaded model: {model_path}")
    print(f"\nGenerating {num_games} games (50 MCTS sims/move)...")
    print("Leela Zero features:")
    print("  - Policy network guides MCTS search")
    print("  - Value network evaluates leaf nodes")
    print("  - Temperature annealing (explore -> exploit)")
    print(f"\nEstimated time: 45-90 minutes\n")
    
    all_positions = []
    all_moves = []
    all_outcomes = []
    
    for game_num in range(num_games):
        try:
            positions, moves, outcome = generate_self_play_game(model, device)
            
            for pos, move in zip(positions, moves):
                all_positions.append(pos)
                all_moves.append(move)
                all_outcomes.append(outcome)
            
            if (game_num + 1) % 10 == 0:
                print(f"  Games: {game_num + 1}/{num_games} | "
                      f"Positions: {len(all_positions)} | "
                      f"Avg length: {len(all_positions)/(game_num+1):.1f}")
        except Exception as e:
            print(f"  Game {game_num + 1} failed: {e}")
            continue
    
    # Save training data
    import json
    data = {
        'positions': all_positions,
        'moves': all_moves,
        'outcomes': all_outcomes
    }
    
    output = Path("self_play_data.json")
    with open(output, 'w') as f:
        json.dump(data, f)
    
    print(f"\nSaved {len(all_positions)} training examples to {output}")
    return all_positions, all_moves, all_outcomes

if __name__ == "__main__":
    generate_training_data(num_games=300)
