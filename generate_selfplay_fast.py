"""Fast self-play without MCTS - just use neural network policy"""
import torch
import chess
import json
from pathlib import Path
from src.training.trainer import ChessNeuralNetwork, board_to_encoding

def generate_fast_game(model, device, max_moves=150):
    """Play one game using just the neural network (no MCTS)"""
    board = chess.Board()
    positions = []
    moves = []
    
    move_count = 0
    while not board.is_game_over() and move_count < max_moves:
        positions.append(board.fen())
        
        # Get policy from neural network
        encoding = board_to_encoding(board)
        x = torch.FloatTensor([encoding]).to(device)
        
        with torch.no_grad():
            policy_logits, _ = model(x)
            policy = torch.softmax(policy_logits[0], dim=0).cpu().numpy()
        
        # Pick best legal move according to policy
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            break
        
        # Score each legal move
        move_scores = []
        for move in legal_moves:
            from_sq = move.from_square
            to_sq = move.to_square
            move_idx = min(from_sq * 64 + to_sq, 4671)
            move_scores.append((move, policy[move_idx]))
        
        # Add exploration: temperature sampling
        temp = 0.8 if move_count < 20 else 0.3
        import numpy as np
        scores = np.array([s for _, s in move_scores])
        scores = scores ** (1.0 / temp)
        probs = scores / scores.sum()
        
        chosen_idx = np.random.choice(len(legal_moves), p=probs)
        chosen_move = legal_moves[chosen_idx]
        
        moves.append(chosen_move.uci())
        board.push(chosen_move)
        move_count += 1
    
    # Outcome
    if board.is_checkmate():
        outcome = -1.0 if board.turn == chess.WHITE else 1.0
    else:
        outcome = 0.0
    
    return positions, moves, outcome

def main():
    print("=" * 60)
    print("FAST SELF-PLAY (Neural Network Only - No MCTS)")
    print("=" * 60)
    
    device = torch.device('cpu')
    model_path = Path("src/models/chess_model.pth")
    
    encoding_size = len(board_to_encoding(chess.Board()))
    model = ChessNeuralNetwork(encoding_size)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    print(f"Loaded: {model_path}")
    print("Generating 300 games (2-3 seconds per game)")
    print("Estimated time: 10-15 minutes\n")
    
    all_positions = []
    all_moves = []
    all_outcomes = []
    
    num_games = 300
    for game_num in range(num_games):
        try:
            positions, moves, outcome = generate_fast_game(model, device)
            
            for pos, move in zip(positions, moves):
                all_positions.append(pos)
                all_moves.append(move)
                all_outcomes.append(outcome)
            
            if (game_num + 1) % 20 == 0:
                print(f"Games: {game_num + 1}/{num_games} | "
                      f"Positions: {len(all_positions)} | "
                      f"Avg length: {len(all_positions)/(game_num+1):.1f}")
        except Exception as e:
            print(f"Game {game_num + 1} failed: {e}")
            continue
    
    # Save
    data = {
        'positions': all_positions,
        'moves': all_moves,
        'outcomes': all_outcomes
    }
    
    output = Path("self_play_data.json")
    with open(output, 'w') as f:
        json.dump(data, f)
    
    print(f"\nDONE! Saved {len(all_positions)} positions to {output}")

if __name__ == "__main__":
    main()
