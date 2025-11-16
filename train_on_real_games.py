"""Train on real master games with Stockfish evaluation - YOUR ORIGINAL IDEA"""
import torch
import chess
import chess.pgn
from pathlib import Path
import io
import random
from src.training.trainer import ChessNeuralNetwork, board_to_encoding
from src.engines.stockfish_evaluator import StockfishEvaluator

def load_games_from_file(filepath):
    """Load games from text file"""
    games = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
            # Split on double newline or game separators
            game_texts = content.split('\n\n')
            for game_text in game_texts:
                if '[Event' in game_text or '1.' in game_text:
                    try:
                        pgn = io.StringIO(game_text)
                        game = chess.pgn.read_game(pgn)
                        if game:
                            games.append(game)
                    except:
                        continue
    except Exception as e:
        print(f"  Error loading {filepath}: {e}")
    return games

print("="*60)
print("TRAINING ON REAL MASTER GAMES (Your Original Idea!)")
print("="*60)

# Load all game files
game_files = [
    'src/resources/magnus_games.txt',
    'src/resources/alphazero_games.txt',
    'src/resources/TCECGames.txt',
    'src/resources/puzzles.txt'
]

all_games = []
for filepath in game_files:
    path = Path(filepath)
    if path.exists():
        print(f"\nLoading {path.name}...")
        games = load_games_from_file(path)
        print(f"  Loaded {len(games)} games")
        all_games.extend(games)

print(f"\nTotal games loaded: {len(all_games)}")

# Sample positions from games
print("\nSampling positions from games...")
positions = []
for game in random.sample(all_games, min(500, len(all_games))):
    board = game.board()
    moves = list(game.mainline_moves())
    
    # Sample 3-5 positions per game
    for i in range(0, len(moves), max(1, len(moves)//4)):
        if len(positions) >= 2000:
            break
        board.push(moves[i])
        positions.append(board.fen())
        
    if len(positions) >= 2000:
        break

print(f"Sampled {len(positions)} positions")

# Evaluate with Stockfish
print("\nEvaluating positions with Stockfish (depth 10)...")
stockfish_path = r"C:\Users\aquat\Downloads\stockfish-windows-x86-64-avx2 (3)\stockfish\stockfish-windows-x86-64-avx2.exe"
evaluator = StockfishEvaluator(stockfish_path)

board_encodings = []
move_indices = []
values = []

for i, fen in enumerate(positions):
    if i % 100 == 0:
        print(f"  Progress: {i}/{len(positions)}")
    
    board = chess.Board(fen)
    
    # Get Stockfish evaluation
    try:
        cp_score, best_move = evaluator.evaluate(board, depth=10)
        
        # Convert centipawns to value [-1, 1]
        value = max(-1.0, min(1.0, cp_score / 500.0))
        
        # Encode position
        encoding = board_to_encoding(board)
        
        # Move index
        if best_move:
            move_idx = best_move.from_square * 64 + best_move.to_square
        else:
            move_idx = 0
        
        board_encodings.append(encoding)
        move_indices.append(move_idx)
        values.append(value)
    except:
        continue

evaluator.close()

print(f"\nEvaluated {len(board_encodings)} positions")

# Train model
print("\nTraining neural network...")
device = torch.device('cpu')

X = torch.FloatTensor(board_encodings).to(device)
y_policy = torch.LongTensor(move_indices).to(device)
y_value = torch.FloatTensor(values).unsqueeze(1).to(device)

model = ChessNeuralNetwork(X.shape[1])
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
policy_criterion = torch.nn.CrossEntropyLoss()
value_criterion = torch.nn.MSELoss()

epochs = 20
batch_size = 64
num_batches = len(X) // batch_size

for epoch in range(epochs):
    total_policy_loss = 0
    total_value_loss = 0
    
    # Shuffle
    indices = torch.randperm(len(X))
    X_shuffled = X[indices]
    y_policy_shuffled = y_policy[indices]
    y_value_shuffled = y_value[indices]
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = start_idx + batch_size
        
        X_batch = X_shuffled[start_idx:end_idx]
        y_policy_batch = y_policy_shuffled[start_idx:end_idx]
        y_value_batch = y_value_shuffled[start_idx:end_idx]
        
        optimizer.zero_grad()
        
        policy_pred, value_pred = model(X_batch)
        
        policy_loss = policy_criterion(policy_pred, y_policy_batch)
        value_loss = value_criterion(value_pred, y_value_batch)
        
        loss = policy_loss + value_loss
        loss.backward()
        optimizer.step()
        
        total_policy_loss += policy_loss.item()
        total_value_loss += value_loss.item()
    
    avg_policy = total_policy_loss / num_batches
    avg_value = total_value_loss / num_batches
    print(f"Epoch {epoch+1}/{epochs} - Policy: {avg_policy:.4f}, Value: {avg_value:.4f}")

# Save model
model_path = Path("src/models/chess_model.pth")
torch.save(model.state_dict(), model_path)
print(f"\nModel saved: {model_path}")

print("\n" + "="*60)
print("TRAINING COMPLETE - Real master game knowledge learned!")
print("="*60)
