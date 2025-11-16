"""
Simple script to train on Lichess puzzles

USAGE:
1. Download puzzles: https://database.lichess.org/lichess_db_puzzle.csv.bz2
2. Extract the CSV file
3. Run: python train_on_lichess.py lichess_db_puzzle.csv

OR use the sample puzzles I'll create
"""

import json
import csv
import chess
import torch
from pathlib import Path
from src.training.trainer import ChessNeuralNetwork, board_to_encoding

# Sample high-quality tactical puzzles
SAMPLE_PUZZLES = [
    # Scholar's Mate pattern
    ("rnbqkb1r/pppp1ppp/5n2/4p3/2B1P3/8/PPPP1PPP/RNBQK1NR w KQkq -", "c4f7", 1200),
    # Back rank mate
    ("6k1/5ppp/8/8/8/8/5PPP/R5K1 w - -", "a1a8", 1300),
    # Fork with knight
    ("r2qkb1r/ppp2ppp/2n2n2/3pp3/2B1P1b1/2NP1N2/PPP2PPP/R1BQK2R w KQkq -", "c4f7", 1500),
    # Pin and win
    ("r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R b KQkq -", "c8g4", 1400),
    # Discovered attack
    ("r1bqkb1r/pppp1ppp/2n2n2/4p3/2BPP3/5N2/PPP2PPP/RNBQK2R b KQkq -", "f6e4", 1600),
    # Remove defender
    ("r2qkb1r/ppp2ppp/2n5/3pN3/3Pn1b1/2P5/PP3PPP/RNBQKB1R w KQkq -", "e5f7", 1700),
    # Skewer
    ("6k1/5ppp/8/8/8/2Q5/5PPP/6K1 w - -", "c3c8", 1400),
    # Mate in 1 - back rank
    ("6k1/5ppp/8/8/8/8/5PPP/5RK1 w - -", "f1f8", 1100),
    # Mate in 1 - queen
    ("r4rk1/5ppp/8/8/8/4Q3/5PPP/6K1 w - -", "e3e8", 1200),
    # Mate in 1 - smothered
    ("6rk/5Npp/8/8/8/8/5PPP/6K1 w - -", "f7h6", 1300),
]

def load_or_create_puzzles(csv_file=None, num_puzzles=1000, min_rating=1200, max_rating=2000):
    """Load puzzles from Lichess CSV or use samples"""
    
    if csv_file and Path(csv_file).exists():
        print(f"Loading puzzles from {csv_file}...")
        print(f"Target: {num_puzzles} puzzles (rating {min_rating}-{max_rating})")
        puzzles = []
        
        with open(csv_file, 'r', encoding='utf-8') as f:
            # Skip header
            header = f.readline()
            
            for line_num, line in enumerate(f):
                if len(puzzles) >= num_puzzles:
                    break
                
                try:
                    parts = line.strip().split(',')
                    if len(parts) < 4:
                        continue
                    
                    # CSV format: PuzzleId,FEN,Moves,Rating,RatingDeviation,Popularity,NbPlays,Themes,GameUrl
                    fen = parts[1]
                    moves = parts[2].split()
                    rating = int(parts[3])
                    
                    if min_rating <= rating <= max_rating and len(moves) >= 1:
                        # First move is the solution
                        puzzles.append((fen, moves[0], rating))
                    
                    if (line_num + 1) % 10000 == 0:
                        print(f"  Scanned {line_num + 1:,} lines, found {len(puzzles)} valid puzzles...")
                        
                except Exception as e:
                    continue
        
        print(f"Loaded {len(puzzles)} puzzles from CSV")
    else:
        print("Using sample tactical puzzles...")
        puzzles = SAMPLE_PUZZLES
        print(f"Loaded {len(puzzles)} sample puzzles")
    
    return puzzles

def train_on_puzzles(puzzles, epochs=100):
    """Train bot to solve tactical puzzles"""
    
    print("\n" + "="*60)
    print("TACTICAL TRAINING - Make your bot a tactics beast!")
    print("="*60)
    
    # Convert puzzles to training data
    print("\n[1/3] Processing puzzles...")
    board_encodings = []
    move_indices = []
    
    for fen, solution_uci, rating in puzzles:
        try:
            board = chess.Board(fen)
            solution_move = chess.Move.from_uci(solution_uci)
            
            if solution_move not in board.legal_moves:
                continue
            
            # Encode position
            encoding = board_to_encoding(board)
            board_encodings.append(encoding)
            
            # Target move
            move_idx = solution_move.from_square * 64 + solution_move.to_square
            move_indices.append(move_idx)
            
        except Exception as e:
            print(f"  Skipping puzzle: {e}")
            continue
    
    print(f"  Processed {len(board_encodings)} valid puzzles")
    
    if len(board_encodings) == 0:
        print("ERROR: No valid puzzles found!")
        return
    
    # Train model
    print(f"\n[2/3] Training for {epochs} epochs...")
    device = torch.device('cpu')
    
    X = torch.FloatTensor(board_encodings).to(device)
    y = torch.LongTensor(move_indices).to(device)
    
    # Load existing model
    model_path = Path("src/models/chess_model.pth")
    model = ChessNeuralNetwork(X.shape[1])
    
    if model_path.exists():
        print("  Fine-tuning existing model on tactics...")
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        print("  Training new model from scratch...")
    
    model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()
    
    batch_size = min(32, len(X))
    num_batches = max(1, len(X) // batch_size)
    
    for epoch in range(epochs):
        total_loss = 0
        
        # Shuffle data
        indices = torch.randperm(len(X))
        X_shuffled = X[indices]
        y_shuffled = y[indices]
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(X))
            
            X_batch = X_shuffled[start_idx:end_idx]
            y_batch = y_shuffled[start_idx:end_idx]
            
            optimizer.zero_grad()
            
            policy_pred, _ = model(X_batch)
            loss = criterion(policy_pred, y_batch)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / num_batches
        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f}")
    
    # Save model
    print("\n[3/3] Saving tactical bot...")
    torch.save(model.state_dict(), model_path)
    file_size = model_path.stat().st_size / (1024 * 1024)
    print(f"  Saved: {model_path} ({file_size:.1f} MB)")
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print(f"Bot trained on {len(board_encodings)} tactical positions")
    print("Your bot should now be much better at tactics!")
    print("\nNext: Restart serve.py and test at localhost:3000")
    print("="*60)

if __name__ == "__main__":
    import sys
    
    # Check for CSV file argument
    csv_file = sys.argv[1] if len(sys.argv) > 1 else None
    
    # Load puzzles
    puzzles = load_or_create_puzzles(
        csv_file=csv_file,
        num_puzzles=5000,
        min_rating=1200,
        max_rating=2000
    )
    
    # Train
    train_on_puzzles(puzzles, epochs=100)
