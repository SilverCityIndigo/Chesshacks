"""
Simple retraining script - retrains model on existing dataset.
"""

import sys
import torch
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.training.trainer import ChessNeuralNetwork, ChessTrainer, board_to_encoding
from src.data.dataset import DatasetBuilder
import chess


def main():
    print("\n" + "="*60)
    print("RETRAINING CHESS MODEL")
    print("="*60)
    
    # Load training data
    print("\nLoading training data...")
    builder = DatasetBuilder()
    
    resources_path = Path("src/resources")
    if resources_path.exists():
        games_file = resources_path / "alphazero_games.txt"
        puzzles_file = resources_path / "puzzles.txt"
        
        if games_file.exists():
            print(f"  Loading games...")
            builder.add_pgn_games([str(games_file)])  # Pass as list!
        
        if puzzles_file.exists():
            print(f"  Loading puzzles...")
            builder.add_puzzles([str(puzzles_file)])  # Pass as list!
    
    training_examples = builder.get_training_examples()
    stats = builder.get_statistics()
    
    print(f"\nLoaded {len(training_examples)} examples")
    print(f"  Games: {stats.get('games', 0)}")
    print(f"  Puzzles: {stats.get('puzzles', 0)}")
    print(f"  Unique positions: {stats.get('unique_positions', 0)}")
    
    if not training_examples:
        print("\nNo training data found. Exiting.")
        return
    
    # Prepare tensors
    print("\nPreparing training data...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    boards = []
    moves = []
    
    for ex in training_examples:
        try:
            board = chess.Board(ex.fen)
            encoding = board_to_encoding(board)
            boards.append(encoding)
            
            # Get move index
            from_sq = (ord(ex.best_move_uci[0]) - ord('a')) + (int(ex.best_move_uci[1]) - 1) * 8
            to_sq = (ord(ex.best_move_uci[2]) - ord('a')) + (int(ex.best_move_uci[3]) - 1) * 8
            move_idx = from_sq * 64 + to_sq
            moves.append(min(move_idx, 4671))
        except:
            pass
    
    boards_tensor = torch.FloatTensor(np.array(boards)).to(device)
    moves_tensor = torch.LongTensor(moves).to(device)
    
    print(f"Prepared {len(boards)} training examples")
    
    # Initialize model
    print("\nInitializing model...")
    model = ChessNeuralNetwork()
    model.to(device)
    
    # Load existing model if available
    model_path = Path("src/models/chess_model.pth")
    if model_path.exists():
        print(f"Loading existing model from {model_path}")
        try:
            model.load_state_dict(torch.load(model_path, map_location=device))
        except Exception as e:
            print(f"Warning: {e}")
    
    # Train
    print("\nTraining model...")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = torch.nn.CrossEntropyLoss()
    
    epochs = 5
    batch_size = 32
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        # Shuffle
        indices = torch.randperm(len(boards_tensor))
        boards_shuffled = boards_tensor[indices]
        moves_shuffled = moves_tensor[indices]
        
        for i in range(0, len(boards_tensor), batch_size):
            batch_boards = boards_shuffled[i:i+batch_size]
            batch_moves = moves_shuffled[i:i+batch_size]
            
            # Forward
            policy_out, value_out = model(batch_boards)
            loss = loss_fn(policy_out, batch_moves)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / (len(boards_tensor) // batch_size)
        print(f"Epoch {epoch+1}/{epochs}: Loss = {avg_loss:.4f}")
    
    # Save model
    print("\nSaving model...")
    output_path = Path("src/models/chess_model.pth")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_path)
    
    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"[OK] Model saved to {output_path} ({file_size_mb:.1f} MB)")
    print(f"[OK] Retraining complete!")
    print(f"\nTo test:")
    print(f"  1. Restart backend: python serve.py")
    print(f"  2. Start frontend: cd devtools && npm run dev")
    print(f"  3. Visit http://localhost:3000")


if __name__ == "__main__":
    main()
