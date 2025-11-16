"""Train bot on tactical puzzles - become a tactics beast!"""
import torch
import chess
import json
from pathlib import Path
from src.training.trainer import ChessNeuralNetwork, board_to_encoding

def load_puzzles(puzzle_file):
    """Load puzzles from JSON file"""
    with open(puzzle_file, 'r') as f:
        return json.load(f)

def train_on_puzzles(puzzle_file="lichess_puzzles.json", epochs=50):
    """
    Train neural network specifically on tactical puzzles
    This makes the bot good at tactics even if strategy is weak
    """
    print("="*60)
    print("TACTICAL PUZZLE TRAINING")
    print("="*60)
    
    # Load puzzles
    print(f"\n[1/4] Loading puzzles from {puzzle_file}...")
    puzzles = load_puzzles(puzzle_file)
    print(f"Loaded {len(puzzles)} puzzles")
    
    # Encode positions and solutions
    print("\n[2/4] Encoding puzzle positions...")
    board_encodings = []
    move_indices = []
    
    for i, puzzle in enumerate(puzzles):
        try:
            board = chess.Board(puzzle['fen'])
            
            # Get the first solution move (the key tactical blow)
            solution_move_uci = puzzle['moves'][0]
            solution_move = chess.Move.from_uci(solution_move_uci)
            
            # Encode position
            encoding = board_to_encoding(board)
            board_encodings.append(encoding)
            
            # Move index
            move_idx = solution_move.from_square * 64 + solution_move.to_square
            move_indices.append(move_idx)
            
            if (i + 1) % 100 == 0:
                print(f"  Processed {i + 1}/{len(puzzles)} puzzles")
        except Exception as e:
            print(f"  Skipping puzzle {i}: {e}")
            continue
    
    print(f"\nEncoded {len(board_encodings)} valid puzzles")
    
    # Train model
    print("\n[3/4] Training neural network on tactics...")
    device = torch.device('cpu')
    
    X = torch.FloatTensor(board_encodings).to(device)
    y = torch.LongTensor(move_indices).to(device)
    
    # Load existing model or create new
    model_path = Path("src/models/chess_model.pth")
    model = ChessNeuralNetwork(X.shape[1])
    
    if model_path.exists():
        print("  Loading existing model to fine-tune...")
        model.load_state_dict(torch.load(model_path, map_location=device))
    
    model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()
    
    batch_size = 32
    num_batches = len(X) // batch_size
    
    for epoch in range(epochs):
        total_loss = 0
        
        # Shuffle
        indices = torch.randperm(len(X))
        X_shuffled = X[indices]
        y_shuffled = y[indices]
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = start_idx + batch_size
            
            X_batch = X_shuffled[start_idx:end_idx]
            y_batch = y_shuffled[start_idx:end_idx]
            
            optimizer.zero_grad()
            
            policy_pred, _ = model(X_batch)
            loss = criterion(policy_pred, y_batch)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / num_batches
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f}")
    
    # Save model
    print("\n[4/4] Saving tactical bot...")
    torch.save(model.state_dict(), model_path)
    print(f"Saved to: {model_path}")
    
    print("\n" + "="*60)
    print("TACTICS TRAINING COMPLETE!")
    print("Your bot should now be much better at tactics!")
    print("="*60)

if __name__ == "__main__":
    # Check if puzzle file exists
    puzzle_file = "lichess_puzzles.json"
    if not Path(puzzle_file).exists():
        print(f"ERROR: {puzzle_file} not found!")
        print("\nRun 'python download_puzzles.py' first")
        print("Or manually download puzzles from:")
        print("  https://database.lichess.org/#puzzles")
    else:
        train_on_puzzles(puzzle_file, epochs=50)
