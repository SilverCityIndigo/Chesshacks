"""
Retrain chess model with PROPER move encoding.
The original trainer had a fatal bug - it trained on legal move indices,
but inference tried to use absolute UCI indices. This fixes that.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import chess
import numpy as np
from pathlib import Path
from src.training.trainer import ChessNeuralNetwork, board_to_encoding
from src.data.pgn_loader import PGNLoader, PGNToFENConverter
from src.data.puzzle_loader import PuzzleLoader


def uci_to_index(uci: str) -> int:
    """
    Convert UCI move to absolute index (0-4671).
    This matches what the model expects during inference.
    """
    from_square = chess.parse_square(uci[:2])
    to_square = chess.parse_square(uci[2:4])
    
    # Simple encoding: from_square * 64 + to_square
    # This gives us 64*64 = 4096 possible moves
    move_idx = from_square * 64 + to_square
    return min(move_idx, 4671)  # Clamp to model output size


def prepare_training_data():
    """Load and prepare training data with CORRECT move encoding."""
    print("Loading training data...")
    
    # Load games
    pgn_file = Path("src/resources/alphazero_games.txt")
    games = PGNLoader.load_from_file(str(pgn_file)) if pgn_file.exists() else []
    print(f"Loaded {len(games)} games")
    
    # Load puzzles  
    puzzle_file = Path("src/resources/puzzles.json")
    puzzles = PuzzleLoader.load_from_file(str(puzzle_file)) if puzzle_file.exists() else []
    print(f"Loaded {len(puzzles)} puzzles")
    
    # Prepare training examples
    board_encodings = []
    move_indices = []
    
    # Process games
    for game in games:
        board = chess.Board()
        for move_san in game.move_stack:
            try:
                # Encode current position
                encoding = board_to_encoding(board)
                board_encodings.append(encoding)
                
                # Parse move and get index
                move = board.parse_san(move_san)
                move_idx = uci_to_index(move.uci())
                move_indices.append(move_idx)
                
                # Apply move
                board.push(move)
            except Exception as e:
                print(f"Error processing move {move_san}: {e}")
                continue
    
    print(f"Added {len(board_encodings)} examples from games")
    
    # Process puzzles
    for puzzle in puzzles:
        board = chess.Board(puzzle.fen)
        for move_uci in puzzle.solution:
            try:
                # Encode position
                encoding = board_to_encoding(board)
                board_encodings.append(encoding)
                
                # Get move index
                move_idx = uci_to_index(move_uci)
                move_indices.append(move_idx)
                
                # Apply move
                move = chess.Move.from_uci(move_uci)
                board.push(move)
            except Exception as e:
                print(f"Error processing puzzle move {move_uci}: {e}")
                continue
    
    print(f"Total training examples: {len(board_encodings)}")
    
    return np.array(board_encodings), np.array(move_indices)


def train_model(epochs=5):
    """Train model with correct move encoding."""
    # Load data
    X, y = prepare_training_data()
    
    # Convert to tensors
    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.LongTensor(y)
    
    # Create dataset
    dataset = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Initialize model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = ChessNeuralNetwork(num_moves=4672).to(device)
    
    # Try to load existing model as starting point
    model_path = Path("src/models/chess_model.pth")
    if model_path.exists():
        print(f"Loading existing model from {model_path}")
        try:
            model.load_state_dict(torch.load(model_path, map_location=device))
        except Exception as e:
            print(f"Could not load existing model: {e}")
            print("Starting from scratch")
    
    # Optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    print(f"\nTraining for {epochs} epochs...")
    model.train()
    
    for epoch in range(epochs):
        total_loss = 0
        num_batches = 0
        
        for batch_X, batch_y in dataloader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            
            # Forward pass
            features = model.feature_layers(batch_X)
            policy_logits = model.policy_head(features)
            
            # Remove softmax from model output for CrossEntropyLoss
            # CrossEntropyLoss expects raw logits
            # We need to get the layer before softmax
            policy_logits = features
            policy_logits = model.policy_head[0](policy_logits)  # First linear layer
            policy_logits = model.policy_head[1](policy_logits)  # ReLU
            policy_logits = model.policy_head[2](policy_logits)  # Second linear layer (logits)
            
            # Calculate loss
            loss = criterion(policy_logits, batch_y)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch + 1}/{epochs}: Loss = {avg_loss:.4f}")
    
    # Save model
    output_path = Path("src/models/chess_model.pth")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_path)
    
    # Get file size
    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"\nModel saved to {output_path} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    train_model(epochs=5)
