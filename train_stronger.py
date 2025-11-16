"""Train bot to play GOOD chess, not just solve puzzles when losing"""
import torch
import chess
import json
from pathlib import Path
from src.training.trainer import ChessNeuralNetwork, board_to_encoding
import sys

def train_on_both_sides(puzzle_file, epochs=150):
    """
    Train bot on BOTH sides of puzzles - 
    not just the winning tactic, but also how to avoid getting into bad positions
    """
    print("="*60)
    print("TRAINING BOT TO PLAY STRONG CHESS (BOTH SIDES)")
    print("="*60)
    
    # Load Lichess puzzles
    print(f"\nLoading puzzles from {puzzle_file}...")
    puzzles = []
    
    with open(puzzle_file, 'r', encoding='utf-8') as f:
        header = f.readline()
        
        for line_num, line in enumerate(f):
            if len(puzzles) >= 10000:  # Use 10k puzzles
                break
            
            try:
                parts = line.strip().split(',')
                if len(parts) < 4:
                    continue
                
                fen = parts[1]
                moves = parts[2].split()
                rating = int(parts[3])
                
                if 1200 <= rating <= 2200 and len(moves) >= 2:
                    puzzles.append((fen, moves, rating))
                
                if (line_num + 1) % 10000 == 0:
                    print(f"  Scanned {line_num + 1:,} lines, found {len(puzzles)} valid puzzles...")
                    
            except Exception:
                continue
    
    print(f"Loaded {len(puzzles)} puzzles\n")
    
    # Create training data from BOTH winning and losing positions
    print("[1/3] Creating training data from both sides...")
    board_encodings = []
    move_indices = []
    
    for fen, moves, rating in puzzles:
        try:
            board = chess.Board(fen)
            
            # Train on the WINNING move (attacker's perspective)
            winning_move = chess.Move.from_uci(moves[0])
            if winning_move in board.legal_moves:
                encoding = board_to_encoding(board)
                board_encodings.append(encoding)
                move_idx = winning_move.from_square * 64 + winning_move.to_square
                move_indices.append(move_idx)
            
            # ALSO train on position BEFORE the blunder (defender's perspective)
            # Make the losing move, then train bot to find defensive moves
            if len(moves) >= 2:
                board_before = chess.Board(fen)
                board_before.push(winning_move)  # After winning blow
                
                # The response move (often forced or best defense)
                if len(board_before.legal_moves) > 0:
                    defender_move = chess.Move.from_uci(moves[1]) if len(moves) > 1 else list(board_before.legal_moves)[0]
                    if defender_move in board_before.legal_moves:
                        encoding = board_to_encoding(board_before)
                        board_encodings.append(encoding)
                        move_idx = defender_move.from_square * 64 + defender_move.to_square
                        move_indices.append(move_idx)
            
        except Exception:
            continue
    
    print(f"Created {len(board_encodings)} training positions (attack + defense)")
    
    if len(board_encodings) == 0:
        print("ERROR: No training data!")
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
        print("  Fine-tuning existing model...")
        model.load_state_dict(torch.load(model_path, map_location=device))
    
    model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)  # Lower LR for fine-tuning
    criterion = torch.nn.CrossEntropyLoss()
    
    batch_size = 64
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
        if (epoch + 1) % 30 == 0:
            print(f"  Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f}")
    
    # Save
    print("\n[3/3] Saving improved model...")
    torch.save(model.state_dict(), model_path)
    file_size = model_path.stat().st_size / (1024 * 1024)
    print(f"  Saved: {model_path} ({file_size:.1f} MB)")
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print(f"Trained on {len(board_encodings)} positions (both sides)")
    print("Bot should now play STRONGER chess overall")
    print("="*60)

if __name__ == "__main__":
    puzzle_file = r"C:\Users\aquat\Downloads\lichess_db_puzzle.csv\lichess_db_puzzle.csv"
    
    if not Path(puzzle_file).exists():
        print(f"ERROR: Puzzle file not found: {puzzle_file}")
        sys.exit(1)
    
    train_on_both_sides(puzzle_file, epochs=150)
