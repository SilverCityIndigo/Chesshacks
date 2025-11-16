"""Quick retraining script without unicode issues"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import chess
import numpy as np
from pathlib import Path
import json
import os
from src.training.trainer import ChessNeuralNetwork, board_to_encoding
from src.data.pgn_loader import PGNLoader
from src.engines.stockfish_evaluator import StockfishEvaluator


def uci_to_index(uci: str) -> int:
    """Convert UCI move to index (0-4671)."""
    from_square = chess.parse_square(uci[:2])
    to_square = chess.parse_square(uci[2:4])
    return min(from_square * 64 + to_square, 4671)


def load_evaluation_cache():
    cache_file = Path("evals_cache.json")
    if cache_file.exists():
        try:
            with open(cache_file, 'r') as f:
                return json.load(f)
        except:
            return {}
    return {}


def save_evaluation_cache(cache):
    with open("evals_cache.json", 'w') as f:
        json.dump(cache, f, indent=2)


def prepare_training_data_with_evals():
    print("\n" + "="*60)
    print("QUICK TRAINING DATA PREPARATION")
    print("="*60)
    
    print("\n[1/4] Loading games...")
    alphazero_games = PGNLoader.load_from_file("src/resources/AlphaZero.pgn")
    print(f"  AlphaZero games: {len(alphazero_games)}")
    
    print("\n[2/4] Extracting positions...")
    board_encodings = []
    move_indices = []
    fens_for_eval = []
    
    for game in alphazero_games[:100]:  # Quick subset
        board = chess.Board()
        for i, move in enumerate(game.move_stack):
            if i % 3 == 0:  # Sample every 3rd move
                fen = board.fen()
                fens_for_eval.append(fen)
                board_encodings.append(board_to_encoding(board))
                uci_move = board.san_to_uci(move)
                move_indices.append(uci_to_index(uci_move))
            try:
                board.push_san(move)
            except:
                break
    
    print(f"  Extracted {len(board_encodings)} positions")
    
    print("\n[3/4] Evaluating with Stockfish...")
    evaluator = StockfishEvaluator()
    cache = load_evaluation_cache()
    position_values = []
    evaluated = 0
    
    for fen in fens_for_eval:
        if fen in cache:
            centipawns = cache[fen]
        else:
            centipawns = evaluator.evaluate_position(fen, depth=8)
            cache[fen] = centipawns
            evaluated += 1
            if evaluated % 50 == 0:
                print(f"    Evaluated {evaluated}/{len(fens_for_eval)}")
                save_evaluation_cache(cache)
        
        value = max(-1, min(1, centipawns / 1000))
        position_values.append(value)
    
    save_evaluation_cache(cache)
    print(f"  Evaluation complete! ({evaluated} new, {len(cache)} cached)")
    
    print("\n[4/4] Preparing tensors...")
    X = torch.FloatTensor(board_encodings)
    y_moves = torch.LongTensor(move_indices)
    y_values = torch.FloatTensor(position_values)
    
    print(f"  Board encodings: {len(board_encodings)}")
    print(f"  Move labels: {len(move_indices)}")
    print(f"  Position values: {len(position_values)}")
    
    return X, y_moves, y_values


def train_model(epochs=5):
    print("\n" + "="*60)
    print("NEURAL NETWORK TRAINING")
    print("="*60)
    
    X, y_moves, y_values = prepare_training_data_with_evals()
    
    input_size = X.shape[1]
    print(f"\nInput size: {input_size}")
    
    model = ChessNeuralNetwork(input_size)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model.to(device)
    
    policy_criterion = nn.CrossEntropyLoss()
    value_criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    dataset = TensorDataset(X, y_moves, y_values)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    print(f"\nTraining for {epochs} epochs...")
    for epoch in range(epochs):
        model.train()
        total_policy_loss = 0
        total_value_loss = 0
        batches = 0
        
        for batch_X, batch_moves, batch_values in dataloader:
            batch_X = batch_X.to(device)
            batch_moves = batch_moves.to(device)
            batch_values = batch_values.to(device)
            
            optimizer.zero_grad()
            policy_logits, value_pred = model(batch_X)
            
            policy_loss = policy_criterion(policy_logits, batch_moves)
            value_loss = value_criterion(value_pred.squeeze(), batch_values)
            total_loss = policy_loss + 0.5 * value_loss
            
            total_loss.backward()
            optimizer.step()
            
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            batches += 1
        
        avg_policy_loss = total_policy_loss / batches
        avg_value_loss = total_value_loss / batches
        print(f"Epoch {epoch+1}/{epochs} - Policy Loss: {avg_policy_loss:.4f}, Value Loss: {avg_value_loss:.4f}")
    
    print("\n" + "="*60)
    print("SAVING MODEL")
    print("="*60)
    
    output_path = Path("src/models/chess_model_complete.pth")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_path)
    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"Model saved to {output_path} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    epochs = int(os.getenv('EPOCHS', '5'))
    train_model(epochs=epochs)
