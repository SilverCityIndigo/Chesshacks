"""Bootstrap value network from real games with Stockfish labels"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import chess
import numpy as np
from pathlib import Path
import json
import random
from src.training.trainer import ChessNeuralNetwork, board_to_encoding
from src.engines.stockfish_evaluator import StockfishEvaluator

print("="*60)
print("BOOTSTRAP TRAINING FROM REAL GAMES")
print("="*60)

# Parse Magnus games
def parse_games(filepath):
    games = []
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    blocks = [b.strip() for b in content.split('\n\n') if b.strip()]
    for block in blocks:
        moves = []
        for line in block.split('\n'):
            line = line.strip()
            if not line or line.startswith('['):
                continue
            parts = line.replace('.', ' ').split()
            for p in parts:
                p = p.strip()
                if p and not p.isdigit() and p not in ['1-0', '0-1', '1/2-1/2', '*']:
                    moves.append(p)
        if moves:
            games.append(moves)
    return games

print("\n[1/5] Loading games...")
magnus = parse_games("src/resources/magnus_games.txt")
print(f"Loaded {len(magnus)} Magnus games")

print("\n[2/5] Sampling positions...")
positions = []
for game in random.sample(magnus, min(50, len(magnus))):
    board = chess.Board()
    for i, move_san in enumerate(game):
        if i % 5 == 0:  # Every 5th move
            positions.append(board.fen())
        try:
            board.push_san(move_san)
        except:
            break

positions = random.sample(positions, min(500, len(positions)))
print(f"Selected {len(positions)} positions")

print("\n[3/5] Evaluating with Stockfish (depth 10)...")
evaluator = StockfishEvaluator()
board_encodings = []
values = []

for i, fen in enumerate(positions):
    if i % 50 == 0:
        print(f"  Progress: {i}/{len(positions)}")
    
    board = chess.Board(fen)
    encoding = board_to_encoding(board)
    cp, _ = evaluator.evaluate(board)
    if cp is None:
        cp = 0
    value = max(-1.0, min(1.0, cp / 100.0))
    
    board_encodings.append(encoding)
    values.append(value)

print(f"Evaluation complete!")

print("\n[4/5] Training value network...")
X = torch.FloatTensor(board_encodings)
y_val = torch.FloatTensor(values)

model = ChessNeuralNetwork(X.shape[1])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")
model.to(device)

value_criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

dataset = TensorDataset(X, y_val)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

epochs = 20
for epoch in range(epochs):
    model.train()
    total_loss = 0
    batches = 0
    
    for batch_X, batch_val in dataloader:
        batch_X = batch_X.to(device)
        batch_val = batch_val.to(device)
        
        optimizer.zero_grad()
        _, value_pred = model(batch_X)
        loss = value_criterion(value_pred.squeeze(), batch_val)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        batches += 1
    
    if (epoch + 1) % 5 == 0:
        print(f"Epoch {epoch+1}/{epochs} - Value Loss: {total_loss/batches:.4f}")

print("\n[5/5] Saving model...")
output_path = Path("src/models/chess_model.pth")
torch.save(model.state_dict(), output_path)
size_mb = output_path.stat().st_size / (1024*1024)
print(f"Bootstrap model saved: {output_path} ({size_mb:.1f} MB)")
print("\nReady for self-play training!")
