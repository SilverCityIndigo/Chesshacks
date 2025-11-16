"""Retrain on self-play data with policy + value loss"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import chess
import json
from pathlib import Path
from src.training.trainer import ChessNeuralNetwork, board_to_encoding

def uci_to_index(uci):
    move = chess.Move.from_uci(uci)
    return min(move.from_square * 64 + move.to_square, 4671)

print("="*60)
print("SELF-PLAY RETRAINING")
print("="*60)

print("\n[1/4] Loading self-play data...")
with open("self_play_data.json", 'r') as f:
    data = json.load(f)

positions = data['positions']
moves = data['moves']
outcomes = data['outcomes']

print(f"Loaded {len(positions)} training examples")

print("\n[2/4] Encoding positions...")
board_encodings = []
move_indices = []
values = []

for i, (fen, uci, outcome) in enumerate(zip(positions, moves, outcomes)):
    if i % 1000 == 0:
        print(f"  Progress: {i}/{len(positions)}")
    
    try:
        board = chess.Board(fen)
        encoding = board_to_encoding(board)
        move_idx = uci_to_index(uci)
        
        board_encodings.append(encoding)
        move_indices.append(move_idx)
        values.append(outcome)
    except:
        continue

print(f"Encoded {len(board_encodings)} positions")

print("\n[3/4] Training neural network...")
X = torch.FloatTensor(board_encodings)
y_policy = torch.LongTensor(move_indices)
y_value = torch.FloatTensor(values)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

model = ChessNeuralNetwork(X.shape[1])
model.to(device)

policy_criterion = nn.CrossEntropyLoss()
value_criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

dataset = TensorDataset(X, y_policy, y_value)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

epochs = 10
for epoch in range(epochs):
    model.train()
    total_policy_loss = 0
    total_value_loss = 0
    batches = 0
    
    for batch_X, batch_policy, batch_value in dataloader:
        batch_X = batch_X.to(device)
        batch_policy = batch_policy.to(device)
        batch_value = batch_value.to(device)
        
        optimizer.zero_grad()
        policy_logits, value_pred = model(batch_X)
        
        policy_loss = policy_criterion(policy_logits, batch_policy)
        value_loss = value_criterion(value_pred.squeeze(), batch_value)
        total_loss = policy_loss + value_loss
        
        total_loss.backward()
        optimizer.step()
        
        total_policy_loss += policy_loss.item()
        total_value_loss += value_loss.item()
        batches += 1
    
    avg_policy = total_policy_loss / batches
    avg_value = total_value_loss / batches
    print(f"Epoch {epoch+1}/{epochs} - Policy: {avg_policy:.4f}, Value: {avg_value:.4f}")

print("\n[4/4] Saving final model...")
output_path = Path("src/models/chess_model.pth")
torch.save(model.state_dict(), output_path)
size_mb = output_path.stat().st_size / (1024*1024)
print(f"Model saved: {output_path} ({size_mb:.1f} MB)")
print("\n" + "="*60)
print("TRAINING COMPLETE - Bot ready for deployment!")
print("="*60)
