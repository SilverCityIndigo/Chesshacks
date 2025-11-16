"""Quick test to see what the model is actually predicting."""
import chess
from src.training.trainer import ChessNeuralNetwork, board_to_encoding
import torch
import numpy as np

# Load model
model = ChessNeuralNetwork()
model.load_state_dict(torch.load('src/models/chess_model.pth', map_location='cpu'))
model.eval()

# Test on starting position
board = chess.Board()
encoding = board_to_encoding(board)
tensor = torch.FloatTensor(encoding).unsqueeze(0)

# Get predictions
features = model.feature_layers(tensor)
policy_output = model.policy_head(features)[0]

print("Starting position - Testing model predictions:\n")
print(f"Policy output shape: {policy_output.shape}")
print(f"Sum of probabilities: {policy_output.sum().item():.6f}")
print(f"Max probability: {policy_output.max().item():.6f}")
print(f"Min probability: {policy_output.min().item():.6f}")

# Check legal moves
legal_moves = list(board.legal_moves)
print(f"\nNumber of legal moves: {len(legal_moves)}")

# Get probabilities for legal moves
move_probs = []
for move in legal_moves:
    from_sq = move.from_square
    to_sq = move.to_square
    idx = from_sq * 64 + to_sq
    prob = policy_output[idx].item()
    move_probs.append((move.uci(), prob))

# Sort by probability
move_probs.sort(key=lambda x: x[1], reverse=True)

print("\nTop 10 legal moves by model probability:")
for i, (move_uci, prob) in enumerate(move_probs[:10], 1):
    print(f"{i}. {move_uci}: {prob:.8f}")

print("\nBottom 5 legal moves:")
for i, (move_uci, prob) in enumerate(move_probs[-5:], 1):
    print(f"{i}. {move_uci}: {prob:.8f}")

# Check what move gets selected
legal_probs = [p for _, p in move_probs]
total = sum(legal_probs)
print(f"\nSum of legal move probabilities: {total:.8f}")

if total > 0:
    normalized_probs = [(m, p/total) for m, p in move_probs]
    print("\nTop 5 after normalizing to legal moves only:")
    for i, (move_uci, prob) in enumerate(normalized_probs[:5], 1):
        print(f"{i}. {move_uci}: {prob:.6f} ({prob*100:.2f}%)")
