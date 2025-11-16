"""Debug script to see what the model is actually predicting"""
import chess
import torch
from src.training.trainer import ChessNeuralNetwork, board_to_encoding

# Load model
model = ChessNeuralNetwork(num_moves=4672)
model.load_state_dict(torch.load('src/models/chess_model.pth', map_location='cpu'))
model.eval()

# Test position: Starting position after 1.e4
board = chess.Board()
board.push_san('e4')

print("Position after 1.e4:")
print(board)
print("\nLegal moves:", [m.uci() for m in board.legal_moves])

# Get model predictions
encoding = board_to_encoding(board)
board_tensor = torch.FloatTensor(encoding).unsqueeze(0)

with torch.no_grad():
    features = model.feature_layers(board_tensor)
    policy_logits = model.policy_head(features)[0]  # [4672]
    
    # Get probabilities for ALL legal moves
    move_probs = {}
    for move in board.legal_moves:
        move_idx = move.from_square * 64 + move.to_square
        prob = torch.softmax(policy_logits, dim=0)[move_idx].item()
        move_probs[move.uci()] = prob
    
    # Sort by probability
    sorted_moves = sorted(move_probs.items(), key=lambda x: x[1], reverse=True)
    
    print("\n" + "="*60)
    print("MODEL PREDICTIONS (Top 10):")
    print("="*60)
    for uci, prob in sorted_moves[:10]:
        print(f"{uci:6s} -> {prob*100:6.3f}%")
    
    print("\n" + "="*60)
    print("ANALYSIS:")
    print("="*60)
    
    # Check if top move makes sense
    top_move = sorted_moves[0][0]
    top_prob = sorted_moves[0][1]
    
    print(f"Top pick: {top_move} ({top_prob*100:.3f}%)")
    
    # Check spread
    total_top5 = sum(p for _, p in sorted_moves[:5])
    print(f"Top 5 moves total: {total_top5*100:.2f}%")
    
    # Check if distribution is too flat (sign of untrained model)
    if top_prob < 0.05:
        print("\n⚠️ WARNING: Probability too low! Model is essentially random.")
        print("   This means training didn't work properly.")
    
    # Check good moves
    good_moves = ['e7e5', 'c7c5', 'e7e6', 'c7c6']  # Normal responses to e4
    print(f"\nProbabilities for standard responses:")
    for gm in good_moves:
        if gm in move_probs:
            print(f"  {gm}: {move_probs[gm]*100:.3f}%")
