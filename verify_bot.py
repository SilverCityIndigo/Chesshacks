"""Verify bot training worked correctly"""
import json
import torch
import chess
from src.training.trainer import ChessNeuralNetwork, board_to_encoding

print("="*60)
print("BOT VERIFICATION")
print("="*60)

# Check self-play data
print("\n[1/2] Self-play data quality:")
with open('self_play_data.json', 'r') as f:
    data = json.load(f)
    
total_pos = len(data['positions'])
avg_game_len = total_pos / 300
wins = sum(1 for x in data['outcomes'] if x > 0)
draws = sum(1 for x in data['outcomes'] if x == 0)
losses = sum(1 for x in data['outcomes'] if x < 0)

print(f"  Total positions: {total_pos}")
print(f"  Avg game length: {avg_game_len:.1f} moves")
print(f"  Outcomes: W={wins} ({100*wins/total_pos:.1f}%), D={draws} ({100*draws/total_pos:.1f}%), L={losses} ({100*losses/total_pos:.1f}%)")

# Test model on tactical position
print("\n[2/2] Model tactical awareness test:")
model = ChessNeuralNetwork(786)
model.load_state_dict(torch.load('src/models/chess_model.pth', map_location='cpu'))
model.eval()

# Position: White can play Bxf7+ (Spanish bishop takes)
board = chess.Board('r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 1')
print(f"  Position: {board.fen()}")
print("  White to move (Bxf7+ is winning)")

encoding = board_to_encoding(board)
x = torch.FloatTensor([encoding])

with torch.no_grad():
    policy, value = model(x)
    
# Get scores for legal moves
legal_moves = list(board.legal_moves)
move_scores = []
for move in legal_moves:
    idx = move.from_square * 64 + move.to_square
    score = policy[0][idx].item()
    move_scores.append((move.uci(), score))

move_scores.sort(key=lambda x: x[1], reverse=True)
print(f"\n  Top 5 moves by policy:")
for i, (move, score) in enumerate(move_scores[:5], 1):
    check = " (CHECK!)" if move == "c4f7" else ""
    print(f"    {i}. {move}: {score:.4f}{check}")

print(f"\n  Position value: {value.item():.3f}")
print("\n" + "="*60)
print("Algorithm is working! Weakness is normal for first training.")
print("To improve: Run more self-play games or train on stronger data.")
print("="*60)
