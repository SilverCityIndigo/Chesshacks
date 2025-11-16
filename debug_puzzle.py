"""Debug puzzle parsing"""
import chess

# Puzzle 1
fen = "r4r1k/6pp/8/3QN3/8/8/5PPP/6K1 w - - 0 1"
board = chess.Board(fen)

print("Board:")
print(board)
print("\nLegal moves:")
for move in board.legal_moves:
    print(f"  {move.uci()}: {board.san(move)}")

# Test the first move from puzzles.txt: 3,4-1,5
from_square = 35  # d5
to_square = 41  # b6

move = chess.Move(from_square, to_square)
print(f"\nTesting move: {chess.square_name(from_square)}{chess.square_name(to_square)}")
print(f"UCI: {move.uci()}")
print(f"Legal: {move in board.legal_moves}")

# What piece is on d5?
print(f"\nPiece on d5: {board.piece_at(35)}")
print(f"Piece on b6: {board.piece_at(41)}")
