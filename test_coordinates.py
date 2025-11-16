"""Figure out your coordinate format"""
import chess

# Puzzle 1
fen = "r4r1k/6pp/8/3QN3/8/8/5PPP/6K1 w - - 0 1"
board = chess.Board(fen)
first_move_str = "3,4-1,5"

print("BOARD:")
print(board)
print()

def try_interpretation(name, from_sq, to_sq):
    """Test a coordinate interpretation"""
    move = chess.Move(from_sq, to_sq)
    legal = move in board.legal_moves
    piece = board.piece_at(from_sq)
    print(f"{name}:")
    print(f"  From {chess.square_name(from_sq)} (piece: {piece})")
    print(f"  To {chess.square_name(to_sq)}")
    print(f"  UCI: {move.uci()}")
    print(f"  Legal: {legal}")
    if legal:
        print(f"  ** THIS IS THE ANSWER! **")
    print()
    return legal

# Try different interpretations of "3,4-1,5"
col1, row1 = 3, 4
col2, row2 = 1, 5

print(f"TESTING: {first_move_str}")
print()

# Interpretation 1: (file, rank) where file=0-7 for a-h, rank=0-7 for 1-8
try_interpretation(
    "1. (file, rank) from bottom",
    col1 + row1*8,  # file=3 (d-file), rank=4 (rank 5)
    col2 + row2*8
)

# Interpretation 2: (col, row) from top-left, row=0 is rank 8
try_interpretation(
    "2. (col, row) from top",
    col1 + (7-row1)*8,
    col2 + (7-row2)*8
)

# Interpretation 3: (rank, file)
try_interpretation(
    "3. (rank, file) swapped",
    row1 + col1*8,
    row2 + col2*8
)

# Interpretation 4: (row from top, col from left)
try_interpretation(
    "4. (row, col) from top",
    col1 + (7-row1)*8,
    col2 + (7-row2)*8
)

print("\nWINNING MOVES:")
for move in board.legal_moves:
    if '+' in board.san(move) or '#' in board.san(move):
        print(f"  {move.uci()}: {board.san(move)}")
