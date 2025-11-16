"""Parse your puzzles.txt format and convert to training data"""
import json
import chess
from pathlib import Path

def parse_coordinate(coord_str):
    """Convert '3,4' to UCI move coordinates"""
    file_idx, rank_idx = map(int, coord_str.split(','))
    return file_idx, rank_idx

def coordinate_to_square(file_idx, rank_idx):
    """Convert (file, rank) indices to square index (0-63)
    Your format: file 0-7 (a-h), rank 0-7 (1-8)
    Chess square numbering: a1=0, h1=7, a8=56, h8=63
    """
    return rank_idx * 8 + file_idx

def parse_puzzles(puzzle_file="src/resources/puzzles.txt"):
    """
    Parse your puzzle format:
    # Puzzle N
    FEN: ...
    from_file,from_rank-to_file,to_rank
    ...
    """
    print("="*60)
    print("PARSING YOUR PUZZLES")
    print("="*60)
    
    with open(puzzle_file, 'r') as f:
        lines = f.readlines()
    
    puzzles = []
    current_puzzle = None
    
    for line in lines:
        line = line.strip()
        
        if not line:
            continue
            
        if line.startswith('# Puzzle'):
            # Save previous puzzle
            if current_puzzle and current_puzzle.get('moves'):
                puzzles.append(current_puzzle)
            
            # Start new puzzle
            current_puzzle = {
                'moves': []
            }
            
        elif line.startswith('FEN:'):
            fen = line.replace('FEN:', '').strip()
            current_puzzle['fen'] = fen
            
        elif '-' in line and ',' in line:
            # This is a move in your coordinate format
            try:
                from_coord, to_coord = line.split('-')
                from_file, from_rank = parse_coordinate(from_coord)
                to_file, to_rank = parse_coordinate(to_coord)
                
                # Convert to square indices
                from_square = coordinate_to_square(from_file, from_rank)
                to_square = coordinate_to_square(to_file, to_rank)
                
                # Create UCI move
                from_sq = chess.square_name(from_square)
                to_sq = chess.square_name(to_square)
                move_uci = f"{from_sq}{to_sq}"
                
                current_puzzle['moves'].append(move_uci)
            except Exception as e:
                print(f"Skipping invalid move format: {line} - {e}")
    
    # Don't forget last puzzle
    if current_puzzle and current_puzzle.get('moves'):
        puzzles.append(current_puzzle)
    
    print(f"\nParsed {len(puzzles)} puzzles successfully!")
    
    # Validate puzzles
    valid_puzzles = []
    for i, puzzle in enumerate(puzzles):
        try:
            board = chess.Board(puzzle['fen'])
            
            # Verify first move is legal
            first_move_uci = puzzle['moves'][0]
            first_move = chess.Move.from_uci(first_move_uci)
            
            if first_move in board.legal_moves:
                puzzle['rating'] = 1500  # Default rating
                puzzle['themes'] = ['tactic']
                valid_puzzles.append(puzzle)
            else:
                print(f"  Puzzle {i+1}: First move {first_move_uci} not legal")
        except Exception as e:
            print(f"  Puzzle {i+1}: Invalid - {e}")
    
    print(f"\n{len(valid_puzzles)} puzzles validated and ready for training!")
    
    # Save to JSON
    output_file = Path("lichess_puzzles.json")
    with open(output_file, 'w') as f:
        json.dump(valid_puzzles, f, indent=2)
    
    print(f"\nSaved to: {output_file}")
    print("="*60)
    
    return valid_puzzles

if __name__ == "__main__":
    puzzles = parse_puzzles()
    
    # Show sample
    if puzzles:
        print("\nSample puzzle:")
        print(f"FEN: {puzzles[0]['fen']}")
        print(f"Solution: {' -> '.join(puzzles[0]['moves'][:3])}")
