"""
Data Pipeline Demo: Load Magnus games, AlphaZero games, and puzzles.
Verify the dataset integration and check statistics.
"""
import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from data.pgn_loader import PGNLoader, PGNToFENConverter
from data.puzzle_loader import PuzzleLoader
from data.dataset import DatasetBuilder


def demo_pgn_loading():
    """Demonstrate PGN loading from Magnus and AlphaZero games."""
    print("=" * 60)
    print("DEMO: PGN Loading")
    print("=" * 60)
    
    # Load AlphaZero games
    alphazero_path = "src/resources/alphazero_games.txt"
    print(f"\nLoading AlphaZero games from: {alphazero_path}")
    
    try:
        games = PGNLoader.load_from_file(alphazero_path)
        print(f"✓ Loaded {len(games)} AlphaZero games")
        
        if games:
            first_game = games[0]
            print(f"\nFirst game info:")
            print(f"  Event: {first_game.headers.get('Event', 'N/A')}")
            print(f"  Date: {first_game.headers.get('Date', 'N/A')}")
            print(f"  Result: {first_game.headers.get('Result', 'N/A')}")
            print(f"  Moves: {len(first_game.move_stack)}")
            print(f"  Move sequence: {' '.join(first_game.move_stack[:10])}...")
            
            # Enrich with FEN
            PGNToFENConverter.enrich_game_data(first_game)
            print(f"  FEN-move pairs: {len(first_game.fen_sequence)}")
            if first_game.fen_sequence:
                print(f"  First position: {first_game.fen_sequence[0][0][:40]}...")
                print(f"  First move: {first_game.fen_sequence[0][1]}")
    except FileNotFoundError as e:
        print(f"✗ File not found: {e}")
    except Exception as e:
        print(f"✗ Error: {e}")


def demo_puzzle_loading():
    """Demonstrate puzzle loading."""
    print("\n" + "=" * 60)
    print("DEMO: Puzzle Loading")
    print("=" * 60)
    
    puzzle_path = "src/resources/puzzles.txt"
    print(f"\nLoading puzzles from: {puzzle_path}")
    
    try:
        puzzles = PuzzleLoader.load_from_file(puzzle_path)
        print(f"✓ Loaded {len(puzzles)} puzzles")
        
        if puzzles:
            first_puzzle = puzzles[0]
            print(f"\nFirst puzzle info:")
            print(f"  ID: {first_puzzle.puzzle_id}")
            print(f"  FEN: {first_puzzle.fen}")
            print(f"  Solution moves: {len(first_puzzle.solution_moves)}")
            print(f"  First move: {first_puzzle.solution_moves[0] if first_puzzle.solution_moves else 'N/A'}")
            
            # Convert coordinate moves to UCI
            PuzzleLoader.convert_coordinate_moves_to_uci(first_puzzle)
            print(f"  Converted to UCI: {first_puzzle.solution_moves[:3]}")
            
    except FileNotFoundError as e:
        print(f"✗ File not found: {e}")
    except Exception as e:
        print(f"✗ Error: {e}")


def demo_dataset_builder():
    """Demonstrate building a unified dataset."""
    print("\n" + "=" * 60)
    print("DEMO: Dataset Builder")
    print("=" * 60)
    
    builder = DatasetBuilder()
    
    # Load PGN games
    pgn_files = ["src/resources/alphazero_games.txt", "src/resources/magnus_games.txt"]
    print(f"\nLoading {len(pgn_files)} PGN file(s)...")
    pgn_count = builder.add_pgn_games(pgn_files, weight=1.0)
    
    # Load puzzles
    puzzle_files = ["src/resources/puzzles.txt"]
    print(f"\nLoading {len(puzzle_files)} puzzle file(s)...")
    puzzle_count = builder.add_puzzles(puzzle_files, weight=2.0)
    
    # Compute move probabilities
    print("\nComputing move probability distributions...")
    builder.compute_move_probabilities()
    
    # Print statistics
    stats = builder.get_statistics()
    print("\n" + "=" * 60)
    print("DATASET STATISTICS")
    print("=" * 60)
    print(f"Total training examples: {stats['total_examples']}")
    print(f"  - From games: {stats['game_examples']} ({100 * stats['game_examples'] / max(stats['total_examples'], 1):.1f}%)")
    print(f"  - From puzzles: {stats['puzzle_examples']} ({stats['puzzle_percentage']:.1f}%)")
    print(f"Unique FEN positions: {stats['unique_positions']}")
    
    # Show sample examples
    if builder.training_examples:
        print("\nSample training examples (first 5):")
        for i, example in enumerate(builder.training_examples[:5]):
            print(f"  {i+1}. FEN[:{30}]... -> {example.best_move_uci} ({example.source})")
            print(f"     Prob dist: {len(example.move_probabilities)} moves")


if __name__ == "__main__":
    print("\n" + "🔥 " * 20)
    print("CHESSHACKS BOT - DATA PIPELINE DEMO")
    print("🔥 " * 20 + "\n")
    
    demo_pgn_loading()
    demo_puzzle_loading()
    demo_dataset_builder()
    
    print("\n" + "=" * 60)
    print("DEMO COMPLETE")
    print("=" * 60)
