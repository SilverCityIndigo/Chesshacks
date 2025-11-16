"""Test PGN loading to see what's happening"""
from src.data.pgn_loader import PGNLoader

print("Loading PGN file...")
games = PGNLoader.load_from_file('src/resources/alphazero_games.txt')

print(f"\n✓ Loaded {len(games)} games\n")

for i, game in enumerate(games, 1):
    event = game.headers.get('Event', 'Unknown')
    white = game.headers.get('White', '?')
    black = game.headers.get('Black', '?')
    result = game.headers.get('Result', '?')
    num_moves = len(game.move_stack)
    
    print(f"{i}. {event}")
    print(f"   {white} vs {black} - {result}")
    print(f"   Moves: {num_moves}")
    print()
