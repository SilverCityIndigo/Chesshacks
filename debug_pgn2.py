#!/usr/bin/env python
import sys
import re
sys.path.insert(0, 'src')

from data.pgn_loader import PGNLoader

# Debug loading
with open('src/resources/alphazero_games', 'r', encoding='utf-8') as f:
    content = f.read()

print(f"File length: {len(content)} bytes")
print(f"First 50 chars: {repr(content[:50])}")

# Try the split methods
blocks1 = re.split(r'\n\n(?=\[Event)', content)
print(f"\nStandard split result: {len(blocks1)} blocks")

blocks2 = re.split(r'\nN\s*\n', content)
print(f"N separator split result: {len(blocks2)} blocks")

# Test actual loading
games = PGNLoader.load_from_file('src/resources/alphazero_games')
print(f"\nLoaded games: {len(games)}")

if games:
    for i, game in enumerate(games[:2]):
        print(f"\nGame {i+1}:")
        print(f"  Result: {game.headers.get('Result')}")
        print(f"  Moves: {len(game.move_stack)}")
        print(f"  First 5 moves: {game.move_stack[:5]}")
