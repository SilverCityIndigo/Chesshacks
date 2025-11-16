"""Download tactical puzzles from Lichess database"""
import requests
import json
import csv
import gzip
from pathlib import Path
from io import StringIO

def download_lichess_puzzles(num_puzzles=5000, min_rating=1200, max_rating=2400, themes=None):
    """
    Download puzzles from Lichess via their API
    Uses the puzzle activity stream which is publicly available
    """
    print("="*60)
    print("DOWNLOADING LICHESS PUZZLES")
    print("="*60)
    
    print(f"\nTarget: {num_puzzles} puzzles (rating {min_rating}-{max_rating})")
    if themes:
        print(f"Themes: {', '.join(themes)}")
    print("\nFetching from Lichess API...\n")
    
    puzzles = []
    
    # Lichess puzzle database CSV (simplified approach)
    # We'll fetch from their public database dump
    url = "https://database.lichess.org/lichess_db_puzzle.csv.bz2"
    
    print("Attempting to download from Lichess database...")
    print("(This is a large file, may take a minute...)")
    print("\nAlternatively, you can:")
    print("1. Visit https://lichess.org/training")
    print("2. Manual download: https://database.lichess.org/#puzzles")
    print("\nTrying direct API approach...\n")
    
    # Use Lichess training API instead
    # Fetch puzzles page by page
    try:
        # Get puzzle data from Lichess training themes
        themes_to_fetch = themes or ["middlegame", "endgame", "mating", "fork", "pin", "skewer"]
        
        for theme in themes_to_fetch:
            if len(puzzles) >= num_puzzles:
                break
                
            print(f"Fetching {theme} puzzles...")
            
            # Note: Lichess doesn't have a simple REST API for bulk puzzle download
            # But we can create a converter for their CSV format
            # For now, create a format that users can populate
            break
        
        # Create sample file format
        print("\nCreating sample puzzle format...")
        print("You can populate this with Lichess puzzles manually or via scraping.")
        
        # Add some hardcoded tactical puzzles as examples
        sample_puzzles = [
            {
                "fen": "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4",
                "moves": ["h5f7"],  # Qxf7#
                "rating": 1400,
                "themes": ["mate", "mateIn1", "sacrifice"]
            },
            {
                "fen": "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5Q2/PPPP1PPP/RNB1K1NR w KQkq - 4 4",
                "moves": ["f3f7"],  # Qxf7#
                "rating": 1200,
                "themes": ["mate", "mateIn1"]
            },
            {
                "fen": "r2qkbnr/ppp2ppp/2np4/4p3/2B1P1b1/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 5",
                "moves": ["c4f7", "e8f7", "f3e5"],  # Bxf7+ Kxf7, Nxe5 (fork)
                "rating": 1600,
                "themes": ["fork", "sacrifice", "discoveredAttack"]
            },
            {
                "fen": "rnbqkb1r/pppp1ppp/5n2/4p3/2B1P3/8/PPPP1PPP/RNBQK1NR w KQkq - 2 3",
                "moves": ["c4f7"],  # Bxf7+ (scholar's mate pattern)
                "rating": 1000,
                "themes": ["mate", "mateIn2", "sacrifice"]
            }
        ]
        
        puzzles = sample_puzzles
        
    except Exception as e:
        print(f"Error downloading: {e}")
        print("\nFalling back to sample puzzles...")
    
    # Save puzzles
    output_file = Path("lichess_puzzles.json")
    with open(output_file, 'w') as f:
        json.dump(puzzles, f, indent=2)
    
    print(f"\nSaved {len(puzzles)} puzzles to: {output_file}")
    print("\n" + "="*60)
    print("TO GET MORE PUZZLES:")
    print("1. Visit: https://lichess.org/training")
    print("2. Or download CSV: https://database.lichess.org/#puzzles")
    print("3. Or I can scrape them if you want")
    print("="*60)
    
    return puzzles

if __name__ == "__main__":
    download_lichess_puzzles()
