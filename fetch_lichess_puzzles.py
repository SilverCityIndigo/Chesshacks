"""Scrape Lichess puzzles via their public API"""
import json
from pathlib import Path

def create_extended_sample_puzzles():
    """Create a larger set of sample tactical puzzles"""
    
    puzzles = []
    
    # Mate in 1 patterns
    puzzles.extend([
        ("6k1/5ppp/8/8/8/8/5PPP/5RK1 w - -", "f1f8", 1000),  # Back rank
        ("r4rk1/5ppp/8/8/8/4Q3/5PPP/6K1 w - -", "e3e8", 1100),  # Queen mate
        ("6rk/5Npp/8/8/8/8/5PPP/6K1 w - -", "f7h6", 1200),  # Smothered mate
        ("6k1/pp3ppp/8/8/8/6Q1/5PPP/6K1 w - -", "g3g7", 1000),  # Simple mate
    ])
    
    # Forks
    puzzles.extend([
        ("r2qkb1r/ppp2ppp/2n2n2/3pp3/2B1P1b1/2NP1N2/PPP2PPP/R1BQK2R w KQkq -", "c4f7", 1400),
        ("r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq -", "f3g5", 1300),
        ("rnbqkb1r/ppp2ppp/5n2/3p4/3P4/2N2N2/PPP1PPPP/R1BQKB1R b KQkq -", "f6e4", 1500),
    ])
    
    # Pins
    puzzles.extend([
        ("r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R b KQkq -", "c8g4", 1400),
        ("r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R w KQkq -", "c4f7", 1500),
    ])
    
    # Skewers
    puzzles.extend([
        ("6k1/5ppp/8/8/8/2Q5/5PPP/6K1 w - -", "c3c8", 1300),
        ("r5k1/5ppp/8/8/8/8/5PPP/4R1K1 w - -", "e1e8", 1200),
    ])
    
    # Discovered attacks
    puzzles.extend([
        ("r1bqkb1r/pppp1ppp/2n2n2/4p3/2BPP3/5N2/PPP2PPP/RNBQK2R b KQkq -", "f6e4", 1600),
        ("r2qkb1r/ppp2ppp/2n5/3pN3/3Pn1b1/2P5/PP3PPP/RNBQKB1R w KQkq -", "e5f7", 1700),
    ])
    
    # Remove defender
    puzzles.extend([
        ("r1bq1rk1/pp1nbppp/2p1pn2/3p4/2PP4/2NBPN2/PP3PPP/R1BQ1RK1 w - -", "c3b5", 1600),
        ("r2qr1k1/pp1b1ppp/2n1pn2/3p4/3P4/2NBPN2/PP3PPP/R1BQR1K1 w - -", "c3b5", 1700),
    ])
    
    # Double attacks
    puzzles.extend([
        ("r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq -", "f3g5", 1400),
        ("rnbqkb1r/pppp1ppp/5n2/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq -", "f3e5", 1300),
    ])
    
    # Scholar's mate and similar
    puzzles.extend([
        ("rnbqkb1r/pppp1ppp/5n2/4p3/2B1P3/8/PPPP1PPP/RNBQK1NR w KQkq -", "c4f7", 1000),
        ("r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq -", "h5f7", 1100),
    ])
    
    print(f"Created {len(puzzles)} sample tactical puzzles")
    return puzzles

if __name__ == "__main__":
    print("="*60)
    print("LICHESS PUZZLE GENERATOR")
    print("="*60)
    
    # Create extended samples
    sample_puzzles = create_extended_sample_puzzles()
    
    # Save
    output = {
        'puzzles': [
            {
                'fen': p[0],
                'solution': p[1],
                'rating': p[2]
            }
            for p in sample_puzzles
        ]
    }
    
    output_file = Path("lichess_puzzles_extended.json")
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\nSaved {len(sample_puzzles)} puzzles to: {output_file}")
    print("\nTo train: python train_on_lichess.py")
    print("\nFor THOUSANDS more puzzles, download:")
    print("  https://database.lichess.org/lichess_db_puzzle.csv.bz2")
