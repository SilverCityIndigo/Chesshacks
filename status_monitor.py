"""Real-time training status monitor"""
import time
import os
from pathlib import Path
from datetime import datetime, timedelta

def get_file_age(filepath):
    if not Path(filepath).exists():
        return None
    mtime = Path(filepath).stat().st_mtime
    return time.time() - mtime

def format_time(seconds):
    if seconds is None:
        return "N/A"
    return str(timedelta(seconds=int(seconds)))

print("="*70)
print("CHESS BOT TRAINING STATUS MONITOR")
print("="*70)

while True:
    os.system('cls' if os.name == 'nt' else 'clear')
    
    print("\n" + "="*70)
    print(f"Status Update - {datetime.now().strftime('%H:%M:%S')}")
    print("="*70)
    
    # Check bootstrap completion
    bootstrap_model = Path("src/models/chess_model.pth")
    if bootstrap_model.exists():
        age = get_file_age(bootstrap_model)
        print(f"\n[PHASE 1] Bootstrap Training: COMPLETE")
        print(f"  Model: {bootstrap_model} ({bootstrap_model.stat().st_size / (1024*1024):.1f} MB)")
        print(f"  Last modified: {format_time(age)} ago")
    else:
        print(f"\n[PHASE 1] Bootstrap Training: IN PROGRESS...")
        print(f"  Evaluating Magnus game positions with Stockfish")
        print(f"  Target: 500 positions, 20 epochs")
    
    # Check self-play progress
    selfplay_data = Path("self_play_data.json")
    if selfplay_data.exists():
        age = get_file_age(selfplay_data)
        size_mb = selfplay_data.stat().st_size / (1024*1024)
        
        # Try to read game count
        try:
            import json
            with open(selfplay_data, 'r') as f:
                data = json.load(f)
                positions = len(data.get('positions', []))
                games_est = positions // 50  # Rough estimate
            print(f"\n[PHASE 2] Self-Play Generation: COMPLETE")
            print(f"  Data: {selfplay_data} ({size_mb:.1f} MB)")
            print(f"  Games: ~{games_est} | Positions: {positions}")
            print(f"  Last modified: {format_time(age)} ago")
        except:
            print(f"\n[PHASE 2] Self-Play Generation: IN PROGRESS...")
            print(f"  Current data: {size_mb:.1f} MB")
            print(f"  MCTS: 100 simulations per move")
    else:
        if bootstrap_model.exists():
            print(f"\n[PHASE 2] Self-Play Generation: STARTING SOON...")
            print(f"  Waiting for bootstrap to complete")
        else:
            print(f"\n[PHASE 2] Self-Play Generation: PENDING")
            print(f"  Requires Phase 1 completion")
    
    # Check final training
    if selfplay_data.exists():
        final_age = get_file_age(bootstrap_model)
        selfplay_age = get_file_age(selfplay_data)
        
        if final_age and selfplay_age and final_age < selfplay_age:
            print(f"\n[PHASE 3] Final Retraining: COMPLETE")
            print(f"  Model ready for deployment!")
        else:
            print(f"\n[PHASE 3] Final Retraining: PENDING")
            print(f"  Will start after Phase 2")
    else:
        print(f"\n[PHASE 3] Final Retraining: PENDING")
        print(f"  Waiting for self-play data")
    
    # Estimate completion
    print("\n" + "-"*70)
    if not bootstrap_model.exists():
        print("ESTIMATED TIME REMAINING: 10-15 minutes (Bootstrap)")
    elif not selfplay_data.exists():
        print("ESTIMATED TIME REMAINING: 2-4 hours (Self-Play)")
    else:
        print("ESTIMATED TIME REMAINING: 5-10 minutes (Final Training)")
    
    print("-"*70)
    print("\nPress CTRL+C to exit monitor")
    print("="*70)
    
    time.sleep(10)  # Update every 10 seconds
