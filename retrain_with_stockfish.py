"""
Retrain chess model with Stockfish position evaluations.
This trains both the policy head (move selection) and value head (position evaluation).
"""
import os
import time
import json
import torch
import numpy as np
from pathlib import Path
from src.data.dataset import DatasetBuilder, TrainingExample
from src.training.trainer import ChessNeuralNetwork, board_to_encoding
from src.engines.stockfish_evaluator import StockfishEvaluator
import chess

print("=" * 60)
print("RETRAINING WITH STOCKFISH EVALUATIONS")
print("=" * 60)

# Configuration
EPOCHS = int(os.environ.get('EPOCHS', '10'))
BATCH_SIZE = int(os.environ.get('BATCH_SIZE', '32'))
LEARNING_RATE = float(os.environ.get('LEARNING_RATE', '0.001'))
EVAL_DEPTH = int(os.environ.get('EVAL_DEPTH', '10'))  # Stockfish search depth
EVAL_SAMPLE_RATE = float(os.environ.get('EVAL_SAMPLE_RATE', '0.3'))  # Legacy: percent of positions (used if STRIDE not set)
EVAL_STRIDE = int(os.environ.get('EVAL_STRIDE', '0'))  # Evaluate every Nth position if >0
MAX_EVAL_POSITIONS = int(os.environ.get('MAX_EVAL_POSITIONS', '1500'))  # Hard cap on engine calls
DEVICE = os.environ.get('DEVICE', 'cpu')

print(f"\nConfiguration:")
print(f"  Epochs: {EPOCHS}")
print(f"  Batch size: {BATCH_SIZE}")
print(f"  Evaluation depth: {EVAL_DEPTH}")
if EVAL_STRIDE > 0:
    print(f"  Evaluation stride: every {EVAL_STRIDE}th position (max {MAX_EVAL_POSITIONS})")
else:
    print(f"  Sample rate: {int(EVAL_SAMPLE_RATE * 100)}% of positions (max {MAX_EVAL_POSITIONS})")

# Load training data
print("\n" + "-" * 60)
print("Loading training data...")
print("-" * 60)

builder = DatasetBuilder()

games_file = Path("src/resources/alphazero_games.txt")
magnus_file = Path("src/resources/magnus_games.txt")
puzzles_file = Path("src/resources/puzzles.txt")

if games_file.exists():
    print(f"  Loading AlphaZero games from {games_file}...")
    builder.add_pgn_games([str(games_file)])

# Lightweight parser for plain text Magnus games without PGN headers
def load_plain_games(path: Path) -> int:
    if not path.exists():
        return 0
    print(f"  Loading Magnus games from {path}...")
    raw = path.read_text(encoding='utf-8', errors='ignore')
    # Split on blank lines followed by a line starting with digit and move
    # We treat each block containing move numbers as a game
    blocks = [b.strip() for b in raw.split('\n\n') if b.strip() and any(ch.isdigit() for ch in b[:4])]
    added = 0
    for block in blocks:
        # Extract move line(s)
        lines = [l.strip() for l in block.splitlines() if l and l[0].isdigit()]
        if not lines:
            continue
        moves_text = ' '.join(lines)
        # Remove result token at end for parsing then capture separately
        result = None
        for res in ['1-0','0-1','1/2-1/2','½-½','*']:
            if moves_text.endswith(' ' + res) or moves_text.endswith(res):
                result = res
                moves_text = moves_text.replace(' ' + res, '').replace(res, '')
                break
        # Tokenize SAN moves by stripping move numbers
        tokens = []
        for part in moves_text.split():
            if part.endswith('.') or part.replace('.', '').isdigit():
                continue
            # Skip ellipsis marker
            if part == '...':
                continue
            tokens.append(part)
        if len(tokens) < 8:  # Skip very short / malformed games
            continue
        board = chess.Board()
        for san in tokens:
            try:
                move = board.parse_san(san)
            except Exception:
                # Skip invalid tokens (annotations like !!, ?!, etc.)
                continue
            fen_before = board.fen()
            board.push(move)
            # Store best move UCI for this position
            training_ex = TrainingExample(fen_before, move.uci(), source='magnus')
            builder.training_examples.append(training_ex)
            builder.fen_to_examples[fen_before].append(training_ex)
            added += 1
    print(f"    Added {added} training examples from Magnus games")
    return added

load_plain_games(magnus_file)

if puzzles_file.exists():
    print(f"  Loading puzzles from {puzzles_file}...")
    builder.add_puzzles([str(puzzles_file)])

training_examples = builder.get_training_examples()
print(f"\n  Total: {len(training_examples)} examples loaded")

# Initialize Stockfish
print("\n" + "-" * 60)
print("Initializing Stockfish...")
print("-" * 60)

evaluator = StockfishEvaluator(depth=EVAL_DEPTH, time_limit_ms=100)

if not evaluator.stockfish_path:
    print("\n[WARNING] Stockfish not found!")
    print("  Training will use basic material evaluation only.")
    print("  For best results, install Stockfish:")
    print("  - Windows: https://stockfishchess.org/download/")
    print("  - Or use: winget install stockfish")
else:
    print(f"[OK] Stockfish ready")

# Prepare training data with evaluations (sampling optimized + cache)
print("\n" + "-" * 60)
print("Preparing enhanced training data...")
print("-" * 60)

enhanced_examples = []
positions_evaluated = 0
total_positions = len(training_examples)

# Load existing evaluation cache
cache_path = Path('src/resources/evals_cache.json')
eval_cache = {}
if cache_path.exists():
    try:
        eval_cache = json.loads(cache_path.read_text())
        print(f"  Loaded evaluation cache: {len(eval_cache)} positions")
    except Exception as e:
        print(f"  Failed to read cache: {e}")
        eval_cache = {}
if EVAL_STRIDE > 0:
    candidate_indices = list(range(0, total_positions, EVAL_STRIDE))
    # Cap the candidates
    candidate_indices = candidate_indices[:MAX_EVAL_POSITIONS]
    planned = len(candidate_indices)
else:
    planned = min(int(total_positions * EVAL_SAMPLE_RATE), MAX_EVAL_POSITIONS)
    candidate_indices = set(range(planned))  # first N examples as old logic

start_eval_time = time.time()

for i, example in enumerate(training_examples):
    try:
        board = chess.Board(example.fen)
        board_encoding = board_to_encoding(board)
        
        # Get move index
        try:
            move = chess.Move.from_uci(example.best_move_uci)
            from_sq = move.from_square
            to_sq = move.to_square
            move_idx = from_sq * 64 + to_sq
        except:
            continue
        
        # Decide if this position should be evaluated / cached
        position_eval = None
        fen_key = example.fen
        if fen_key in eval_cache:
            position_eval = eval_cache[fen_key]
        else:
            should_eval = False
            if evaluator.stockfish_path:
                if EVAL_STRIDE > 0:
                    if i in candidate_indices:
                        should_eval = True
                else:
                    if i in candidate_indices:
                        should_eval = True
            if should_eval and positions_evaluated < MAX_EVAL_POSITIONS:
                position_eval, _ = evaluator.evaluate(board)
                if position_eval is not None:
                    positions_evaluated += 1
                    eval_cache[fen_key] = position_eval
                    if positions_evaluated % 100 == 0:
                        elapsed = time.time() - start_eval_time
                        rate = positions_evaluated / max(elapsed, 1)
                        pct = (positions_evaluated / planned * 100) if planned else 0
                        print(f"  Evaluated {positions_evaluated}/{planned} ({pct:.1f}%) | {rate:.1f} pos/sec")
        
        enhanced_examples.append({
            'board_encoding': board_encoding,
            'move_idx': move_idx,
            'position_eval': position_eval
        })
        
    except Exception as e:
        continue

evaluator.close()

print(f"\n  Prepared {len(enhanced_examples)} training examples")
coverage_pct = (positions_evaluated/len(enhanced_examples)*100) if enhanced_examples else 0
print(f"  Positions evaluated: {positions_evaluated}/{planned} ({coverage_pct:.1f}% of all, {planned} planned)")

# Persist updated cache
try:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(eval_cache))
    print(f"  Saved evaluation cache: {len(eval_cache)} positions -> {cache_path}")
except Exception as e:
    print(f"  Failed to save cache: {e}")

# Convert to tensors
print("\n" + "-" * 60)
print("Converting to tensors...")
print("-" * 60)

board_encodings = torch.FloatTensor([ex['board_encoding'] for ex in enhanced_examples])
move_indices = torch.LongTensor([ex['move_idx'] for ex in enhanced_examples])

# Position evaluations (with mask for missing values)
position_evals = []
eval_mask = []
for ex in enhanced_examples:
    if ex['position_eval'] is not None:
        position_evals.append(ex['position_eval'])
        eval_mask.append(1.0)
    else:
        position_evals.append(0.5)  # Neutral if not evaluated
        eval_mask.append(0.0)

position_evals = torch.FloatTensor(position_evals).unsqueeze(1)
eval_mask = torch.FloatTensor(eval_mask).unsqueeze(1)

print(f"  Board encodings: {board_encodings.shape}")
print(f"  Move indices: {move_indices.shape}")
print(f"  Position evaluations: {position_evals.shape} ({eval_mask.sum().item():.0f} valid)")

# Initialize model
print("\n" + "-" * 60)
print("Initializing model...")
print("-" * 60)

model_path = Path("src/models/chess_model.pth")
model = ChessNeuralNetwork()

if model_path.exists():
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    print(f"[OK] Loaded existing model from {model_path}")
else:
    print(f"[OK] Created new model")

model.to(DEVICE)
board_encodings = board_encodings.to(DEVICE)
move_indices = move_indices.to(DEVICE)
position_evals = position_evals.to(DEVICE)
eval_mask = eval_mask.to(DEVICE)

# Training setup
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
policy_criterion = torch.nn.CrossEntropyLoss()
value_criterion = torch.nn.MSELoss()

# Training loop
print("\n" + "-" * 60)
print("Training model...")
print("-" * 60)

model.train()

for epoch in range(EPOCHS):
    total_policy_loss = 0.0
    total_value_loss = 0.0
    total_loss = 0.0
    num_batches = 0
    
    # Shuffle data
    indices = torch.randperm(len(board_encodings))
    
    # Train in batches
    for i in range(0, len(board_encodings), BATCH_SIZE):
        batch_indices = indices[i:i+BATCH_SIZE]
        
        batch_boards = board_encodings[batch_indices]
        batch_moves = move_indices[batch_indices]
        batch_evals = position_evals[batch_indices]
        batch_mask = eval_mask[batch_indices]
        
        # Forward pass
        optimizer.zero_grad()
        policy_logits, value_output = model(batch_boards)
        
        # Policy loss (move prediction)
        policy_loss = policy_criterion(policy_logits, batch_moves)
        
        # Value loss (position evaluation) - only for evaluated positions
        value_loss = torch.tensor(0.0).to(DEVICE)
        if batch_mask.sum() > 0:
            masked_pred = value_output * batch_mask
            masked_target = batch_evals * batch_mask
            value_loss = value_criterion(masked_pred, masked_target)
        
        # Combined loss (60% policy, 40% value)
        loss = 0.6 * policy_loss + 0.4 * value_loss
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_policy_loss += policy_loss.item()
        total_value_loss += value_loss.item()
        total_loss += loss.item()
        num_batches += 1
    
    avg_policy_loss = total_policy_loss / num_batches
    avg_value_loss = total_value_loss / num_batches
    avg_total_loss = total_loss / num_batches
    
    print(f"Epoch {epoch+1}/{EPOCHS}: Loss = {avg_total_loss:.4f} (Policy: {avg_policy_loss:.4f}, Value: {avg_value_loss:.4f})")

# Save model
print("\n" + "-" * 60)
print("Saving model...")
print("-" * 60)

model_path.parent.mkdir(parents=True, exist_ok=True)
torch.save(model.state_dict(), model_path)
model_size_mb = model_path.stat().st_size / (1024 * 1024)
print(f"[OK] Model saved to {model_path} ({model_size_mb:.1f} MB)")

print("\n" + "=" * 60)
print("RETRAINING COMPLETE!")
print("=" * 60)
print("\nTo test:")
print("  1. Restart backend: python serve.py")
print("  2. Visit http://localhost:3000")
print("  3. Play against the improved bot!")
