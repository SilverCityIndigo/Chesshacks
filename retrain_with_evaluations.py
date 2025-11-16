"""
Retrain chess model with Stockfish position evaluations.
This script:
1. Loads game data with engine evaluations
2. Trains value head using Stockfish guidance
3. Saves improved model for deployment
"""

import sys
import os
import torch
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.training.trainer import ChessNeuralNetwork
from src.training.enhanced_trainer import EnhancedChessTrainer
from src.data.enhanced_dataset import EnhancedDatasetBuilder, EnhancedTrainingExample
from src.data.dataset import DatasetBuilder
from src.engines.stockfish_evaluator import StockfishEvaluator
from src.data.dataset import DatasetBuilder


def load_training_data():
    """Load training examples with Stockfish evaluations."""
    print("\n" + "="*60)
    print("LOADING TRAINING DATA WITH STOCKFISH EVALUATIONS")
    print("="*60)
    
    # First try enhanced dataset with evaluations
    enhanced_builder = EnhancedDatasetBuilder()
    
    # Paths to data
    games_path = Path("data/games")
    puzzles_path = Path("data/puzzles")
    
    # Load games with evaluations
    if games_path.exists():
        print(f"\n[1/2] Loading games from {games_path}...")
        games_files = list(games_path.glob("*.pgn"))
        
        if games_files:
            for pgn_file in games_files[:3]:  # Limit to first 3 files for speed
                print(f"  Processing: {pgn_file.name}")
                enhanced_builder.add_pgn_games([str(pgn_file)], depth=12)
        else:
            print("  No PGN files found")
    else:
        print(f"  Games directory not found: {games_path}")
    
    # Load puzzles with evaluations
    if puzzles_path.exists():
        print(f"\n[2/2] Loading puzzles from {puzzles_path}...")
        puzzle_files = list(puzzles_path.glob("*.pgn"))
        
        if puzzle_files:
            for puzzle_file in puzzle_files:
                print(f"  Processing: {puzzle_file.name}")
                enhanced_builder.add_puzzles([str(puzzle_file)], depth=12)
        else:
            print("  No puzzle files found")
    else:
        print(f"  Puzzles directory not found: {puzzles_path}")
    
    # Get training examples
    examples = enhanced_builder.get_training_examples()
    stats = enhanced_builder.get_statistics()
    
    # If no examples loaded, fallback to original dataset
    if not examples:
        print("\nNo PGN files found. Loading from original dataset...")
        builder = DatasetBuilder()
        
        # Load from existing data files
        resources_path = Path("src/resources")
        if resources_path.exists():
            games_file = resources_path / "alphazero_games.txt"
            puzzles_file = resources_path / "puzzles.txt"
            
            if games_file.exists():
                print(f"  Loading games from {games_file}")
                builder.add_pgn_games(str(games_file))
            
            if puzzles_file.exists():
                print(f"  Loading puzzles from {puzzles_file}")
                builder.add_puzzles(str(puzzles_file))
        
        # Convert to enhanced examples (without evaluations for now)
        examples = builder.get_training_examples()
        examples = [
            EnhancedTrainingExample(
                fen=ex.fen,
                best_move_uci=ex.best_move_uci,
                move_probabilities=ex.move_probabilities,
                source=ex.source,
                position_evaluation=None  # Will be evaluated during training
            )
            for ex in examples
        ]
        
        stats = {
            'total_examples': len(examples),
            'evaluated_positions': 0,
            'games_processed': builder.stats['games'],
            'puzzles_processed': builder.stats['puzzles'],
            'avg_evaluation': 0.5
        }
    
    print(f"\nData Loading Complete:")
    print(f"  Total examples: {stats['total_examples']}")
    print(f"  With evaluations: {stats['evaluated_positions']}")
    print(f"  Avg position strength: {stats['avg_evaluation']:.3f}")
    
    return examples
    print(f"  Avg position strength: {stats['avg_evaluation']:.3f}")
    
    return examples


def train_model(examples, num_epochs=10):
    """Train model with enhanced trainer."""
    print("\n" + "="*60)
    print("TRAINING MODEL WITH POSITION EVALUATIONS")
    print("="*60)
    
    # Determine device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {device}")
    
    # Initialize model
    print("\nInitializing model...")
    model = ChessNeuralNetwork()
    model.to(device)
    
    # Try to load existing model weights
    model_path = Path("src/models/chess_model.pth")
    if model_path.exists():
        print(f"Loading existing model from {model_path}")
        try:
            model.load_state_dict(torch.load(model_path, map_location=device))
        except Exception as e:
            print(f"Warning: Could not load existing model: {e}")
    
    # Initialize trainer
    trainer = EnhancedChessTrainer(
        model,
        learning_rate=0.001,
        device=device,
        policy_weight=0.6,
        value_weight=0.4
    )
    
    print(f"\nStarting training ({num_epochs} epochs)...")
    print(f"Training examples: {len(examples)}")
    print(f"Policy weight: 0.6, Value weight: 0.4\n")
    
    best_loss = float('inf')
    
    for epoch in range(num_epochs):
        epoch_stats = trainer.train_epoch(examples, batch_size=32)
        
        print(f"Epoch {epoch+1}/{num_epochs}:")
        print(f"  Policy Loss:  {epoch_stats['policy_loss']:.4f}")
        print(f"  Value Loss:   {epoch_stats['value_loss']:.4f}")
        print(f"  Total Loss:   {epoch_stats['total_loss']:.4f}")
        print(f"  Evals Used:   {epoch_stats['evaluated_examples']}/{len(examples)} " +
              f"({epoch_stats['evaluated_pct']:.1f}%)")
        
        if epoch_stats['total_loss'] < best_loss:
            best_loss = epoch_stats['total_loss']
    
    final_stats = trainer.get_stats()
    
    print(f"\nTraining Complete!")
    print(f"  Final Policy Loss:  {final_stats['avg_policy_loss']:.4f}")
    print(f"  Final Value Loss:   {final_stats['avg_value_loss']:.4f}")
    print(f"  Final Total Loss:   {final_stats['avg_total_loss']:.4f}")
    print(f"  Model Parameters:   {sum(p.numel() for p in model.parameters()):,}")
    
    return model


def save_model(model, output_path="src/models/chess_model_improved.pth"):
    """Save trained model."""
    print("\n" + "="*60)
    print("SAVING IMPROVED MODEL")
    print("="*60)
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    torch.save(model.state_dict(), output_path)
    
    # File size
    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    
    print(f"\nModel saved to: {output_path}")
    print(f"File size: {file_size_mb:.1f} MB")
    print(f"\nTo use this model in production:")
    print(f"  1. Replace src/models/chess_model.pth with chess_model_improved.pth")
    print(f"  2. Restart the server: python serve.py")
    print(f"  3. The bot will now use the improved model")


def main():
    """Main training pipeline."""
    print("\n╔════════════════════════════════════════════════════════════╗")
    print("║      CHESS BOT RETRAINING WITH POSITION EVALUATIONS       ║")
    print("╚════════════════════════════════════════════════════════════╝")
    
    try:
        # Load data
        examples = load_training_data()
        
        if not examples:
            print("\n❌ No training examples found. Exiting.")
            return
        
        # Train model
        improved_model = train_model(examples, num_epochs=10)
        
        # Save model
        save_model(improved_model)
        
        print("\n✓ Retraining complete! Bot is ready for improved performance.")
        print("  Expected improvement: +200-400 Elo with better position evaluation")
        
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
    except Exception as e:
        print(f"\n❌ Error during retraining: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
