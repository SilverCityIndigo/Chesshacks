"""
End-to-End Training Pipeline
Loads data, trains model, evaluates, and saves checkpoint.
"""
import sys
import torch
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.data.pgn_loader import PGNLoader
from src.data.puzzle_loader import PuzzleLoader
from src.data.dataset import DatasetBuilder
from src.training.trainer import ChessTrainer, ChessNeuralNetwork
from src.agents.supervised_agent import SupervisedAgent


def main():
    """Run the complete training pipeline."""
    
    print("=" * 60)
    print("CHESS BOT TRAINING PIPELINE")
    print("=" * 60)
    
    # 1. Load data
    print("\n[1/5] Loading training data...")
    builder = DatasetBuilder()
    
    pgn_files = [
        "src/resources/alphazero_games.txt",
        "src/resources/magnus_games.txt"
    ]
    puzzle_files = ["src/resources/puzzles.txt"]
    
    pgn_count = builder.add_pgn_games(pgn_files, weight=1.0)
    puzzle_count = builder.add_puzzles(puzzle_files, weight=2.0)
    
    print(f"  ✓ Loaded {pgn_count} PGN games")
    print(f"  ✓ Loaded {puzzle_count} puzzle examples")
    
    # Get final dataset
    examples = builder.get_training_examples()
    stats = builder.get_statistics()
    
    print(f"  ✓ Total training examples: {stats['total_examples']}")
    print(f"  ✓ Unique FEN positions: {stats['unique_positions']}")
    
    # 2. Initialize model and trainer
    print("\n[2/5] Initializing neural network...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"  Device: {device}")
    
    model = ChessNeuralNetwork(num_moves=4672)
    trainer = ChessTrainer(model, device=device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  ✓ Model initialized ({total_params:,} parameters)")
    
    # 3. Prepare training data
    print("\n[3/5] Preparing training data...")
    try:
        train_loader = trainer.prepare_training_data(examples, batch_size=32)
        print(f"  ✓ DataLoader created with {len(examples)} examples")
    except Exception as e:
        print(f"  ✗ Failed to prepare training data: {e}")
        return
    
    # 4. Train model
    print("\n[4/5] Training model (10 epochs)...")
    try:
        history = trainer.train(train_loader, epochs=10, validation_loader=None)
        
        # Print final metrics
        final_loss = history['total_loss'][-1]
        print(f"  ✓ Training complete")
        print(f"  ✓ Final loss: {final_loss:.4f}")
        
    except Exception as e:
        print(f"  ✗ Training failed: {e}")
        return
    
    # 5. Save model and create supervised agent
    print("\n[5/5] Saving model and creating agent...")
    
    model_path = "src/models/chess_model.pth"
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    
    trainer.save_model(model_path)
    
    # Create supervised agent
    agent = SupervisedAgent()
    agent.set_model(model)
    
    print(f"  ✓ Supervised agent created and ready")
    
    # Summary
    # Summary
    game_percentage = 100.0 * stats['game_examples'] / stats['total_examples']
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Training Examples: {stats['total_examples']}")
    print(f"Game Examples: {stats['game_examples']} ({game_percentage:.1f}%)")
    print(f"Puzzle Examples: {stats['puzzle_examples']} ({stats['puzzle_percentage']:.1f}%)")
    print(f"Final Loss: {final_loss:.4f}")
    print(f"Model Path: {model_path}")
    print("\nNext steps:")
    print("  1. Update serve.py to use SupervisedAgent instead of test_func")
    print("  2. Deploy to production")
    print("  3. Monitor move quality and win rate")
    print("=" * 60)


if __name__ == "__main__":
    main()

