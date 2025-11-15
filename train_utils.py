#!/usr/bin/env python3
"""
Utility functions for training workflow.

Includes helpers for:
- Pre-encoding datasets
- Managing checkpoints
- Analyzing training metrics
- Loading trained models
"""

import torch
from pathlib import Path
from typing import Optional, Dict, Any, List
import json
from datetime import datetime


class TrainingUtils:
    """Utilities for training management."""
    
    @staticmethod
    def list_checkpoints(checkpoint_dir: str | Path = "./checkpoints") -> List[Dict[str, Any]]:
        """
        List all checkpoints with metadata.
        
        Returns:
            List of dicts with checkpoint info
        """
        checkpoint_dir = Path(checkpoint_dir)
        checkpoints = []
        
        for ckpt_file in sorted(checkpoint_dir.glob("*.pt")):
            try:
                ckpt = torch.load(ckpt_file, map_location='cpu')
                info = {
                    'path': str(ckpt_file),
                    'name': ckpt_file.name,
                    'epoch': ckpt.get('epoch', 'N/A'),
                    'step': ckpt.get('step', 'N/A'),
                    'size_mb': ckpt_file.stat().st_size / (1024 ** 2),
                }
                checkpoints.append(info)
            except Exception as e:
                print(f"Warning: Could not load {ckpt_file}: {e}")
        
        return checkpoints
    
    @staticmethod
    def delete_old_checkpoints(
        checkpoint_dir: str | Path = "./checkpoints",
        keep_last_n: int = 5,
        confirm: bool = True
    ) -> int:
        """
        Delete old checkpoints, keeping only the last N.
        
        Args:
            checkpoint_dir: Directory containing checkpoints
            keep_last_n: Number of recent checkpoints to keep
            confirm: Ask for confirmation before deleting
        
        Returns:
            Number of checkpoints deleted
        """
        checkpoint_dir = Path(checkpoint_dir)
        checkpoints = sorted(checkpoint_dir.glob("*.pt"), key=lambda p: p.stat().st_mtime)
        
        to_delete = checkpoints[:-keep_last_n] if len(checkpoints) > keep_last_n else []
        
        if not to_delete:
            print(f"✓ Keeping all {len(checkpoints)} checkpoints (within limit of {keep_last_n})")
            return 0
        
        total_size = sum(f.stat().st_size for f in to_delete) / (1024 ** 2)
        
        if confirm:
            print(f"\nWill delete {len(to_delete)} checkpoints, freeing {total_size:.1f} MB:")
            for f in to_delete:
                print(f"  - {f.name}")
            response = input("\nProceed? (yes/no): ").lower().strip()
            if response != 'yes':
                print("Cancelled")
                return 0
        
        for f in to_delete:
            f.unlink()
            print(f"✓ Deleted {f.name}")
        
        print(f"✓ Deleted {len(to_delete)} checkpoints, freed {total_size:.1f} MB")
        return len(to_delete)
    
    @staticmethod
    def get_latest_checkpoint(checkpoint_dir: str | Path = "./checkpoints") -> Optional[Path]:
        """Get path to most recent checkpoint."""
        checkpoint_dir = Path(checkpoint_dir)
        checkpoints = list(checkpoint_dir.glob("*.pt"))
        
        if not checkpoints:
            return None
        
        return max(checkpoints, key=lambda p: p.stat().st_mtime)
    
    @staticmethod
    def find_checkpoint_by_epoch(
        epoch: int,
        checkpoint_dir: str | Path = "./checkpoints"
    ) -> Optional[Path]:
        """Find checkpoint for a specific epoch."""
        checkpoint_dir = Path(checkpoint_dir)
        
        for ckpt_file in checkpoint_dir.glob("*.pt"):
            try:
                ckpt = torch.load(ckpt_file, map_location='cpu')
                if ckpt.get('epoch') == epoch:
                    return ckpt_file
            except:
                pass
        
        return None


def list_checkpoints_cmd():
    """CLI command to list checkpoints."""
    import argparse
    parser = argparse.ArgumentParser(description="List training checkpoints")
    parser.add_argument("--dir", default="./checkpoints", help="Checkpoint directory")
    args = parser.parse_args()
    
    checkpoints = TrainingUtils.list_checkpoints(args.dir)
    
    if not checkpoints:
        print(f"No checkpoints found in {args.dir}")
        return
    
    print(f"\nCheckpoints in {args.dir}:")
    print("-" * 80)
    for i, ckpt in enumerate(checkpoints, 1):
        print(f"{i}. {ckpt['name']}")
        print(f"   Epoch: {ckpt['epoch']}, Step: {ckpt['step']}, Size: {ckpt['size_mb']:.1f} MB")
    print("-" * 80)


def cleanup_checkpoints_cmd():
    """CLI command to cleanup old checkpoints."""
    import argparse
    parser = argparse.ArgumentParser(description="Clean up old checkpoints")
    parser.add_argument("--dir", default="./checkpoints", help="Checkpoint directory")
    parser.add_argument("--keep", type=int, default=5, help="Number of checkpoints to keep")
    parser.add_argument("--force", action="store_true", help="Delete without confirmation")
    args = parser.parse_args()
    
    TrainingUtils.delete_old_checkpoints(
        checkpoint_dir=args.dir,
        keep_last_n=args.keep,
        confirm=not args.force
    )


def get_latest_cmd():
    """CLI command to get latest checkpoint path."""
    import argparse
    parser = argparse.ArgumentParser(description="Get latest checkpoint path")
    parser.add_argument("--dir", default="./checkpoints", help="Checkpoint directory")
    args = parser.parse_args()
    
    latest = TrainingUtils.get_latest_checkpoint(args.dir)
    if latest:
        print(latest)
    else:
        print("No checkpoints found")


# Example usage script
EXAMPLE_USAGE = """
# List all checkpoints with metadata
python -c "from train_utils import TrainingUtils; print(TrainingUtils.list_checkpoints())"

# Delete checkpoints older than the last 3
python -c "from train_utils import TrainingUtils; TrainingUtils.delete_old_checkpoints(keep_last_n=3)"

# Get path to latest checkpoint
python -c "from train_utils import TrainingUtils; print(TrainingUtils.get_latest_checkpoint())"

# Find checkpoint for specific epoch
python -c "from train_utils import TrainingUtils; print(TrainingUtils.find_checkpoint_by_epoch(5))"
"""


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "list":
            list_checkpoints_cmd()
        elif command == "cleanup":
            cleanup_checkpoints_cmd()
        elif command == "latest":
            get_latest_cmd()
        else:
            print(f"Unknown command: {command}")
            print("\nAvailable commands:")
            print("  list      - List all checkpoints")
            print("  cleanup   - Delete old checkpoints")
            print("  latest    - Show latest checkpoint path")
    else:
        # Show usage
        print("Training Utilities")
        print("=" * 60)
        print("\nUsage: python train_utils.py <command> [options]")
        print("\nCommands:")
        print("  list      - List all checkpoints")
        print("  cleanup   - Delete old checkpoints (keeps last N)")
        print("  latest    - Print path to latest checkpoint")
        print("\nExamples:")
        print("  python train_utils.py list")
        print("  python train_utils.py cleanup --keep 3 --force")
        print("  python train_utils.py latest")
        print("\nProgrammatic usage:")
        print(EXAMPLE_USAGE)
