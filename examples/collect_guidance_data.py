"""
Data Collection Script for Training Guidance Models

This script helps you collect training data for guidance models by:
1. Running the base diffusion model and saving intermediate states
2. Computing target corrections based on reward signals
3. Creating a dataset for supervised training
"""

import torch
import torch.nn as nn
from pathlib import Path
import json
import numpy as np
from typing import List, Dict, Any, Callable
from tqdm import tqdm
import pickle


class GuidanceDataCollector:
    """
    Collects training data for guidance models by recording denoising states.
    """
    
    def __init__(
        self,
        save_dir: str = "guidance_training_data",
        reward_model: nn.Module | None = None,
    ):
        """
        Args:
            save_dir: Directory to save collected data
            reward_model: Optional reward model to score generations
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.reward_model = reward_model
        self.collected_samples = []
    
    def collect_from_generation(
        self,
        model,
        prompts: List[str],
        num_samples_per_prompt: int = 5,
        num_steps: int = 28,
        **generation_kwargs,
    ):
        """
        Collect data by generating images and recording all intermediate states.
        
        Args:
            model: Flux model
            prompts: List of prompts to generate from
            num_samples_per_prompt: How many samples per prompt
            num_steps: Number of denoising steps
            **generation_kwargs: Other generation parameters
        """
        
        print(f"Collecting data for {len(prompts)} prompts...")
        
        for prompt_idx, prompt in enumerate(tqdm(prompts)):
            for sample_idx in range(num_samples_per_prompt):
                # Generate and collect states
                states = self._generate_with_state_collection(
                    model=model,
                    prompt=prompt,
                    num_steps=num_steps,
                    seed=prompt_idx * 1000 + sample_idx,
                    **generation_kwargs,
                )
                
                # Compute reward if available
                if self.reward_model is not None:
                    final_latent = states[-1]['img_after_step']
                    reward = self._compute_reward(final_latent)
                else:
                    reward = None
                
                # Save this generation's data
                sample_data = {
                    'prompt': prompt,
                    'prompt_idx': prompt_idx,
                    'sample_idx': sample_idx,
                    'states': states,
                    'reward': reward,
                }
                
                self.collected_samples.append(sample_data)
        
        print(f"Collected {len(self.collected_samples)} samples")
    
    def _generate_with_state_collection(
        self,
        model,
        prompt: str,
        num_steps: int,
        seed: int,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """
        Run generation and collect state at each step.
        
        Returns list of dictionaries containing:
        - step_idx: step index
        - timestep: current timestep
        - img_before_step: latent before this step
        - pred: model prediction
        - img_after_step: latent after this step
        - txt, vec, img_ids, txt_ids: conditioning
        """
        
        # This is a simplified version - adapt to your actual model API
        # In practice, you'd need to modify the denoise loop to save states
        
        states = []
        
        # Placeholder - replace with actual generation code
        # You would need to instrument the denoise() function to save states
        
        # Example structure:
        for step in range(num_steps):
            state = {
                'step_idx': step,
                'timestep': None,  # Fill from actual generation
                'img_before_step': None,  # Save from denoise loop
                'pred': None,  # Save from denoise loop
                'img_after_step': None,  # Save from denoise loop
                'txt': None,  # Conditioning
                'vec': None,  # Conditioning
                'img_ids': None,
                'txt_ids': None,
            }
            states.append(state)
        
        return states
    
    def _compute_reward(self, latent: torch.Tensor) -> float:
        """Compute reward for a generated latent."""
        if self.reward_model is None:
            return 0.0
        
        with torch.no_grad():
            reward = self.reward_model(latent)
        
        return reward.item()
    
    def create_supervised_dataset(
        self,
        strategy: str = 'high_reward_only',
        reward_threshold: float = 0.7,
    ):
        """
        Create supervised training dataset from collected samples.
        
        Args:
            strategy: How to compute target corrections
                - 'high_reward_only': Only use high-reward samples, target is zero correction
                - 'reward_difference': Use pairs of high/low reward, target is difference
                - 'optimal_direction': Compute optimal direction via reward gradients
            reward_threshold: Threshold for high-reward samples
        """
        
        if strategy == 'high_reward_only':
            return self._create_high_reward_dataset(reward_threshold)
        elif strategy == 'reward_difference':
            return self._create_difference_dataset(reward_threshold)
        elif strategy == 'optimal_direction':
            return self._create_gradient_dataset()
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
    
    def _create_high_reward_dataset(self, threshold: float):
        """
        Create dataset from high-reward samples.
        Target is zero correction (model should not change good predictions).
        """
        
        dataset = []
        
        for sample in self.collected_samples:
            if sample['reward'] is None or sample['reward'] < threshold:
                continue
            
            # For each step in this high-reward sample
            for state in sample['states']:
                training_example = {
                    'img': state['img_before_step'],
                    'pred': state['pred'],
                    'txt': state['txt'],
                    'vec': state['vec'],
                    'timestep': state['timestep'],
                    'step_idx': state['step_idx'],
                    'target_correction': torch.zeros_like(state['pred']),  # No correction needed
                    'reward': sample['reward'],
                }
                dataset.append(training_example)
        
        return dataset
    
    def _create_difference_dataset(self, threshold: float):
        """
        Create dataset by pairing high and low reward samples.
        Target is the difference in predictions.
        """
        
        # Separate high and low reward samples
        high_reward = [s for s in self.collected_samples if s['reward'] >= threshold]
        low_reward = [s for s in self.collected_samples if s['reward'] < threshold]
        
        dataset = []
        
        # Match samples with same prompt
        for low_sample in low_reward:
            prompt = low_sample['prompt']
            
            # Find high reward sample with same prompt
            matching_high = [
                s for s in high_reward 
                if s['prompt'] == prompt
            ]
            
            if not matching_high:
                continue
            
            high_sample = matching_high[0]  # Take first match
            
            # For each step, compute target as difference
            for low_state, high_state in zip(low_sample['states'], high_sample['states']):
                target_correction = high_state['pred'] - low_state['pred']
                
                training_example = {
                    'img': low_state['img_before_step'],
                    'pred': low_state['pred'],
                    'txt': low_state['txt'],
                    'vec': low_state['vec'],
                    'timestep': low_state['timestep'],
                    'step_idx': low_state['step_idx'],
                    'target_correction': target_correction,
                    'low_reward': low_sample['reward'],
                    'high_reward': high_sample['reward'],
                }
                dataset.append(training_example)
        
        return dataset
    
    def _create_gradient_dataset(self):
        """
        Create dataset by computing optimal guidance direction via gradients.
        Requires reward model with gradients.
        """
        
        if self.reward_model is None:
            raise ValueError("Reward model required for gradient-based dataset")
        
        dataset = []
        
        for sample in self.collected_samples:
            for state in sample['states']:
                # Enable gradients
                img = state['img_before_step'].requires_grad_(True)
                pred = state['pred'].requires_grad_(True)
                
                # Compute predicted next state
                timestep_size = 1.0 / len(sample['states'])  # Approximate
                next_img = img + timestep_size * pred
                
                # Compute reward
                reward = self.reward_model(next_img)
                
                # Compute gradient w.r.t. prediction
                pred_grad = torch.autograd.grad(reward, pred)[0]
                
                # Target correction is gradient direction (scaled)
                target_correction = 0.1 * pred_grad  # Scale factor
                
                training_example = {
                    'img': state['img_before_step'].detach(),
                    'pred': state['pred'].detach(),
                    'txt': state['txt'],
                    'vec': state['vec'],
                    'timestep': state['timestep'],
                    'step_idx': state['step_idx'],
                    'target_correction': target_correction.detach(),
                    'reward': sample['reward'],
                }
                dataset.append(training_example)
        
        return dataset
    
    def save_dataset(self, dataset, name: str = "training_data"):
        """Save dataset to disk."""
        save_path = self.save_dir / f"{name}.pkl"
        
        with open(save_path, 'wb') as f:
            pickle.dump(dataset, f)
        
        print(f"Saved dataset with {len(dataset)} examples to {save_path}")
        
        # Also save metadata
        metadata = {
            'num_examples': len(dataset),
            'num_prompts': len(set(s.get('prompt', '') for s in self.collected_samples)),
            'num_samples': len(self.collected_samples),
        }
        
        with open(self.save_dir / f"{name}_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def load_dataset(self, name: str = "training_data"):
        """Load dataset from disk."""
        load_path = self.save_dir / f"{name}.pkl"
        
        with open(load_path, 'rb') as f:
            dataset = pickle.load(f)
        
        print(f"Loaded dataset with {len(dataset)} examples from {load_path}")
        return dataset


class GuidanceDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset for training guidance models.
    """
    
    def __init__(self, data: List[Dict[str, Any]]):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        return {
            'img': item['img'],
            'pred': item['pred'],
            'txt': item['txt'],
            'vec': item['vec'],
            'timestep': item['timestep'],
            'step_idx': item['step_idx'],
            'target': item['target_correction'],
        }


# Example usage
def example_data_collection():
    """Example of how to use the data collector."""
    
    # Create collector with optional reward model
    class DummyRewardModel(nn.Module):
        def forward(self, latent):
            # Placeholder reward
            return torch.randn(1)
    
    collector = GuidanceDataCollector(
        save_dir="guidance_data",
        reward_model=DummyRewardModel(),
    )
    
    # Define prompts
    prompts = [
        "A beautiful sunset over mountains",
        "A serene lake with reflections",
        "An ancient castle on a hill",
        # ... more prompts
    ]
    
    # Collect data
    # Note: You'll need to implement _generate_with_state_collection
    # to actually save states during generation
    # collector.collect_from_generation(
    #     model=your_flux_model,
    #     prompts=prompts,
    #     num_samples_per_prompt=5,
    # )
    
    # Create training dataset
    # dataset = collector.create_supervised_dataset(
    #     strategy='high_reward_only',
    #     reward_threshold=0.7,
    # )
    
    # Save it
    # collector.save_dataset(dataset, name="aesthetic_guidance_data")
    
    print("Data collection setup complete!")


if __name__ == "__main__":
    example_data_collection()
