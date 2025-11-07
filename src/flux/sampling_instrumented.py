"""
Instrumented Sampling for Data Collection

This module provides a modified denoise function that saves intermediate states
during the denoising process, which can be used to collect training data for
guidance models.
"""

import torch
from torch import Tensor
from typing import List, Dict, Any, Callable
from pathlib import Path
import pickle


def denoise_with_state_collection(
    model,
    # model input
    img: Tensor,
    img_ids: Tensor,
    txt: Tensor,
    txt_ids: Tensor,
    vec: Tensor,
    # sampling parameters
    timesteps: list[float],
    guidance: float = 1.0,
    # extra img tokens (channel-wise)
    img_cond: Tensor | None = None,
    # extra img tokens (sequence-wise)
    img_cond_seq: Tensor | None = None,
    img_cond_seq_ids: Tensor | None = None,
    # state collection
    save_states: bool = True,
    # unconditional generation
    unconditional: bool = False,
) -> tuple[Tensor, List[Dict[str, Any]]]:
    """
    Modified denoise function that saves intermediate states.
    
    Returns:
        img: Final denoised latent
        states: List of dictionaries containing state at each step
    """
    
    guidance_vec = torch.full((img.shape[0],), guidance, device=img.device, dtype=img.dtype)
    
    states = []
    
    for step_idx, (t_curr, t_prev) in enumerate(zip(timesteps[:-1], timesteps[1:])):
        t_vec = torch.full((img.shape[0],), t_curr, dtype=img.dtype, device=img.device)
        
        # Save state before step
        if save_states:
            state_before = {
                'step_idx': step_idx,
                't_curr': t_curr,
                't_prev': t_prev,
                'img_before_step': img.detach().cpu().clone(),
                'txt': txt.detach().cpu().clone(),
                'vec': vec.detach().cpu().clone(),
                'img_ids': img_ids.detach().cpu().clone(),
                'txt_ids': txt_ids.detach().cpu().clone(),
                'timestep': t_vec.detach().cpu().clone(),
            }
        
        img_input = img
        img_input_ids = img_ids
        if img_cond is not None:
            img_input = torch.cat((img, img_cond), dim=-1)
        if img_cond_seq is not None:
            assert (
                img_cond_seq_ids is not None
            ), "You need to provide either both or neither of the sequence conditioning"
            img_input = torch.cat((img_input, img_cond_seq), dim=1)
            img_input_ids = torch.cat((img_input_ids, img_cond_seq_ids), dim=1)
        
        pred = model(
            img=img_input,
            img_ids=img_input_ids,
            txt=txt,
            txt_ids=txt_ids,
            y=vec,
            timesteps=t_vec,
            guidance=guidance_vec,
            unconditional=unconditional,
        )
        if img_input_ids is not None:
            pred = pred[:, : img.shape[1]]
        
        # Update image
        img = img + (t_prev - t_curr) * pred
        
        # Save state after step
        if save_states:
            state_before['pred'] = pred.detach().cpu().clone()
            state_before['img_after_step'] = img.detach().cpu().clone()
            states.append(state_before)
    
    return img, states


class StateCollectionCallback:
    """
    Callback for collecting states during multiple generations.
    """
    
    def __init__(self, save_dir: str = "collected_states"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.all_generations = []
    
    def on_generation_complete(
        self,
        prompt: str,
        states: List[Dict[str, Any]],
        final_image: Tensor = None,
        metadata: Dict[str, Any] = None,
    ):
        """
        Called after each generation completes.
        
        Args:
            prompt: Text prompt used
            states: List of states from denoise_with_state_collection
            final_image: Final generated image (optional)
            metadata: Additional metadata (seed, parameters, etc.)
        """
        
        generation_data = {
            'prompt': prompt,
            'states': states,
            'final_image': final_image.detach().cpu() if final_image is not None else None,
            'metadata': metadata or {},
            'num_steps': len(states),
        }
        
        self.all_generations.append(generation_data)
    
    def save_to_disk(self, filename: str = "collected_states.pkl"):
        """Save all collected generations to disk."""
        save_path = self.save_dir / filename
        
        with open(save_path, 'wb') as f:
            pickle.dump(self.all_generations, f)
        
        print(f"Saved {len(self.all_generations)} generations to {save_path}")
        
        # Save summary
        summary = {
            'num_generations': len(self.all_generations),
            'prompts': [g['prompt'] for g in self.all_generations],
            'total_steps': sum(g['num_steps'] for g in self.all_generations),
        }
        
        import json
        with open(self.save_dir / "summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
    
    def load_from_disk(self, filename: str = "collected_states.pkl"):
        """Load collected generations from disk."""
        load_path = self.save_dir / filename
        
        with open(load_path, 'rb') as f:
            self.all_generations = pickle.load(f)
        
        print(f"Loaded {len(self.all_generations)} generations from {load_path}")
    
    def get_states_by_step(self, step_idx: int) -> List[Dict[str, Any]]:
        """
        Get all states for a specific step index across all generations.
        
        Useful for analyzing or training on specific denoising steps.
        """
        states_at_step = []
        
        for gen in self.all_generations:
            if step_idx < len(gen['states']):
                state = gen['states'][step_idx].copy()
                state['prompt'] = gen['prompt']
                state['metadata'] = gen['metadata']
                states_at_step.append(state)
        
        return states_at_step
    
    def get_states_by_timestep_range(
        self,
        t_min: float,
        t_max: float,
    ) -> List[Dict[str, Any]]:
        """
        Get all states within a timestep range.
        
        Useful for training step-specific guidance models.
        """
        states_in_range = []
        
        for gen in self.all_generations:
            for state in gen['states']:
                t = state['t_curr']
                if t_min <= t <= t_max:
                    state_copy = state.copy()
                    state_copy['prompt'] = gen['prompt']
                    state_copy['metadata'] = gen['metadata']
                    states_in_range.append(state_copy)
        
        return states_in_range
    
    def filter_by_quality(
        self,
        quality_fn: Callable[[Tensor], float],
        threshold: float,
    ) -> 'StateCollectionCallback':
        """
        Create a new callback with only high-quality generations.
        
        Args:
            quality_fn: Function that takes final_image and returns quality score
            threshold: Minimum quality score to keep
        
        Returns:
            New StateCollectionCallback with filtered generations
        """
        filtered_callback = StateCollectionCallback(save_dir=self.save_dir)
        
        for gen in self.all_generations:
            if gen['final_image'] is None:
                continue
            
            quality = quality_fn(gen['final_image'])
            if quality >= threshold:
                filtered_callback.all_generations.append(gen)
        
        print(f"Filtered to {len(filtered_callback.all_generations)} / {len(self.all_generations)} generations")
        
        return filtered_callback


def create_training_pairs(
    states: List[Dict[str, Any]],
    strategy: str = 'identity',
) -> List[Dict[str, Any]]:
    """
    Convert collected states into training pairs for guidance models.
    
    Args:
        states: List of states from StateCollectionCallback
        strategy: How to create training pairs
            - 'identity': Target is zero correction (for high-quality samples)
            - 'small_noise': Add small noise as correction target
            - 'temporal_consistency': Use next step's prediction as target
    
    Returns:
        List of training examples with inputs and targets
    """
    
    training_pairs = []
    
    for state in states:
        if strategy == 'identity':
            # Target is zero correction
            target = torch.zeros_like(state['pred'])
        
        elif strategy == 'small_noise':
            # Small random correction (for regularization)
            target = torch.randn_like(state['pred']) * 0.01
        
        elif strategy == 'temporal_consistency':
            # Would need next step's data - placeholder
            target = torch.zeros_like(state['pred'])
        
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        training_pair = {
            'img': state['img_before_step'],
            'pred': state['pred'],
            'txt': state['txt'],
            'vec': state['vec'],
            'timestep': state['timestep'],
            'step_idx': state['step_idx'],
            'target_correction': target,
            'prompt': state.get('prompt', ''),
        }
        
        training_pairs.append(training_pair)
    
    return training_pairs


# Example usage
def example_state_collection():
    """
    Example showing how to collect states during generation.
    """
    
    print("Example: State Collection for Guidance Training")
    print("=" * 70)
    
    # Setup
    callback = StateCollectionCallback(save_dir="example_states")
    
    # Simulate generation (replace with actual model)
    prompts = [
        "A beautiful landscape",
        "A portrait of a person",
        "An abstract painting",
    ]
    
    print(f"\nCollecting states for {len(prompts)} prompts...")
    
    for i, prompt in enumerate(prompts):
        # In real usage:
        # img, states = denoise_with_state_collection(
        #     model=your_model,
        #     img=noise,
        #     img_ids=img_ids,
        #     txt=txt,
        #     txt_ids=txt_ids,
        #     vec=vec,
        #     timesteps=timesteps,
        #     save_states=True,
        # )
        
        # Dummy data for demonstration
        num_steps = 28
        states = [
            {
                'step_idx': j,
                't_curr': 1.0 - j / num_steps,
                't_prev': 1.0 - (j + 1) / num_steps,
                'img_before_step': torch.randn(1, 256, 64),
                'pred': torch.randn(1, 256, 64),
                'img_after_step': torch.randn(1, 256, 64),
                'txt': torch.randn(1, 77, 4096),
                'vec': torch.randn(1, 768),
                'img_ids': torch.zeros(1, 256, 3),
                'txt_ids': torch.zeros(1, 77, 3),
                'timestep': torch.tensor([1.0 - j / num_steps]),
            }
            for j in range(num_steps)
        ]
        
        final_image = torch.randn(1, 3, 512, 512)
        
        callback.on_generation_complete(
            prompt=prompt,
            states=states,
            final_image=final_image,
            metadata={'seed': i, 'guidance': 3.5},
        )
    
    print(f"Collected states from {len(callback.all_generations)} generations")
    
    # Save to disk
    callback.save_to_disk("example_data.pkl")
    
    # Analyze collected data
    print(f"\nAnalysis:")
    print(f"  Total generations: {len(callback.all_generations)}")
    print(f"  Total steps collected: {sum(g['num_steps'] for g in callback.all_generations)}")
    
    # Get states for specific steps
    early_steps = callback.get_states_by_step(0)
    print(f"  States at step 0: {len(early_steps)}")
    
    mid_steps = callback.get_states_by_timestep_range(0.4, 0.6)
    print(f"  States in t=0.4-0.6: {len(mid_steps)}")
    
    # Create training pairs
    all_states = []
    for gen in callback.all_generations:
        all_states.extend(gen['states'])
    
    training_pairs = create_training_pairs(all_states, strategy='identity')
    print(f"\n  Created {len(training_pairs)} training pairs")
    
    print("\n" + "=" * 70)
    print("State collection example complete!")


if __name__ == "__main__":
    example_state_collection()
