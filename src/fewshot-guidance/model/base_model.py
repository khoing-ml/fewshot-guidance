import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Dict, Any



class BaseGuidanceModel(nn.Module): 
    def forward(
        self,
        img: Tensor,
        timestep: Tensor,
        step_idx: int,
        pred: Tensor | None,
        txt: Tensor | None,
        vec: Tensor | None,
        fewshot_img: Tensor | None = None,
    ) -> Tensor | Dict[str, Tensor]:
        """
        Args:
            img: Current latent image [batch, seq_len, channels]
            pred: Base model prediction [batch, seq_len, channels]
            txt: Text embeddings [batch, txt_seq_len, txt_dim]
            vec: CLIP pooled embeddings [batch, vec_dim]
            timestep: Current timestep value [batch]
            step_idx: Integer step index in the denoising process
            fewshot_img: Fewshot reference latent image(s) from dataset [batch, seq_len, channels]
                        or [num_shots, seq_len, channels]
        
        Returns:
            Guidance signal to add to pred, same shape as pred [batch, seq_len, channels]
            OR dictionary with 'guidance' key and optional other outputs
        """
        raise NotImplementedError