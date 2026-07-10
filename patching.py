from __future__ import annotations
import torch
from jaxtyping import Float, Int
from transformer_lens import HookedTransformer, ActivationCache


# ... keep the rest of the file exactly the same
# ... rest of your code stays exactly the same

def compute_logit_diff(
    logits: Float[torch.Tensor, "batch seq vocab"],
    correct_token_ids: Int[torch.Tensor, "batch"],
    incorrect_token_ids: Int[torch.Tensor, "batch"],
    seq_pos: int = -1
) -> Float[torch.Tensor, ""]:
    """
    Computes the difference between correct and incorrect token logits at a target position.
    """
    # Isolate the logits at the specific sequence position (usually the final token)
    target_logits = logits[:, seq_pos, :] # shape: [batch, vocab]
    
    # Gather the specific logit values using advanced indexing
    batch_indices = torch.arange(logits.size(0), device=logits.device)
    correct_logits = target_logits[batch_indices, correct_token_ids]
    incorrect_logits = target_logits[batch_indices, incorrect_token_ids]
    
    # Return the average logit difference across the batch
    return (correct_logits - incorrect_logits).mean()

def patch_residual_stream_layer(
    model: HookedTransformer,
    clean_cache: ActivationCache,
    corrupted_tokens: torch.Tensor,
    target_layer: int,
    target_pos: int,
    metric_fn
) -> float:
    """
    Intervenes at a specific layer-token position by swapping the corrupted 
    residual stream representation with the clean cached activation.
    """
    def hook_function(
        corrupted_activation: Float[torch.Tensor, "batch seq d_model"], 
        hook
    ):
        # Swap clean activations only into the exact target sequence position
        corrupted_activation[:, target_pos, :] = clean_cache[hook.name][:, target_pos, :]
        return corrupted_activation

    hook_name = f"blocks.{target_layer}.hook_resid_pre"
    
    # Safely run the model with the temporary hook context manager
    with model.hooks(fwd_hooks=[(hook_name, hook_function)]):
        patched_logits = model(corrupted_tokens)
        
    return metric_fn(patched_logits).item()