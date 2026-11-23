"""
Analytical (theoretical) complexity of a network, computed with forward hooks.

For every model we derive, per input sample:
  * P  - number of trainable parameters                (space: weights)
  * F  - multiply-accumulate operations (MACs) of one forward pass (time)
  * A  - number of activation values produced by all leaf layers
         (these must be kept for back-propagation -> training memory)

Formulas used (k = kernel size, g = groups):
  Conv2d : MACs = C_out * H_out * W_out * (C_in / g) * k_h * k_w
  Linear : MACs = d_in * d_out
  Other layers (BN, ReLU, pooling) are O(activations) and are ignored for F
  because they are negligible compared with Conv/Linear MACs.

Theoretical costs used in the paper:
  Training   C_train(n) = 3 * F * n * E      (forward + backward ~ 3x forward)
  Inference  C_inf(m)   = F * m
  Memory     M_train    = P * (w + g + s) * 4 bytes + B * A * b_act
"""
import torch
import torch.nn as nn


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@torch.no_grad()
def profile_model(model: nn.Module, input_shape=(2, 3, 32, 32)) -> dict:
    macs = {"conv": 0, "dwconv": 0, "linear": 0}
    activations = [0]
    hooks = []

    def conv_hook(m: nn.Conv2d, inp, out):
        k = m.kernel_size[0] * m.kernel_size[1]
        mac = out.numel() * (m.in_channels // m.groups) * k
        key = "dwconv" if m.groups > 1 else "conv"
        macs[key] += mac

    def linear_hook(m: nn.Linear, inp, out):
        macs["linear"] += m.in_features * m.out_features * out.shape[0]

    def act_hook(m, inp, out):
        if isinstance(out, torch.Tensor):
            activations[0] += out.numel()

    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            hooks.append(m.register_forward_hook(conv_hook))
        elif isinstance(m, nn.Linear):
            hooks.append(m.register_forward_hook(linear_hook))
        if len(list(m.children())) == 0:  # leaf module
            hooks.append(m.register_forward_hook(act_hook))

    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    model(torch.zeros(*input_shape, device=device))
    model.train(was_training)
    for h in hooks:
        h.remove()

    bs = input_shape[0]
    total = sum(macs.values()) / bs
    n_layers = sum(1 for m in model.modules() if isinstance(m, (nn.Conv2d, nn.Linear)))
    return {
        "params": count_params(model),
        "macs_per_sample": total,
        "macs_conv": macs["conv"] / bs,
        "macs_dwconv": macs["dwconv"] / bs,
        "macs_linear": macs["linear"] / bs,
        "activations_per_sample": activations[0] / bs,
        "weight_layers": n_layers,
        "model_size_mb": count_params(model) * 4 / 2**20,
    }


def theoretical_train_memory_mb(params, activations_per_sample, batch_size, act_bytes=2):
    """Weights + grads + SGD momentum (fp32) + stored activations (fp16 under AMP)."""
    weights_states = params * 4 * 3
    acts = batch_size * activations_per_sample * act_bytes
    return (weights_states + acts) / 2**20


if __name__ == "__main__":
    from models import MODEL_REGISTRY
    for name, cls in MODEL_REGISTRY.items():
        p = profile_model(cls())
        print(f"{name:12s} params={p['params']/1e6:7.3f}M  MACs={p['macs_per_sample']/1e6:8.2f}M  "
              f"acts={p['activations_per_sample']/1e3:8.1f}K  layers={p['weight_layers']}")
