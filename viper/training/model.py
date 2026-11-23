"""The models that can be fine-tuned, and the choice of trainable parameters."""
import re

from viper.megaloc.model import load_model as load_megaloc

# model name -> function weights_or_None -> torch module (SALAD is added with its package)
MODELS = {
    "megaloc": load_megaloc,
}

BLOCK = re.compile(r"blocks\.(\d+)\.")


def select_trainable(model, unfreeze_blocks=0, head_prefixes=None):
    """Freeze everything, then unfreeze the aggregation head and optionally the last backbone blocks.

    The head is every parameter whose name does not contain "backbone", unless head_prefixes names the head's
    modules. Prints the parameter counts per top-level module and returns the trainable parameters.
    """
    if head_prefixes:
        def is_head(name):
            return any(name.startswith(prefix) for prefix in head_prefixes)
    else:
        def is_head(name):
            return "backbone" not in name

    names = [name for name, _ in model.named_parameters()]
    blocks = sorted({int(m.group(1)) for name in names for m in [BLOCK.search(name)] if m and not is_head(name)})
    unfrozen = set(blocks[-unfreeze_blocks:]) if unfreeze_blocks > 0 else set()

    counts = {}                    # top-level module -> [total, trainable]
    for name, param in model.named_parameters():
        block = BLOCK.search(name)
        param.requires_grad = bool(is_head(name) or (block and not is_head(name) and int(block.group(1)) in unfrozen))
        count = counts.setdefault(name.split(".")[0], [0, 0])
        count[0] += param.numel()
        count[1] += param.numel() if param.requires_grad else 0

    print("[model] parameters by top-level module (total / trainable):")
    for module, (total, trainable) in counts.items():
        print(f"    {module:<20}{total / 1e6:8.2f}M {trainable / 1e6:8.2f}M")
    
    n_trainable = sum(c[1] for c in counts.values())
    n_total = sum(c[0] for c in counts.values())

    if n_trainable == 0:
        raise SystemExit("nothing is trainable: pass --head-prefixes with the module name(s) of the aggregation head listed above")
    
    if n_trainable == n_total and unfreeze_blocks == 0:
        print("[model] WARNING: every parameter is trainable, so the backbone was not recognised by name. "
              "Pass --head-prefixes with the head's module name(s) from the list above.")
    
    if unfreeze_blocks > 0:
        print(f"[model] backbone blocks unfrozen: {sorted(unfrozen)}")
    
    return [p for p in model.parameters() if p.requires_grad]
