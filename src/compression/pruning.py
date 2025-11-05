import torch
import torch.nn.utils.prune as prune

def prune_model(model, amount=0.3):
   
    for module in model.modules():
        if isinstance(module, torch.nn.Linear):
            prune.l1_unstructured(module, name='weight', amount=amount)
    print(f"Pruned Linear layers with amount={amount}")
    return model
