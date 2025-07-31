import numpy as np
import torch

def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100
    return acc


def logits_to_label(pred_logits: torch.tensor):
    pred_probs=torch.softmax(pred_logits,dim=1)
    return torch.argmax(pred_probs)