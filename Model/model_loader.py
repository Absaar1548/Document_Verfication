import torch
import torch.nn as nn
import torch.optim as optim
import os
import sys
from cnn import MODELS

def load_model(model_name: str, model_path: str , device : str):

    model = MODELS[model_name]
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        model.to(device)
        print(f"Model weights loaded from {model_path}")
    else:
        print(f"No weights found for model {model_name}. Using untrained model.")

    return model

# Example of how to use this:
# model = load_model('third_model_name', model_type='third_model')
