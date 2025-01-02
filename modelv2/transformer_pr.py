import math
import torch
import torch.nn as nn


""" 
    TODO: Create a transformer model that can read complex handwritten text
    - As much as possible make it encoder only for simplicity
    Rule 1: Simply the architecture compare to the previous model architecture
    Rule 2: Better accuracy compare to previous model
    Rule 3: Prevent Overfitting!
    Rule 4: Training with mixed dataset (Real data, Generated data, Medical words)
"""

class PositionalEncoding(nn.Module):
    def __init__(self):
        super().__init__()
        position = torch.arange(512).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, 768, 2) * (-math.log(10000.0) / 768))
        pe = torch.zeros(512, 1, 768)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x.transpose(0, 1)
        x = x + self.pe[:x.size(0), :]
        return x.transpose(0, 1)

class InputEmbeddings(nn.Module):
    def __init__(self):
        super().__init__()
        pass

    def forward(self, x):
        pass

class AttentionLayer(nn.Module):
    def __init__(self):
        super().__init__()
        pass

    def forward(self, x):
        pass

class EncoderMLPLayer(nn.Module):
    def __init__(self):
        super().__init__()
        pass

    def forward(self, x):
        pass

class EncoderLayer():
    def __init__(self):
        super().__init__()
        pass

    def forward(self, x):
        pass

class MLPLayer(nn.Module):
    def __init__(self):
        super().__init__()
        pass

    def forward(self, x):
        pass
