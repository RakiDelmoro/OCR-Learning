import torch
from model.constant import CHAR_TO_INDEX, PAD_TOKEN, DEVICE

def length_of_actual_tokens(batch):
    mask = (batch != CHAR_TO_INDEX[PAD_TOKEN]).sum(1)
    return mask

def generate_square_mask(size):
    mask = (torch.triu(torch.ones((size, size), device=DEVICE)) == 1).transpose(1, 0)
    mask = mask.float().masked_fill(mask == 0, float("-inf")).masked_fill(mask == 1, float(0.0)).to(DEVICE)
    return mask
