import torch.nn as nn
from model.constant import CHARS, PAD_TOKEN, CHAR_TO_INDEX

INPUT_IMAGE_SIZE = (128, 1024)
MAX_SEQUENCE_LENGTH = 512
NUM_CLASSES = 128
ENCODER_EMB = 256
NUM_ENCODER_LAYER = 8
NUM_ENC_ATTN_HEADS = 8
MLP_DIMENSION = ENCODER_EMB*4
ENC_DROPOUT = 0.2

DECODER_EMB = 256
DIM_FEED_FORWARD = DECODER_EMB*4
DEC_DROPOUT = 0.2
NUM_DECODER_LAYER = 8
NUM_DEC_ATTN_HEADS = 8

class EncoderConfig:
    image_size=INPUT_IMAGE_SIZE
    num_classes=len(CHARS)
    max_sequence_length=MAX_SEQUENCE_LENGTH
    embedding_dimension=ENCODER_EMB
    num_encoder_layers=NUM_ENCODER_LAYER
    num_attention_heads=NUM_ENC_ATTN_HEADS
    intermediate_size=MLP_DIMENSION
    encoder_dropout=ENC_DROPOUT

class DecoderConfig:
    vocab_size=len(CHARS)
    embedding_dimension=DECODER_EMB
    num_decoder_layers=NUM_DECODER_LAYER
    num_attention_heads=NUM_DEC_ATTN_HEADS
    intermediate_size=DIM_FEED_FORWARD
    embedding_act=nn.GELU()
    dec_dropout=DEC_DROPOUT
    max_sequence_length=MAX_SEQUENCE_LENGTH
    layer_norm_eps=1e-12
    pad_token_id=CHAR_TO_INDEX[PAD_TOKEN]
