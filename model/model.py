import torch
import torch.nn as nn

from model.utils import generate_square_mask
from model.configurations import DecoderConfig, EncoderConfig
from model.encoder import Encoder
from model.decoder import Decoder

class TransformerOCR(nn.Module):
    def __init__(self, encoder=Encoder(), decoder=Decoder()):
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder

        self.encoder_embedding = EncoderConfig.embedding_dimension
        self.decoder_embedding = DecoderConfig.embedding_dimension

        self.loss_function = nn.CrossEntropyLoss()

    def forward(self, image, expected_text):
        encoder_label = expected_text[0].to("cuda")
        decoder_label = expected_text[1].to("cuda")

        encoder_output, encoder_loss = self.encoder(image, encoder_label)
        batch, length, _ = encoder_output.shape
        encoder_attn_mask = torch.ones((batch, length), device="cuda")

        expected_mask = generate_square_mask(decoder_label.shape[1])
        decoder_output = self.decoder(decoder_label, encoder_output, expected_mask, encoder_attn_mask)

        shifted_prediction_score = decoder_output[:, :-1, :].contiguous()
        expected_text = decoder_label[:, 1:].contiguous()
        decoder_loss = self.loss_function(shifted_prediction_score.view(-1, DecoderConfig.vocab_size), expected_text.view(-1))

        return decoder_loss, encoder_loss, decoder_output, expected_text

    def encode(self, image):
        return self.encoder(image)
    
    def decode(self, start_token, encoder_hidden_state, decoder_mask, encoder_mask):
        return self.decoder(start_token, encoder_hidden_state, decoder_mask, encoder_mask)

class HybridLoss(nn.Module):
    def __init__(self, lambda_val=0.5):
        super().__init__()
        self.lambda_val = lambda_val

    def forward(self, ctc_loss, ce_loss):
        return self.lambda_val * ctc_loss + (1 - self.lambda_val) * ce_loss
