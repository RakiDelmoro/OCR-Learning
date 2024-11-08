import torch
import torch.nn as nn
import math

from model.configurations import EncoderConfig, DecoderConfig
from model.constant import PAD_TOKEN, CHAR_TO_INDEX
from model.utils import length_of_actual_tokens

class PositionalEncoding(nn.Module):

    def __init__(self, config=EncoderConfig):
        super().__init__()
        self.dropout = nn.Dropout(p=0.1)

        position = torch.arange(config.max_sequence_length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, config.embedding_dimension, 2) * (-math.log(10000.0) / config.embedding_dimension))
        pe = torch.zeros(config.max_sequence_length, 1, config.embedding_dimension)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x.transpose(0, 1)
        x = x + self.pe[:x.size(0)]
        
        return self.dropout(x).transpose(0, 1)

class CnnFeatureExtractor(nn.Module):
    def __init__(self, config=EncoderConfig):
        super().__init__()
        image_height, image_width = config.image_size
        self.conv1_image_h, self.conv1_image_w = (image_height-2), (image_width-2)
        self.first_layer_conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3,3), stride=1),
            nn.LeakyReLU(),
            nn.LayerNorm((8, self.conv1_image_h, self.conv1_image_w)),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=0.2))
        self.conv2_image_h, self.conv2_image_w = (self.conv1_image_h//2)-2, (self.conv1_image_w//2)-2,
        self.second_layer_conv = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3,3), stride=1),
            nn.LeakyReLU(),
            nn.LayerNorm((16, self.conv2_image_h, self.conv2_image_w)),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=0.2))
        self.conv3_image_h, self.conv3_image_w = (self.conv2_image_h//2)-2, (self.conv2_image_w//2)-2,
        self.third_layer_conv = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3,3), stride=1),
            nn.LeakyReLU(),
            nn.LayerNorm((32, self.conv3_image_h, self.conv3_image_w)),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=0.2))    
        self.conv4_image_h, self.conv4_image_w = (self.conv3_image_h//2)-2, (self.conv3_image_w//2)-2,
        self.fourth_layer_conv = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3,3), stride=1),
            nn.LeakyReLU(),
            nn.LayerNorm((64, self.conv4_image_h, self.conv4_image_w)),
            nn.Dropout(p=0.2))
        self.conv5_image_h, self.conv5_image_w = self.conv4_image_h-6, self.conv4_image_w-6
        self.fifth_layer_conv = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(7, 7), stride=1),
            nn.LeakyReLU(),
            nn.LayerNorm((128, self.conv5_image_h, self.conv5_image_w)),
            nn.Dropout(p=0.2)
        )
        self.dense_layer = nn.Linear(768, config.embedding_dimension)
        self.position_embeddings = nn.Parameter(torch.zeros(1, 118, config.embedding_dimension))
    def forward(self, x):
        # batch | channels | height | width
        conv_1 = self.first_layer_conv(x)
        conv_2 = self.second_layer_conv(conv_1)
        conv_3 = self.third_layer_conv(conv_2)
        conv_4 = self.fourth_layer_conv(conv_3)
        conv_5 = self.fifth_layer_conv(conv_4)
        # batch | channels | height | width -> batch | channels | embedding_dimension
        batch, _, _, width = conv_5.shape
        collapse_layer = conv_5.view(batch, -1, width).transpose(1, 2)
        dense_layer = self.dense_layer(collapse_layer)
        pos_encoding = dense_layer + self.position_embeddings
        return pos_encoding
    
class SelfAttention(nn.Module):
    def __init__(self, config=EncoderConfig):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.atttention_embedding_dim = int(config.embedding_dimension // config.num_attention_heads)
        self.combine_embedding_size = self.num_heads * self.atttention_embedding_dim
        self.query = nn.Linear(config.embedding_dimension, self.combine_embedding_size)
        self.key = nn.Linear(config.embedding_dimension, self.combine_embedding_size)
        self.value = nn.Linear(config.embedding_dimension, self.combine_embedding_size)
        self.dropout = nn.Dropout(config.encoder_dropout)
    def transpose_for_attn_scores(self, x: torch.Tensor) -> torch.Tensor:
        # batch | patches | attn_heads | attention_dimension
        new_x_shape = x.shape[:-1] + (self.num_heads, self.atttention_embedding_dim)
        x = x.view(new_x_shape)
        # batch | attn_heads | patches | attention_dimension
        return x.permute(0, 2, 1, 3)
    def forward(self, hidden_states: torch.Tensor):
        query_layer = self.transpose_for_attn_scores(self.query(hidden_states))
        key_layer = self.transpose_for_attn_scores(self.key(hidden_states))
        value_layer = self.transpose_for_attn_scores(self.value(hidden_states))
        # Take the dot product of "query" and "key" to the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-2, -1))
        attention_scores = attention_scores / math.sqrt(self.atttention_embedding_dim)
        # normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.shape[:-2] + (self.combine_embedding_size,)
        context_layer = context_layer.view(new_context_layer_shape)
        return context_layer

class SelfAttetionOutput(nn.Module):
    def __init__(self, config=EncoderConfig):
        super().__init__()
        self.dense = nn.Linear(config.embedding_dimension, config.embedding_dimension)
        self.dropout = nn.Dropout(config.encoder_dropout)
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states

class MultiHeadAttention(nn.Module):
    def __init__(self, config=EncoderConfig):
        super().__init__()
        self.attention = SelfAttention(config)
        self.output = SelfAttetionOutput(config)
    def forward(self, hidden_states: torch.Tensor):
        self_attention = self.attention(hidden_states)
        attention_output = self.output(self_attention)
        return attention_output

class FeedForward(nn.Module):
    def __init__(self, config=EncoderConfig):
        super().__init__()
        self.ff_neural_network = nn.Sequential(
            nn.Linear(config.embedding_dimension, config.intermediate_size),
            nn.GELU(),
            nn.Linear(config.intermediate_size, config.embedding_dimension),
            nn.Dropout(config.encoder_dropout))
    def forward(self, hidden_states: torch.Tensor):
        return self.ff_neural_network(hidden_states)

class Layer(nn.Module):
    def __init__(self, config=EncoderConfig):
        super().__init__()
        self.attention = MultiHeadAttention(config)
        self.ff_layer = FeedForward(config)
        self.layer_norm_before = nn.LayerNorm(config.embedding_dimension)
        self.layer_norm_after = nn.LayerNorm(config.embedding_dimension)
    def forward(self, hidden_states: torch.Tensor):
        self_attention_outputs = self.attention(self.layer_norm_before(hidden_states))
        # first residual connection
        hidden_states = self_attention_outputs + hidden_states
        # Apply Layer Norm
        attention_output = self.layer_norm_after(hidden_states)
        # Feed Forward
        feed_forward_output = self.ff_layer(attention_output)
        # second residual connection
        feed_forward_with_residual_connection = feed_forward_output + attention_output
        return feed_forward_with_residual_connection
    
class Encoder(nn.Module):
    def __init__(self, config=EncoderConfig):
        super().__init__()
        self.config = config
        self.embeddings = CnnFeatureExtractor()
        self.layer = Layer()
        self.decoder_embedding = nn.Linear(config.embedding_dimension, DecoderConfig.embedding_dimension)
        self.positional_encoding_for_decoder = PositionalEncoding(DecoderConfig)
        self.encoder_layer = nn.ModuleList([self.layer for _ in range(config.num_encoder_layers)])
        self.character_alignment = nn.Linear(config.embedding_dimension, config.num_classes)
        self.activation_function = nn.LogSoftmax(dim=-1)
    def forward(self, image: torch.Tensor, target=None):
        hidden_states = self.embeddings(image)
        for layer_module in self.encoder_layer:
            layer_outputs = layer_module(hidden_states)
        encoder_hidden_state = self.positional_encoding_for_decoder(self.decoder_embedding(layer_outputs))
        prediction_score = self.activation_function(self.character_alignment(layer_outputs))
        training_mode = target is not None
        if training_mode:
            transpose_for_loss = prediction_score.transpose(0, 1)
            length, batch, _ = transpose_for_loss.shape
            image_patches_length = torch.full(size=(batch,), fill_value=length, dtype=torch.long)
            target_length = torch.zeros(size=(batch,), dtype=torch.long)
            actual_tokens_length = length_of_actual_tokens(target)
            for each in range(batch): target_length[each] = actual_tokens_length[each]
            loss_func = nn.CTCLoss(blank=CHAR_TO_INDEX[PAD_TOKEN])
            loss = loss_func(transpose_for_loss, target, image_patches_length, target_length)
            return encoder_hidden_state, loss            
        else:
            return encoder_hidden_state
