import math
import torch
import torch.nn as nn
from model.configurations import DecoderConfig

def create_position_ids_from_input_ids(input_ids: torch.Tensor, padding_idx, past_key_values_length=0):
    mask = input_ids.ne(padding_idx).int()
    incremental_indices = (torch.cumsum(mask, dim=1).type_as(mask) + past_key_values_length) * mask
    return incremental_indices.long() + padding_idx

class PositionalEncoding(nn.Module):
    def __init__(self, config=DecoderConfig):
        super().__init__()
        self.dropout = nn.Dropout(p=config.dec_dropout)
        position = torch.arange(config.max_sequence_length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, config.embedding_dimension, 2) * (-math.log(10000.0) / config.embedding_dimension))
        pe = torch.zeros(config.max_sequence_length, 1, config.embedding_dimension)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Arguments: x: Tensor, shape ``[seq_len, batch_size, embedding_dim]`` """
        x = x.transpose(0, 1)
        x = x + self.pe[:x.size(0)]
        return self.dropout(x).transpose(0, 1)

class CharacterEmbeddings(nn.Module):
    def __init__(self, config=DecoderConfig):
        super().__init__()
        self.character_embeddings = nn.Embedding(config.vocab_size, config.embedding_dimension,
                                                 config.pad_token_id)
        self.register_buffer("position_ids", torch.arange(config.max_sequence_length).expand((1, -1)), persistent=False)

        self.padding_idx = config.pad_token_id
        self.position_embeddings = nn.Embedding(
            config.max_sequence_length, config.embedding_dimension, padding_idx=self.padding_idx)
    def forward(self, input_ids):
        character_embedding = self.character_embeddings(input_ids)
        position_ids = create_position_ids_from_input_ids(input_ids, self.padding_idx)
        position_embeddings = self.position_embeddings(position_ids)

        embeddings = character_embedding + position_embeddings
        
        return embeddings

class AttentionMechanism(nn.Module):
    def __init__(self, config=DecoderConfig):
        super().__init__()

        self.num_heads = config.num_attention_heads
        self.attention_embedding_dim = int(config.embedding_dimension // config.num_attention_heads)
        self.combine_embedding_size = self.num_heads * self.attention_embedding_dim

        self.query = nn.Linear(config.embedding_dimension, self.combine_embedding_size)
        self.key = nn.Linear(config.embedding_dimension, self.combine_embedding_size)
        self.value = nn.Linear(config.embedding_dimension, self.combine_embedding_size)

        self.dense = nn.Linear(config.embedding_dimension, config.embedding_dimension)
        self.dropout = nn.Dropout(config.dec_dropout)

        self.max_position_embeddings = config.max_sequence_length
        self.distance_embedding = nn.Embedding(2 * config.max_sequence_length - 1, self.attention_embedding_dim)
    
    def transpose_for_attn_scores(self, x: torch.Tensor) -> torch.Tensor:
        # batch | patches | attn_heads | attention_dimension
        new_x_shape = x.shape[:-1] + (self.num_heads, self.attention_embedding_dim)
        x = x.view(new_x_shape)

        # batch | attn_heads | patches | attention_dimension
        return x.permute(0, 2, 1, 3)
    
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_state=None, attn_mask=None):
        is_cross_attention = encoder_hidden_state is not None
        
        if is_cross_attention:
            query_layer = self.transpose_for_attn_scores(self.query(hidden_states))
            key_layer = self.transpose_for_attn_scores(self.key(encoder_hidden_state))
            value_layer = self.transpose_for_attn_scores(self.value(encoder_hidden_state))
            attn_mask = attn_mask.unsqueeze(1).unsqueeze(2)
        else:
            query_layer = self.transpose_for_attn_scores(self.query(hidden_states))
            key_layer = self.transpose_for_attn_scores(self.key(hidden_states))
            value_layer = self.transpose_for_attn_scores(self.value(hidden_states))

        # Take the dot product of "query" and "key" to the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-2, -1))

        query_length, key_length = query_layer.shape[2], key_layer.shape[2]
        position_ids_left = torch.arange(query_length, dtype=torch.long,
                                            device=hidden_states.device).view(-1, 1)
        position_ids_right = torch.arange(key_length, dtype=torch.long,
                                            device=hidden_states.device).view(1, -1)
        distance = position_ids_left - position_ids_right
        position_embedding = self.distance_embedding(distance + self.max_position_embeddings - 1)
        positional_embedding = position_embedding.type(query_layer.dtype)

        relative_position_scores_query = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
        relative_position_scores_key = torch.einsum("bhrd,lrd->bhlr", key_layer, positional_embedding) 
        attention_scores = attention_scores + relative_position_scores_query + relative_position_scores_key
        
        attention_scores = attention_scores / math.sqrt(self.attention_embedding_dim)
        if attn_mask is not None:
            attention_scores = attention_scores + attn_mask

        # normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.shape[:-2] + (self.combine_embedding_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        hidden_state = self.dense(context_layer)
        return self.dropout(hidden_state)
    
class FeedForward(nn.Module):
    def __init__(self, config=DecoderConfig):
        super().__init__()
        self.ff_neural_network = nn.Sequential(
            nn.Linear(config.embedding_dimension, config.intermediate_size),
            nn.GELU(),
            nn.Linear(config.intermediate_size, config.embedding_dimension),
            nn.Dropout(config.dec_dropout))

    def forward(self, hidden_states: torch.Tensor):
        return self.ff_neural_network(hidden_states)

class Layer(nn.Module):
    def __init__(self, config=DecoderConfig):
        super().__init__()
        
        self.self_attention = AttentionMechanism()
        self.cross_attention = AttentionMechanism()
        self.ff_layer = FeedForward()
        self.layer_norm = nn.LayerNorm(config.embedding_dimension)

    def forward(self, hidden_states: torch.Tensor, encoder_hidden_state: torch.Tensor, decoder_mask: torch.Tensor, encoder_mask: torch.Tensor):
        # Self Attention
        self_attention_output = self.self_attention(self.layer_norm(hidden_states), None, decoder_mask) # Encoder hidden state set to None because it's self attention.
        first_residual_hidden_state = self_attention_output + hidden_states
        self_attention_output = self.layer_norm(first_residual_hidden_state)
        # Cross attention
        cross_attention = self.cross_attention(self_attention_output, encoder_hidden_state, encoder_mask)
        second_residual_hidden_state = cross_attention + self_attention_output
        cross_attention_output = self.layer_norm(second_residual_hidden_state)
        # Feed Forward
        feed_forward_output = self.ff_layer(cross_attention_output)
        third_residual_connection = feed_forward_output + cross_attention_output
        feed_forward_output = self.layer_norm(third_residual_connection)
        return feed_forward_output
    
class DecoderLayer(nn.Module):
    def __init__(self, config=DecoderConfig):
        super().__init__()
        self.decoder_layers = nn.ModuleList([Layer() for _ in range(config.num_decoder_layers)])

    def forward(self, input_embeddings, encoder_hidden_state, decoder_mask, encoder_mask):
        layer_output = input_embeddings
        for each in self.decoder_layers:
            layer_output = each(layer_output, encoder_hidden_state, decoder_mask, encoder_mask)
        return layer_output

class Decoder(nn.Module):
    def __init__(self, config=DecoderConfig):
        super().__init__()
        self.config = config
        self.embeddings = CharacterEmbeddings()
        self.decoder_layers = DecoderLayer()
        self.dense = nn.Linear(config.embedding_dimension, config.vocab_size)

    def forward(self, target, encoder_hidden_state, decoder_mask, encoder_mask):
        hidden_states = self.embeddings(target)
        layer_outputs = self.decoder_layers(hidden_states, encoder_hidden_state, decoder_mask, encoder_mask)
        return self.dense(layer_outputs)
