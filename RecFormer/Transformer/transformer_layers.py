import torch.nn as nn
import torch
from .transformer_submodules import MultiHeadAttention, PositionwiseFeedForward


class EncoderLayer(nn.Module):
    ''' Compose with two layers '''

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head=n_head, d_model=d_model, d_k=d_k, d_v=d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_in=d_model, d_hid=d_inner, dropout=dropout)

    def forward(self, encode_input, hours, slf_attn_mask=None):
        encode_output, enc_slf_attn = self.slf_attn(q=encode_input, k=encode_input, v=encode_input, hours=hours, mask=slf_attn_mask)
        encode_output = self.pos_ffn(encode_output)
        return encode_output, enc_slf_attn