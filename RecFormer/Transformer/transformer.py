import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .transformer_layers import EncoderLayer


PAD = 0


def get_pad_mask(seq):
    return (seq != PAD).unsqueeze(-2)


def get_subsequent_mask(seq):
    ''' For masking out the subsequent info. '''
    sz_b, len_s = seq.size()
    subsequent_mask = (1 - torch.triu(
        torch.ones((1, len_s, len_s), device=seq.device), diagonal=1)).bool()
    return subsequent_mask


class PositionalEncoding(nn.Module):
    
    def __init__(self, d_hid, n_position=200):
        super(PositionalEncoding, self).__init__()
        # Not a parameter
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        ''' Sinusoid position encoding table '''

        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x):
        return x + self.pos_table[:, :x.size(1)].clone().detach()


class TransformerEncoder(nn.Module):
    def __init__(self, config):
        super(TransformerEncoder, self).__init__()

        self.n_layers = config['n_layers']

        n_heads = config['n_heads']
        d_k = config['encode_dim']
        d_v = config['encode_dim']
        d_model = config['encode_dim']
        d_inner = config['encode_dim'] * 4
        dropout = config['dropout']
        self.scale_emb = True
        self.d_model = d_model

        # + 1 for mask
        self.track_embedding = nn.Embedding(config['track_num']+1, config['encode_dim'], scale_grad_by_freq=True)
        # self.artist_embedding = nn.Embedding(config['artist_num']+1, config['encode_dim'])
        self.gender_embedding = nn.Embedding(config['gender_num'], config['encode_dim'])
        self.country_embedding = nn.Embedding(config['country_num'], config['encode_dim'])
        self.position_embedding = PositionalEncoding(config['encode_dim'], n_position=config['max_len'])

        # self.novelty_artist_embedding = nn.Sequential(
        #     nn.Linear(config['user_numerical_num'], config['encode_dim'])
        # )

        self.dropout = nn.Dropout(p=dropout)

        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model=d_model, d_inner=d_inner, n_head=n_heads, d_k=d_k, d_v=d_v, dropout=dropout)
            for _ in range(self.n_layers)])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

        self.classifier = nn.Linear(config['encode_dim'], config['track_num'])

        # # two layer achieved better hit-rate but worse in overall score
        # self.classifier = nn.Sequential(
        #     nn.Linear(config['encode_dim'], config['encode_dim']*2),
        #     nn.GELU(),
        #     nn.Linear(config['encode_dim']*2, config['track_num'])
        # )
        

    def forward(self, sequences, artists, genders, countrys, novelty_artists, src_mask=None, return_attns=False):
        enc_slf_attn_list = []

        # mask = get_pad_mask(sequences) & get_subsequent_mask(sequences)

        embedded_tracks = self.track_embedding(sequences)
        # embedded_artists = self.artist_embedding(artists)
        embedded_genders = self.gender_embedding(genders)
        embedded_countrys = self.country_embedding(countrys)

        # # add this will become nan loss
        # novelty_artists = torch.cat([novelty_artists[0].unsqueeze(-1), novelty_artists[1].unsqueeze(-1), novelty_artists[2].unsqueeze(-1)], dim=-1).to(sequences.device)
        # embedded_novelty_artists = self.novelty_artist_embedding(novelty_artists)

        embeddings = embedded_tracks + embedded_genders + embedded_countrys

        # Forward
        if self.scale_emb:
            embeddings *= self.d_model ** 0.5

        encode_output = self.dropout(self.position_embedding(embeddings))
        encode_output = self.layer_norm(encode_output)

        for enc_layer in self.layer_stack:
            encode_output, enc_slf_attn = enc_layer(encode_output)

        logits = self.classifier(encode_output)

        if return_attns:
            return logits, enc_slf_attn_list
        return logits