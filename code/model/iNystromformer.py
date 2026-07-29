import torch
import torch.nn as nn
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.Embed import DataEmbedding_inverted
from nystrom_attention import NystromAttention


class NystromAttentionLayer(nn.Module):
    """Wraps lucidrains NystromAttention to match AttentionLayer interface.

    NystromAttention handles its own Q/K/V projections and head splitting,
    so this layer simply delegates to it while conforming to the
    (queries, keys, values, attn_mask) -> (output, attn) interface
    expected by EncoderLayer.
    """

    def __init__(self, d_model, n_heads, num_landmarks=64, dropout=0.1):
        super().__init__()
        self.nystrom = NystromAttention(
            dim=d_model,
            dim_head=d_model // n_heads,
            heads=n_heads,
            num_landmarks=num_landmarks,
            dropout=dropout,
        )

    def forward(self, queries, keys, values, attn_mask=None, tau=None, delta=None):
        # NystromAttention is self-attention: uses only `queries` (= x)
        out = self.nystrom(queries)
        return out, None


class Model(nn.Module):
    """iTransformer with Nystrom Attention (O(N*m) complexity).

    Replaces standard O(N^2) attention with Nystrom approximation
    using landmark-based kernel approximation.
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        self.use_norm = configs.use_norm

        # Embedding
        self.enc_embedding = DataEmbedding_inverted(
            configs.seq_len, configs.d_model, configs.embed, configs.freq, configs.dropout
        )

        num_landmarks = getattr(configs, 'num_landmarks', 64)

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    NystromAttentionLayer(
                        d_model=configs.d_model,
                        n_heads=configs.n_heads,
                        num_landmarks=num_landmarks,
                        dropout=configs.dropout,
                    ),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for _ in range(configs.e_layers)
            ],
            norm_layer=nn.LayerNorm(configs.d_model),
        )
        self.projector = nn.Linear(configs.d_model, configs.pred_len, bias=True)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        if self.use_norm:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev

        _, _, N = x_enc.shape

        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        dec_out = self.projector(enc_out).permute(0, 2, 1)[:, :, :N]

        if self.use_norm:
            dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        return dec_out, attns

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        dec_out, attns = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        if self.output_attention:
            return dec_out[:, -self.pred_len :, :], attns
        else:
            return dec_out[:, -self.pred_len :, :]
