import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.Embed import DataEmbedding_inverted
from math import sqrt


class SDPAttention(nn.Module):
    """Scaled Dot-Product Attention using PyTorch's native SDPA.

    On Ampere+ GPUs (RTX 3090, A100, etc.) with CUDA 11.6+,
    SDPA automatically dispatches to the FlashAttention CUDA kernel
    for O(N) memory and faster wall-clock time.
    """

    def __init__(self, mask_flag=True, scale=None, attention_dropout=0.1, output_attention=False):
        super().__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout_p = attention_dropout

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        # Input: queries/keys/values (B, L, H, D) from AttentionLayer
        # SDPA expects (B, H, L, D)
        q = queries.transpose(1, 2)
        k = keys.transpose(1, 2)
        v = values.transpose(1, 2)

        dropout_p = self.dropout_p if self.training else 0.0

        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,  # no causal mask for iTransformer (cross-variate)
            dropout_p=dropout_p,
            scale=self.scale,
        )

        # Back to (B, L, H, D)
        out = out.transpose(1, 2).contiguous()
        return out, None


class SDPAttentionLayer(nn.Module):
    """Full attention layer with Q/K/V projections + SDPA."""

    def __init__(self, d_model, n_heads, attention_dropout=0.1):
        super().__init__()
        d_keys = d_model // n_heads
        d_values = d_model // n_heads

        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads
        self.dropout_p = attention_dropout

    def forward(self, queries, keys, values, attn_mask=None, tau=None, delta=None):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        q = self.query_projection(queries).view(B, L, H, -1).transpose(1, 2)
        k = self.key_projection(keys).view(B, S, H, -1).transpose(1, 2)
        v = self.value_projection(values).view(B, S, H, -1).transpose(1, 2)

        dropout_p = self.dropout_p if self.training else 0.0

        out = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)

        out = out.transpose(1, 2).contiguous().view(B, L, -1)
        return self.out_projection(out), None


class Model(nn.Module):
    """iTransformer with FlashAttention via PyTorch SDPA.

    Uses torch.nn.functional.scaled_dot_product_attention which
    automatically dispatches to FlashAttention v2 CUDA kernel
    on compatible hardware (Ampere+, CUDA 11.6+).
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

        # Encoder with SDPA attention
        self.encoder = Encoder(
            [
                EncoderLayer(
                    SDPAttentionLayer(
                        d_model=configs.d_model,
                        n_heads=configs.n_heads,
                        attention_dropout=configs.dropout,
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
