import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import AttentionLayer
from layers.Embed import DataEmbedding_inverted
from layers.Hierarchical_Attention import HierarchicalAttention


class Model(nn.Module):
    """VG-iFlashformer: Variable Grouping applied to iFlashformer.

    Uses HierarchicalAttention with SDPA (FlashAttention kernel) for
    intra-group local attention. Supports Shifted Grouping and FiLM broadcast.
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        self.use_norm = configs.use_norm

        # VG config
        self.num_groups = getattr(configs, 'num_groups', 8)
        self.pooling = getattr(configs, 'pooling', 'statistical')
        self.use_reorder = getattr(configs, 'use_reorder', 0)
        self.variate_indices = None
        self.group_sizes = None
        self.group_partition = getattr(configs, 'group_partition', 'window')

        # Key difference: always use SDPA for local attention
        self.use_sdpa = True

        # Embedding
        self.enc_embedding = DataEmbedding_inverted(
            configs.seq_len, configs.d_model, configs.embed, configs.freq, configs.dropout
        )
        self.class_strategy = configs.class_strategy

        # Encoder with Hierarchical Attention (SDPA local kernel)
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        HierarchicalAttention(
                            configs.enc_in, self.num_groups, configs.d_model,
                            n_heads=configs.n_heads,
                            attention_dropout=configs.dropout,
                            output_attention=configs.output_attention,
                            pooling=self.pooling,
                            use_variable_resolution=getattr(configs, 'use_variable_resolution', True),
                            use_interaction_bridge=getattr(configs, 'use_interaction_bridge', True),
                            use_global_interact=getattr(configs, 'use_global_interact', 1),
                            partition_strategy=getattr(configs, 'partition_strategy', 'softmax'),
                            dynamic_tokens_per_group=getattr(configs, 'dynamic_tokens_per_group', 1),
                            layer_idx=l,
                            use_shifted_grouping=getattr(configs, 'use_shifted_grouping', False),
                            use_gated_broadcast=getattr(configs, 'use_gated_broadcast', False),
                            num_layers=configs.e_layers,
                            use_multi_shift=getattr(configs, 'use_multi_shift', False),
                            use_film_broadcast=getattr(configs, 'use_film_broadcast', False),
                            use_sdpa=self.use_sdpa),
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        self.projector = nn.Linear(configs.d_model, configs.pred_len, bias=True)

    def _sync_group_plan_to_encoder(self, n_tokens):
        if getattr(self, '_synced_n_tokens', None) == n_tokens:
            return
        custom_sizes = None
        if self.group_partition in ('ragged', 'pack') and self.group_sizes is not None:
            if len(self.group_sizes) == self.num_groups:
                total_variates = sum(self.group_sizes)
                if total_variates <= n_tokens:
                    custom_sizes = list(self.group_sizes)
                    extra = n_tokens - total_variates
                    if extra > 0:
                        custom_sizes[-1] += extra
        device = next(self.parameters()).device
        for layer in self.encoder.attn_layers:
            inner = layer.attention.inner_attention
            if hasattr(inner, 'set_custom_group_sizes'):
                inner.set_custom_group_sizes(custom_sizes, device=device)
        self._synced_n_tokens = n_tokens

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        if self.use_norm:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev

        B, T, N = x_enc.shape

        if self.use_reorder:
            if self.variate_indices is None or self.variate_indices.shape[0] != N:
                g = torch.Generator()
                g.manual_seed(42)
                self.variate_indices = torch.randperm(N, generator=g).to(x_enc.device)
                self.inverse_indices = torch.argsort(self.variate_indices)
            x_enc = x_enc[:, :, self.variate_indices]
            if self.use_norm:
                stdev = stdev[:, :, self.variate_indices]
                means = means[:, :, self.variate_indices]

        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        if not torch.compiler.is_compiling():
            self._sync_group_plan_to_encoder(enc_out.shape[1])

        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        dec_out = self.projector(enc_out).permute(0, 2, 1)[:, :, :N]

        if self.use_norm:
            dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        if self.use_reorder:
            dec_out = dec_out[:, :, self.inverse_indices]

        return dec_out, attns

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        dec_out, attns = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns
        else:
            return dec_out[:, -self.pred_len:, :]
