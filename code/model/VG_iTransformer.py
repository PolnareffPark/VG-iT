import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import AttentionLayer, FullAttention
from layers.Embed import DataEmbedding_inverted
from layers.Hierarchical_Attention import HierarchicalAttention
from layers.Learnable_Grouping import LearnableGrouping, VariateReconstruction


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        self.use_norm = configs.use_norm
        
        # VG-iT Specific
        self.num_groups = getattr(configs, 'num_groups', 8)
        self.use_learnable_grouping = getattr(configs, 'use_learnable_grouping', False)
        self.pooling = getattr(configs, 'pooling', 'statistical')
        self.use_reorder = getattr(configs, 'use_reorder', 0)
        self.variate_indices = None # Will be initialized in forecast based on N
        self.group_sizes = None
        self.group_partition = getattr(configs, 'group_partition', 'window')
        self.use_sdpa = getattr(configs, 'use_sdpa', False)

        # Embedding
        self.enc_embedding = DataEmbedding_inverted(configs.seq_len, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
        self.class_strategy = configs.class_strategy
        
        # Optional: Learnable Grouping Layer (Plan B)
        if self.use_learnable_grouping:
            self.grouping_layer = LearnableGrouping(configs.enc_in, self.num_groups, configs.d_model, configs.dropout)
            self.reconstruction_layer = VariateReconstruction(configs.d_model)
            self.recon_norm = nn.LayerNorm(configs.d_model)

            # Encoder that operates on group tokens (length = G)
            self.group_encoder = Encoder(
                [
                    EncoderLayer(
                        AttentionLayer(
                            FullAttention(
                                mask_flag=False,  # Important: Disable causal mask for encoder
                                attention_dropout=configs.dropout,
                                output_attention=configs.output_attention
                            ),
                            configs.d_model, configs.n_heads
                        ),
                        configs.d_model,
                        configs.d_ff,
                        dropout=configs.dropout,
                        activation=configs.activation
                    ) for _ in range(configs.e_layers)
                ],
                norm_layer=nn.LayerNorm(configs.d_model)
            )
            self.last_group_stats = {}

        # Encoder with Hierarchical Attention
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                         HierarchicalAttention(configs.enc_in, self.num_groups, configs.d_model,
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
                                            film_rank=getattr(configs, 'film_rank', 'full'),
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

    def _project_forecast(self, enc_out, x_enc, n_variates):
        return self.projector(enc_out).permute(0, 2, 1)[:, :, :n_variates]

    def _sync_group_plan_to_encoder(self, n_tokens):
        # Skip if already synced for this n_tokens
        if getattr(self, '_synced_n_tokens', None) == n_tokens:
            return
        custom_sizes = None
        if self.group_partition in ('ragged', 'pack') and self.group_sizes is not None:
            if len(self.group_sizes) == self.num_groups:
                total_variates = sum(self.group_sizes)
                # Exact match or extra tokens from embedding (e.g. time features)
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
        
        # Ablation 1: Deterministic variate reordering
        if self.use_reorder:
            if self.variate_indices is None or self.variate_indices.shape[0] != N:
                # Use a fixed seed for shuffling based on the existing seed + a constant
                g = torch.Generator()
                g.manual_seed(42) # Fixed seed for the shuffle itself to be consistent across batches
                self.variate_indices = torch.randperm(N, generator=g).to(x_enc.device)
                self.inverse_indices = torch.argsort(self.variate_indices)

            x_enc = x_enc[:, :, self.variate_indices]
            if self.use_norm:
                stdev = stdev[:, :, self.variate_indices]
                means = means[:, :, self.variate_indices]

        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        if not torch.compiler.is_compiling():
            self._sync_group_plan_to_encoder(enc_out.shape[1])
        
        if self.use_learnable_grouping:
            # 1) Grouping
            group_reps, weights = self.grouping_layer(enc_out) # (B, G, E), (B, N, G)

            # --- A3 logging for Plan B ---
            with torch.no_grad():
                w = weights.clamp_min(1e-9)
                ent = -(w * w.log()).sum(dim=-1).mean()
                eff = ent.exp()
                load = weights.sum(dim=(0, 1))
                load = load / (load.sum() + 1e-9)
                imb = load.var(unbiased=False)
            self.last_group_stats = {"entropy": float(ent), "eff_groups": float(eff), "load_imbalance": float(imb)}

            # 2) Encode on group tokens
            group_reps, group_attns = self.group_encoder(group_reps, attn_mask=None)

            # 3) Reconstruct back to N tokens
            enc_out = self.reconstruction_layer(group_reps, weights) # (B, N, E)
            enc_out = self.recon_norm(enc_out)
            attns = {"group": group_attns}
        else:
            enc_out, attns = self.encoder(enc_out, attn_mask=None)
        
        # Project and filter covariates (if any). Optional anchor-residual
        # head keeps the output in the same normalized space before de-normalization.
        dec_out = self._project_forecast(enc_out, x_enc, N)
        
        # De-Normalize while in current order (shuffled or original)
        if self.use_norm:
            dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        # Restore original order if reordered (Ablation 1)
        if self.use_reorder:
            dec_out = dec_out[:, :, self.inverse_indices]

        return dec_out, attns

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        dec_out, attns = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns
        else:
            return dec_out[:, -self.pred_len:, :]
