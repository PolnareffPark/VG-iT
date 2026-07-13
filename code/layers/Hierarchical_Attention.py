import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt, ceil

class _InjectiveIndexPut(torch.autograd.Function):
    """Scatter src into zero tensor at injective indices (CUDA-graph safe).
    Uses scatter_/gather only — no index_put_ in forward or backward."""
    @staticmethod
    def forward(ctx, src, idx_4d, total_len):
        # src: (B, N, H, D), idx_4d: (1, N, 1, 1)
        idx_exp = idx_4d.expand_as(src)
        ctx.save_for_backward(idx_exp)
        out = src.new_zeros(src.shape[0], total_len, *src.shape[2:])
        out.scatter_(1, idx_exp, src)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        (idx_exp,) = ctx.saved_tensors
        return grad_output.gather(1, idx_exp), None, None


class _InjectiveIndexSelect(torch.autograd.Function):
    """Gather from src at injective indices (CUDA-graph safe).
    Uses gather/scatter_ only — no index_put_ in forward or backward."""
    @staticmethod
    def forward(ctx, src, idx_4d):
        # src: (B, L, H, D), idx_4d: (1, N, 1, 1)
        idx_exp = idx_4d.expand(src.shape[0], -1, *src.shape[2:])
        ctx.save_for_backward(idx_exp)
        ctx.src_shape = src.shape
        return src.gather(1, idx_exp)

    @staticmethod
    def backward(ctx, grad_output):
        (idx_exp,) = ctx.saved_tensors
        grad_src = grad_output.new_zeros(ctx.src_shape)
        grad_src.scatter_(1, idx_exp, grad_output)
        return grad_src, None


class HierarchicalAttention(nn.Module):
    """
    Hybrid Hierarchical Attention (Phase 12)
    Logic: Local Direct Interaction + Global Bridge
    Complexity: O(N^2/G + G^2)
    """
    def __init__(self, n_vars, num_groups, d_model, n_heads=8, attention_dropout=0.1, output_attention=False,
                 pooling='statistical', use_variable_resolution=True, use_interaction_bridge=True,
                 use_global_interact=True, partition_strategy='softmax', dynamic_tokens_per_group=1,
                 layer_idx=0, use_shifted_grouping=False, use_gated_broadcast=False,
                 num_layers=1, use_multi_shift=False, use_film_broadcast=False,
                 film_rank='full', use_sdpa=False):
        super(HierarchicalAttention, self).__init__()
        self.num_groups = num_groups
        self.d_model = d_model
        self.output_attention = output_attention
        self.pooling = pooling
        self.use_variable_resolution = use_variable_resolution
        self.use_interaction_bridge = use_interaction_bridge
        self.use_global_interact = use_global_interact
        self.partition_strategy = partition_strategy
        self.dynamic_tokens_per_group = dynamic_tokens_per_group
        self.dropout = nn.Dropout(attention_dropout)
        self.layer_idx = layer_idx
        self.use_shifted_grouping = use_shifted_grouping
        self.use_gated_broadcast = use_gated_broadcast
        self.use_multi_shift = use_multi_shift
        self.num_layers = num_layers
        self.use_film_broadcast = use_film_broadcast

        # Local Attention (Intra-group): scaled dot-product (SDPA kernel optional)
        self.local_attn = MultiHeadAttention(attention_dropout, use_sdpa=use_sdpa)

        # Global Attention (Inter-group): scaled dot-product
        self.global_attn = MultiHeadAttention(attention_dropout)
        
        # Dynamic Grouping (Phase 17)
        if pooling == 'dynamic':
            # 1. Dynamic Partitioning: Learnable mapping
            self.score_projection = nn.Linear(d_model, num_groups * dynamic_tokens_per_group)
            self.temp = nn.Parameter(torch.ones(1) * 0.1) 
            
            # 2. Variable Resolution: Salience Gate
            if use_variable_resolution:
                self.salience_gate = nn.Linear(d_model, 1)
            
            # 3. Cross-Interaction Bridge: Gated Integration
            if use_interaction_bridge:
                self.interaction_gate = nn.Linear(d_model * 2, d_model)
            self.refinement = nn.Linear(d_model, d_model)
        
        # Statistical Pooling: feature dim concat + learned projection
        if pooling == 'statistical':
            D = d_model // n_heads
            self.stat_proj = nn.Linear(3 * D, D)

        # Learnable Pooling: group-specific queries + dedicated attention
        if pooling == 'learnable':
            D = d_model // n_heads
            self.query_gen = nn.Parameter(torch.randn(1, num_groups, 1, D) / sqrt(D))
            self.pool_attn = MultiHeadAttention(attention_dropout)

        # Gated Broadcast: variable-wise gate for global context integration
        if use_gated_broadcast:
            D = d_model // n_heads
            self.broadcast_gate = nn.Linear(D, 1, bias=True)
            nn.init.zeros_(self.broadcast_gate.weight)
            nn.init.constant_(self.broadcast_gate.bias, 2.0)

        # FiLM Broadcast: multiplicative modulation gamma*local + beta
        # Supports rank-constrained variants for ablation (film_rank parameter).
        if use_film_broadcast:
            D = d_model // n_heads
            self._film_mode = film_rank
            if film_rank == 'diagonal':
                # Per-dimension affine: gamma = 1 + diag * x + bias (no cross-dim mixing)
                self._film_gamma_diag = nn.Parameter(torch.zeros(D))
                self._film_gamma_bias = nn.Parameter(torch.zeros(D))
                self._film_beta_diag = nn.Parameter(torch.zeros(D))
                self._film_beta_bias = nn.Parameter(torch.zeros(D))
            elif film_rank.startswith('rank_'):
                # Bottleneck: Linear(D, r) [-> ReLU] -> Linear(r, D)
                parts = film_rank.split('_')
                r = int(parts[1])
                use_relu = len(parts) > 2 and parts[2] == 'relu'
                self.film_gamma = self._build_bottleneck_film(D, r, use_relu)
                self.film_beta = self._build_bottleneck_film(D, r, use_relu)
            else:
                # Full rank (original): Linear(D, D) with AdaLN-Zero init
                self.film_gamma = nn.Linear(D, D, bias=True)
                self.film_beta = nn.Linear(D, D, bias=True)
                nn.init.zeros_(self.film_gamma.weight)
                nn.init.zeros_(self.film_gamma.bias)
                nn.init.zeros_(self.film_beta.weight)
                nn.init.zeros_(self.film_beta.bias)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.last_stats = {}
        self.custom_group_sizes = None
        self._ragged_cache = None

    @staticmethod
    def _build_bottleneck_film(D, r, use_relu):
        """Build a rank-constrained FiLM projection with AdaLN-Zero init.

        Architecture: Linear(D, r) [-> ReLU] -> Linear(r, D)
        The output layer is zero-initialized so the module outputs 0 at init,
        preserving the AdaLN-Zero residual property (gamma=1, beta=passthrough).
        """
        layers = [nn.Linear(D, r, bias=False)]
        if use_relu:
            layers.append(nn.ReLU())
        layers.append(nn.Linear(r, D, bias=True))
        module = nn.Sequential(*layers)
        nn.init.zeros_(module[-1].weight)
        nn.init.zeros_(module[-1].bias)
        return module

    def _apply_film(self, global_ctx, local_out):
        """Apply FiLM modulation: out = gamma * local + beta.

        Dispatches to the appropriate implementation based on self._film_mode.
        All modes satisfy AdaLN-Zero at init: gamma=1, beta=global_ctx.

        Args:
            global_ctx: (..., D) global context (already broadcast to match local)
            local_out:  (..., D) local attention output
        Returns:
            (..., D) modulated output
        """
        if self._film_mode == 'diagonal':
            gamma = 1.0 + self._film_gamma_diag * global_ctx + self._film_gamma_bias
            beta = global_ctx + self._film_beta_diag * global_ctx + self._film_beta_bias
        else:  # 'full' or 'rank_*'
            gamma = 1.0 + self.film_gamma(global_ctx)
            beta = global_ctx + self.film_beta(global_ctx)
        return gamma * local_out + self.dropout(beta)

    def set_custom_group_sizes(self, group_sizes, device=None):
        if group_sizes is None:
            self.custom_group_sizes = None
            self._ragged_cache = None
            return
        sizes = [int(s) for s in group_sizes]
        if len(sizes) != int(self.num_groups):
            self.custom_group_sizes = None
            self._ragged_cache = None
            return
        if any(s < 0 for s in sizes):
            self.custom_group_sizes = None
            self._ragged_cache = None
            return
        self.custom_group_sizes = sizes

        # Pre-compute cached tensors for vectorized ragged path
        G = len(sizes)
        M_max = max(sizes)
        sizes_t = torch.tensor(sizes, dtype=torch.long, device=device)

        # pad_mask: (G, M_max) — True for padded positions
        arange = torch.arange(M_max, device=device).unsqueeze(0)       # (1, M_max)
        pad_mask = arange >= sizes_t.unsqueeze(1)        # (G, M_max)

        # scatter_idx: (N,) — maps original token pos to padded flat pos
        offsets = torch.arange(G, device=device).unsqueeze(1) * M_max   # (G, 1)
        full_idx = offsets + arange                       # (G, M_max)
        scatter_idx = full_idx[~pad_mask]                 # (N,)

        # valid_counts: (G,) float — for masked mean
        valid_counts = sizes_t.float()

        self._ragged_cache = {
            'M_max': M_max,
            'pad_mask': pad_mask,
            'scatter_idx': scatter_idx,
            'scatter_idx_4d': scatter_idx.view(1, -1, 1, 1),
            'valid_counts': valid_counts,
        }
        
    def _partition(self, logits):
        """Helper for different partitioning strategies"""
        temp = torch.abs(self.temp) + 1e-4
        
        if self.partition_strategy == 'gumbel':
            # Gumbel-Softmax for hard categorical assignment with gradients
            return F.gumbel_softmax(logits, tau=temp, hard=True, dim=-1)
        
        elif self.partition_strategy == 'topk':
            # Keep only top-k groups, zero out others (sharp sparsity)
            k = max(1, self.num_groups // 4) # Heuristic: keep 25%
            values, indices = torch.topk(logits, k, dim=-1)
            mask = torch.zeros_like(logits).scatter_(-1, indices, 1.0)
            weights = torch.softmax(logits / temp, dim=-1)
            return weights * mask / (weights * mask).sum(dim=-1, keepdim=True)
            
        else: # 'softmax' (standard)
            return torch.softmax(logits / temp, dim=-1)

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        """
        queries/keys/values: (B, N, H, D)
        """
        B, N, H, D = queries.shape
        G = self.num_groups
        K = self.dynamic_tokens_per_group
        
        if self.pooling == 'dynamic':
            # --- [Phase 17+: Multi-Token Dynamic Grouping] ---
            x_raw = queries.reshape(B, N, H * D)
            v_raw = values.reshape(B, N, H * D)
            
            # 1. Salience Gate
            if self.use_variable_resolution:
                salience = torch.sigmoid(self.salience_gate(x_raw))
            else:
                salience = torch.zeros((B, N, 1), device=x_raw.device)
            
            # 2. Dynamic Partitioning with Multi-Token support
            # logits: (B, N, G*K)
            logits = self.score_projection(x_raw)
            weights = self._partition(logits) # (B, N, G*K)
            
            # --- A3 logging (collapse/entropy) ---
            with torch.no_grad():
                w = weights.clamp_min(1e-9)  # (B, N, G*K)
                entropy = -(w * w.log()).sum(dim=-1).mean()
                eff_groups = entropy.exp()

                # load per group token: (G*K,)
                load = weights.sum(dim=(0, 1))
                load = load / (load.sum() + 1e-9)
                load_imbalance = load.var(unbiased=False)

            self.last_stats = {
                "entropy": float(entropy),
                "eff_groups": float(eff_groups),
                "load_imbalance": float(load_imbalance),
            }

            # 3. Weighted Sum Pooling (Representative Calculation)
            # weights: (B, N, G*K), x_raw: (B, N, H*D) -> group_reps: (B, G*K, H*D)
            group_reps = torch.matmul(weights.transpose(1, 2), x_raw)
            group_reps = group_reps.reshape(B, G * K, H, D)
            
            # 4. Global Attention: Inter-Token Interaction
            if self.use_global_interact:
                out_global, attn_global = self.global_attn(group_reps, group_reps, group_reps)
                out_global_raw = out_global.reshape(B, G * K, H * D)
                out_context = torch.matmul(weights, out_global_raw)
            else:
                # Skip global interaction, contextual bias is zero
                out_context = torch.zeros((B, N, H * D), device=x_raw.device)

            # 6. Cross-Interaction Bridge (Gated Integration)
            if self.use_interaction_bridge:
                combined = torch.cat([out_context, v_raw], dim=-1)
                interaction = torch.sigmoid(self.interaction_gate(combined))
                gate = interaction * (1.0 - salience) 
                out_raw = gate * out_context + (1.0 - gate) * v_raw
            else:
                # Fallback to simple addition/residual if bridge is off
                out_raw = out_context + v_raw
            
            out = self.refinement(out_raw).reshape(B, N, H, D)
            
            if self.output_attention:
                res = {'attn': attn_global, 'salience': salience}
                if self.use_interaction_bridge:
                    res['interaction'] = interaction
                return out, res
            else:
                return out, None

        # Shifted Grouping: compute shift for this layer
        shift = 0
        if self.use_multi_shift and self.layer_idx > 0:
            M_est = max((N + G - 1) // G, 2)
            shift = (self.layer_idx * M_est) // self.num_layers
            shift = max(shift, 1)
        elif self.use_shifted_grouping and self.layer_idx % 2 == 1:
            M_est = max((N + G - 1) // G, 2)
            shift = max(M_est // 2, 1)

        # Vectorized ragged non-uniform grouping path (pad-and-batch).
        if self.custom_group_sizes is not None and self._ragged_cache is not None:
            group_sizes = self.custom_group_sizes
            cache = self._ragged_cache
            if len(group_sizes) == G and int(sum(group_sizes)) == int(N):
                M_max = cache['M_max']
                device = queries.device
                pad_mask = cache['pad_mask']           # (G, M_max)
                scatter_idx = cache['scatter_idx']     # (N,)
                valid_counts = cache['valid_counts']   # (G,)

                # --- Shifted Grouping: roll inputs before scatter ---
                if shift > 0:
                    queries = torch.roll(queries, shifts=shift, dims=1)
                    keys = torch.roll(keys, shifts=shift, dims=1)
                    values = torch.roll(values, shifts=shift, dims=1)

                # --- Scatter into padded layout: (B, G*M_max, H, D) ---
                scatter_idx_4d = cache['scatter_idx_4d']
                total_len = G * M_max
                padded_q = _InjectiveIndexPut.apply(queries, scatter_idx_4d, total_len)
                padded_k = _InjectiveIndexPut.apply(keys, scatter_idx_4d, total_len)
                padded_v = _InjectiveIndexPut.apply(values, scatter_idx_4d, total_len)

                # Reshape to (B*G, M_max, H, D) for batched attention
                q_local = padded_q.view(B, G, M_max, H, D).reshape(B * G, M_max, H, D)
                k_local = padded_k.view(B, G, M_max, H, D).reshape(B * G, M_max, H, D)
                v_local = padded_v.view(B, G, M_max, H, D).reshape(B * G, M_max, H, D)

                # key_padding_mask: (B*G, M_max) — True for padded positions
                kpm = pad_mask.unsqueeze(0).expand(B, -1, -1).reshape(B * G, M_max)

                # --- Single batched local attention call ---
                out_local, _ = self.local_attn(q_local, k_local, v_local, key_padding_mask=kpm)
                out_local = out_local.view(B, G, M_max, H, D)

                # Sanitize NaN from any fully-padded groups (size=0 edge case)
                out_local = out_local.nan_to_num(0.0)

                # --- Global attention (inter-group) ---
                attn_global = None
                if self.use_global_interact:
                    # valid_mask: (1, G, M_max, 1, 1) for broadcasting with (B, G, M_max, H, D)
                    valid_mask = (~pad_mask).float().unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                    counts = valid_counts.view(1, G, 1, 1).clamp_min(1)  # (1, G, 1, 1)

                    if self.pooling == 'statistical':
                        masked_out = out_local * valid_mask
                        r_mean = masked_out.sum(dim=2) / counts                      # (B, G, H, D)
                        masked_max = out_local.masked_fill(
                            pad_mask.unsqueeze(0).unsqueeze(-1).unsqueeze(-1), float('-inf'))
                        r_max = masked_max.max(dim=2)[0]                              # (B, G, H, D)
                        # For empty groups (all -inf), replace with 0 to match original behavior
                        r_max = r_max.masked_fill(~torch.isfinite(r_max), 0.0)
                        diff = (out_local - r_mean.unsqueeze(2)) * valid_mask
                        r_std = (diff.pow(2).sum(dim=2) / counts).sqrt()              # (B, G, H, D)

                        enriched = torch.cat([r_mean, r_max, r_std], dim=-1)          # (B, G, H, 3D)
                        group_reps = self.stat_proj(enriched)                         # (B, G, H, D)
                        out_global, attn_global = self.global_attn(group_reps, group_reps, group_reps)

                    elif self.pooling == 'learnable':
                        q_pool = self.query_gen.expand(B, G, 1, D)                    # (B, G, 1, D)
                        q_pool = q_pool.reshape(B * G, 1, 1, D).expand(B * G, 1, H, D)
                        rep, _ = self.pool_attn(q_pool, k_local, v_local, key_padding_mask=kpm)
                        group_reps = rep.reshape(B, G, H, D).nan_to_num(0.0)
                        out_global, attn_global = self.global_attn(group_reps, group_reps, group_reps)

                    elif self.pooling == 'max':
                        masked_max = out_local.masked_fill(
                            pad_mask.unsqueeze(0).unsqueeze(-1).unsqueeze(-1), float('-inf'))
                        group_reps = masked_max.max(dim=2)[0]                         # (B, G, H, D)
                        group_reps = group_reps.masked_fill(~torch.isfinite(group_reps), 0.0)
                        out_global, attn_global = self.global_attn(group_reps, group_reps, group_reps)

                    else:  # mean
                        masked_out = out_local * valid_mask
                        group_reps = masked_out.sum(dim=2) / counts                   # (B, G, H, D)
                        out_global, attn_global = self.global_attn(group_reps, group_reps, group_reps)

                    # Integration: broadcast global context to local tokens
                    out_global_exp = out_global.unsqueeze(2).expand(-1, -1, M_max, -1, -1)
                    if self.use_film_broadcast:
                        out_padded = self._apply_film(out_global_exp, out_local)
                    elif self.use_gated_broadcast:
                        gate = torch.sigmoid(self.broadcast_gate(out_local))
                        out_padded = out_local + gate * self.dropout(out_global_exp)
                    else:
                        out_padded = out_local + self.dropout(out_global_exp)
                else:
                    out_padded = out_local

                # --- Gather back to original positions ---
                out_flat = out_padded.reshape(B, G * M_max, H, D)
                out = _InjectiveIndexSelect.apply(out_flat, scatter_idx_4d)              # (B, N, H, D)
                # --- Shifted Grouping: unroll output ---
                if shift > 0:
                    out = torch.roll(out, shifts=-shift, dims=1)

                if self.output_attention:
                    return out, attn_global
                return out, None

        # --- [Original Hierarchical Flow (Fixed Grouping)] ---
        # 1. Padding if N is not divisible by G
        pad_len = (G - (N % G)) % G
        if pad_len > 0:
            queries = F.pad(queries, (0, 0, 0, 0, 0, pad_len))
            keys = F.pad(keys, (0, 0, 0, 0, 0, pad_len))
            values = F.pad(values, (0, 0, 0, 0, 0, pad_len))
        
        N_padded = N + pad_len
        M = N_padded // G # Number of variates per group

        # W2 stability fixes
        assert N_padded % G == 0
        queries = queries.contiguous()
        keys = keys.contiguous()
        values = values.contiguous()

        # Shifted Grouping: roll inputs before grouping
        if shift > 0:
            queries = torch.roll(queries, shifts=shift, dims=1)
            keys = torch.roll(keys, shifts=shift, dims=1)
            values = torch.roll(values, shifts=shift, dims=1)

        # --- [Step 1: Local Attention (Intra-group)] ---
        # Reshape to (B*G, M, H, D)
        q_local = queries.view(B, G, M, H, D).transpose(1, 0).reshape(G * B, M, H, D)
        k_local = keys.view(B, G, M, H, D).transpose(1, 0).reshape(G * B, M, H, D)
        v_local = values.view(B, G, M, H, D).transpose(1, 0).reshape(G * B, M, H, D)

        # Perform self-attention within each group: O(G * M^2) = O(N^2/G)
        out_local, attn_local = self.local_attn(q_local, k_local, v_local)

        # Reshape back to (B, G, M, H, D)
        out_local = out_local.view(G, B, M, H, D).transpose(0, 1)
        
        # --- [Step 2: Global Attention (Inter-group)] ---
        if self.use_global_interact:
            if self.pooling == 'statistical':
                rep_mean = out_local.mean(dim=2)
                rep_max = out_local.max(dim=2)[0]
                rep_std = out_local.std(dim=2, unbiased=False)

                enriched = torch.cat([rep_mean, rep_max, rep_std], dim=-1)  # (B, G, H, 3D)
                group_reps = self.stat_proj(enriched)  # (B, G, H, D)
                out_global, attn_global = self.global_attn(group_reps, group_reps, group_reps)

            elif self.pooling == 'learnable':
                q_pool = self.query_gen.expand(B, G, 1, D)          # (B, G, 1, D)
                q_pool = q_pool.permute(1, 0, 2, 3)                 # (G, B, 1, D)
                q_pool = q_pool.reshape(G * B, 1, 1, D)             # (G*B, 1, 1, D)
                q_pool = q_pool.expand(G * B, 1, H, D)              # (G*B, 1, H, D)
                rep_learnable, _ = self.pool_attn(q_pool, k_local, v_local)
                group_reps = rep_learnable.view(G, B, 1, H, D).transpose(0, 1).reshape(B, G, H, D)

                out_global, attn_global = self.global_attn(group_reps, group_reps, group_reps)

            elif self.pooling == 'max':
                group_reps = out_local.max(dim=2)[0]
                out_global, attn_global = self.global_attn(group_reps, group_reps, group_reps)

            else: # 'mean' (Baseline)
                group_reps = out_local.mean(dim=2)
                out_global, attn_global = self.global_attn(group_reps, group_reps, group_reps)
            
            # --- [Step 3: Integration] ---
            out_global = out_global.unsqueeze(2).expand(-1, -1, M, -1, -1)
            if self.use_film_broadcast:
                out = self._apply_film(out_global, out_local)
            elif self.use_gated_broadcast:
                gate = torch.sigmoid(self.broadcast_gate(out_local))  # (B,G,M,H,1)
                out = out_local + gate * self.dropout(out_global)
            else:
                out = out_local + self.dropout(out_global)
        else:
            # Skip Step 2 & 3, just use local info
            out = out_local
            attn_global = None
        
        out = out.reshape(B, N_padded, H, D)
        # Shifted Grouping: unroll output
        if shift > 0:
            out = torch.roll(out, shifts=-shift, dims=1)
        if pad_len > 0:
            out = out[:, :N, :, :]
            
        if self.output_attention:
            # attn_local: (G*B, H, M, M), reshape to (B, G, H, M, M)
            if attn_local is not None:
                H_attn = attn_local.shape[1]
                attn_local = attn_local.view(G, B, H_attn, M, M).transpose(0, 1)
            return out, (attn_local, attn_global)
        else:
            return out, None


class MultiHeadAttention(nn.Module):
    """Refined Multi-head Attention logic"""
    def __init__(self, dropout=0.1, use_sdpa=False):
        super(MultiHeadAttention, self).__init__()
        self.use_sdpa = use_sdpa
        self.dropout = nn.Dropout(dropout)
        self.dropout_p = dropout  # store float for SDPA

    def forward(self, queries, keys, values, key_padding_mask=None):
        # queries: (B, L, H, D), keys: (B, S, H, D)
        # key_padding_mask: (B, S) — True for padded positions (optional)
        B, L, H, D = queries.shape

        if self.use_sdpa:
            q = queries.transpose(1, 2)  # (B, H, L, D)
            k = keys.transpose(1, 2)
            v = values.transpose(1, 2)
            dp = self.dropout_p if self.training else 0.0
            attn_mask = None
            if key_padding_mask is not None:
                # key_padding_mask: (B, S), True=padded → SDPA needs float mask with -inf
                attn_mask = key_padding_mask.unsqueeze(1).unsqueeze(2).float()
                attn_mask = attn_mask.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(2), float('-inf'))
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=dp)
            return out.transpose(1, 2).contiguous(), None

        scale = 1. / sqrt(D)

        # Attention scores: (B, H, L, S)
        scores = torch.einsum("blhd,bshd->bhls", queries, keys)
        if key_padding_mask is not None:
            scores = scores.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2), float('-inf'))
        A = torch.softmax(scale * scores, dim=-1)
        A = A.nan_to_num(0.0)    # size=0 group defense: all-masked rows → NaN → 0
        A = self.dropout(A)
        V = torch.einsum("bhls,bshd->blhd", A, values)

        return V.contiguous(), A
