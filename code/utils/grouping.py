import torch
import numpy as np
from sklearn.cluster import KMeans

class GroupingManager:
    EXTRACTOR_METHODS = [
        'ordered', 'random',
        'correlation', 'correlation_abs', 'spearman',
        'feature', 'feature_plus',
        'adversarial',
        'horizon', 'horizon_v2', 'horizon_curve',
        'lag_profile', 'corr_lag',
        'spectral_corr', 'spectral_lag', 'spectral_corr_lag'
    ]
    ADV_SUFFIX_METHODS = {
        'correlation', 'correlation_abs', 'spearman',
        'feature', 'feature_plus',
        'horizon', 'horizon_v2', 'horizon_curve',
        'lag_profile', 'corr_lag'
    }
    HORIZON_AWARE_METHODS = {
        'horizon', 'horizon_v2', 'horizon_curve', 'lag_profile',
        'corr_lag', 'spectral_lag', 'spectral_corr_lag'
    }
    GROUP_STRATEGIES = [
        'A_window',
        'B_pack',
        'B_ragged',
        'C_horizon',
        'D_feature',
    ]

    @staticmethod
    def list_supported_methods():
        methods = list(GroupingManager.EXTRACTOR_METHODS)
        methods.extend([f"{m}_adv" for m in sorted(GroupingManager.ADV_SUFFIX_METHODS)])
        return sorted(set(methods))

    @staticmethod
    def parse_method(method):
        if not isinstance(method, str) or len(method.strip()) == 0:
            raise ValueError("grouping method must be a non-empty string")

        raw = method.strip()
        use_adv = False
        if raw.endswith('_adv'):
            base_method = raw[:-4]
            use_adv = True
        else:
            base_method = raw

        if base_method not in GroupingManager.EXTRACTOR_METHODS:
            raise ValueError(
                f"Unknown grouping method '{raw}'. "
                f"Supported: {', '.join(GroupingManager.list_supported_methods())}"
            )
        if use_adv and base_method not in GroupingManager.ADV_SUFFIX_METHODS:
            raise ValueError(
                f"Method '{raw}' is invalid: '_adv' suffix is only supported for "
                f"{sorted(GroupingManager.ADV_SUFFIX_METHODS)}"
            )
        return base_method, use_adv

    @staticmethod
    def list_supported_group_strategies(include_auto=True):
        items = list(GroupingManager.GROUP_STRATEGIES)
        if include_auto:
            items = ['auto'] + items
        return items

    @staticmethod
    def parse_group_strategy(strategy):
        if strategy is None:
            return 'auto'
        if not isinstance(strategy, str):
            raise ValueError("group strategy must be a string")

        raw = strategy.strip()
        if len(raw) == 0:
            return 'auto'

        aliases = {
            'auto': 'auto',
            'a': 'A_window',
            'a_window': 'A_window',
            'window': 'A_window',
            'relation_window': 'A_window',
            'b_pack': 'B_pack',
            'pack': 'B_pack',
            'b_ragged': 'B_ragged',
            'ragged': 'B_ragged',
            'c_horizon': 'C_horizon',
            'horizon': 'C_horizon',
            'horizon_dependent': 'C_horizon',
            'd_feature': 'D_feature',
            'feature': 'D_feature',
            'feature_embedding': 'D_feature',
            'feature_embed': 'D_feature',
        }

        key = raw.lower()
        if key not in aliases:
            raise ValueError(
                f"Unknown group strategy '{strategy}'. "
                f"Supported: {', '.join(GroupingManager.list_supported_group_strategies(include_auto=True))}"
            )
        return aliases[key]

    @staticmethod
    def _safe_corrcoef(x):
        corr = np.corrcoef(x.T)
        corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
        corr = np.clip(corr, -1.0, 1.0)
        np.fill_diagonal(corr, 1.0)
        return corr

    @staticmethod
    def _safe_row_corrcoef(x):
        corr = np.corrcoef(x)
        corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
        corr = np.clip(corr, -1.0, 1.0)
        np.fill_diagonal(corr, 1.0)
        return corr

    @staticmethod
    def _rank_transform(x):
        t, n = x.shape
        ranks = np.empty((t, n), dtype=np.float64)
        base = np.arange(t, dtype=np.float64)
        for j in range(n):
            order = np.argsort(x[:, j], kind='mergesort')
            ranks[order, j] = base
        return ranks

    @staticmethod
    def _lag_curve_features(data, max_lag, num_lags=8):
        t, n = data.shape
        max_lag = max(1, min(int(max_lag), t - 2))
        lags = np.unique(np.linspace(1, max_lag, num=num_lags, dtype=int))

        z = data - data.mean(axis=0, keepdims=True)
        z = z / (data.std(axis=0, keepdims=True) + 1e-6)
        curves = np.zeros((n, len(lags)), dtype=np.float64)

        for li, lag in enumerate(lags):
            x = z[:-lag]
            y = z[lag:]
            num = np.mean(x * y, axis=0)
            den = np.sqrt(np.mean(x * x, axis=0) * np.mean(y * y, axis=0)) + 1e-8
            curves[:, li] = num / den

        return np.nan_to_num(curves, nan=0.0, posinf=0.0, neginf=0.0)

    @staticmethod
    def _cross_lag_similarity(data, max_lag, num_lags=8):
        t, n = data.shape
        max_lag = max(1, min(int(max_lag), t - 2))
        lags = np.unique(np.linspace(1, max_lag, num=num_lags, dtype=int))

        z = data - data.mean(axis=0, keepdims=True)
        z = z / (data.std(axis=0, keepdims=True) + 1e-6)

        accum = np.zeros((n, n), dtype=np.float64)
        weight_sum = 0.0
        for lag in lags:
            x0 = z[:-lag]
            x1 = z[lag:]
            c = (x0.T @ x1) / max(float(x0.shape[0]), 1.0)  # directed lag correlation
            c = np.clip(c, -1.0, 1.0)
            # Symmetric lead-lag interaction strength
            s = np.maximum(np.abs(c), np.abs(c.T))
            w = 1.0 / np.sqrt(float(lag))
            accum += w * s
            weight_sum += w

        if weight_sum <= 0:
            sim = np.eye(n, dtype=np.float64)
        else:
            sim = accum / weight_sum
        return GroupingManager._normalize_similarity(sim, force_abs=False)

    @staticmethod
    def _spectral_corr_lag_similarity(data, H=None):
        corr = GroupingManager._safe_corrcoef(data)
        corr_sim = GroupingManager._normalize_similarity(np.abs(corr), force_abs=False)

        lag_max = H if H is not None else min(96, data.shape[0] - 2)
        lag_sim = GroupingManager._cross_lag_similarity(data, max_lag=lag_max, num_lags=8)

        # Remove the component of lag similarity that is collinear with corr similarity
        # to keep spectral_corr_lag distinct from spectral_corr in A_window ordering.
        a = corr_sim - corr_sim.mean()
        b = lag_sim - lag_sim.mean()
        alpha = float(np.sum(a * b) / (np.sum(a * a) + 1e-9))
        lag_orth = b - alpha * a

        lag_orth = lag_orth - lag_orth.min()
        denom = lag_orth.max()
        if denom > 1e-9:
            lag_orth = lag_orth / denom
        else:
            lag_orth = np.zeros_like(lag_orth)

        # Keep some zero-lag structure while emphasizing non-collinear lag topology.
        sim = 0.25 * corr_sim + 0.75 * lag_orth
        return GroupingManager._normalize_similarity(sim, force_abs=False)

    @staticmethod
    def _spectral_order(similarity):
        similarity = np.nan_to_num(similarity, nan=0.0, posinf=0.0, neginf=0.0)
        similarity = 0.5 * (similarity + similarity.T)
        np.fill_diagonal(similarity, 0.0)
        similarity = np.clip(similarity, 0.0, None)

        degree = similarity.sum(axis=1)
        if np.all(degree < 1e-8):
            return np.arange(similarity.shape[0])

        lap = np.diag(degree) - similarity
        inv_sqrt_degree = 1.0 / np.sqrt(np.clip(degree, 1e-8, None))
        lap_norm = (inv_sqrt_degree[:, None] * lap) * inv_sqrt_degree[None, :]
        _, evecs = np.linalg.eigh(lap_norm)

        if evecs.shape[1] < 2:
            return np.arange(similarity.shape[0])
        fiedler = evecs[:, 1]
        return np.argsort(fiedler)

    @staticmethod
    def _safe_standardize(features):
        mu = features.mean(axis=0, keepdims=True)
        sd = features.std(axis=0, keepdims=True)
        return (features - mu) / (sd + 1e-6)

    @staticmethod
    def _cosine_similarity(features):
        x = GroupingManager._safe_standardize(features)
        norm = np.linalg.norm(x, axis=1, keepdims=True) + 1e-9
        x = x / norm
        return x @ x.T

    @staticmethod
    def _normalize_similarity(similarity, force_abs=True):
        sim = np.nan_to_num(similarity, nan=0.0, posinf=0.0, neginf=0.0)
        sim = 0.5 * (sim + sim.T)
        if force_abs:
            sim = np.abs(sim)
        sim = np.clip(sim, 0.0, 1.0)
        np.fill_diagonal(sim, 1.0)
        return sim

    @staticmethod
    def _reduce_feature_dim(features, max_dim=12):
        if features.ndim != 2:
            raise ValueError("features must be 2D")
        if features.shape[1] <= max_dim:
            return features
        x = GroupingManager._safe_standardize(features)
        try:
            _, _, vt = np.linalg.svd(x, full_matrices=False)
            k = max(1, min(int(max_dim), int(vt.shape[0])))
            return x @ vt[:k].T
        except np.linalg.LinAlgError:
            return x[:, :max_dim]

    @staticmethod
    def _adversarial_indices(labels, G, N):
        sorted_indices = np.argsort(labels)
        groups = [[] for _ in range(G)]
        for idx in sorted_indices:
            g = int(labels[idx])
            if g < 0 or g >= G:
                g = max(0, min(G - 1, g))
            groups[g].append(int(idx))

        new_indices = []
        max_len = max(len(g) for g in groups) if groups else 0
        for i in range(max_len):
            for g in range(G):
                if i < len(groups[g]):
                    new_indices.append(groups[g][i])
        if len(new_indices) != int(N):
            raise RuntimeError("adversarial interleave produced invalid index length")
        return np.array(new_indices, dtype=np.int64)

    @staticmethod
    def _horizon_dependent_affinity(data, base_method, H):
        if H is None:
            raise ValueError("H is required for C_horizon strategy")
        base = GroupingManager._similarity_from_method(data, base_method, H=H)
        lag = GroupingManager._similarity_from_method(data, 'lag_profile', H=H)
        hor = GroupingManager._similarity_from_method(data, 'horizon_v2', H=H)
        if base_method in GroupingManager.HORIZON_AWARE_METHODS:
            w_base, w_lag, w_hor = 0.50, 0.30, 0.20
        else:
            w_base, w_lag, w_hor = 0.35, 0.35, 0.30
        fused = w_base * base + w_lag * lag + w_hor * hor
        return GroupingManager._normalize_similarity(fused, force_abs=False)

    @staticmethod
    def _feature_embedding_matrix(data, base_method, H=None):
        n = data.shape[1]
        universal_feat = GroupingManager._feature_matrix(data, 'feature_plus', H=H)
        lag_feat = GroupingManager._feature_matrix(data, 'lag_profile', H=H if H is not None else min(96, data.shape[0] - 2))
        base_feat = GroupingManager._feature_matrix(data, base_method, H=H)
        if base_feat is None:
            base_feat = np.eye(n)
        base_sim = GroupingManager._similarity_from_method(data, base_method, H=H)

        chunks = [
            GroupingManager._safe_standardize(universal_feat),
            GroupingManager._safe_standardize(lag_feat),
            GroupingManager._safe_standardize(GroupingManager._reduce_feature_dim(base_feat, max_dim=12)),
            GroupingManager._safe_standardize(GroupingManager._reduce_feature_dim(base_sim, max_dim=6)),
        ]
        embedding = np.concatenate(chunks, axis=1)
        return GroupingManager._safe_standardize(embedding)

    @staticmethod
    def _feature_embedding_order(data, base_method, G, H=None, use_adv=False):
        n = data.shape[1]
        if n <= 0:
            return np.array([], dtype=np.int64)

        embedding = GroupingManager._feature_embedding_matrix(data, base_method, H=H)
        n_clusters = min(int(G), int(n))
        if n_clusters == 1:
            labels = np.zeros(n, dtype=np.int64)
            centers = np.zeros((1, embedding.shape[1]), dtype=np.float64)
        else:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit(embedding)
            labels = kmeans.labels_
            centers = kmeans.cluster_centers_

        if use_adv:
            return GroupingManager._adversarial_indices(labels, G=max(int(G), n_clusters), N=n)

        base_affinity = GroupingManager._similarity_from_method(data, base_method, H=H)
        if centers.shape[0] <= 1:
            cluster_order = [0]
        else:
            cluster_order = list(np.argsort(centers[:, 0]))

        ordered_groups = []
        for lab in cluster_order:
            members = np.where(labels == lab)[0].astype(np.int64)
            if members.size == 0:
                continue
            members = GroupingManager._order_group_members(members, base_affinity)
            ordered_groups.append(members)

        if not ordered_groups:
            return np.arange(n, dtype=np.int64)
        return np.concatenate(ordered_groups, axis=0).astype(np.int64)

    @staticmethod
    def _feature_matrix(data, base_method, H=None):
        N = data.shape[1]

        if base_method == 'correlation':
            corr = GroupingManager._safe_corrcoef(data)
            return 1.0 - corr

        if base_method == 'correlation_abs':
            corr = GroupingManager._safe_corrcoef(data)
            return 1.0 - np.abs(corr)

        if base_method == 'spearman':
            ranks = GroupingManager._rank_transform(data)
            corr = GroupingManager._safe_corrcoef(ranks)
            return 1.0 - np.abs(corr)

        if base_method == 'feature':
            means = data.mean(axis=0)
            stds = data.std(axis=0)
            if np.abs(means).mean() < 0.1 and np.abs(stds - 1.0).mean() < 0.1:
                import warnings
                warnings.warn(
                    "grouping_method='feature' has weak discrimination on standardized data. "
                    "Consider using 'feature_plus' instead.",
                    stacklevel=2
                )
            return np.stack([means, stds], axis=1)

        if base_method == 'feature_plus':
            means = data.mean(axis=0)
            stds = data.std(axis=0)
            q25 = np.percentile(data, 25, axis=0)
            q50 = np.percentile(data, 50, axis=0)
            q75 = np.percentile(data, 75, axis=0)
            iqr = q75 - q25

            centered = data - means
            skew = np.mean(centered ** 3, axis=0) / (stds ** 3 + 1e-6)
            kurt = np.mean(centered ** 4, axis=0) / (stds ** 4 + 1e-6)

            if data.shape[0] > 1:
                acf1 = np.array([np.corrcoef(data[:-1, i], data[1:, i])[0, 1] for i in range(N)])
            else:
                acf1 = np.zeros(N)

            lag24 = min(24, max(1, data.shape[0] - 1))
            if data.shape[0] > lag24:
                acf24 = np.array([np.corrcoef(data[:-lag24, i], data[lag24:, i])[0, 1] for i in range(N)])
            else:
                acf24 = np.zeros(N)

            return np.stack(
                [means, stds, q25, q50, q75, iqr, skew, kurt, np.nan_to_num(acf1), np.nan_to_num(acf24)],
                axis=1
            )

        if base_method == 'horizon':
            if H is None:
                raise ValueError("H required for horizon method")
            if data.shape[0] <= H:
                return np.eye(N)
            x_t, x_th = data[:-H], data[H:]
            return np.stack(
                [np.nan_to_num(np.array([np.corrcoef(x_t[:, i], x_th[:, j])[0, 1] for j in range(N)])) for i in range(N)]
            )

        if base_method == 'horizon_v2':
            if H is None:
                raise ValueError("H required")
            window = 5
            data_ma = np.array([np.convolve(data[:, i], np.ones(window) / window, mode='same') for i in range(N)]).T
            if data_ma.shape[0] <= H:
                return np.eye(N)
            x_t, x_th = data_ma[:-H], data_ma[H:]
            return np.stack(
                [np.nan_to_num(np.array([np.corrcoef(x_t[:, i], x_th[:, j])[0, 1] for j in range(N)])) for i in range(N)]
            )

        if base_method == 'horizon_curve':
            if H is None:
                raise ValueError("H required")
            lags = np.unique(np.linspace(0, H, num=5, dtype=int))
            curves = []
            for i in range(N):
                curve = []
                for lag in lags:
                    if lag == 0:
                        curve.append(1.0)
                    elif data.shape[0] > lag:
                        c = np.corrcoef(data[:-lag, i], data[lag:, i])[0, 1]
                        curve.append(np.nan_to_num(c))
                    else:
                        curve.append(0.0)
                curves.append(np.array(curve))
            return np.stack(curves)

        if base_method == 'lag_profile':
            max_lag = H if H is not None else min(96, data.shape[0] - 2)
            return GroupingManager._lag_curve_features(data, max_lag=max_lag, num_lags=8)

        if base_method == 'corr_lag':
            corr = GroupingManager._safe_corrcoef(data)
            lag_max = H if H is not None else min(96, data.shape[0] - 2)
            lag_curves = GroupingManager._lag_curve_features(data, max_lag=lag_max, num_lags=8)
            corr_reduced = GroupingManager._reduce_feature_dim(np.abs(corr), max_dim=8)
            return np.concatenate([corr_reduced, lag_curves], axis=1)

        return None

    @staticmethod
    def _similarity_from_method(data, base_method, H=None):
        N = data.shape[1]

        if base_method == 'ordered':
            idx = np.arange(N)
            dist = np.abs(idx[:, None] - idx[None, :])
            sim = np.exp(-dist / max(1.0, N / 32.0))
            return GroupingManager._normalize_similarity(sim, force_abs=False)

        if base_method == 'random':
            rnd = np.random.rand(N, N)
            sim = 0.5 * (rnd + rnd.T)
            return GroupingManager._normalize_similarity(sim, force_abs=False)

        if base_method == 'correlation':
            corr = GroupingManager._safe_corrcoef(data)
            # Signed similarity: positive correlation should be close, negative should be far.
            sim = 0.5 * (corr + 1.0)  # [-1,1] -> [0,1]
            sim = np.clip(sim, 0.0, 1.0)
            np.fill_diagonal(sim, 1.0)
            return GroupingManager._normalize_similarity(sim, force_abs=False)

        if base_method == 'correlation_abs':
            corr = GroupingManager._safe_corrcoef(data)
            return GroupingManager._normalize_similarity(np.abs(corr), force_abs=False)

        if base_method == 'spearman':
            ranks = GroupingManager._rank_transform(data)
            corr = GroupingManager._safe_corrcoef(ranks)
            return GroupingManager._normalize_similarity(np.abs(corr), force_abs=False)

        if base_method == 'adversarial':
            corr = GroupingManager._safe_corrcoef(data)
            sim = 1.0 - np.abs(corr)
            return GroupingManager._normalize_similarity(sim, force_abs=False)

        if base_method == 'spectral_corr':
            corr = GroupingManager._safe_corrcoef(data)
            base = GroupingManager._normalize_similarity(np.abs(corr), force_abs=False)
            # Build a spectral-order locality kernel so spectral_corr is not identical
            # to correlation_abs under pack/ragged partition.
            order = GroupingManager._spectral_order(base)
            pos = np.empty(N, dtype=np.float64)
            pos[order] = np.arange(N, dtype=np.float64)
            dist = np.abs(pos[:, None] - pos[None, :])
            scale = max(1.0, N / 32.0)
            local = np.exp(-dist / scale)
            sim = 0.5 * base + 0.5 * local
            return GroupingManager._normalize_similarity(sim, force_abs=False)

        if base_method == 'spectral_lag':
            lag_max = H if H is not None else min(96, data.shape[0] - 2)
            lag_curves = GroupingManager._lag_curve_features(data, max_lag=lag_max, num_lags=8)
            lag_sim = GroupingManager._safe_row_corrcoef(lag_curves)
            return GroupingManager._normalize_similarity(np.abs(lag_sim), force_abs=False)

        if base_method == 'spectral_corr_lag':
            return GroupingManager._spectral_corr_lag_similarity(data, H=H)

        features = GroupingManager._feature_matrix(data, base_method, H=H)
        if features is None:
            raise ValueError(
                f"Cannot build similarity for unknown method '{base_method}'. "
                f"Supported: {', '.join(GroupingManager.EXTRACTOR_METHODS)}"
            )

        if features.shape[1] == N:
            # N x N relation-like matrix (e.g., horizon relation matrix)
            row_sim = GroupingManager._safe_row_corrcoef(features)
            return GroupingManager._normalize_similarity(row_sim, force_abs=True)

        sim = GroupingManager._cosine_similarity(features)
        return GroupingManager._normalize_similarity(sim, force_abs=True)

    @staticmethod
    def _fixed_window_sizes(N, G):
        if G <= 0:
            raise ValueError("G must be positive")
        if N < G:
            return [1] * N + [0] * (G - N)
        M = int(np.ceil(float(N) / float(G)))
        sizes = [M] * (G - 1)
        last = N - M * (G - 1)
        if last <= 0:
            base = N // G
            rem = N % G
            sizes = [base + 1] * rem + [base] * (G - rem)
        else:
            sizes.append(last)
        return sizes

    @staticmethod
    def _order_group_members(nodes, affinity):
        if len(nodes) <= 2:
            return np.array(nodes, dtype=np.int64)
        sub = affinity[np.ix_(nodes, nodes)]
        local = GroupingManager._spectral_order(sub)
        return np.array(nodes, dtype=np.int64)[local]

    @staticmethod
    def _balanced_graph_pack(affinity, capacities, balance_alpha=0.15):
        N = affinity.shape[0]
        G = len(capacities)
        capacities = [max(0, int(c)) for c in capacities]
        if sum(capacities) < N:
            raise ValueError("pack capacities must cover all variables")

        groups = [[] for _ in range(G)]
        remaining = capacities[:]
        degree = affinity.sum(axis=1)
        order = list(np.argsort(-degree))
        unassigned = set(order)

        # Seed groups with high-degree nodes first.
        for g in range(G):
            if remaining[g] <= 0 or not unassigned:
                continue
            seed = max(unassigned, key=lambda i: degree[i])
            groups[g].append(seed)
            remaining[g] -= 1
            unassigned.remove(seed)

        for i in order:
            if i not in unassigned:
                continue
            candidates = [g for g in range(G) if remaining[g] > 0]
            if not candidates:
                raise RuntimeError("no remaining capacity while nodes are unassigned")
            best_g = None
            best_score = -1e18
            for g in candidates:
                if groups[g]:
                    rel = float(np.mean(affinity[i, groups[g]]))
                else:
                    rel = float(degree[i] / max(N, 1))
                cap = max(capacities[g], 1)
                balance_penalty = balance_alpha * (len(groups[g]) / cap)
                score = rel - balance_penalty
                if score > best_score:
                    best_score = score
                    best_g = g
            groups[best_g].append(i)
            remaining[best_g] -= 1
            unassigned.remove(i)

        return [GroupingManager._order_group_members(g, affinity).tolist() if g else [] for g in groups]

    @staticmethod
    def _balanced_graph_ragged(affinity, G, balance_alpha=0.15):
        N = affinity.shape[0]
        if G <= 0:
            raise ValueError("G must be positive")

        degree = affinity.sum(axis=1)
        order = list(np.argsort(-degree))
        groups = [[] for _ in range(G)]
        unassigned = set(order)

        seeds = min(G, N)
        for g in range(seeds):
            seed = order[g]
            groups[g].append(seed)
            unassigned.remove(seed)

        target = max(float(N) / float(G), 1.0)
        for i in order:
            if i not in unassigned:
                continue
            best_g = None
            best_score = -1e18
            for g in range(G):
                if groups[g]:
                    rel = float(np.mean(affinity[i, groups[g]]))
                else:
                    rel = float(degree[i] / max(N, 1))
                balance_penalty = balance_alpha * (len(groups[g]) / target)
                score = rel - balance_penalty
                if score > best_score:
                    best_score = score
                    best_g = g
            groups[best_g].append(i)
            unassigned.remove(i)

        # Keep deterministic ordering inside each group.
        return [GroupingManager._order_group_members(g, affinity).tolist() if g else [] for g in groups]

    @staticmethod
    def get_indices(data, method, G, H=None):
        """
        data: (N_samples, N_vars) numpy array or torch tensor
        method: str, one of
          ['ordered', 'random', 'correlation', 'correlation_abs', 'spearman',
           'feature', 'feature_plus', 'adversarial', 'horizon', 'horizon_v2',
           'horizon_curve', 'lag_profile', 'corr_lag',
           'spectral_corr', 'spectral_lag', 'spectral_corr_lag']
        G: int, number of groups
        H: int, horizon for horizon-specific clustering
        """
        if isinstance(data, torch.Tensor):
            data = data.cpu().numpy()
        if not isinstance(data, np.ndarray) or data.ndim != 2:
            raise ValueError("data must be a 2D array-like object with shape (T, N)")
        if int(G) <= 0:
            raise ValueError("G must be a positive integer")

        N = data.shape[1]
        if N <= 0:
            return np.array([], dtype=np.int64)
        n_clusters = min(int(G), int(N))
        
        base_method, use_adv = GroupingManager.parse_method(method)

        if base_method == 'ordered':
            return np.arange(N)
        
        if base_method == 'random':
            indices = np.arange(N)
            np.random.shuffle(indices)
            return indices
        
        features = None
        spectral_similarity = None
        
        if base_method in {
            'correlation', 'correlation_abs', 'spearman',
            'feature', 'feature_plus',
            'horizon', 'horizon_v2', 'horizon_curve',
            'lag_profile', 'corr_lag'
        }:
            features = GroupingManager._feature_matrix(data, base_method, H=H)

        elif base_method == 'adversarial': # Legacy support
            corr = GroupingManager._safe_corrcoef(data)
            dist = 1 - corr 
            if n_clusters == 1:
                labels = np.zeros(N, dtype=np.int64)
            else:
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit(dist)
                labels = kmeans.labels_
            return GroupingManager._adversarial_indices(labels, G, N)

        elif base_method == 'spectral_corr':
            corr = GroupingManager._safe_corrcoef(data)
            spectral_similarity = np.abs(corr)

        elif base_method == 'spectral_lag':
            lag_max = H if H is not None else min(96, data.shape[0] - 2)
            lag_curves = GroupingManager._lag_curve_features(data, max_lag=lag_max, num_lags=8)
            lag_sim = GroupingManager._safe_row_corrcoef(lag_curves)
            spectral_similarity = np.abs(lag_sim)

        elif base_method == 'spectral_corr_lag':
            spectral_similarity = GroupingManager._spectral_corr_lag_similarity(data, H=H)

        if spectral_similarity is not None:
            return GroupingManager._spectral_order(spectral_similarity)

        if features is not None:
            if n_clusters == 1:
                labels = np.zeros(N, dtype=np.int64)
            else:
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit(features)
                labels = kmeans.labels_
            if use_adv:
                return GroupingManager._adversarial_indices(labels, G, N)
            return np.argsort(labels)

        raise ValueError(
            f"Unknown grouping method '{method}'. "
            f"Supported: {', '.join(GroupingManager.list_supported_methods())}"
        )

    @staticmethod
    def get_group_plan(
        data,
        method,
        G,
        H=None,
        partition='window',
        balance_alpha=0.15,
        strategy='auto',
    ):
        """
        Build a static grouping plan.
        Returns:
          {
            "indices": np.ndarray[int64],           # global permutation
            "group_sizes": list[int] or None,       # only used for ragged partition
            "partition": str,                       # window | pack | ragged
            "group_strategy": str,                  # A_window | B_pack | B_ragged | C_horizon | D_feature
            "base_method": str
          }
        """
        if isinstance(data, torch.Tensor):
            data = data.cpu().numpy()
        if not isinstance(data, np.ndarray) or data.ndim != 2:
            raise ValueError("data must be a 2D array-like object with shape (T, N)")
        if int(G) <= 0:
            raise ValueError("G must be a positive integer")

        N = data.shape[1]
        base_method, use_adv = GroupingManager.parse_method(method)
        if partition not in {'window', 'pack', 'ragged'}:
            raise ValueError(f"Unknown partition '{partition}'. Choose from [window, pack, ragged].")

        strategy_key = GroupingManager.parse_group_strategy(strategy)
        if strategy_key == 'auto':
            if partition == 'window':
                strategy_key = 'A_window'
            elif partition == 'pack':
                strategy_key = 'B_pack'
            elif partition == 'ragged':
                strategy_key = 'B_ragged'

        def _validate_indices(indices):
            idx = np.array(indices, dtype=np.int64)
            if idx.shape[0] != int(N):
                raise RuntimeError("group plan produced invalid index length")
            if np.unique(idx).shape[0] != int(N):
                raise RuntimeError("group plan produced non-permutation indices")
            return idx

        if strategy_key == 'A_window':
            indices = GroupingManager.get_indices(data, method, G, H=H)
            indices = _validate_indices(indices)
            return {
                "indices": indices,
                "group_sizes": None,
                "partition": "window",
                "group_strategy": strategy_key,
                "base_method": base_method,
                "use_adv": bool(use_adv),
            }

        if strategy_key == 'B_pack':
            if use_adv:
                raise ValueError(f"'{method}' is not supported with strategy='{strategy_key}'. Use non-_adv method.")
            affinity = GroupingManager._similarity_from_method(data, base_method, H=H)
            capacities = GroupingManager._fixed_window_sizes(N, G)
            groups = GroupingManager._balanced_graph_pack(
                affinity, capacities=capacities, balance_alpha=balance_alpha
            )
            flat = [np.array(g, dtype=np.int64) for g in groups if len(g) > 0]
            indices = np.concatenate(flat, axis=0) if flat else np.arange(N, dtype=np.int64)
            indices = _validate_indices(indices)
            return {
                "indices": indices,
                "group_sizes": [len(g) for g in groups if len(g) > 0],
                "partition": "pack",
                "group_strategy": strategy_key,
                "base_method": base_method,
                "use_adv": bool(use_adv),
            }

        if strategy_key == 'B_ragged':
            if use_adv:
                raise ValueError(f"'{method}' is not supported with strategy='{strategy_key}'. Use non-_adv method.")
            affinity = GroupingManager._similarity_from_method(data, base_method, H=H)
            groups = GroupingManager._balanced_graph_ragged(
                affinity,
                G=G,
                balance_alpha=balance_alpha,
            )
            flat = [np.array(g, dtype=np.int64) for g in groups if len(g) > 0]
            indices = np.concatenate(flat, axis=0) if flat else np.arange(N, dtype=np.int64)
            group_sizes = [len(g) for g in groups]
            if int(np.sum(group_sizes)) != int(N):
                raise RuntimeError("ragged grouping produced invalid group sizes")
            indices = _validate_indices(indices)
            return {
                "indices": indices,
                "group_sizes": group_sizes,
                "partition": "ragged",
                "group_strategy": strategy_key,
                "base_method": base_method,
                "use_adv": bool(use_adv),
            }

        if strategy_key == 'C_horizon':
            if use_adv:
                raise ValueError(f"'{method}' is not supported with strategy='{strategy_key}'. Use non-_adv method.")
            affinity = GroupingManager._horizon_dependent_affinity(data, base_method, H=H)
            indices = GroupingManager._spectral_order(affinity).astype(np.int64)
            indices = _validate_indices(indices)
            return {
                "indices": indices,
                "group_sizes": None,
                "partition": "window",
                "group_strategy": strategy_key,
                "base_method": base_method,
                "use_adv": bool(use_adv),
            }

        if strategy_key == 'D_feature':
            indices = GroupingManager._feature_embedding_order(
                data, base_method, G=G, H=H, use_adv=use_adv
            )
            indices = _validate_indices(indices)
            return {
                "indices": indices,
                "group_sizes": None,
                "partition": "window",
                "group_strategy": strategy_key,
                "base_method": base_method,
                "use_adv": bool(use_adv),
            }

        raise ValueError(
            f"Unknown group strategy '{strategy_key}'. "
            f"Supported: {', '.join(GroupingManager.list_supported_group_strategies(include_auto=False))}"
        )
