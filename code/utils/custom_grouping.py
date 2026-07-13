"""
Custom Variable Grouping Strategies for VG-iT
==============================================

Implements 8 grouping methods:

No strategy:
  1. Ordered: Naive sequential partitioning (baseline)

Cohesion (minimize intra-group distance):
  2. FINCH-like: MST + single-linkage cut (1-NN intuition)
  3. Coarsening: Heavy-edge matching (HEM) graph coarsening
  4. MI-based: HEM with mutual information similarity

Diversity (maximize intra-group distance):
  5. Random: Shuffle then partition (null hypothesis baseline)
  6. Anti-clustering: Maximize total intra-group diversity (MAX-SUM, greedy swap)
  7. Score-stratified: Snake-draft by naive forecast difficulty
  8. MaxMin Dispersion: Maximize minimum intra-group pairwise distance (MAX-MIN)

All methods output:
{
    "groups": List[List[int]],   # G clusters
    "perm": List[int],            # Permutation for contiguity
    "group_sizes": List[int]      # Variable sizes (len=G, sum=N)
}

Usage:
    from utils.custom_grouping import compute_grouping_plan

    result = compute_grouping_plan(
        X_train,                    # (T, N) numpy array
        method="coarsening",        # or "ordered", "finch_like", "mi_based"
        G=32,                       # Number of groups
        seed=2021                   # For reproducibility
    )

    # Apply permutation
    x_permuted = x[:, :, result["perm"]]

    # Configure model
    model.group_partition = "ragged"
    model.group_sizes = result["group_sizes"]
    model.use_reorder = False
"""

import numpy as np
from scipy import stats
from scipy.sparse.csgraph import minimum_spanning_tree, connected_components
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, fcluster
import itertools
from typing import List, Dict, Tuple, Optional
import warnings


# ==============================================================================
# Stage 1: Feature Extraction (Similarity/Distance Matrices)
# ==============================================================================

def spearman_correlation_matrix(X: np.ndarray, absolute: bool = True, method: str = "auto") -> np.ndarray:
    """
    Compute Spearman correlation matrix between variables.

    Args:
        X: (T, N) array, T timepoints, N variables
        absolute: If True, return |correlation|
        method: "auto" (default), "vectorized", or "loop"
                auto: use vectorized if T*N < 10M, else loop
                vectorized: fast (931x speedup) using rankdata + np.corrcoef
                loop: original nested loop (backward compatibility)

    Returns:
        S: (N, N) similarity matrix, S[i,j] ∈ [0,1] if absolute else [-1,1]
    """
    from scipy.stats import rankdata

    T, N = X.shape

    # Auto-select method (vectorized is almost always faster)
    if method == "auto":
        method = "vectorized"  # Default to vectorized (931x faster)

    if method == "vectorized":
        # Vectorized approach: 931x faster (22min → 1.4s for N=862, T=12089)
        # Step 1: Convert each variable to ranks (Spearman = Pearson on ranks)
        X_ranked = np.apply_along_axis(rankdata, 0, X)  # (T, N) → (T, N) ranked

        # Step 2: Compute Pearson correlation on ranked data
        S = np.corrcoef(X_ranked.T)  # (N, N) correlation matrix

        # Step 3: Numerical stability fixes
        np.fill_diagonal(S, 1.0)  # Ensure diagonal is exactly 1.0
        S = np.clip(S, -1.0, 1.0)  # Clip to valid correlation range

        # Step 4: Handle NaN (constant variables)
        S = np.nan_to_num(S, nan=0.0)

        # Step 5: Take absolute value if requested
        if absolute:
            S = np.abs(S)

    else:  # method == "loop"
        # Original nested loop (backward compatibility)
        S = np.zeros((N, N))
        for i in range(N):
            for j in range(i, N):
                if i == j:
                    S[i, j] = 1.0
                else:
                    corr, _ = stats.spearmanr(X[:, i], X[:, j])
                    if np.isnan(corr):
                        corr = 0.0
                    if absolute:
                        corr = abs(corr)
                    S[i, j] = corr
                    S[j, i] = corr

    return S


def mi_proxy_matrix(X: np.ndarray) -> np.ndarray:
    """
    Mutual Information proxy based on rank correlation.

    Fast approximation: MI ≈ -0.5 * log(1 - ρ²) where ρ is Spearman correlation
    This captures monotonic dependencies (linear + nonlinear monotonic).

    Args:
        X: (T, N) array

    Returns:
        MI: (N, N) similarity matrix (higher = more dependent)
    """
    # Get Spearman correlation (NOT absolute)
    S_corr = spearman_correlation_matrix(X, absolute=False)

    # MI proxy formula
    # Clamp correlation to avoid log(0)
    S_corr_clamped = np.clip(S_corr, -0.9999, 0.9999)
    MI = -0.5 * np.log(1 - S_corr_clamped ** 2)

    # Ensure symmetry and non-negative
    MI = (MI + MI.T) / 2
    MI = np.maximum(MI, 0)

    # Diagonal should be maximum
    np.fill_diagonal(MI, np.max(MI) + 1)

    return MI


def mi_full_matrix(
    X: np.ndarray,
    k: int = 3,
    downsample_factor: Optional[int] = None
) -> np.ndarray:
    """
    Full Mutual Information using KSG estimator (optional, expensive).

    Note: This requires sklearn and is computationally intensive.
    Only use if computational budget allows.

    Args:
        X: (T, N) array
        k: Number of nearest neighbors for KSG estimator
        downsample_factor: If provided, use every nth sample

    Returns:
        MI: (N, N) similarity matrix
    """
    try:
        from sklearn.feature_selection import mutual_info_regression
    except ImportError:
        warnings.warn("sklearn not available, falling back to MI proxy")
        return mi_proxy_matrix(X)

    # Downsample if needed
    if downsample_factor is not None and downsample_factor > 1:
        X = X[::downsample_factor, :]

    N = X.shape[1]
    MI = np.zeros((N, N))

    for i in range(N):
        # Compute MI(X_i, X_j) for all j
        mi_scores = mutual_info_regression(
            X[:, [j for j in range(N) if j != i]],  # Features
            X[:, i],  # Target
            n_neighbors=k,
            random_state=42
        )

        # Fill matrix
        j_idx = 0
        for j in range(N):
            if i == j:
                MI[i, j] = np.max(mi_scores) + 1  # Diagonal max
            else:
                MI[i, j] = mi_scores[j_idx]
                j_idx += 1

    # Ensure symmetry
    MI = (MI + MI.T) / 2

    return MI


# ==============================================================================
# Stage 2: Clustering Algorithms
# ==============================================================================

def ordered_grouping(N: int, G: int) -> Dict:
    """
    Naive ordered grouping (baseline).

    Sequentially partition [0, 1, ..., N-1] into G groups with nearly equal sizes.

    Args:
        N: Number of variables
        G: Number of groups

    Returns:
        Dict with keys: groups, perm, group_sizes
    """
    perm = list(range(N))

    base_size = N // G
    remainder = N % G

    group_sizes = [base_size + 1] * remainder + [base_size] * (G - remainder)

    groups = []
    start = 0
    for size in group_sizes:
        groups.append(list(range(start, start + size)))
        start += size

    return {
        "groups": groups,
        "perm": perm,
        "group_sizes": group_sizes
    }


def random_grouping(N: int, G: int, seed: int = 2021) -> Dict:
    """Random variable grouping: shuffle then partition."""
    rng = np.random.RandomState(seed)
    perm = rng.permutation(N).tolist()
    base_size = N // G
    remainder = N % G
    group_sizes = [base_size + 1] * remainder + [base_size] * (G - remainder)
    groups, start = [], 0
    for size in group_sizes:
        groups.append(perm[start:start + size])
        start += size
    return {"groups": groups, "perm": perm, "group_sizes": group_sizes}


def mst_cut_clustering(D: np.ndarray, G: int) -> Dict:
    """
    Hierarchical clustering with average-linkage (distance-based).

    Uses scipy's hierarchical clustering with average-linkage method.
    Average-linkage produces more balanced clusters than single-linkage
    while still being based on pairwise distances.

    Steps:
    1. Compute average-linkage hierarchical clustering
    2. Cut dendrogram to get exactly G clusters

    Args:
        D: (N, N) distance matrix (lower = more similar)
        G: Number of groups

    Returns:
        Dict with keys: groups, perm, group_sizes

    Note:
        Changed from single-linkage to average-linkage to avoid
        the "chaining" effect that produces highly unbalanced clusters.
    """
    N = D.shape[0]

    # Convert distance matrix to condensed form (upper triangle)
    # scipy.cluster.hierarchy.linkage expects condensed distance matrix
    condensed_D = squareform(D, checks=False)

    # Perform hierarchical clustering with average-linkage
    # average-linkage is more stable than single-linkage (avoids chaining)
    Z = linkage(condensed_D, method='average')

    # Cut dendrogram to get exactly G clusters
    labels = fcluster(Z, G, criterion='maxclust')

    # Group by labels (labels are 1-indexed, convert to 0-indexed)
    groups = [[] for _ in range(G)]
    for idx in range(N):
        groups[labels[idx] - 1].append(idx)

    # Filter empty groups (shouldn't happen with maxclust, but safety check)
    groups = [g for g in groups if len(g) > 0]

    # Verify we have exactly G groups
    if len(groups) != G:
        warnings.warn(f"Expected {G} groups, got {len(groups)}. Data may have degenerate structure.")

    return clusters_to_perm_and_sizes(groups)


def hem_coarsening(S: np.ndarray, G: int, seed: int = 2021) -> Dict:
    """
    Heavy-Edge Matching (HEM) graph coarsening.

    Iteratively contracts nodes with highest similarity until G supernodes remain.
    Uses greedy matching: for each node, find most similar unmatched neighbor.

    Args:
        S: (N, N) similarity matrix (higher = more similar)
        G: Target number of supernodes (groups)
        seed: Random seed for deterministic order

    Returns:
        Dict with keys: groups, perm, group_sizes
    """
    np.random.seed(seed)
    N = S.shape[0]

    # Initialize: each node is its own supernode
    supernodes = [{i} for i in range(N)]

    # Precompute supernode similarities (will be updated)
    # For efficiency, maintain a similarity matrix for supernodes
    S_super = S.copy()

    while len(supernodes) > G:
        num_super = len(supernodes)

        # If close to target, merge only one pair at a time
        max_merges = num_super - G if (num_super - G) < num_super // 2 else None

        # Greedy heavy-edge matching
        used = set()
        pairs = []

        # Random permutation for tie-breaking
        order = np.random.permutation(num_super)

        for a_idx in order:
            if a_idx in used:
                continue

            # Stop if reached max merges for this iteration
            if max_merges is not None and len(pairs) >= max_merges:
                break

            # Find most similar unmatched supernode
            best_sim = -1
            best_b = None

            for b_idx in range(num_super):
                if b_idx == a_idx or b_idx in used:
                    continue
                sim = S_super[a_idx, b_idx]
                if sim > best_sim:
                    best_sim = sim
                    best_b = b_idx

            if best_b is not None:
                pairs.append((a_idx, best_b))
                used.add(a_idx)
                used.add(best_b)

        # If no pairs formed (shouldn't happen), force merge two closest
        if len(pairs) == 0:
            max_sim = -1
            best_pair = (0, 1)
            for i in range(num_super):
                for j in range(i+1, num_super):
                    if S_super[i, j] > max_sim:
                        max_sim = S_super[i, j]
                        best_pair = (i, j)
            pairs = [best_pair]
            used = {best_pair[0], best_pair[1]}

        # Contract pairs
        new_supernodes = []

        # Merged pairs
        for a_idx, b_idx in pairs:
            merged = supernodes[a_idx] | supernodes[b_idx]
            new_supernodes.append(merged)

        # Unmatched (carry forward)
        for idx in range(num_super):
            if idx not in used:
                new_supernodes.append(supernodes[idx])

        # Update similarity matrix
        new_num = len(new_supernodes)
        new_S_super = np.zeros((new_num, new_num))

        for i in range(new_num):
            for j in range(i, new_num):
                if i == j:
                    new_S_super[i, j] = 1.0
                else:
                    # Average similarity between members
                    sim = np.mean([
                        S[node_i, node_j]
                        for node_i in new_supernodes[i]
                        for node_j in new_supernodes[j]
                    ])
                    new_S_super[i, j] = sim
                    new_S_super[j, i] = sim

        supernodes = new_supernodes
        S_super = new_S_super

    # Convert to groups format
    groups = [sorted(list(supernode)) for supernode in supernodes]

    return clusters_to_perm_and_sizes(groups)


def score_stratified(X_train: np.ndarray, G: int, pred_len: int = 96, seed: int = 2021) -> Dict:
    """
    Score-stratified interleaving: balance groups by prediction difficulty.

    Ranks variables by naive forecast MSE (shift-by-pred_len), then assigns
    them to groups via snake-pattern round-robin so each group contains a
    mix of easy and hard variables.

    Academic basis: Stratified sampling (Neyman 1934), snake-draft
    anticlustering (Papenberg & Klau 2021).

    Args:
        X_train: (T, N) training data
        G: Number of groups
        pred_len: Prediction horizon for naive forecast scoring
        seed: Random seed (unused, deterministic algorithm)

    Returns:
        Dict with keys: groups, perm, group_sizes
    """
    N = X_train.shape[1]
    scores = np.zeros(N)
    for i in range(N):
        series = X_train[:, i]
        # Naive forecast: shift by pred_len
        pred = series[:-pred_len]
        actual = series[pred_len:]
        scores[i] = np.mean((pred - actual) ** 2)

    sorted_idx = np.argsort(scores)  # easy → hard

    # Snake-pattern round-robin
    groups = [[] for _ in range(G)]
    for rank, var_idx in enumerate(sorted_idx):
        cycle = rank // G
        pos = rank % G
        if cycle % 2 == 0:
            groups[pos].append(var_idx)
        else:
            groups[G - 1 - pos].append(var_idx)

    return clusters_to_perm_and_sizes(groups)


def residual_correlation_grouping(X_train: np.ndarray, G: int,
                                   window: int = 24, seed: int = 2021,
                                   max_iter: int = 1000) -> Dict:
    """
    Residual correlation grouping with anti-clustering.

    Removes trend (moving average) from each variable, computes Spearman
    correlation on residuals, then applies anti-clustering to disperse
    highly correlated residual pairs across groups.

    Academic basis: Moving average detrending (Box et al. 2015; Hamilton
    1994), residual analysis (Draper & Smith 1998), anti-clustering
    (Papenberg & Klau 2021).

    Args:
        X_train: (T, N) training data
        G: Number of groups
        window: Moving average window for trend removal
        seed: Random seed
        max_iter: Max anti-clustering iterations

    Returns:
        Dict with keys: groups, perm, group_sizes
    """
    N = X_train.shape[1]

    # 1. Extract residuals: remove moving average (trend) from each variable
    residuals = np.zeros_like(X_train)
    for i in range(N):
        ma = np.convolve(X_train[:, i], np.ones(window) / window, mode='same')
        residuals[:, i] = X_train[:, i] - ma

    # 2. Spearman correlation on residuals (reuse existing function)
    S_resid = spearman_correlation_matrix(residuals, absolute=True)

    # 3. Anti-clustering: disperse high-correlation residual pairs
    result = anti_clustering(S_resid, G, seed=seed, max_iter=max_iter)
    return result


def lead_lag_grouping(X_train: np.ndarray, G: int, max_lag: int = 12,
                       seed: int = 2021, max_iter: int = 1000) -> Dict:
    """
    Lead-lag anti-clustering: disperse temporally related variables.

    Computes cross-correlation at multiple lags to capture lead-lag
    relationships, then applies anti-clustering so each group contains
    variables with diverse temporal roles.

    Academic basis: Cross-correlation (Box et al. 2015), lead-lag
    analysis (Lo & MacKinlay 1990), Granger causality (Granger 1969).

    Args:
        X_train: (T, N) training data
        G: Number of groups
        max_lag: Maximum lag for cross-correlation
        seed: Random seed
        max_iter: Max anti-clustering iterations

    Returns:
        Dict with keys: groups, perm, group_sizes
    """
    N = X_train.shape[1]

    # 1. Cross-correlation matrix (vectorized per lag)
    # Standardize each variable
    X_std = (X_train - X_train.mean(axis=0)) / (X_train.std(axis=0) + 1e-10)

    S_lag = np.zeros((N, N))
    for lag in range(-max_lag, max_lag + 1):
        if lag == 0:
            C = np.corrcoef(X_std.T)  # (N, N)
        elif lag > 0:
            C = np.corrcoef(X_std[lag:].T, X_std[:-lag].T)[:N, N:]
        else:
            C = np.corrcoef(X_std[:lag].T, X_std[-lag:].T)[:N, N:]
        np.maximum(S_lag, np.abs(C), out=S_lag)

    # Handle NaN from constant variables
    S_lag = np.nan_to_num(S_lag, nan=0.0)
    np.fill_diagonal(S_lag, 1.0)

    # 2. Anti-clustering: disperse temporally related pairs
    result = anti_clustering(S_lag, G, seed=seed, max_iter=max_iter)
    return result


def anti_clustering(S: np.ndarray, G: int, seed: int = 2021, max_iter: int = 1000) -> Dict:
    """
    Anti-clustering: maximize intra-group diversity via greedy pairwise swaps.

    Produces groups where variables within each group are as dissimilar as
    possible, ensuring each group captures a diverse cross-section of the
    variable space.

    Algorithm:
    1. Initialize with ordered (sequential) assignment into G equal groups
    2. Convert similarity S to dissimilarity D = 1 - S
    3. Greedy swap: for every pair of variables in different groups, swap if
       it increases total intra-group dissimilarity
    4. Repeat until convergence or max_iter

    Args:
        S: (N, N) similarity matrix (higher = more similar), values in [0, 1]
        G: Number of groups
        seed: Random seed for deterministic tie-breaking
        max_iter: Maximum number of full sweeps

    Returns:
        Dict with keys: groups, perm, group_sizes
    """
    rng = np.random.RandomState(seed)
    N = S.shape[0]

    # Dissimilarity matrix
    D = 1.0 - S

    # Initialize: sequential assignment into G groups (same as ordered)
    base_size = N // G
    remainder = N % G
    group_sizes = [base_size + 1] * remainder + [base_size] * (G - remainder)

    # assignments[i] = group index of variable i
    assignments = np.empty(N, dtype=np.int32)
    start = 0
    for g_idx, size in enumerate(group_sizes):
        assignments[start:start + size] = g_idx
        start += size

    def intra_dissimilarity_contribution(var_idx, group_members):
        """Sum of dissimilarities of var_idx to all other members in its group."""
        if len(group_members) <= 1:
            return 0.0
        return D[var_idx, group_members].sum() - D[var_idx, var_idx]

    # Build group member lists for fast access
    group_members = [[] for _ in range(G)]
    for i in range(N):
        group_members[assignments[i]].append(i)
    group_members = [np.array(g, dtype=np.int32) for g in group_members]

    for iteration in range(max_iter):
        improved = False

        # Shuffle variable order for fairness
        var_order = rng.permutation(N)

        for i in var_order:
            g_i = assignments[i]
            best_delta = 0.0
            best_j = -1

            # Current contribution of i in its group
            contrib_i_current = D[i, group_members[g_i]].sum()

            # Try swapping with a variable from each other group
            for g_other in range(G):
                if g_other == g_i:
                    continue

                members_other = group_members[g_other]

                # Current contribution of each candidate j in g_other
                for j in members_other:
                    contrib_j_current = D[j, members_other].sum()

                    # After swap: i goes to g_other, j goes to g_i
                    # New contribution of i in g_other (replace j with i)
                    contrib_i_new = D[i, members_other].sum() - D[i, j] + D[i, i]
                    # But we need: sum of D[i, members_other \ {j}]
                    contrib_i_new = D[i, members_other].sum() - D[i, j]

                    # New contribution of j in g_i (replace i with j)
                    contrib_j_new = D[j, group_members[g_i]].sum() - D[j, i]

                    # Delta = (new total) - (old total)
                    # We only care about contributions of i and j
                    delta = (contrib_i_new + contrib_j_new) - (contrib_i_current + contrib_j_current)

                    if delta > best_delta:
                        best_delta = delta
                        best_j = j

            if best_j >= 0:
                # Perform swap
                j = best_j
                g_j = assignments[j]

                # Update assignments
                assignments[i] = g_j
                assignments[j] = g_i

                # Update group member arrays
                mask_i = group_members[g_i] != i
                group_members[g_i] = np.append(group_members[g_i][mask_i], j)

                mask_j = group_members[g_j] != j
                group_members[g_j] = np.append(group_members[g_j][mask_j], i)

                improved = True

        if not improved:
            if iteration > 0:
                print(f">>> Anti-clustering converged after {iteration + 1} iterations")
            break
    else:
        print(f">>> Anti-clustering reached max_iter={max_iter}")

    # Convert to groups format
    groups = [sorted(group_members[g].tolist()) for g in range(G)]

    return clusters_to_perm_and_sizes(groups)


def maximin_dispersion(S: np.ndarray, G: int, seed: int = 2021, max_iter: int = 1000) -> Dict:
    """
    MaxMin Dispersion: maximize minimum within-group pairwise distance.

    Produces groups where the closest pair within each group is as far apart
    as possible (bottleneck objective). This is complementary to anti_clustering
    which maximizes total diversity (MAX-SUM); maximin maximizes worst-case
    diversity (MAX-MIN).

    Algorithm (Ravi, Rosenkrantz & Tayi 1994):
    1. Initialize with ordered (sequential) assignment into G equal groups
    2. Convert similarity S to dissimilarity D = 1 - S
    3. Greedy swap: for every pair of variables in different groups, swap if
       it increases the minimum of the two groups' bottleneck distances
    4. Repeat until convergence or max_iter

    Args:
        S: (N, N) similarity matrix (higher = more similar), values in [0, 1]
        G: Number of groups
        seed: Random seed for deterministic tie-breaking
        max_iter: Maximum number of full sweeps

    Returns:
        Dict with keys: groups, perm, group_sizes
    """
    rng = np.random.RandomState(seed)
    N = S.shape[0]

    # Dissimilarity matrix
    D = 1.0 - S

    # Initialize: sequential assignment into G groups (same as ordered)
    base_size = N // G
    remainder = N % G
    group_sizes = [base_size + 1] * remainder + [base_size] * (G - remainder)

    assignments = np.empty(N, dtype=np.int32)
    start = 0
    for g_idx, size in enumerate(group_sizes):
        assignments[start:start + size] = g_idx
        start += size

    # Build group member lists for fast access
    group_members = [[] for _ in range(G)]
    for i in range(N):
        group_members[assignments[i]].append(i)
    group_members = [np.array(g, dtype=np.int32) for g in group_members]

    def min_intra_distance(g_idx):
        """Minimum pairwise distance within group (bottleneck)."""
        members = group_members[g_idx]
        if len(members) <= 1:
            return np.inf
        # Vectorized: extract submatrix, mask diagonal with inf, take min
        sub = D[np.ix_(members, members)]
        np.fill_diagonal(sub, np.inf)
        return sub.min()

    for iteration in range(max_iter):
        improved = False

        # Shuffle variable order for fairness
        var_order = rng.permutation(N)

        for i in var_order:
            g_i = assignments[i]
            current_min_i = min_intra_distance(g_i)

            best_gain = 0.0
            best_j = -1

            for g_other in range(G):
                if g_other == g_i:
                    continue

                current_min_other = min_intra_distance(g_other)
                current_bottleneck = min(current_min_i, current_min_other)

                members_other = group_members[g_other]
                for j in members_other:
                    # Simulate swap: i→g_other, j→g_i
                    # New members of g_i: replace i with j
                    new_gi = group_members[g_i][group_members[g_i] != i]
                    new_gi = np.append(new_gi, j)
                    # New members of g_other: replace j with i
                    new_go = members_other[members_other != j]
                    new_go = np.append(new_go, i)

                    # Compute new bottlenecks
                    if len(new_gi) <= 1:
                        new_min_gi = np.inf
                    else:
                        sub_gi = D[np.ix_(new_gi, new_gi)]
                        np.fill_diagonal(sub_gi, np.inf)
                        new_min_gi = sub_gi.min()

                    if len(new_go) <= 1:
                        new_min_go = np.inf
                    else:
                        sub_go = D[np.ix_(new_go, new_go)]
                        np.fill_diagonal(sub_go, np.inf)
                        new_min_go = sub_go.min()

                    new_bottleneck = min(new_min_gi, new_min_go)
                    gain = new_bottleneck - current_bottleneck

                    if gain > best_gain:
                        best_gain = gain
                        best_j = j

            if best_j >= 0:
                # Perform swap
                j = best_j
                g_j = assignments[j]

                # Update assignments
                assignments[i] = g_j
                assignments[j] = g_i

                # Update group member arrays
                mask_i = group_members[g_i] != i
                group_members[g_i] = np.append(group_members[g_i][mask_i], j)

                mask_j = group_members[g_j] != j
                group_members[g_j] = np.append(group_members[g_j][mask_j], i)

                improved = True

        if not improved:
            print(f">>> MaxMin Dispersion converged after {iteration + 1} iterations")
            break
    else:
        print(f">>> MaxMin Dispersion reached max_iter={max_iter}")

    # Convert to groups format
    groups = [sorted(group_members[g].tolist()) for g in range(G)]

    return clusters_to_perm_and_sizes(groups)


# ==============================================================================
# Helper Functions
# ==============================================================================

def clusters_to_perm_and_sizes(groups: List[List[int]]) -> Dict:
    """
    Convert cluster assignments to permutation and group sizes.

    Args:
        groups: List of G clusters, each cluster is a list of variable indices

    Returns:
        Dict with keys:
        - groups: original groups
        - perm: concatenation of groups (for contiguity)
        - group_sizes: size of each group
    """
    # Flatten groups to get permutation
    perm = []
    for group in groups:
        perm.extend(group)

    # Get sizes
    group_sizes = [len(group) for group in groups]

    return {
        "groups": groups,
        "perm": perm,
        "group_sizes": group_sizes
    }


def compute_group_metrics(
    groups: List[List[int]],
    S: Optional[np.ndarray] = None
) -> Dict:
    """
    Compute group structure quality metrics.

    Args:
        groups: List of G clusters
        S: (N, N) similarity matrix (optional, needed for intra/inter)

    Returns:
        Dict with metrics:
        - CV_size: Coefficient of variation of group sizes
        - intra_cohesion: Average intra-group similarity (if S provided)
        - inter_separation: Average inter-group similarity (if S provided)
    """
    group_sizes = [len(g) for g in groups]

    metrics = {
        "CV_size": np.std(group_sizes) / np.mean(group_sizes)
    }

    if S is not None:
        # Intra-group cohesion
        intra_cohesions = []
        for g in groups:
            if len(g) > 1:
                sims = [S[i, j] for i in g for j in g if i < j]
                if sims:
                    intra_cohesions.append(np.mean(sims))

        if intra_cohesions:
            metrics["intra_cohesion"] = np.mean(intra_cohesions)
        else:
            metrics["intra_cohesion"] = 0.0

        # Inter-group separation
        inter_seps = []
        for g1, g2 in itertools.combinations(groups, 2):
            sims = [S[i, j] for i in g1 for j in g2]
            if sims:
                inter_seps.append(np.mean(sims))

        if inter_seps:
            metrics["inter_separation"] = np.mean(inter_seps)
        else:
            metrics["inter_separation"] = 0.0

    return metrics


# ==============================================================================
# Rebalancing
# ==============================================================================

def rebalance_groups(
    groups: List[List[int]],
    S: np.ndarray,
    target_G: int,
    alpha: float = 1.5,
    min_size: int = 1,
    verbose: bool = True
) -> Dict:
    """
    Post-process groups to enforce size upper bound derived from VG-iT
    efficiency constraint.

    Upper bound: M_max = (N/G) * (1 + sqrt((alpha-1) * (G-1)))
    where alpha = max allowed cost degradation factor.

    Algorithm (Phase 1 — Oversized resolution):
      While max(sizes) > ceil(M_max):
        From the largest group, move the variate most similar (S-based)
        to the smallest group into that smallest group.

    Phase 2 (group count restoration) is unnecessary since only moves occur.
    Phase 3 (undersized): min_size=1 allows singletons; if min_size>1,
    merge small groups into nearest neighbor then split largest to restore G.

    Args:
        groups: List of G clusters, each a list of variable indices
        S: (N, N) similarity matrix (higher = more similar)
        target_G: Target number of groups
        alpha: Max efficiency degradation factor (default 1.5)
        min_size: Minimum group size (default 1, singletons allowed)
        verbose: Print rebalancing info

    Returns:
        Dict with keys: groups, perm, group_sizes
    """
    import math

    # Deep copy to avoid mutation
    groups = [list(g) for g in groups]
    N = sum(len(g) for g in groups)
    G = len(groups)

    # Compute M_max
    avg = N / G
    M_max_raw = avg * (1 + math.sqrt((alpha - 1) * (G - 1)))
    M_max = math.ceil(M_max_raw)

    sizes_before = [len(g) for g in groups]
    max_before = max(sizes_before)

    if verbose:
        print(f">>> Rebalancing: N={N}, G={G}, alpha={alpha:.2f}")
        print(f">>> M_max = ceil({avg:.1f} * (1 + sqrt({alpha-1:.2f} * {G-1}))) = {M_max}")
        print(f">>> Before: max={max_before}, sizes={sizes_before}")

    if max_before <= M_max:
        if verbose:
            print(f">>> No rebalancing needed (max {max_before} <= M_max {M_max})")
        return clusters_to_perm_and_sizes(groups)

    # Phase 1: Move variates from oversized groups to smallest groups
    iteration = 0
    max_iterations = N * G  # Safety bound
    while max(len(g) for g in groups) > M_max and iteration < max_iterations:
        iteration += 1

        # Find largest and smallest group
        sizes = [len(g) for g in groups]
        largest_idx = int(np.argmax(sizes))
        smallest_idx = int(np.argmin(sizes))

        # From the largest group, find the variate most similar to the smallest group
        largest_group = groups[largest_idx]
        smallest_group = groups[smallest_idx]

        best_var = None
        best_sim = -np.inf

        for var in largest_group:
            # Average similarity of this variate to members of the smallest group
            if len(smallest_group) > 0:
                sim = np.mean([S[var, member] for member in smallest_group])
            else:
                sim = 0.0
            if sim > best_sim:
                best_sim = sim
                best_var = var

        # Move the variate
        groups[largest_idx].remove(best_var)
        groups[smallest_idx].append(best_var)

    # Phase 3: Undersized handling (only if min_size > 1)
    if min_size > 1:
        changed = True
        while changed:
            changed = False
            for i in range(len(groups)):
                if len(groups[i]) < min_size and len(groups[i]) > 0:
                    # Find most similar neighbor group
                    best_neighbor = None
                    best_sim = -np.inf
                    for j in range(len(groups)):
                        if j == i or len(groups[j]) == 0:
                            continue
                        sim = np.mean([
                            S[vi, vj]
                            for vi in groups[i]
                            for vj in groups[j]
                        ])
                        if sim > best_sim:
                            best_sim = sim
                            best_neighbor = j
                    if best_neighbor is not None:
                        groups[best_neighbor].extend(groups[i])
                        groups[i] = []
                        changed = True

            # Remove empty groups
            groups = [g for g in groups if len(g) > 0]

            # Restore G by splitting the largest group
            while len(groups) < target_G:
                sizes = [len(g) for g in groups]
                largest_idx = int(np.argmax(sizes))
                g = groups[largest_idx]
                mid = len(g) // 2
                groups[largest_idx] = g[:mid]
                groups.insert(largest_idx + 1, g[mid:])

    # Remove any empty groups (safety)
    groups = [g for g in groups if len(g) > 0]

    sizes_after = [len(g) for g in groups]
    if verbose:
        print(f">>> After:  max={max(sizes_after)}, sizes={sizes_after}")
        # Compute actual alpha
        cost_actual = sum(s ** 2 for s in sizes_after)
        cost_balanced = N ** 2 / G
        alpha_actual = cost_actual / cost_balanced if cost_balanced > 0 else 1.0
        print(f">>> Actual alpha = {alpha_actual:.3f} (limit={alpha:.2f})")

    return clusters_to_perm_and_sizes(groups)


# ==============================================================================
# Main API
# ==============================================================================

def compute_grouping_plan(
    X_train: np.ndarray,
    method: str = "coarsening",
    G: int = 32,
    seed: int = 2021,
    verbose: bool = True,
    alpha: float = 1.5,
    min_size: int = 1,
    pred_len: int = 96
) -> Dict:
    """
    Compute variable grouping plan using specified method.

    Args:
        X_train: (T, N) training data
        method: One of ["ordered", "finch_like", "coarsening", "mi_based", "random",
                        "anti_clustering", "score_stratified", "maximin_dispersion"]
        G: Number of groups
        seed: Random seed for reproducibility
        verbose: Print progress
        alpha: Max efficiency degradation factor for group size bound (default 1.5)
        min_size: Minimum group size (default 1, singletons allowed)
        pred_len: Prediction horizon (used by score_stratified)

    Returns:
        Dict with keys:
        - method: method name
        - N: number of variables
        - G: number of groups
        - seed: random seed
        - groups: List[List[int]] - G clusters
        - perm: List[int] - permutation for contiguity
        - group_sizes: List[int] - sizes (len=G, sum=N)
        - metrics: Dict with CV_size, intra_cohesion, inter_separation
    """
    N = X_train.shape[1]

    if verbose:
        print(f">>> Computing grouping plan: {method}")
        print(f">>> N={N}, G={G}, seed={seed}")

    # Stage 1: Feature extraction (if needed)
    S = None
    if method == "ordered":
        if verbose:
            print(">>> No feature extraction (ordered baseline)")
        result = ordered_grouping(N, G)

    elif method == "finch_like":
        if verbose:
            print(">>> Feature: Spearman correlation")
        S = spearman_correlation_matrix(X_train, absolute=True)
        D = 1 - S  # Convert similarity to distance
        if verbose:
            print(">>> Clustering: Hierarchical (average-linkage)")
        result = mst_cut_clustering(D, G)

    elif method == "coarsening":
        if verbose:
            print(">>> Feature: Spearman correlation")
        S = spearman_correlation_matrix(X_train, absolute=True)
        if verbose:
            print(f">>> Clustering: HEM coarsening (target G={G})")
        result = hem_coarsening(S, G, seed=seed)

    elif method == "mi_based":
        if verbose:
            print(">>> Feature: MI-proxy (rank-based)")
        S = mi_proxy_matrix(X_train)
        if verbose:
            print(f">>> Clustering: HEM coarsening (target G={G})")
        result = hem_coarsening(S, G, seed=seed)

    elif method == "random":
        if verbose:
            print(">>> Feature: None (random assignment)")
        result = random_grouping(N, G, seed=seed)

    elif method == "anti_clustering":
        if verbose:
            print(">>> Feature: Spearman correlation")
        S = spearman_correlation_matrix(X_train, absolute=True)
        if verbose:
            print(f">>> Clustering: Anti-clustering (maximize intra-group diversity)")
        result = anti_clustering(S, G, seed=seed)

    elif method == "score_stratified":
        if verbose:
            print(f">>> Strategy: Score-stratified interleaving (pred_len={pred_len})")
        result = score_stratified(X_train, G, pred_len=pred_len, seed=seed)

    elif method == "residual_correlation":
        if verbose:
            print(">>> Feature: Residual Spearman correlation (trend-removed)")
        S = None  # S_resid computed internally, not needed for rebalancing
        result = residual_correlation_grouping(X_train, G, seed=seed)

    elif method == "maximin_dispersion":
        if verbose:
            print(">>> Feature: Spearman correlation")
        S = spearman_correlation_matrix(X_train, absolute=True)
        if verbose:
            print(f">>> Clustering: MaxMin Dispersion (maximize min intra-group distance)")
        result = maximin_dispersion(S, G, seed=seed)

    elif method == "lead_lag":
        if verbose:
            print(">>> Feature: Cross-correlation (max over lags)")
        S = None  # S_lag computed internally, not needed for rebalancing
        result = lead_lag_grouping(X_train, G, seed=seed)

    else:
        raise ValueError(f"Unknown method: {method}")

    # Rebalancing: enforce M_max upper bound (only for similarity-based methods)
    if S is not None:
        import math
        avg = N / G
        M_max = math.ceil(avg * (1 + math.sqrt((alpha - 1) * (G - 1))))
        max_size = max(len(g) for g in result["groups"])
        if max_size > M_max:
            if verbose:
                print(f">>> Applying rebalancing (max_size={max_size} > M_max={M_max})")
            result = rebalance_groups(
                result["groups"], S, target_G=G,
                alpha=alpha, min_size=min_size, verbose=verbose
            )
        else:
            if verbose:
                print(f">>> Rebalancing skipped (max_size={max_size} <= M_max={M_max})")

    # Compute metrics
    metrics = compute_group_metrics(result["groups"], S)

    if verbose:
        print(f">>> Grouping complete: {len(result['groups'])} groups")
        print(f">>> Group sizes: {result['group_sizes']}")
        print(f">>> CV(size) = {metrics['CV_size']:.3f}")
        if "intra_cohesion" in metrics:
            print(f">>> Intra-ρ = {metrics['intra_cohesion']:.3f}")
            print(f">>> Inter-ρ = {metrics['inter_separation']:.3f}")

    # Assemble final result
    result.update({
        "method": method,
        "N": N,
        "G": G,
        "seed": seed,
        "metrics": metrics
    })

    return result


# ==============================================================================
# Validation
# ==============================================================================

def validate_grouping_plan(result: Dict) -> bool:
    """
    Validate that grouping plan is well-formed.

    Returns:
        True if valid, raises AssertionError otherwise
    """
    N = result["N"]
    G = result["G"]
    perm = result["perm"]
    group_sizes = result["group_sizes"]
    groups = result["groups"]

    # Check perm
    assert len(perm) == N, f"perm length {len(perm)} != N {N}"
    assert set(perm) == set(range(N)), "perm missing indices"

    # Check group_sizes
    assert len(group_sizes) == G, f"group_sizes length {len(group_sizes)} != G {G}"
    assert sum(group_sizes) == N, f"sum(group_sizes) {sum(group_sizes)} != N {N}"

    # Check groups
    assert len(groups) == G, f"groups length {len(groups)} != G {G}"
    all_vars = set()
    for g in groups:
        all_vars.update(g)
    assert all_vars == set(range(N)), "groups missing variables"

    # Check contiguity
    perm_check = []
    for g in groups:
        perm_check.extend(sorted(g))  # Note: perm may reorder within group
    # Can't directly compare perm and perm_check due to within-group ordering
    # Just ensure all variables covered
    assert set(perm) == set(perm_check), "perm and groups mismatch"

    return True


if __name__ == "__main__":
    # Quick test
    print("Testing custom_grouping.py...")

    # Generate random data
    np.random.seed(42)
    T, N, G = 1000, 100, 10
    X = np.random.randn(T, N)

    for method in ["ordered", "finch_like", "coarsening", "mi_based", "random", "anti_clustering",
                    "score_stratified", "residual_correlation", "lead_lag"]:
        print(f"\n{'='*60}")
        print(f"Testing method: {method}")
        print('='*60)

        result = compute_grouping_plan(X, method=method, G=G, seed=42, verbose=True, pred_len=96)
        validate_grouping_plan(result)

        print("✓ Validation passed")

    print("\n" + "="*60)
    print("All tests passed!")
