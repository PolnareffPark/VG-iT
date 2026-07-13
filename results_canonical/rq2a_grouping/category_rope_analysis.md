# Category-Level Bayesian ROPE & Statistical Analysis

**Generated**: 2026-03-12 17:46:05
**Data**: `grouping_invariance_results.csv` (2880 rows)
**methodology**: Bayesian signed-rank ROPE (Benavoli et al., 2017 JMLR), Wilcoxon signed-rank, Friedman test

## 0. Data Summary

- methods: ['anti_clustering', 'coarsening', 'finch_like', 'maximin_dispersion', 'mi_based', 'ordered', 'random', 'score_stratified']
- Datasets: ['electricity', 'solar_AL', 'traffic']
- Pred_lens: [np.int64(96), np.int64(192), np.int64(336), np.int64(720)]
- Seeds: 30 (2021-2050)
- Total runs: 2880

### 4-Way Category Classification

| Category | method | Description |
|----------|------|------|
| No strategy (No strategy) | ordered | deterministic equal-width partition over CSV column order, deterministic |
| Control (Control) | random | balanced random assignment |
| Diversity (Diversity) | anti_clustering, score_stratified, maximin_dispersion | explicit variable-dispersion strategies |
| Cohesion (Cohesion) | finch_like, coarsening, mi_based | HEM/hierarchical cohesion strategies |

### Convergence outliers (MSE > 0.6)

- Outlier count: 25 / 2880 (0.9%)
- All occur on Traffic and in non-cohesion methods
- Cohesion outliers: 0 (0%)

- Number of (dataset, pred_len, seed) triplets containing outliers: 14

### ROPE Parameters

| Condition | Mean MSE | ROPE (+-1%) |
|------|----------|------------|
| All 8 (raw) | 0.283100 | +-0.002831 |
| Diversity 3 (raw) | 0.285496 | +-0.002855 |
| Cohesion 3 (raw) | 0.278870 | +-0.002789 |
| Non-cohesion 5 (raw) | 0.285639 | +-0.002856 |
| All 8 (clean) | 0.271864 | +-0.002719 |
| Diversity 3 (clean) | 0.271858 | +-0.002719 |
| Non-cohesion 5 (clean) | 0.271650 | +-0.002716 |

---

## 1. Within-Category ROPE (Category within-category equivalence)

### 1.1 Cohesion(Cohesion) Comparison

### Cohesion Internal (raw)

- N conditions (dataset x pred_len x seed): 360
- ROPE = +-0.002831

| Pair | Mean Diff | Cohen's d | Interp | P(left) | P(ROPE) | P(right) | ROPE Decision | Wilcoxon p |
|------|-----------|-----------|--------|---------|---------|----------|---------------|------------|
| finch_like vs coarsening | -0.000330 | -0.1521 | negligible | 0.0000 | 1.0000 | 0.0000 | EQUIVALENT | 0.000689 |
| finch_like vs mi_based | -0.000668 | -0.2632 | small | 0.0000 | 1.0000 | 0.0000 | EQUIVALENT | 0.000167 |
| coarsening vs mi_based | -0.000338 | -0.1257 | negligible | 0.0000 | 1.0000 | 0.0000 | EQUIVALENT | 0.032455 |

**Summary**: EQUIVALENT 3/3, INCONCLUSIVE 0/3, DIFFERENT 0/3
- Mean P(ROPE): 1.0000
- Max |Cohen's d|: 0.2632 (small)

### Cohesion Internal (clean)

- N conditions (dataset x pred_len x seed): 346
- ROPE = +-0.002719

| Pair | Mean Diff | Cohen's d | Interp | P(left) | P(ROPE) | P(right) | ROPE Decision | Wilcoxon p |
|------|-----------|-----------|--------|---------|---------|----------|---------------|------------|
| finch_like vs coarsening | -0.000295 | -0.1361 | negligible | 0.0000 | 1.0000 | 0.0000 | EQUIVALENT | 0.002241 |
| finch_like vs mi_based | -0.000603 | -0.2454 | small | 0.0000 | 1.0000 | 0.0000 | EQUIVALENT | 0.000513 |
| coarsening vs mi_based | -0.000307 | -0.1175 | negligible | 0.0000 | 1.0000 | 0.0000 | EQUIVALENT | 0.044317 |

**Summary**: EQUIVALENT 3/3, INCONCLUSIVE 0/3, DIFFERENT 0/3
- Mean P(ROPE): 1.0000
- Max |Cohen's d|: 0.2454 (small)

### 1.2 Diversity(Diversity) Comparison

### Diversity Internal (3 methods) (raw)

- N conditions (dataset x pred_len x seed): 360
- ROPE = +-0.002855

| Pair | Mean Diff | Cohen's d | Interp | P(left) | P(ROPE) | P(right) | ROPE Decision | Wilcoxon p |
|------|-----------|-----------|--------|---------|---------|----------|---------------|------------|
| anti_clustering vs score_stratified | -0.000245 | -0.0044 | negligible | 0.2643 | 0.5085 | 0.2272 | INCONCLUSIVE | 0.000000 |
| anti_clustering vs maximin_dispersion | -0.003428 | -0.0498 | negligible | 0.5444 | 0.3448 | 0.1108 | INCONCLUSIVE | 0.016731 |
| score_stratified vs maximin_dispersion | -0.003183 | -0.0416 | negligible | 0.5229 | 0.3320 | 0.1451 | INCONCLUSIVE | 0.000064 |

**Summary**: EQUIVALENT 0/3, INCONCLUSIVE 3/3, DIFFERENT 0/3
- Mean P(ROPE): 0.3951
- Max |Cohen's d|: 0.0498 (negligible)

### Diversity Internal (3 methods) (clean)

- N conditions (dataset x pred_len x seed): 346
- ROPE = +-0.002719

| Pair | Mean Diff | Cohen's d | Interp | P(left) | P(ROPE) | P(right) | ROPE Decision | Wilcoxon p |
|------|-----------|-----------|--------|---------|---------|----------|---------------|------------|
| anti_clustering vs score_stratified | +0.000575 | +0.3903 | small | 0.0000 | 1.0000 | 0.0000 | EQUIVALENT | 0.000000 |
| anti_clustering vs maximin_dispersion | +0.000258 | +0.1414 | negligible | 0.0000 | 1.0000 | 0.0000 | EQUIVALENT | 0.011023 |
| score_stratified vs maximin_dispersion | -0.000318 | -0.2169 | small | 0.0000 | 1.0000 | 0.0000 | EQUIVALENT | 0.000050 |

**Summary**: EQUIVALENT 3/3, INCONCLUSIVE 0/3, DIFFERENT 0/3
- Mean P(ROPE): 1.0000
- Max |Cohen's d|: 0.3903 (small)

### 1.3 All 8 methods -- all C(8,2)=28 Pair (Grouping invariance)

ROPE = +-1% of all 8 methods mean MSE. All 28 pairwise ROPE tests validate grouping invariance.

### All 8 methods (28 pairs) (raw)

- N conditions (dataset x pred_len x seed): 360
- ROPE = +-0.002831

| Pair | Type | Mean Diff | Cohen's d | Interp | P(left) | P(ROPE) | P(right) | ROPE Decision | Wilcoxon p |
|------|------|-----------|-----------|--------|---------|---------|----------|---------------|------------|
| finch_like vs coarsening | within-Cohesion | -0.000330 | -0.1521 | negligible | 0.0000 | 1.0000 | 0.0000 | EQUIVALENT | 0.000689 |
| finch_like vs mi_based | within-Cohesion | -0.000668 | -0.2632 | small | 0.0000 | 1.0000 | 0.0000 | EQUIVALENT | 0.000167 |
| finch_like vs anti_clustering | Cohesion × Diversity | -0.005735 | -0.1102 | negligible | 0.7727 | 0.2134 | 0.0139 | INCONCLUSIVE | 0.045490 |
| finch_like vs score_stratified | Cohesion × Diversity | -0.005980 | -0.1161 | negligible | 0.7935 | 0.1953 | 0.0112 | INCONCLUSIVE | 0.002544 |
| finch_like vs maximin_dispersion | Cohesion × Diversity | -0.009163 | -0.1395 | negligible | 0.9015 | 0.0911 | 0.0074 | INCONCLUSIVE | 0.778412 |
| finch_like vs ordered | Cohesion × No strategy | -0.005648 | -0.0832 | negligible | 0.7109 | 0.2417 | 0.0474 | INCONCLUSIVE | 0.000000 |
| finch_like vs random | Cohesion × Control | -0.008982 | -0.1096 | negligible | 0.8425 | 0.1305 | 0.0271 | INCONCLUSIVE | 0.831670 |
| coarsening vs mi_based | within-Cohesion | -0.000338 | -0.1257 | negligible | 0.0000 | 1.0000 | 0.0000 | EQUIVALENT | 0.032455 |
| coarsening vs anti_clustering | Cohesion × Diversity | -0.005405 | -0.1038 | negligible | 0.7461 | 0.2366 | 0.0173 | INCONCLUSIVE | 0.482076 |
| coarsening vs score_stratified | Cohesion × Diversity | -0.005650 | -0.1100 | negligible | 0.7689 | 0.2174 | 0.0137 | INCONCLUSIVE | 0.000095 |
| coarsening vs maximin_dispersion | Cohesion × Diversity | -0.008833 | -0.1345 | negligible | 0.8893 | 0.1018 | 0.0089 | INCONCLUSIVE | 0.050339 |
| coarsening vs ordered | Cohesion × No strategy | -0.005318 | -0.0786 | negligible | 0.6887 | 0.2576 | 0.0537 | INCONCLUSIVE | 0.000000 |
| coarsening vs random | Cohesion × Control | -0.008651 | -0.1059 | negligible | 0.8299 | 0.1399 | 0.0302 | INCONCLUSIVE | 0.114971 |
| mi_based vs anti_clustering | Cohesion × Diversity | -0.005067 | -0.0979 | negligible | 0.7186 | 0.2606 | 0.0207 | INCONCLUSIVE | 0.201287 |
| mi_based vs score_stratified | Cohesion × Diversity | -0.005312 | -0.1032 | negligible | 0.7408 | 0.2419 | 0.0173 | INCONCLUSIVE | 0.000005 |
| mi_based vs maximin_dispersion | Cohesion × Diversity | -0.008495 | -0.1295 | negligible | 0.8762 | 0.1132 | 0.0106 | INCONCLUSIVE | 0.012976 |
| mi_based vs ordered | Cohesion × No strategy | -0.004980 | -0.0737 | negligible | 0.6650 | 0.2740 | 0.0609 | INCONCLUSIVE | 0.000000 |
| mi_based vs random | Cohesion × Control | -0.008314 | -0.1015 | negligible | 0.8148 | 0.1506 | 0.0345 | INCONCLUSIVE | 0.028607 |
| anti_clustering vs score_stratified | within-Diversity | -0.000245 | -0.0044 | negligible | 0.2662 | 0.5048 | 0.2289 | INCONCLUSIVE | 0.000000 |
| anti_clustering vs maximin_dispersion | within-Diversity | -0.003428 | -0.0498 | negligible | 0.5463 | 0.3420 | 0.1117 | INCONCLUSIVE | 0.016731 |
| anti_clustering vs ordered | Diversity × No strategy | +0.000087 | +0.0015 | negligible | 0.2542 | 0.4789 | 0.2670 | INCONCLUSIVE | 0.000000 |
| anti_clustering vs random | Diversity × Control | -0.003247 | -0.0351 | negligible | 0.5240 | 0.2864 | 0.1896 | INCONCLUSIVE | 0.054650 |
| score_stratified vs maximin_dispersion | within-Diversity | -0.003183 | -0.0416 | negligible | 0.5246 | 0.3293 | 0.1461 | INCONCLUSIVE | 0.000064 |
| score_stratified vs ordered | Diversity × No strategy | +0.000332 | +0.0043 | negligible | 0.2919 | 0.3755 | 0.3326 | INCONCLUSIVE | 0.000000 |
| score_stratified vs random | Diversity × Control | -0.003002 | -0.0436 | negligible | 0.5132 | 0.3584 | 0.1284 | INCONCLUSIVE | 0.000000 |
| maximin_dispersion vs ordered | Diversity × No strategy | +0.003515 | +0.0398 | negligible | 0.1680 | 0.2907 | 0.5413 | INCONCLUSIVE | 0.000000 |
| maximin_dispersion vs random | Diversity × Control | +0.000181 | +0.0023 | negligible | 0.3036 | 0.3708 | 0.3256 | INCONCLUSIVE | 0.439935 |
| ordered vs random | No strategy × Control | -0.003334 | -0.0326 | negligible | 0.5263 | 0.2641 | 0.2096 | INCONCLUSIVE | 0.000000 |

**Summary**: EQUIVALENT 3/28, INCONCLUSIVE 25/28, DIFFERENT 0/28
- Mean P(ROPE): 0.3381
- Max |Cohen's d|: 0.2632 (small)

### All 8 methods (28 pairs) (clean)

- N conditions (dataset x pred_len x seed): 346
- ROPE = +-0.002719

| Pair | Type | Mean Diff | Cohen's d | Interp | P(left) | P(ROPE) | P(right) | ROPE Decision | Wilcoxon p |
|------|------|-----------|-----------|--------|---------|---------|----------|---------------|------------|
| finch_like vs coarsening | within-Cohesion | -0.000295 | -0.1361 | negligible | 0.0000 | 1.0000 | 0.0000 | EQUIVALENT | 0.002241 |
| finch_like vs mi_based | within-Cohesion | -0.000603 | -0.2454 | small | 0.0000 | 1.0000 | 0.0000 | EQUIVALENT | 0.000513 |
| finch_like vs anti_clustering | Cohesion × Diversity | -0.000214 | -0.0961 | negligible | 0.0000 | 1.0000 | 0.0000 | EQUIVALENT | 0.295829 |
| finch_like vs score_stratified | Cohesion × Diversity | +0.000361 | +0.1812 | negligible | 0.0000 | 1.0000 | 0.0000 | EQUIVALENT | 0.000051 |
| finch_like vs maximin_dispersion | Cohesion × Diversity | +0.000044 | +0.0189 | negligible | 0.0000 | 1.0000 | 0.0000 | EQUIVALENT | 0.155364 |
| finch_like vs ordered | Cohesion × No strategy | +0.001239 | +0.5968 | medium | 0.0000 | 1.0000 | 0.0000 | EQUIVALENT | 0.000000 |
| finch_like vs random | Cohesion × Control | -0.000069 | -0.0288 | negligible | 0.0000 | 1.0000 | 0.0000 | EQUIVALENT | 0.506973 |
| coarsening vs mi_based | within-Cohesion | -0.000307 | -0.1175 | negligible | 0.0000 | 1.0000 | 0.0000 | EQUIVALENT | 0.044317 |
| coarsening vs anti_clustering | Cohesion × Diversity | +0.000081 | +0.0270 | negligible | 0.0000 | 1.0000 | 0.0000 | EQUIVALENT | 0.183849 |
| coarsening vs score_stratified | Cohesion × Diversity | +0.000657 | +0.2379 | small | 0.0000 | 1.0000 | 0.0000 | EQUIVALENT | 0.000001 |
| coarsening vs maximin_dispersion | Cohesion × Diversity | +0.000339 | +0.1099 | negligible | 0.0000 | 1.0000 | 0.0000 | EQUIVALENT | 0.007344 |
| coarsening vs ordered | Cohesion × No strategy | +0.001534 | +0.6093 | medium | 0.0000 | 1.0000 | 0.0000 | EQUIVALENT | 0.000000 |
| coarsening vs random | Cohesion × Control | +0.000226 | +0.0720 | negligible | 0.0000 | 1.0000 | 0.0000 | EQUIVALENT | 0.018680 |
| mi_based vs anti_clustering | Cohesion × Diversity | +0.000389 | +0.1245 | negligible | 0.0000 | 1.0000 | 0.0000 | EQUIVALENT | 0.069351 |
| mi_based vs score_stratified | Cohesion × Diversity | +0.000964 | +0.3275 | small | 0.0000 | 1.0000 | 0.0000 | EQUIVALENT | 0.000000 |
| mi_based vs maximin_dispersion | Cohesion × Diversity | +0.000646 | +0.2007 | small | 0.0000 | 1.0000 | 0.0000 | EQUIVALENT | 0.001402 |
| mi_based vs ordered | Cohesion × No strategy | +0.001842 | +0.6778 | medium | 0.0000 | 1.0000 | 0.0000 | EQUIVALENT | 0.000000 |
| mi_based vs random | Cohesion × Control | +0.000534 | +0.1649 | negligible | 0.0000 | 1.0000 | 0.0000 | EQUIVALENT | 0.006454 |
| anti_clustering vs score_stratified | within-Diversity | +0.000575 | +0.3903 | small | 0.0000 | 1.0000 | 0.0000 | EQUIVALENT | 0.000000 |
| anti_clustering vs maximin_dispersion | within-Diversity | +0.000258 | +0.1414 | negligible | 0.0000 | 1.0000 | 0.0000 | EQUIVALENT | 0.011023 |
| anti_clustering vs ordered | Diversity × No strategy | +0.001453 | +0.6638 | medium | 0.0000 | 1.0000 | 0.0000 | EQUIVALENT | 0.000000 |
| anti_clustering vs random | Diversity × Control | +0.000145 | +0.0871 | negligible | 0.0000 | 1.0000 | 0.0000 | EQUIVALENT | 0.030659 |
| score_stratified vs maximin_dispersion | within-Diversity | -0.000318 | -0.2169 | small | 0.0000 | 1.0000 | 0.0000 | EQUIVALENT | 0.000050 |
| score_stratified vs ordered | Diversity × No strategy | +0.000877 | +0.4141 | small | 0.0000 | 1.0000 | 0.0000 | EQUIVALENT | 0.000000 |
| score_stratified vs random | Diversity × Control | -0.000430 | -0.2886 | small | 0.0000 | 1.0000 | 0.0000 | EQUIVALENT | 0.000000 |
| maximin_dispersion vs ordered | Diversity × No strategy | +0.001195 | +0.5144 | medium | 0.0000 | 1.0000 | 0.0000 | EQUIVALENT | 0.000000 |
| maximin_dispersion vs random | Diversity × Control | -0.000112 | -0.0664 | negligible | 0.0000 | 1.0000 | 0.0000 | EQUIVALENT | 0.291386 |
| ordered vs random | No strategy × Control | -0.001308 | -0.5316 | medium | 0.0000 | 1.0000 | 0.0000 | EQUIVALENT | 0.000000 |

**Summary**: EQUIVALENT 28/28, INCONCLUSIVE 0/28, DIFFERENT 0/28
- Mean P(ROPE): 1.0000
- Max |Cohen's d|: 0.6778 (medium)

#### 28 Pairs Decision by Cross-Category Type (clean)

| Type | Pairs | EQUIVALENT | INCONCLUSIVE | DIFFERENT |
|------|-------|------------|--------------|-----------|
| within-Diversity | 3 | 3 | 0 | 0 |
| within-Cohesion | 3 | 3 | 0 | 0 |
| No strategy × Control | 1 | 1 | 0 | 0 |
| Diversity × Control | 3 | 3 | 0 | 0 |
| Diversity × No strategy | 3 | 3 | 0 | 0 |
| Cohesion × Control | 3 | 3 | 0 | 0 |
| Cohesion × No strategy | 3 | 3 | 0 | 0 |
| Cohesion × Diversity | 9 | 9 | 0 | 0 |

---

## 2. Between-Category ROPE (4-way Category between Comparison)

### 2.1 4-Way Aggregate Comparisons

For each condition (dataset, pred_len, seed), paired comparisons are computed from category-level mean MSE.

#### 4-Way Aggregate (raw)

| Comparison | N | Mean A | Mean B | Mean Diff | Cohen's d | Interp | P(A better) | P(ROPE) | P(B better) | Decision | Wilcoxon p |
|------|---|--------|--------|-----------|-----------|--------|-------------|---------|-------------|----------|------------|
| No strategy vs Control | 360 | 0.284185 | 0.287519 | -0.003334 | -0.0326 | negligible | 0.5248 | 0.2666 | 0.2085 | INCONCLUSIVE | 0.000000 |
| No strategy vs Diversity | 360 | 0.284185 | 0.285496 | -0.001311 | -0.0201 | negligible | 0.3759 | 0.4279 | 0.1961 | INCONCLUSIVE | 0.000000 |
| No strategy vs Cohesion | 360 | 0.284185 | 0.278870 | +0.005315 | +0.0785 | negligible | 0.0541 | 0.2563 | 0.6896 | INCONCLUSIVE | 0.000000 |
| Control vs Diversity | 360 | 0.287519 | 0.285496 | +0.002022 | +0.0287 | negligible | 0.1767 | 0.3869 | 0.4364 | INCONCLUSIVE | 0.392944 |
| Control vs Cohesion | 360 | 0.287519 | 0.278870 | +0.008649 | +0.1057 | negligible | 0.0304 | 0.1403 | 0.8293 | INCONCLUSIVE | 0.070405 |
| Diversity vs Cohesion | 360 | 0.285496 | 0.278870 | +0.006627 | +0.1608 | negligible | 0.0011 | 0.1072 | 0.8917 | INCONCLUSIVE | 0.017424 |

**Per-dataset breakdown:**

*No strategy vs Control:*

| Dataset | N | Mean Diff | Cohen's d | Wilcoxon p | Direction |
|---------|---|-----------|-----------|------------|-----------|
| electricity | 120 | -0.003277 | -1.2109 | 0.000000 | No strategy better |
| solar_AL | 120 | -0.000807 | -0.5724 | 0.000000 | No strategy better |
| traffic | 120 | -0.005917 | -0.0333 | 0.002467 | No strategy better |

*No strategy vs Diversity:*

| Dataset | N | Mean Diff | Cohen's d | Wilcoxon p | Direction |
|---------|---|-----------|-----------|------------|-----------|
| electricity | 120 | -0.002757 | -1.1800 | 0.000000 | No strategy better |
| solar_AL | 120 | -0.000820 | -0.8132 | 0.000000 | No strategy better |
| traffic | 120 | -0.000356 | -0.0032 | 0.675202 | No strategy better |

*No strategy vs Cohesion:*

| Dataset | N | Mean Diff | Cohen's d | Wilcoxon p | Direction |
|---------|---|-----------|-----------|------------|-----------|
| electricity | 120 | -0.001232 | -0.5800 | 0.000000 | No strategy better |
| solar_AL | 120 | -0.002418 | -1.2526 | 0.000000 | No strategy better |
| traffic | 120 | +0.019596 | +0.1686 | 0.000006 | No strategy worse |

*Control vs Diversity:*

| Dataset | N | Mean Diff | Cohen's d | Wilcoxon p | Direction |
|---------|---|-----------|-----------|------------|-----------|
| electricity | 120 | +0.000519 | +0.3427 | 0.000906 | Control worse |
| solar_AL | 120 | -0.000013 | -0.0101 | 0.743397 | Control better |
| traffic | 120 | +0.005560 | +0.0454 | 0.012755 | Control worse |

*Control vs Cohesion:*

| Dataset | N | Mean Diff | Cohen's d | Wilcoxon p | Direction |
|---------|---|-----------|-----------|------------|-----------|
| electricity | 120 | +0.002045 | +0.8585 | 0.000000 | Control worse |
| solar_AL | 120 | -0.001611 | -0.8489 | 0.000000 | Control better |
| traffic | 120 | +0.025512 | +0.1814 | 0.000000 | Control worse |

*Diversity vs Cohesion:*

| Dataset | N | Mean Diff | Cohen's d | Wilcoxon p | Direction |
|---------|---|-----------|-----------|------------|-----------|
| electricity | 120 | +0.001526 | +0.7698 | 0.000000 | Diversity worse |
| solar_AL | 120 | -0.001598 | -0.9019 | 0.000000 | Diversity better |
| traffic | 120 | +0.019952 | +0.2867 | 0.000051 | Diversity worse |

#### 4-Way Aggregate (clean)

| Comparison | N | Mean A | Mean B | Mean Diff | Cohen's d | Interp | P(A better) | P(ROPE) | P(B better) | Decision | Wilcoxon p |
|------|---|--------|--------|-----------|-----------|--------|-------------|---------|-------------|----------|------------|
| No strategy vs Control | 346 | 0.270683 | 0.271991 | -0.001308 | -0.5316 | medium | 0.0000 | 1.0000 | 0.0000 | EQUIVALENT | 0.000000 |
| No strategy vs Diversity | 346 | 0.270683 | 0.271858 | -0.001175 | -0.5844 | medium | 0.0000 | 1.0000 | 0.0000 | EQUIVALENT | 0.000000 |
| No strategy vs Cohesion | 346 | 0.270683 | 0.272221 | -0.001538 | -0.7638 | medium | 0.0000 | 1.0000 | 0.0000 | EQUIVALENT | 0.000000 |
| Control vs Diversity | 346 | 0.271991 | 0.271858 | +0.000132 | +0.0995 | negligible | 0.0000 | 1.0000 | 0.0000 | EQUIVALENT | 0.201069 |
| Control vs Cohesion | 346 | 0.271991 | 0.272221 | -0.000230 | -0.0887 | negligible | 0.0000 | 1.0000 | 0.0000 | EQUIVALENT | 0.009611 |
| Diversity vs Cohesion | 346 | 0.271858 | 0.272221 | -0.000363 | -0.1643 | negligible | 0.0000 | 1.0000 | 0.0000 | EQUIVALENT | 0.000221 |

**Per-dataset breakdown:**

*No strategy vs Control:*

| Dataset | N | Mean Diff | Cohen's d | Wilcoxon p | Direction |
|---------|---|-----------|-----------|------------|-----------|
| electricity | 120 | -0.003277 | -1.2109 | 0.000000 | No strategy better |
| solar_AL | 120 | -0.000807 | -0.5724 | 0.000000 | No strategy better |
| traffic | 106 | +0.000355 | +0.2633 | 0.000076 | No strategy worse |

*No strategy vs Diversity:*

| Dataset | N | Mean Diff | Cohen's d | Wilcoxon p | Direction |
|---------|---|-----------|-----------|------------|-----------|
| electricity | 120 | -0.002757 | -1.1800 | 0.000000 | No strategy better |
| solar_AL | 120 | -0.000820 | -0.8132 | 0.000000 | No strategy better |
| traffic | 106 | +0.000214 | +0.2153 | 0.059666 | No strategy worse |

*No strategy vs Cohesion:*

| Dataset | N | Mean Diff | Cohen's d | Wilcoxon p | Direction |
|---------|---|-----------|-----------|------------|-----------|
| electricity | 120 | -0.001232 | -0.5800 | 0.000000 | No strategy better |
| solar_AL | 120 | -0.002418 | -1.2526 | 0.000000 | No strategy better |
| traffic | 106 | -0.000889 | -0.5543 | 0.000000 | No strategy better |

*Control vs Diversity:*

| Dataset | N | Mean Diff | Cohen's d | Wilcoxon p | Direction |
|---------|---|-----------|-----------|------------|-----------|
| electricity | 120 | +0.000519 | +0.3427 | 0.000906 | Control worse |
| solar_AL | 120 | -0.000013 | -0.0101 | 0.743397 | Control better |
| traffic | 106 | -0.000141 | -0.1361 | 0.020438 | Control better |

*Control vs Cohesion:*

| Dataset | N | Mean Diff | Cohen's d | Wilcoxon p | Direction |
|---------|---|-----------|-----------|------------|-----------|
| electricity | 120 | +0.002045 | +0.8585 | 0.000000 | Control worse |
| solar_AL | 120 | -0.001611 | -0.8489 | 0.000000 | Control better |
| traffic | 106 | -0.001244 | -0.7852 | 0.000000 | Control better |

*Diversity vs Cohesion:*

| Dataset | N | Mean Diff | Cohen's d | Wilcoxon p | Direction |
|---------|---|-----------|-----------|------------|-----------|
| electricity | 120 | +0.001526 | +0.7698 | 0.000000 | Diversity worse |
| solar_AL | 120 | -0.001598 | -0.9019 | 0.000000 | Diversity better |
| traffic | 106 | -0.001103 | -0.8577 | 0.000000 | Diversity better |

### 2.2 Cohesion each method vs ordered

#### Cohesion vs ordered (raw)

- N conditions: 360
- ROPE = +-0.002802

| Pair | Mean Diff | Cohen's d | Interp | P(left) | P(ROPE) | P(right) | Decision | Wilcoxon p |
|------|-----------|-----------|--------|---------|---------|----------|----------|------------|
| finch_like vs ordered | -0.005648 | -0.0832 | negligible | 0.7128 | 0.2392 | 0.0480 | INCONCLUSIVE | 0.000000 |
| coarsening vs ordered | -0.005318 | -0.0786 | negligible | 0.6907 | 0.2550 | 0.0543 | INCONCLUSIVE | 0.000000 |
| mi_based vs ordered | -0.004980 | -0.0737 | negligible | 0.6671 | 0.2712 | 0.0616 | INCONCLUSIVE | 0.000000 |

#### Cohesion vs ordered (clean)

- N conditions: 346
- ROPE = +-0.002718

| Pair | Mean Diff | Cohen's d | Interp | P(left) | P(ROPE) | P(right) | Decision | Wilcoxon p |
|------|-----------|-----------|--------|---------|---------|----------|----------|------------|
| finch_like vs ordered | +0.001239 | +0.5968 | medium | 0.0000 | 1.0000 | 0.0000 | EQUIVALENT | 0.000000 |
| coarsening vs ordered | +0.001534 | +0.6093 | medium | 0.0000 | 1.0000 | 0.0000 | EQUIVALENT | 0.000000 |
| mi_based vs ordered | +0.001842 | +0.6778 | medium | 0.0000 | 1.0000 | 0.0000 | EQUIVALENT | 0.000000 |

### 2.3 Diversity x Cohesion Cross-Pairs (9 pairs)

#### Cohesion x Diversity Cross-Pairs (raw)

- N conditions: 360
- ROPE = +-0.002822

| Cohesion | Diversity | Mean Diff | Cohen's d | Interp | P(Cohesion better) | P(ROPE) | P(Diversity better) | Decision | Wilcoxon p |
|---------------|---------------|-----------|-----------|--------|---------------------|---------|---------------------|----------|------------|
| finch_like | anti_clustering | -0.005735 | -0.1102 | negligible | 0.7734 | 0.2126 | 0.0140 | INCONCLUSIVE | 0.045490 |
| finch_like | score_stratified | -0.005980 | -0.1161 | negligible | 0.7942 | 0.1945 | 0.0113 | INCONCLUSIVE | 0.002544 |
| finch_like | maximin_dispersion | -0.009163 | -0.1395 | negligible | 0.9018 | 0.0907 | 0.0075 | INCONCLUSIVE | 0.778412 |
| coarsening | anti_clustering | -0.005405 | -0.1038 | negligible | 0.7469 | 0.2358 | 0.0174 | INCONCLUSIVE | 0.482076 |
| coarsening | score_stratified | -0.005650 | -0.1100 | negligible | 0.7696 | 0.2166 | 0.0138 | INCONCLUSIVE | 0.000095 |
| coarsening | maximin_dispersion | -0.008833 | -0.1345 | negligible | 0.8897 | 0.1014 | 0.0089 | INCONCLUSIVE | 0.050339 |
| mi_based | anti_clustering | -0.005067 | -0.0979 | negligible | 0.7194 | 0.2597 | 0.0208 | INCONCLUSIVE | 0.201287 |
| mi_based | score_stratified | -0.005312 | -0.1032 | negligible | 0.7415 | 0.2411 | 0.0174 | INCONCLUSIVE | 0.000005 |
| mi_based | maximin_dispersion | -0.008495 | -0.1295 | negligible | 0.8766 | 0.1128 | 0.0106 | INCONCLUSIVE | 0.012976 |

**Summary**: EQUIVALENT 0/9, Cohesion_BETTER 0/9, Diversity_BETTER 0/9, INCONCLUSIVE 9/9
- Mean P(ROPE): 0.1850
- Wilcoxon p < 0.05: 5/9
- Wilcoxon p < 0.01: 3/9

#### Cohesion x Diversity Cross-Pairs (clean)

- N conditions: 346
- ROPE = +-0.002720

| Cohesion | Diversity | Mean Diff | Cohen's d | Interp | P(Cohesion better) | P(ROPE) | P(Diversity better) | Decision | Wilcoxon p |
|---------------|---------------|-----------|-----------|--------|---------------------|---------|---------------------|----------|------------|
| finch_like | anti_clustering | -0.000214 | -0.0961 | negligible | 0.0000 | 1.0000 | 0.0000 | EQUIVALENT | 0.295829 |
| finch_like | score_stratified | +0.000361 | +0.1812 | negligible | 0.0000 | 1.0000 | 0.0000 | EQUIVALENT | 0.000051 |
| finch_like | maximin_dispersion | +0.000044 | +0.0189 | negligible | 0.0000 | 1.0000 | 0.0000 | EQUIVALENT | 0.155364 |
| coarsening | anti_clustering | +0.000081 | +0.0270 | negligible | 0.0000 | 1.0000 | 0.0000 | EQUIVALENT | 0.183849 |
| coarsening | score_stratified | +0.000657 | +0.2379 | small | 0.0000 | 1.0000 | 0.0000 | EQUIVALENT | 0.000001 |
| coarsening | maximin_dispersion | +0.000339 | +0.1099 | negligible | 0.0000 | 1.0000 | 0.0000 | EQUIVALENT | 0.007344 |
| mi_based | anti_clustering | +0.000389 | +0.1245 | negligible | 0.0000 | 1.0000 | 0.0000 | EQUIVALENT | 0.069351 |
| mi_based | score_stratified | +0.000964 | +0.3275 | small | 0.0000 | 1.0000 | 0.0000 | EQUIVALENT | 0.000000 |
| mi_based | maximin_dispersion | +0.000646 | +0.2007 | small | 0.0000 | 1.0000 | 0.0000 | EQUIVALENT | 0.001402 |

**Summary**: EQUIVALENT 9/9, Cohesion_BETTER 0/9, Diversity_BETTER 0/9, INCONCLUSIVE 0/9
- Mean P(ROPE): 1.0000
- Wilcoxon p < 0.05: 5/9
- Wilcoxon p < 0.01: 5/9

### 2.4 Control(random) vs Cohesion Cross-Pairs (3 pairs)

#### Cohesion x Control Cross-Pairs (raw)

- N conditions: 360
- ROPE = +-0.002810

| Cohesion | Control | Mean Diff | Cohen's d | Interp | P(Cohesion better) | P(ROPE) | P(Control better) | Decision | Wilcoxon p |
|---------------|---------------|-----------|-----------|--------|---------------------|---------|---------------------|----------|------------|
| finch_like | random | -0.008982 | -0.1096 | negligible | 0.8433 | 0.1294 | 0.0273 | INCONCLUSIVE | 0.831670 |
| coarsening | random | -0.008651 | -0.1059 | negligible | 0.8307 | 0.1389 | 0.0304 | INCONCLUSIVE | 0.114971 |
| mi_based | random | -0.008314 | -0.1015 | negligible | 0.8157 | 0.1495 | 0.0348 | INCONCLUSIVE | 0.028607 |

**Summary**: EQUIVALENT 0/3, Cohesion_BETTER 0/3, Control_BETTER 0/3, INCONCLUSIVE 3/3
- Mean P(ROPE): 0.1393
- Wilcoxon p < 0.05: 1/3
- Wilcoxon p < 0.01: 0/3

#### Cohesion x Control Cross-Pairs (clean)

- N conditions: 346
- ROPE = +-0.002722

| Cohesion | Control | Mean Diff | Cohen's d | Interp | P(Cohesion better) | P(ROPE) | P(Control better) | Decision | Wilcoxon p |
|---------------|---------------|-----------|-----------|--------|---------------------|---------|---------------------|----------|------------|
| finch_like | random | -0.000069 | -0.0288 | negligible | 0.0000 | 1.0000 | 0.0000 | EQUIVALENT | 0.506973 |
| coarsening | random | +0.000226 | +0.0720 | negligible | 0.0000 | 1.0000 | 0.0000 | EQUIVALENT | 0.018680 |
| mi_based | random | +0.000534 | +0.1649 | negligible | 0.0000 | 1.0000 | 0.0000 | EQUIVALENT | 0.006454 |

**Summary**: EQUIVALENT 3/3, Cohesion_BETTER 0/3, Control_BETTER 0/3, INCONCLUSIVE 0/3
- Mean P(ROPE): 1.0000
- Wilcoxon p < 0.05: 2/3
- Wilcoxon p < 0.01: 1/3

### 2.5 Control(random) vs Diversity Cross-Pairs (3 pairs)

#### Control x Diversity Cross-Pairs (raw)

- N conditions: 360
- ROPE = +-0.002860

| Control | Diversity | Mean Diff | Cohen's d | Interp | P(Control better) | P(ROPE) | P(Diversity better) | Decision | Wilcoxon p |
|---------------|---------------|-----------|-----------|--------|---------------------|---------|---------------------|----------|------------|
| random | anti_clustering | +0.003247 | +0.0351 | negligible | 0.1885 | 0.2892 | 0.5223 | INCONCLUSIVE | 0.054650 |
| random | score_stratified | +0.003002 | +0.0436 | negligible | 0.1272 | 0.3618 | 0.5110 | INCONCLUSIVE | 0.000000 |
| random | maximin_dispersion | -0.000181 | -0.0023 | negligible | 0.3238 | 0.3743 | 0.3019 | INCONCLUSIVE | 0.439935 |

**Summary**: EQUIVALENT 0/3, Control_BETTER 0/3, Diversity_BETTER 0/3, INCONCLUSIVE 3/3
- Mean P(ROPE): 0.3418
- Wilcoxon p < 0.05: 1/3
- Wilcoxon p < 0.01: 1/3

#### Control x Diversity Cross-Pairs (clean)

- N conditions: 346
- ROPE = +-0.002719

| Control | Diversity | Mean Diff | Cohen's d | Interp | P(Control better) | P(ROPE) | P(Diversity better) | Decision | Wilcoxon p |
|---------------|---------------|-----------|-----------|--------|---------------------|---------|---------------------|----------|------------|
| random | anti_clustering | -0.000145 | -0.0871 | negligible | 0.0000 | 1.0000 | 0.0000 | EQUIVALENT | 0.030659 |
| random | score_stratified | +0.000430 | +0.2886 | small | 0.0000 | 1.0000 | 0.0000 | EQUIVALENT | 0.000000 |
| random | maximin_dispersion | +0.000112 | +0.0664 | negligible | 0.0000 | 1.0000 | 0.0000 | EQUIVALENT | 0.291386 |

**Summary**: EQUIVALENT 3/3, Control_BETTER 0/3, Diversity_BETTER 0/3, INCONCLUSIVE 0/3
- Mean P(ROPE): 1.0000
- Wilcoxon p < 0.05: 2/3
- Wilcoxon p < 0.01: 1/3

---

## 3. Friedman Test (Category within-category rank consistency)

### 3.1 Cohesion(Cohesion) Friedman

### Cohesion (3 methods) -- raw

- N conditions: 360, K methods: 3
- Friedman chi2 = 10.6167, p = 4.95e-03
- Kendall's W = 0.0147 (negligible agreement)

| method | Mean Rank |
|--------|-----------|
| finch_like | 1.872 |
| coarsening | 2.014 |
| mi_based | 2.114 |

**Wilcoxon-Holm Post-Hoc:**

| Pair | Raw p | Holm-corrected p | Sig |
|------|-------|------------------|-----|
| finch_like vs coarsening | 0.000689 | 0.001378 | ** |
| finch_like vs mi_based | 0.000167 | 0.000502 | *** |
| coarsening vs mi_based | 0.032455 | 0.032455 | * |

### Cohesion (3 methods) -- clean

- N conditions: 346, K methods: 3
- Friedman chi2 = 8.8092, p = 1.22e-02
- Kendall's W = 0.0127 (negligible agreement)

| method | Mean Rank |
|--------|-----------|
| finch_like | 1.884 |
| coarsening | 2.006 |
| mi_based | 2.110 |

**Wilcoxon-Holm Post-Hoc:**

| Pair | Raw p | Holm-corrected p | Sig |
|------|-------|------------------|-----|
| finch_like vs coarsening | 0.002241 | 0.004483 | ** |
| finch_like vs mi_based | 0.000513 | 0.001538 | ** |
| coarsening vs mi_based | 0.044317 | 0.044317 | * |

### 3.2 Diversity(Diversity) Friedman

### Diversity (3 methods) -- raw

- N conditions: 360, K methods: 3
- Friedman chi2 = 32.7167, p = 7.86e-08
- Kendall's W = 0.0454 (negligible agreement)

| method | Mean Rank |
|--------|-----------|
| score_stratified | 1.778 |
| maximin_dispersion | 2.019 |
| anti_clustering | 2.203 |

**Wilcoxon-Holm Post-Hoc:**

| Pair | Raw p | Holm-corrected p | Sig |
|------|-------|------------------|-----|
| anti_clustering vs score_stratified | 0.000000 | 0.000000 | *** |
| anti_clustering vs maximin_dispersion | 0.016731 | 0.016731 | * |
| score_stratified vs maximin_dispersion | 0.000064 | 0.000127 | *** |

### Diversity (3 methods) -- clean

- N conditions: 346, K methods: 3
- Friedman chi2 = 32.2948, p = 9.71e-08
- Kendall's W = 0.0467 (negligible agreement)

| method | Mean Rank |
|--------|-----------|
| score_stratified | 1.775 |
| maximin_dispersion | 2.020 |
| anti_clustering | 2.205 |

**Wilcoxon-Holm Post-Hoc:**

| Pair | Raw p | Holm-corrected p | Sig |
|------|-------|------------------|-----|
| anti_clustering vs score_stratified | 0.000000 | 0.000000 | *** |
| anti_clustering vs maximin_dispersion | 0.011023 | 0.011023 | * |
| score_stratified vs maximin_dispersion | 0.000050 | 0.000099 | *** |

### 3.3 Cohesion excluding All 5 methods Friedman

### Non-cohesion 5 methods (No strategy+Control+Diversity) -- raw

- N conditions: 360, K methods: 5
- Friedman chi2 = 139.6022, p = 3.43e-29
- Kendall's W = 0.0969 (negligible agreement)

| method | Mean Rank |
|--------|-----------|
| ordered | 2.272 |
| score_stratified | 2.742 |
| maximin_dispersion | 3.200 |
| random | 3.283 |
| anti_clustering | 3.503 |

**Wilcoxon-Holm Post-Hoc:**

| Pair | Raw p | Holm-corrected p | Sig |
|------|-------|------------------|-----|
| ordered vs random | 0.000000 | 0.000000 | *** |
| ordered vs anti_clustering | 0.000000 | 0.000000 | *** |
| ordered vs score_stratified | 0.000000 | 0.000000 | *** |
| ordered vs maximin_dispersion | 0.000000 | 0.000000 | *** |
| random vs anti_clustering | 0.054650 | 0.109300 | n.s. |
| random vs score_stratified | 0.000000 | 0.000001 | *** |
| random vs maximin_dispersion | 0.439935 | 0.439935 | n.s. |
| anti_clustering vs score_stratified | 0.000000 | 0.000000 | *** |
| anti_clustering vs maximin_dispersion | 0.016731 | 0.050192 | n.s. |
| score_stratified vs maximin_dispersion | 0.000064 | 0.000254 | *** |

### Non-cohesion 5 methods (No strategy+Control+Diversity) -- clean

- N conditions: 346, K methods: 5
- Friedman chi2 = 140.4231, p = 2.29e-29
- Kendall's W = 0.1015 (weak agreement)

| method | Mean Rank |
|--------|-----------|
| ordered | 2.254 |
| score_stratified | 2.740 |
| maximin_dispersion | 3.199 |
| random | 3.289 |
| anti_clustering | 3.517 |

**Wilcoxon-Holm Post-Hoc:**

| Pair | Raw p | Holm-corrected p | Sig |
|------|-------|------------------|-----|
| ordered vs random | 0.000000 | 0.000000 | *** |
| ordered vs anti_clustering | 0.000000 | 0.000000 | *** |
| ordered vs score_stratified | 0.000000 | 0.000000 | *** |
| ordered vs maximin_dispersion | 0.000000 | 0.000000 | *** |
| random vs anti_clustering | 0.030659 | 0.061319 | n.s. |
| random vs score_stratified | 0.000000 | 0.000000 | *** |
| random vs maximin_dispersion | 0.291386 | 0.291386 | n.s. |
| anti_clustering vs score_stratified | 0.000000 | 0.000000 | *** |
| anti_clustering vs maximin_dispersion | 0.011023 | 0.033068 | * |
| score_stratified vs maximin_dispersion | 0.000050 | 0.000199 | *** |

### 3.4 All 8 method Friedman

### All 8 methods -- raw

- N conditions: 360, K methods: 8
- Friedman chi2 = 213.8269, p = 1.35e-42
- Kendall's W = 0.0849 (negligible agreement)

| method | Mean Rank |
|--------|-----------|
| ordered | 2.975 |
| score_stratified | 3.942 |
| maximin_dispersion | 4.575 |
| random | 4.653 |
| finch_like | 4.800 |
| anti_clustering | 4.997 |
| coarsening | 5.019 |
| mi_based | 5.039 |

**Wilcoxon-Holm Post-Hoc:**

| Pair | Raw p | Holm-corrected p | Sig |
|------|-------|------------------|-----|
| finch_like vs coarsening | 0.000689 | 0.010336 | * |
| finch_like vs mi_based | 0.000167 | 0.002677 | ** |
| finch_like vs anti_clustering | 0.045490 | 0.409413 | n.s. |
| finch_like vs score_stratified | 0.002544 | 0.035610 | * |
| finch_like vs maximin_dispersion | 0.778412 | 1.000000 | n.s. |
| finch_like vs ordered | 0.000000 | 0.000000 | *** |
| finch_like vs random | 0.831670 | 1.000000 | n.s. |
| coarsening vs mi_based | 0.032455 | 0.324549 | n.s. |
| coarsening vs anti_clustering | 0.482076 | 1.000000 | n.s. |
| coarsening vs score_stratified | 0.000095 | 0.001618 | ** |
| coarsening vs maximin_dispersion | 0.050339 | 0.409413 | n.s. |
| coarsening vs ordered | 0.000000 | 0.000000 | *** |
| coarsening vs random | 0.114971 | 0.689827 | n.s. |
| mi_based vs anti_clustering | 0.201287 | 1.000000 | n.s. |
| mi_based vs score_stratified | 0.000005 | 0.000087 | *** |
| mi_based vs maximin_dispersion | 0.012976 | 0.168688 | n.s. |
| mi_based vs ordered | 0.000000 | 0.000000 | *** |
| mi_based vs random | 0.028607 | 0.314672 | n.s. |
| anti_clustering vs score_stratified | 0.000000 | 0.000000 | *** |
| anti_clustering vs maximin_dispersion | 0.016731 | 0.200770 | n.s. |
| anti_clustering vs ordered | 0.000000 | 0.000000 | *** |
| anti_clustering vs random | 0.054650 | 0.409413 | n.s. |
| score_stratified vs maximin_dispersion | 0.000064 | 0.001144 | ** |
| score_stratified vs ordered | 0.000000 | 0.000000 | *** |
| score_stratified vs random | 0.000000 | 0.000002 | *** |
| maximin_dispersion vs ordered | 0.000000 | 0.000000 | *** |
| maximin_dispersion vs random | 0.439935 | 1.000000 | n.s. |
| ordered vs random | 0.000000 | 0.000000 | *** |

### All 8 methods -- clean

- N conditions: 346, K methods: 8
- Friedman chi2 = 235.5376, p = 3.30e-47
- Kendall's W = 0.0972 (negligible agreement)

| method | Mean Rank |
|--------|-----------|
| ordered | 2.890 |
| score_stratified | 3.887 |
| maximin_dispersion | 4.526 |
| random | 4.618 |
| finch_like | 4.922 |
| anti_clustering | 4.968 |
| coarsening | 5.092 |
| mi_based | 5.095 |

**Wilcoxon-Holm Post-Hoc:**

| Pair | Raw p | Holm-corrected p | Sig |
|------|-------|------------------|-----|
| finch_like vs coarsening | 0.002241 | 0.029138 | * |
| finch_like vs mi_based | 0.000513 | 0.007689 | ** |
| finch_like vs anti_clustering | 0.295829 | 0.874157 | n.s. |
| finch_like vs score_stratified | 0.000051 | 0.000846 | *** |
| finch_like vs maximin_dispersion | 0.155364 | 0.776822 | n.s. |
| finch_like vs ordered | 0.000000 | 0.000000 | *** |
| finch_like vs random | 0.506973 | 0.874157 | n.s. |
| coarsening vs mi_based | 0.044317 | 0.310217 | n.s. |
| coarsening vs anti_clustering | 0.183849 | 0.776822 | n.s. |
| coarsening vs score_stratified | 0.000001 | 0.000022 | *** |
| coarsening vs maximin_dispersion | 0.007344 | 0.080787 | n.s. |
| coarsening vs ordered | 0.000000 | 0.000000 | *** |
| coarsening vs random | 0.018680 | 0.168119 | n.s. |
| mi_based vs anti_clustering | 0.069351 | 0.416108 | n.s. |
| mi_based vs score_stratified | 0.000000 | 0.000001 | *** |
| mi_based vs maximin_dispersion | 0.001402 | 0.019626 | * |
| mi_based vs ordered | 0.000000 | 0.000000 | *** |
| mi_based vs random | 0.006454 | 0.077446 | n.s. |
| anti_clustering vs score_stratified | 0.000000 | 0.000000 | *** |
| anti_clustering vs maximin_dispersion | 0.011023 | 0.110225 | n.s. |
| anti_clustering vs ordered | 0.000000 | 0.000000 | *** |
| anti_clustering vs random | 0.030659 | 0.245276 | n.s. |
| score_stratified vs maximin_dispersion | 0.000050 | 0.000846 | *** |
| score_stratified vs ordered | 0.000000 | 0.000000 | *** |
| score_stratified vs random | 0.000000 | 0.000001 | *** |
| maximin_dispersion vs ordered | 0.000000 | 0.000000 | *** |
| maximin_dispersion vs random | 0.291386 | 0.874157 | n.s. |
| ordered vs random | 0.000000 | 0.000000 | *** |

---

## 4. ROPE sensitivity analysis (Sensitivity)

For all 28 pairs, the ROPE width is changed to +-0.5%, +-1%, and +-2% to check result stability under the clean criterion.

### ROPE = +-0.5% (+-0.001359)

| Pair | Type | P(ROPE) | Decision |
|------|------|---------|----------|
| finch_like vs coarsening | within-Cohesion | 1.0000 | EQUIVALENT |
| finch_like vs mi_based | within-Cohesion | 1.0000 | EQUIVALENT |
| finch_like vs anti_clustering | Cohesion × Diversity | 1.0000 | EQUIVALENT |
| finch_like vs score_stratified | Cohesion × Diversity | 1.0000 | EQUIVALENT |
| finch_like vs maximin_dispersion | Cohesion × Diversity | 1.0000 | EQUIVALENT |
| finch_like vs ordered | Cohesion × No strategy | 0.7771 | INCONCLUSIVE |
| finch_like vs random | Cohesion × Control | 1.0000 | EQUIVALENT |
| coarsening vs mi_based | within-Cohesion | 1.0000 | EQUIVALENT |
| coarsening vs anti_clustering | Cohesion × Diversity | 1.0000 | EQUIVALENT |
| coarsening vs score_stratified | Cohesion × Diversity | 0.9995 | EQUIVALENT |
| coarsening vs maximin_dispersion | Cohesion × Diversity | 1.0000 | EQUIVALENT |
| coarsening vs ordered | Cohesion × No strategy | 0.1811 | INCONCLUSIVE |
| coarsening vs random | Cohesion × Control | 1.0000 | EQUIVALENT |
| mi_based vs anti_clustering | Cohesion × Diversity | 1.0000 | EQUIVALENT |
| mi_based vs score_stratified | Cohesion × Diversity | 0.9607 | EQUIVALENT |
| mi_based vs maximin_dispersion | Cohesion × Diversity | 0.9981 | EQUIVALENT |
| mi_based vs ordered | Cohesion × No strategy | 0.0101 | RIGHT_BETTER |
| mi_based vs random | Cohesion × Control | 0.9996 | EQUIVALENT |
| anti_clustering vs score_stratified | within-Diversity | 1.0000 | EQUIVALENT |
| anti_clustering vs maximin_dispersion | within-Diversity | 1.0000 | EQUIVALENT |
| anti_clustering vs ordered | Diversity × No strategy | 0.2871 | INCONCLUSIVE |
| anti_clustering vs random | Diversity × Control | 1.0000 | EQUIVALENT |
| score_stratified vs maximin_dispersion | within-Diversity | 1.0000 | EQUIVALENT |
| score_stratified vs ordered | Diversity × No strategy | 0.9985 | EQUIVALENT |
| score_stratified vs random | Diversity × Control | 1.0000 | EQUIVALENT |
| maximin_dispersion vs ordered | Diversity × No strategy | 0.8230 | INCONCLUSIVE |
| maximin_dispersion vs random | Diversity × Control | 1.0000 | EQUIVALENT |
| ordered vs random | No strategy × Control | 0.6085 | INCONCLUSIVE |

**EQUIVALENT: 22/28, DIFFERENT: 1/28, INCONCLUSIVE: 5/28**

### ROPE = +-1.0% (+-0.002719)

| Pair | Type | P(ROPE) | Decision |
|------|------|---------|----------|
| finch_like vs coarsening | within-Cohesion | 1.0000 | EQUIVALENT |
| finch_like vs mi_based | within-Cohesion | 1.0000 | EQUIVALENT |
| finch_like vs anti_clustering | Cohesion × Diversity | 1.0000 | EQUIVALENT |
| finch_like vs score_stratified | Cohesion × Diversity | 1.0000 | EQUIVALENT |
| finch_like vs maximin_dispersion | Cohesion × Diversity | 1.0000 | EQUIVALENT |
| finch_like vs ordered | Cohesion × No strategy | 1.0000 | EQUIVALENT |
| finch_like vs random | Cohesion × Control | 1.0000 | EQUIVALENT |
| coarsening vs mi_based | within-Cohesion | 1.0000 | EQUIVALENT |
| coarsening vs anti_clustering | Cohesion × Diversity | 1.0000 | EQUIVALENT |
| coarsening vs score_stratified | Cohesion × Diversity | 1.0000 | EQUIVALENT |
| coarsening vs maximin_dispersion | Cohesion × Diversity | 1.0000 | EQUIVALENT |
| coarsening vs ordered | Cohesion × No strategy | 1.0000 | EQUIVALENT |
| coarsening vs random | Cohesion × Control | 1.0000 | EQUIVALENT |
| mi_based vs anti_clustering | Cohesion × Diversity | 1.0000 | EQUIVALENT |
| mi_based vs score_stratified | Cohesion × Diversity | 1.0000 | EQUIVALENT |
| mi_based vs maximin_dispersion | Cohesion × Diversity | 1.0000 | EQUIVALENT |
| mi_based vs ordered | Cohesion × No strategy | 1.0000 | EQUIVALENT |
| mi_based vs random | Cohesion × Control | 1.0000 | EQUIVALENT |
| anti_clustering vs score_stratified | within-Diversity | 1.0000 | EQUIVALENT |
| anti_clustering vs maximin_dispersion | within-Diversity | 1.0000 | EQUIVALENT |
| anti_clustering vs ordered | Diversity × No strategy | 1.0000 | EQUIVALENT |
| anti_clustering vs random | Diversity × Control | 1.0000 | EQUIVALENT |
| score_stratified vs maximin_dispersion | within-Diversity | 1.0000 | EQUIVALENT |
| score_stratified vs ordered | Diversity × No strategy | 1.0000 | EQUIVALENT |
| score_stratified vs random | Diversity × Control | 1.0000 | EQUIVALENT |
| maximin_dispersion vs ordered | Diversity × No strategy | 1.0000 | EQUIVALENT |
| maximin_dispersion vs random | Diversity × Control | 1.0000 | EQUIVALENT |
| ordered vs random | No strategy × Control | 1.0000 | EQUIVALENT |

**EQUIVALENT: 28/28, DIFFERENT: 0/28, INCONCLUSIVE: 0/28**

### ROPE = +-2.0% (+-0.005437)

| Pair | Type | P(ROPE) | Decision |
|------|------|---------|----------|
| finch_like vs coarsening | within-Cohesion | 1.0000 | EQUIVALENT |
| finch_like vs mi_based | within-Cohesion | 1.0000 | EQUIVALENT |
| finch_like vs anti_clustering | Cohesion × Diversity | 1.0000 | EQUIVALENT |
| finch_like vs score_stratified | Cohesion × Diversity | 1.0000 | EQUIVALENT |
| finch_like vs maximin_dispersion | Cohesion × Diversity | 1.0000 | EQUIVALENT |
| finch_like vs ordered | Cohesion × No strategy | 1.0000 | EQUIVALENT |
| finch_like vs random | Cohesion × Control | 1.0000 | EQUIVALENT |
| coarsening vs mi_based | within-Cohesion | 1.0000 | EQUIVALENT |
| coarsening vs anti_clustering | Cohesion × Diversity | 1.0000 | EQUIVALENT |
| coarsening vs score_stratified | Cohesion × Diversity | 1.0000 | EQUIVALENT |
| coarsening vs maximin_dispersion | Cohesion × Diversity | 1.0000 | EQUIVALENT |
| coarsening vs ordered | Cohesion × No strategy | 1.0000 | EQUIVALENT |
| coarsening vs random | Cohesion × Control | 1.0000 | EQUIVALENT |
| mi_based vs anti_clustering | Cohesion × Diversity | 1.0000 | EQUIVALENT |
| mi_based vs score_stratified | Cohesion × Diversity | 1.0000 | EQUIVALENT |
| mi_based vs maximin_dispersion | Cohesion × Diversity | 1.0000 | EQUIVALENT |
| mi_based vs ordered | Cohesion × No strategy | 1.0000 | EQUIVALENT |
| mi_based vs random | Cohesion × Control | 1.0000 | EQUIVALENT |
| anti_clustering vs score_stratified | within-Diversity | 1.0000 | EQUIVALENT |
| anti_clustering vs maximin_dispersion | within-Diversity | 1.0000 | EQUIVALENT |
| anti_clustering vs ordered | Diversity × No strategy | 1.0000 | EQUIVALENT |
| anti_clustering vs random | Diversity × Control | 1.0000 | EQUIVALENT |
| score_stratified vs maximin_dispersion | within-Diversity | 1.0000 | EQUIVALENT |
| score_stratified vs ordered | Diversity × No strategy | 1.0000 | EQUIVALENT |
| score_stratified vs random | Diversity × Control | 1.0000 | EQUIVALENT |
| maximin_dispersion vs ordered | Diversity × No strategy | 1.0000 | EQUIVALENT |
| maximin_dispersion vs random | Diversity × Control | 1.0000 | EQUIVALENT |
| ordered vs random | No strategy × Control | 1.0000 | EQUIVALENT |

**EQUIVALENT: 28/28, DIFFERENT: 0/28, INCONCLUSIVE: 0/28**

---

## 5. Comprehensive Summary (Comprehensive Summary)

### 5.1 All 28 Pairs Grouping invariance (Clean criterion)

| Classification | Pairs | EQUIVALENT | INCONCLUSIVE | DIFFERENT |
|------|-------|------------|--------------|-----------|
| All 28 pairs | 28 | 28 | 0 | 0 |
| within-Diversity | 3 | 3 | 0 | 0 |
| within-Cohesion | 3 | 3 | 0 | 0 |
| No strategy × Control | 1 | 1 | 0 | 0 |
| Diversity × Control | 3 | 3 | 0 | 0 |
| Diversity × No strategy | 3 | 3 | 0 | 0 |
| Cohesion × Control | 3 | 3 | 0 | 0 |
| Cohesion × No strategy | 3 | 3 | 0 | 0 |
| Cohesion × Diversity | 9 | 9 | 0 | 0 |

### 5.2 Within-Category (Clean criterion)

| Category | Pairs | EQUIVALENT | INCONCLUSIVE | DIFFERENT |
|----------|-------|------------|--------------|-----------|
| Diversity (3 pairs) | 3 | 3 | 0 | 0 |
| Cohesion (3 pairs) | 3 | 3 | 0 | 0 |

### 5.3 Between-Category (Clean criterion)

| Comparison | Pairs | EQUIVALENT | A_BETTER | B_BETTER | INCONCLUSIVE |
|------|-------|------------|----------|----------|--------------|
| Diversity x Cohesion | 9 | 9 | 0 (Cohesion) | 0 (Diversity) | 0 |
| Control x Cohesion | 3 | 3 | 0 (Cohesion) | 0 (Control) | 0 |
| Control x Diversity | 3 | 3 | 0 (Control) | 0 (Diversity) | 0 |

### 5.4 Key Conclusion

1. **Grouping invariance (practical equivalence across all eight methods)**: 28/28 pairs among all C(8,2)=28 pairs are ROPE EQUIVALENT under the clean criterion.
2. **Diversity x Cohesion**: 9/9 pairs are EQUIVALENT and 0/9 are DIFFERENT.
3. **Within Cohesion**: 3/3 pairs are EQUIVALENT.
4. **Within Diversity**: 3/3 pairs are EQUIVALENT.

### 5.5 MSE Spread statistics

**Raw:**

| Group | Mean MSE Range | Spread (%) |
|-------|---------------|------------|
| Diversity (3) | [0.284272, 0.287700] | 1.20% |
| Cohesion (3) | [0.278537, 0.279205] | 0.24% |
| Cohesion excluding (5) | [0.284185, 0.287700] | 1.23% |
| All (8) | [0.278537, 0.287700] | 3.24% |

method-level means (Raw):

| method | Category | Mean MSE |
|--------|----------|----------|
| finch_like | Cohesion | 0.278537 |
| coarsening | Cohesion | 0.278867 |
| mi_based | Cohesion | 0.279205 |
| ordered | No strategy | 0.284185 |
| anti_clustering | Diversity | 0.284272 |
| score_stratified | Diversity | 0.284517 |
| random | Control | 0.287519 |
| maximin_dispersion | Diversity | 0.287700 |

**Clean:**

| Group | Mean MSE Range | Spread (%) |
|-------|---------------|------------|
| Diversity (3) | [0.271561, 0.272136] | 0.21% |
| Cohesion (3) | [0.271922, 0.272525] | 0.22% |
| Cohesion excluding (5) | [0.270683, 0.272136] | 0.53% |
| All (8) | [0.270683, 0.272525] | 0.68% |

method-level means (Clean):

| method | Category | Mean MSE |
|--------|----------|----------|
| ordered | No strategy | 0.270683 |
| score_stratified | Diversity | 0.271561 |
| maximin_dispersion | Diversity | 0.271878 |
| finch_like | Cohesion | 0.271922 |
| random | Control | 0.271991 |
| anti_clustering | Diversity | 0.272136 |
| coarsening | Cohesion | 0.272217 |
| mi_based | Cohesion | 0.272525 |

### 5.6 All Mean Rank

**Raw (N=360 conditions):**

| Rank | method | Category | Mean Rank |
|------|--------|----------|-----------|
| 1 | ordered | No strategy | 2.975 |
| 2 | score_stratified | Diversity | 3.942 |
| 3 | maximin_dispersion | Diversity | 4.575 |
| 4 | random | Control | 4.653 |
| 5 | finch_like | Cohesion | 4.800 |
| 6 | anti_clustering | Diversity | 4.997 |
| 7 | coarsening | Cohesion | 5.019 |
| 8 | mi_based | Cohesion | 5.039 |

**Clean (N=346 conditions):**

| Rank | method | Category | Mean Rank |
|------|--------|----------|-----------|
| 1 | ordered | No strategy | 2.890 |
| 2 | score_stratified | Diversity | 3.887 |
| 3 | maximin_dispersion | Diversity | 4.526 |
| 4 | random | Control | 4.618 |
| 5 | finch_like | Cohesion | 4.922 |
| 6 | anti_clustering | Diversity | 4.968 |
| 7 | coarsening | Cohesion | 5.092 |
| 8 | mi_based | Cohesion | 5.095 |

---
*End of analysis*
