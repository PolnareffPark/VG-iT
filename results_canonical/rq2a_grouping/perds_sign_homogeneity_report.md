
# Dataset-Level Sign Test and Win-Rate Homogeneity Chi-Square Test Report

**Data**: `results_canonical/rq2a_grouping/grouping_invariance_results.csv`
**Filtering rule**: remove blocks with maximum within-block MSE > 0.6 → 346 blocks, 2768 runs
**Block definition**: (dataset, pred_len, seed) — seed mean none
**Number of methods**: 8, **Number of pairs**: 28


## 1. Ordered vs Seven Competing Methods: Dataset-Level Win Rates and Sign Tests


### 1-1. Dataset-Level Ordered Win-Rate Summary

Two-sided sign test (H₀: p=0.5). `*` p<0.05, `**` p<0.01, `***` p<0.001

| Competing method | Dataset | Ordered wins | Competitor wins | Ordered win rate | p-value | |
|---|---|---:|---:|---:|---:|---|
| anti_clustering | electricity | 105 | 15 | 0.8750 | 0.000000 | *** |
| anti_clustering | solar_AL | 95 | 25 | 0.7917 | 0.000000 | *** |
| anti_clustering | traffic | 69 | 37 | 0.6509 | 0.002440 | ** |
| coarsening | electricity | 90 | 30 | 0.7500 | 0.000000 | *** |
| coarsening | solar_AL | 105 | 15 | 0.8750 | 0.000000 | *** |
| coarsening | traffic | 77 | 29 | 0.7264 | 0.000003 | *** |
| finch_like | electricity | 105 | 15 | 0.8750 | 0.000000 | *** |
| finch_like | solar_AL | 102 | 18 | 0.8500 | 0.000000 | *** |
| finch_like | traffic | 72 | 34 | 0.6792 | 0.000285 | *** |
| maximin_dispersion | electricity | 103 | 17 | 0.8583 | 0.000000 | *** |
| maximin_dispersion | solar_AL | 93 | 27 | 0.7750 | 0.000000 | *** |
| maximin_dispersion | traffic | 46 | 60 | 0.4340 | 0.206498 |  |
| mi_based | electricity | 104 | 16 | 0.8667 | 0.000000 | *** |
| mi_based | solar_AL | 106 | 14 | 0.8833 | 0.000000 | *** |
| mi_based | traffic | 57 | 49 | 0.5377 | 0.496754 |  |
| random | electricity | 103 | 17 | 0.8583 | 0.000000 | *** |
| random | solar_AL | 94 | 26 | 0.7833 | 0.000000 | *** |
| random | traffic | 32 | 74 | 0.3019 | 0.000055 | *** |
| score_stratified | electricity | 98 | 22 | 0.8167 | 0.000000 | *** |
| score_stratified | solar_AL | 91 | 29 | 0.7583 | 0.000000 | *** |
| score_stratified | traffic | 21 | 85 | 0.1981 | 0.000000 | *** |


### 1-2. Dataset-Level Interpretation


#### electricity

- Mean ordered win rate: 0.8429
- p < 0.05 competing Number of methods: 7/7


#### solar_AL

- Mean ordered win rate: 0.8167
- p < 0.05 competing Number of methods: 7/7


#### traffic

- Mean ordered win rate: 0.5040
- p < 0.05 competing Number of methods: 5/7


## 2. Win-rate homogeneity chi-square test: 3 Dataset (DS-level)

For each pair, the test checks whether win rates are equal across the three datasets (H0: equal win rate across the three datasets).
BH FDR correction applied. `*` BH p < 0.05.

| Pair | chi² | dof | p_raw | p_BH | Significant |
|---|---:|---:|---:|---:|:---:|
| anti_clustering vs coarsening | 76.4053 | 2 | 0.000000 | 0.000000 | ✓ |
| anti_clustering vs finch_like | 41.3767 | 2 | 0.000000 | 0.000000 | ✓ |
| anti_clustering vs mi_based | 71.0804 | 2 | 0.000000 | 0.000000 | ✓ |
| anti_clustering vs score_stratified | 30.1324 | 2 | 0.000000 | 0.000000 | ✓ |
| coarsening vs score_stratified | 112.6389 | 2 | 0.000000 | 0.000000 | ✓ |
| finch_like vs maximin_dispersion | 56.2620 | 2 | 0.000000 | 0.000000 | ✓ |
| coarsening vs random | 119.6152 | 2 | 0.000000 | 0.000000 | ✓ |
| coarsening vs maximin_dispersion | 101.5951 | 2 | 0.000000 | 0.000000 | ✓ |
| mi_based vs score_stratified | 59.7094 | 2 | 0.000000 | 0.000000 | ✓ |
| ordered vs random | 89.9823 | 2 | 0.000000 | 0.000000 | ✓ |
| mi_based vs ordered | 47.5620 | 2 | 0.000000 | 0.000000 | ✓ |
| mi_based vs random | 85.1669 | 2 | 0.000000 | 0.000000 | ✓ |
| maximin_dispersion vs ordered | 53.2057 | 2 | 0.000000 | 0.000000 | ✓ |
| maximin_dispersion vs mi_based | 73.9729 | 2 | 0.000000 | 0.000000 | ✓ |
| finch_like vs score_stratified | 76.4021 | 2 | 0.000000 | 0.000000 | ✓ |
| finch_like vs random | 67.1577 | 2 | 0.000000 | 0.000000 | ✓ |
| ordered vs score_stratified | 107.9179 | 2 | 0.000000 | 0.000000 | ✓ |
| coarsening vs finch_like | 25.9980 | 2 | 0.000002 | 0.000003 | ✓ |
| anti_clustering vs random | 19.4262 | 2 | 0.000060 | 0.000084 | ✓ |
| maximin_dispersion vs score_stratified | 19.4530 | 2 | 0.000060 | 0.000084 | ✓ |
| anti_clustering vs ordered | 16.5451 | 2 | 0.000255 | 0.000340 | ✓ |
| finch_like vs ordered | 16.0535 | 2 | 0.000327 | 0.000416 | ✓ |
| random vs score_stratified | 13.2702 | 2 | 0.001313 | 0.001598 | ✓ |
| maximin_dispersion vs random | 12.7651 | 2 | 0.001691 | 0.001973 | ✓ |
| finch_like vs mi_based | 9.7947 | 2 | 0.007466 | 0.008362 | ✓ |
| coarsening vs ordered | 8.8168 | 2 | 0.012175 | 0.013112 | ✓ |
| coarsening vs mi_based | 3.6818 | 2 | 0.158677 | 0.164554 |  |
| anti_clustering vs maximin_dispersion | 2.5905 | 2 | 0.273824 | 0.273824 |  |

**DS-dependent directional pairs (BH p<0.05): 26/28**

Significant pair list:
- anti_clustering vs coarsening: chi²=76.4053, p_BH=0.000000
- anti_clustering vs finch_like: chi²=41.3767, p_BH=0.000000
- anti_clustering vs mi_based: chi²=71.0804, p_BH=0.000000
- anti_clustering vs ordered: chi²=16.5451, p_BH=0.000340
- anti_clustering vs random: chi²=19.4262, p_BH=0.000084
- anti_clustering vs score_stratified: chi²=30.1324, p_BH=0.000000
- coarsening vs finch_like: chi²=25.9980, p_BH=0.000003
- coarsening vs maximin_dispersion: chi²=101.5951, p_BH=0.000000
- coarsening vs ordered: chi²=8.8168, p_BH=0.013112
- coarsening vs random: chi²=119.6152, p_BH=0.000000
- coarsening vs score_stratified: chi²=112.6389, p_BH=0.000000
- finch_like vs maximin_dispersion: chi²=56.2620, p_BH=0.000000
- finch_like vs mi_based: chi²=9.7947, p_BH=0.008362
- finch_like vs ordered: chi²=16.0535, p_BH=0.000416
- finch_like vs random: chi²=67.1577, p_BH=0.000000
- finch_like vs score_stratified: chi²=76.4021, p_BH=0.000000
- maximin_dispersion vs mi_based: chi²=73.9729, p_BH=0.000000
- maximin_dispersion vs ordered: chi²=53.2057, p_BH=0.000000
- maximin_dispersion vs random: chi²=12.7651, p_BH=0.001973
- maximin_dispersion vs score_stratified: chi²=19.4530, p_BH=0.000084
- mi_based vs ordered: chi²=47.5620, p_BH=0.000000
- mi_based vs random: chi²=85.1669, p_BH=0.000000
- mi_based vs score_stratified: chi²=59.7094, p_BH=0.000000
- ordered vs random: chi²=89.9823, p_BH=0.000000
- ordered vs score_stratified: chi²=107.9179, p_BH=0.000000
- random vs score_stratified: chi²=13.2702, p_BH=0.001598


## 3. Win-rate homogeneity chi-square test: 12 (DS, PL) Condition

each Pairfor 12 (Dataset×prediction length) Condition of  win rateare equal Test.
BH FDR correction applied.

| Pair | chi² | dof | p_raw | p_BH | Significant |
|---|---:|---:|---:|---:|:---:|
| anti_clustering vs coarsening | 95.8329 | 11 | 0.000000 | 0.000000 | ✓ |
| anti_clustering vs finch_like | 114.4188 | 11 | 0.000000 | 0.000000 | ✓ |
| anti_clustering vs mi_based | 100.1449 | 11 | 0.000000 | 0.000000 | ✓ |
| anti_clustering vs ordered | 54.2490 | 11 | 0.000000 | 0.000000 | ✓ |
| coarsening vs score_stratified | 145.9579 | 11 | 0.000000 | 0.000000 | ✓ |
| coarsening vs random | 147.1122 | 11 | 0.000000 | 0.000000 | ✓ |
| coarsening vs ordered | 57.8155 | 11 | 0.000000 | 0.000000 | ✓ |
| coarsening vs maximin_dispersion | 128.8242 | 11 | 0.000000 | 0.000000 | ✓ |
| finch_like vs ordered | 82.7515 | 11 | 0.000000 | 0.000000 | ✓ |
| finch_like vs maximin_dispersion | 122.9229 | 11 | 0.000000 | 0.000000 | ✓ |
| mi_based vs ordered | 68.8700 | 11 | 0.000000 | 0.000000 | ✓ |
| mi_based vs random | 121.1182 | 11 | 0.000000 | 0.000000 | ✓ |
| maximin_dispersion vs ordered | 75.4358 | 11 | 0.000000 | 0.000000 | ✓ |
| maximin_dispersion vs mi_based | 103.8596 | 11 | 0.000000 | 0.000000 | ✓ |
| finch_like vs score_stratified | 178.7094 | 11 | 0.000000 | 0.000000 | ✓ |
| finch_like vs random | 149.5752 | 11 | 0.000000 | 0.000000 | ✓ |
| ordered vs score_stratified | 148.5926 | 11 | 0.000000 | 0.000000 | ✓ |
| mi_based vs score_stratified | 106.1066 | 11 | 0.000000 | 0.000000 | ✓ |
| ordered vs random | 109.0845 | 11 | 0.000000 | 0.000000 | ✓ |
| anti_clustering vs score_stratified | 43.5448 | 11 | 0.000009 | 0.000013 | ✓ |
| random vs score_stratified | 40.9313 | 11 | 0.000025 | 0.000033 | ✓ |
| coarsening vs finch_like | 40.7385 | 11 | 0.000027 | 0.000034 | ✓ |
| anti_clustering vs random | 34.9545 | 11 | 0.000252 | 0.000307 | ✓ |
| maximin_dispersion vs score_stratified | 28.7512 | 11 | 0.002481 | 0.002895 | ✓ |
| finch_like vs mi_based | 21.5220 | 11 | 0.028349 | 0.031751 | ✓ |
| maximin_dispersion vs random | 15.8312 | 11 | 0.147516 | 0.158863 |  |
| anti_clustering vs maximin_dispersion | 8.9861 | 11 | 0.623174 | 0.646255 |  |
| coarsening vs mi_based | 5.6555 | 11 | 0.895333 | 0.895333 |  |

**(DS,PL)-dependent directional pairs (BH p<0.05): 25/28**

Significant pair list:
- anti_clustering vs coarsening: chi²=95.8329, p_BH=0.000000
- anti_clustering vs finch_like: chi²=114.4188, p_BH=0.000000
- anti_clustering vs mi_based: chi²=100.1449, p_BH=0.000000
- anti_clustering vs ordered: chi²=54.2490, p_BH=0.000000
- anti_clustering vs random: chi²=34.9545, p_BH=0.000307
- anti_clustering vs score_stratified: chi²=43.5448, p_BH=0.000013
- coarsening vs finch_like: chi²=40.7385, p_BH=0.000034
- coarsening vs maximin_dispersion: chi²=128.8242, p_BH=0.000000
- coarsening vs ordered: chi²=57.8155, p_BH=0.000000
- coarsening vs random: chi²=147.1122, p_BH=0.000000
- coarsening vs score_stratified: chi²=145.9579, p_BH=0.000000
- finch_like vs maximin_dispersion: chi²=122.9229, p_BH=0.000000
- finch_like vs mi_based: chi²=21.5220, p_BH=0.031751
- finch_like vs ordered: chi²=82.7515, p_BH=0.000000
- finch_like vs random: chi²=149.5752, p_BH=0.000000
- finch_like vs score_stratified: chi²=178.7094, p_BH=0.000000
- maximin_dispersion vs mi_based: chi²=103.8596, p_BH=0.000000
- maximin_dispersion vs ordered: chi²=75.4358, p_BH=0.000000
- maximin_dispersion vs score_stratified: chi²=28.7512, p_BH=0.002895
- mi_based vs ordered: chi²=68.8700, p_BH=0.000000
- mi_based vs random: chi²=121.1182, p_BH=0.000000
- mi_based vs score_stratified: chi²=106.1066, p_BH=0.000000
- ordered vs random: chi²=109.0845, p_BH=0.000000
- ordered vs score_stratified: chi²=148.5926, p_BH=0.000000
- random vs score_stratified: chi²=40.9313, p_BH=0.000033


## 4. Comprehensive Summary

### Key results

- **Data size**: 346 blocks × 8 methods = 2768 runs (no seed averaging)
- **Test 1 (sign test)**: ordered dataset-level win rate, used to check consistency within each dataset.
- **Test 2a (DS homogeneity)**: DS-dependent directional pairs = **26/28** (BH p<0.05)
- **Test 2b (DS×PL homogeneity)**: (DS,PL)-dependent directional pairs = **25/28** (BH p<0.05)

**Note**: condition dependence is detected in 26 pairs at the dataset level and 25 pairs at the dataset-by-prediction-length level.
For those pairs, the win/loss direction varies by dataset, indicating that the grouping effect is condition-dependent.

### ordered method Summary

- **electricity**: mean win rate 0.8429, Significant sign p<0.05 Pair: 7/7
- **solar_AL**: mean win rate 0.8167, Significant sign p<0.05 Pair: 7/7
- **traffic**: mean win rate 0.5040, Significant sign p<0.05 Pair: 5/7

---
*Generated from the archived grouping-invariance results.*
