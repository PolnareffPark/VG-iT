# Epoch-1 Train Loss Diagnostic Analysis

> Data: 2880 public provenance records from the grouping-invariance runs.
> Parse failures: 0 records

## 1. Basic Statistics

- Total runs: 2880
- Convergence anomaly (MSE > 0.6): 25
- Normal: 2855

## 2. Epoch-1 Train Loss Distribution

| Group | N | Mean | Std | Min | Max | Median |
|------|---|------|-----|-----|-----|--------|
| Normal | 2855 | 0.312546 | 0.085965 | 0.195909 | 0.600424 | 0.287413 |
| Convergence anomaly | 25 | 0.632475 | 0.052287 | 0.592045 | 0.843652 | 0.618657 |

**Normal maximum value**: 0.600424
**Minimum anomaly value**: 0.592045
**Gap**: -0.008379
**Overlap exists**: some normal runs have higher epoch-1 loss than anomaly runs

## 3. Epoch-1 Loss Threshold Sensitivity

| Threshold | True Positive | False Positive | Recall | Precision | F1 |
|-----------|--------------|---------------|--------|-----------|-----|
| 0.30 | 25/25 | 946/2855 | 1.000 | 0.026 | 0.050 |
| 0.35 | 25/25 | 935/2855 | 1.000 | 0.026 | 0.051 |
| 0.40 | 25/25 | 637/2855 | 1.000 | 0.038 | 0.073 |
| 0.45 | 25/25 | 193/2855 | 1.000 | 0.115 | 0.206 |
| 0.50 | 25/25 | 41/2855 | 1.000 | 0.379 | 0.549 |
| 0.55 | 25/25 | 12/2855 | 1.000 | 0.676 | 0.806 |
| 0.60 | 23/25 | 1/2855 | 0.920 | 0.958 | 0.939 |
| 0.65 | 3/25 | 0/2855 | 0.120 | 1.000 | 0.214 |
| 0.70 | 3/25 | 0/2855 | 0.120 | 1.000 | 0.214 |

## 4. Convergence anomaly details (ordered by epoch-1 loss)

| method | Dataset | PL | Seed | Epoch-1 Loss | Final MSE |
|--------|---------|-----|------|-------------|-----------|
| random | traffic | 720 | 2049 | 0.843652 | 1.449899 |
| ordered | traffic | 720 | 2037 | 0.712106 | 1.420713 |
| random | traffic | 720 | 2036 | 0.708487 | 1.448803 |
| score_stratified | traffic | 720 | 2036 | 0.632587 | 0.987920 |
| ordered | traffic | 720 | 2035 | 0.632572 | 0.992609 |
| maximin_dispersion | traffic | 720 | 2034 | 0.630514 | 0.985944 |
| anti_clustering | traffic | 720 | 2037 | 0.627646 | 1.026082 |
| maximin_dispersion | traffic | 720 | 2049 | 0.626009 | 0.936242 |
| anti_clustering | traffic | 96 | 2034 | 0.622717 | 0.880263 |
| maximin_dispersion | traffic | 720 | 2038 | 0.622691 | 1.001198 |
| anti_clustering | traffic | 720 | 2031 | 0.620185 | 0.989613 |
| ordered | traffic | 96 | 2033 | 0.618857 | 0.954953 |
| random | traffic | 720 | 2038 | 0.618657 | 0.866656 |
| score_stratified | traffic | 720 | 2039 | 0.618370 | 0.901153 |
| maximin_dispersion | traffic | 96 | 2034 | 0.616787 | 0.829898 |
| score_stratified | traffic | 96 | 2037 | 0.616645 | 0.807637 |
| maximin_dispersion | traffic | 720 | 2045 | 0.616354 | 0.858802 |
| random | traffic | 96 | 2037 | 0.615715 | 0.811752 |
| score_stratified | traffic | 720 | 2031 | 0.609520 | 0.874483 |
| maximin_dispersion | traffic | 336 | 2027 | 0.607264 | 0.967557 |
| score_stratified | traffic | 336 | 2027 | 0.604196 | 0.822731 |
| ordered | traffic | 336 | 2027 | 0.600444 | 0.826558 |
| anti_clustering | traffic | 336 | 2027 | 0.600122 | 0.825379 |
| maximin_dispersion | traffic | 96 | 2021 | 0.597737 | 0.797014 |
| random | traffic | 336 | 2027 | 0.592045 | 0.831171 |

## 5. Epoch-1 Loss Histogram

| Range | Normal | Anomaly | Total |
|-------|--------|---------|-------|
| [0.15, 0.20) | 89 | 0 | 89 |
| [0.20, 0.25) | 631 | 0 | 631 |
| [0.25, 0.30) | 1189 | 0 | 1189 |
| [0.30, 0.35) | 11 | 0 | 11 |
| [0.35, 0.40) | 298 | 0 | 298 |
| [0.40, 0.50) | 596 | 0 | 596 |
| [0.50, 0.60) | 40 | 2 | 42 |
| [0.60, 0.80) | 1 | 22 | 23 |
| [0.80, 1.00) | 0 | 1 | 1 |

## 6. Dataset-Level Analysis

### electricity
- Runs: 960, Anomalies: 0
- Epoch-1 loss: mean=0.232077, std=0.032952, range=[0.195909, 0.286841]

### solar_AL
- Runs: 960, Anomalies: 0
- Epoch-1 loss: mean=0.284935, std=0.015995, range=[0.255670, 0.301319]

### traffic
- Runs: 960, Anomalies: 25
- Epoch-1 loss: mean=0.428957, std=0.050762, range=[0.373887, 0.843652]
- Anomaly epoch-1 loss: mean=0.632475, min=0.592045

## 7. Category-level analysis

### Cohesion
- Runs: 1080, Anomalies: 0 (0.0%)
- Epoch-1 loss: mean=0.306739, std=0.076691

### Non-cohesion
- Runs: 1800, Anomalies: 25 (1.4%)
- Epoch-1 loss: mean=0.320474, std=0.097836

## 8. Conclusion

- **Recall-first screening threshold**: 0.55 (TP=25/25, FP=12/2,855, recall=1.000, overall FPR=0.0042)
- The normal and anomaly distributions overlap; the 0.5962 midpoint is therefore not a separating or prospectively optimal threshold.
- At 0.60: recall=0.920, FP=1.
