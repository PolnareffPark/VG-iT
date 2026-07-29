# Observed experiment environment

This file records the environment observed in the archived VG-iT experiment and efficiency
metadata. It is an audit record, not a claim that other compatible environments reproduce
bit-identical GPU kernels.

## Runtime and hardware

- Python: 3.11.13
- PyTorch: 2.7.1+cu128
- CUDA runtime reported by PyTorch: 12.8
- cuDNN version: 90701
- GPU: NVIDIA GeForce RTX 3090
- Reported GPU memory: 23.56 GB
- Mixed precision: bfloat16
- Efficiency-profile seed: 2021

## Principal Python packages

- NumPy 1.26.4
- pandas 2.3.2
- PyArrow 23.0.1
- SciPy 1.16.3
- scikit-learn 1.7.2
- Matplotlib 3.10.5
- einops 0.8.2
- reformer-pytorch 1.4.4
- nystrom-attention 0.0.14
- scikit-posthocs 0.12.0
- statsmodels 0.14.6
- baycomp 1.0.3

The portable installation list is `requirements.txt`; install the matching PyTorch build
separately. Hardware and efficiency-protocol metadata originate from the archived
`efficiency_profile_real.json` used to build the curated efficiency table.
