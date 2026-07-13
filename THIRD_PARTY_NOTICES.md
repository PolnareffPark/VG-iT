# Third-Party Notices

This repository contains original VG-iT code and selected files adapted from public time-series forecasting implementations. The root MIT license applies to the authors' original VG-iT contributions unless a source file carries a different license header or attribution requirement. The file mapping below is intentionally conservative: a listed file may contain both upstream scaffold and VG-iT-specific changes.

The exact historical upstream Git revisions could not be recovered from the experiment
checkout for every adapted file. The repository URLs below identify the source projects
but are not claimed as commit-pinned references. This limitation is retained in v1.0.0;
no unverified revision is asserted.

## Apache-2.0 Components

- `code/utils/timefeatures.py` is adapted from `gluonts/src/gluonts/time_feature/_base.py`.
  - Copyright: 2018 Amazon.com, Inc. or its affiliates.
  - License: Apache License, Version 2.0.
  - The original license header is retained in the source file.
- ProbAttention code in `code/layers/SelfAttention_Family.py` is adapted from `https://github.com/zhouhaoyi/Informer2020`.
  - License observed from the source repository: Apache-2.0.
  - The repository keeps file-level attribution comments and the Apache-2.0 license copy for redistribution.

A copy of the Apache-2.0 license terms is provided in `LICENSES/Apache-2.0.txt`.

## MIT-Licensed Attributed Attention Implementations

`code/layers/SelfAttention_Family.py` also retains source comments for attention implementations adapted from public repositories:

- FlowAttention: `https://github.com/thuml/Flowformer`.
  - License observed from the source repository: MIT.
  - Copyright notice included in `LICENSES/MIT-THIRD-PARTY.txt`.
- FlashAttention-style implementation: `https://github.com/shreyansh26/FlashAttention-PyTorch`.
  - License observed from the source repository: MIT.
  - Copyright notice included in `LICENSES/MIT-THIRD-PARTY.txt`.

These attribution comments are preserved in the corresponding source file.
## MIT-Licensed Forecasting Framework Components

The iTransformer-family baseline and selected forecasting scaffold files in this
deposit follow the public THUML implementations used by the manuscript:

- Conservative mapped set: `code/run.py`, `code/model/iTransformer.py`,
  `code/layers/Embed.py`, `code/layers/Transformer_EncDec.py`,
  `code/layers/SelfAttention_Family.py`, `code/utils/tools.py`,
  `code/utils/masking.py`, `code/utils/metrics.py`,
  `code/data_provider/data_loader.py`, `code/data_provider/data_factory.py`,
  `code/experiments/exp_basic.py`, `code/experiments/exp_long_term_forecasting.py`,
  and `code/experiments/exp_long_term_forecasting_partial.py` follow or adapt the
  iTransformer and Time-Series-Library code organization.
- Source repositories:
  - `https://github.com/thuml/iTransformer`
  - `https://github.com/thuml/Time-Series-Library`
- License observed from both source repositories: MIT.
- A consolidated copy of the standard MIT terms and the observed upstream copyright
  notices is included in `LICENSES/MIT-THIRD-PARTY.txt`.

- These files include VG-iT-specific modifications for grouping, efficiency logging,
  preprocessing, and repository packaging; the notices above identify the reused public
  scaffold/component structure.
