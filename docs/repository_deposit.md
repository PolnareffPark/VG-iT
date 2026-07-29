# VG-iT Repository Deposit

This document describes the cleaned public repository root and archival payload for VG-iT v1.0.0.

Contents:
- `README.md`: GitHub-facing overview and reproduction entry point.
- `CITATION.cff`: GitHub and archival-repository citation metadata.
- `IMPLEMENTATION_NOTES.md`: observed v1.0.0 behavior retained for exact reruns.
- `LICENSE`: MIT license for the authors' original software contributions.
- `LICENSE_SCOPE.md`: fixed file-level software/data/content rights boundary for v1.0.0.
- `THIRD_PARTY_NOTICES.md` and `LICENSES/`: third-party attribution, license files, and the custom non-software rights statement.
- `code/`: public source code, training scripts, and manuscript-used preprocessing scripts.
- `data_provenance/`: public dataset access/provenance notes.
- `results_canonical/`: curated CSV/JSON evidence files used by the manuscript, organized by analysis family.
- `figure_generation/`: manual-artwork notes for Figures 1-2 and scripts/rendered outputs for Figures 3-6 and Supplementary Figures S1-S2.
- `docs/results_usage_manifest.md`: table/figure/claim mapping for included result files.
- `docs/figure_code_manifest.md`: figure artwork and code mapping.

Full training checkpoints are not included because the complete training checkpoint set is too large for this public package.

## Copyright ownership note

The MIT copyright holder is intentionally listed as Himchan Park in the repository license files. Manuscript authorship and citation metadata remain listed separately in CITATION.cff.

If this release is deposited in an archival repository, its record should describe
the MIT, Apache-2.0, and custom non-software rights scopes. Record-level declarations
do not replace the archive-path mapping in `LICENSE_SCOPE.md`,
`THIRD_PARTY_NOTICES.md`, and `LICENSES/`. No single license applies to the archive
as a whole.
