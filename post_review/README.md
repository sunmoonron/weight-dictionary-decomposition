# After the ICLR 2027 reviews

The paper was submitted to ICLR 2027 and withdrawn after three reviews. These runs answer the reviewers' questions with the paper's own scripts and protocol: GPT-2 small, WikiText-2, 32,768 states, FVU on typical states (norm at most ten times the median). Predictions were written to `prereg.json` before any run, smoke tests included; `analyze.py` scores them (10 of 15 confirmed). Every rerun of a recorded dictionary reproduces the paper's numbers to three decimals.

- `pr1_provenance_random.py` (Reviewer 7MWE): the Section 4.1 provenance controls next to size-matched random directions (three seeds), each control's own Gram-preserving rotation, and the embedding rows alone. At k = 32, with 19,210 atoms and no embeddings: writers 0.375, non-writers 0.490, random 0.570, rotated non-writers 0.578. On the sink states writers and non-writers both reach 0.0002 at k = 64, random 0.33.
- `pr2_z2_variant.py` (Reviewers NMa5 and MgDZ): `exp_z2_ci.py` with two switches, `WDD_LAYER` and `WDD_RANDOM_INIT`. Layer 2: FVU 0.230 against 0.547 rotated, top-1 recall 0.739. Layer 6: the paper's 0.338 and 0.647. Layer 10: 0.345 against 0.517, recall 0.235. Random weights at layer 6: 0.495 against 0.500, recall 0.008.
- `run_pr.sh`, `chain_gpu0.sh`, `chain_gpu1.sh`: how they were run on two 8 GB cards; `results/` holds every output, the per-token records and the logs.

To rescore, run `python post_review/analyze.py`. It computes the typical-state numbers from the per-token records itself, so `scripts/exp_z2_recompute.py` is not needed.
