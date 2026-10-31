#!/bin/bash
# UniMuMo generation for the reproduction and the two experiments (about 1 GPU-hour in total).
set -e
cd "$(dirname "$0")/.."
python generate_batch.py --jobs configs/reproduce.json --out runs/reproduce --seeds 42 --n_cand 4
python generate_batch.py --jobs configs/exp_a_music2dance.json --out runs/exp_a --seeds 0 1 2 --n_cand 16
python generate_batch.py --jobs configs/exp_b_text2musicdance.json --out runs/exp_b --seeds 0 1 2 --n_cand 16
