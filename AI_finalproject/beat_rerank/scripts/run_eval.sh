#!/bin/bash
# Scoring, tables, figures and samples (CPU, plus a few GPU minutes for the audio judges).
set -e
cd "$(dirname "$0")/.."
python motion_style.py --out runs/motion_style.pt
python real_reference.py
python score_candidates.py --run runs/reproduce --jobs configs/reproduce.json
python score_candidates.py --run runs/exp_a --jobs configs/exp_a_music2dance.json
python score_candidates.py --run runs/exp_b --jobs configs/exp_b_text2musicdance.json
python judge_audio.py --run runs/reproduce --jobs configs/reproduce.json
python judge_audio.py --run runs/exp_b --jobs configs/exp_b_text2musicdance.json
python reproduce_report.py
python make_tables.py --exp a
python make_tables.py --exp b
python retime_sweep.py
python plot_results.py --exp a
python plot_results.py --exp b
python render_samples.py
