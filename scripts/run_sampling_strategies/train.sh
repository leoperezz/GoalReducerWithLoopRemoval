#!/bin/bash
export PYTHONPATH=$(pwd)/src:$PYTHONPATH

seed=0
gr=VAE
lr=5e-4
cores=8
graph=all
walk_repeats=50
scalfactor=1.0
klw=0.01

python -m src.experiments.run_sampling_strategies \
    --seed $seed \
    train \
    --gr $gr \
    --lr $lr \
    --cores $cores \
    --graph $graph \
    --walk-repeats $walk_repeats \
    --scalfactor $scalfactor \
    --klw $klw
