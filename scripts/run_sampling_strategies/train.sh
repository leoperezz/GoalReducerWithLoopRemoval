#!/bin/bash


seed=0
gr=VAE
lr=5e-4
cores=8
graph=RD
walk_repeats=50
scalfactor=1.0
klw=0.01
lm=None

python -m src.experiments.run_sampling_strategies \
    --seed $seed \
    train \
    --gr $gr \
    --lr $lr \
    --cores $cores \
    --graph $graph \
    --walk-repeats $walk_repeats \
    --scalfactor $scalfactor \
    --klw $klw \
    --lm $lm
