#!/bin/bash

extra=Demo
analyze=False
debug=False

# first run 11
lr=5e-4                          # learning rate
d_kl_c=0.05                      # kl divergence coefficient
bs=256                           # batch size
agv=11                           # agent view size
size=15                          # environment size
shape=${size}x${size}            # environment shape
maxsteps=1                     # max steps per an agent can execute in a single episode
task=tasks.TVMGFR-$shape-RARG-GI # task name
qh_dim=128                       # qnet hidden dimension
epochs=1
step_per_epoch=10

train_n=1
test_n=1

WANDB_MODE=disabled python -m src.experiments.run_gridworld \
    --seed 0 \
    train \
    -e $task \
    --policy DQL \
    --agent-view-size $agv \
    --max-steps $maxsteps \
    --training-num $train_n \
    --test-num $test_n \
    --extra $extra \
    --epochs $epochs \
    --step-per-epoch $step_per_epoch \
    --lr $lr \
    --d-kl-c $d_kl_c \
    --batch-size $bs \
    --subgoal-on False \
    --planning True \
    --qh-dim $qh_dim \
    --analyze $analyze \
    --sampling-strategy 4 \
    --debug $debug