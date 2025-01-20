#!/bin/bash
export PYTHONPATH=$(pwd)/src:$PYTHONPATH

# Configuración de entorno
size=15
shape=${size}x${size}
task="tasks.TVMGFR-$shape-RARG-GI"

seed=0


extra="Demo"
analyze=False
debug=False

# Parámetros principales

lr=5e-4
d_kl_c=0.05
batch_size=256
agent_view_size=11
qh_dim=128
max_steps=140
epochs=40

policy="DQL"       # DQL, DQLG, NonRL
subgoal_on=False  
planning=True  
sampling_strategy=4  

python -m src.experiments.run_gridworld \
    --seed ${seed} \
    train \
    -e ${task} \
    --policy ${policy} \
    --agent-view-size ${agent_view_size} \
    --max-steps ${max_steps} \
    --extra ${extra} \
    --epochs ${epochs} \
    --lr ${lr} \
    --d-kl-c ${d_kl_c} \
    --batch-size ${batch_size} \
    --subgoal-on ${subgoal_on} \
    --planning ${planning} \
    --qh-dim ${qh_dim} \
    --analyze ${analyze} \
    --sampling-strategy ${sampling_strategy} \
    --debug ${debug}