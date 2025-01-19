#!/bin/bash
export PYTHONPATH=$(pwd)/src:$PYTHONPATH

# Mantenemos las mismas variables del entrenamiento para consistencia
seed=0
gr=VAE
graph=all
klw=0.01

# Variables específicas para analyze
maxk=4  # Número de pasos de reducción

python -m src.experiments.run_sampling_strategies \
--seed $seed \
analyze \
--gr $gr \
--graph $graph \
--maxk $maxk \
--klw $klw