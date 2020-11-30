#!/bin/bash

PYTHON="/scratch1/rkwitt/anaconda3/bin/python"

INFLATION=(20 5)

for run in `seq 10`; do
  for inf in ${INFLATION[*]}; do
    CMD="$PYTHON cc.py \
      --shooting_dim 2 \
      --method rk4 \
      --stepsize 0.1 \
      --pw 1.5 \
      --shooting_model updown_universal \
      --nr_of_particles 20 \
      --gpu 1 \
      --batch_size 128 \
      --inflation_factor ${inf}\
      --sim_weight 1.0 \
      --lr 5e-3 \
      --niters 20 \
      --save_model model_${inf}_run_${run}"
    echo $CMD
    $CMD
  done
done

