#!/bin/bash

PYTHON="/scratch1/rkwitt/anaconda3/bin/python"

PARAMETER_WEIGHT=(1.0 5.0 10)
NR_OF_PARTICLES=(2 4 6 8 10 12 14 16 18 20)
NORM_PENALTY=(1e-3)

for pw in ${PARAMETER_WEIGHT[*]}; do
	for nrp in ${NR_OF_PARTICLES[*]}; do
    for np in ${NORM_PENALTY[*]}; do
      CMD="$PYTHON demos/cc.py --parameter_weight ${pw} \
                          --norm_penalty_weight ${np} \
                          --nr_of_particles ${nrp} \
                          --save_model model/model_debug_${nrp}_${pw}_${np} \
                          --nepochs 10 \
                          --inflation_factor 4 \
                          --wd 1e-2 \
                          --lr 5e-2 \
                          --verbose"
      echo $CMD
      $CMD
    done
  done
done

