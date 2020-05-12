#!/bin/bash

PYTHON="/scratch1/rkwitt/anaconda3/bin/python"

PW=10.0 # parameter weight
IF=40 # inflation factor
NP=20 # number of particles

for subject in $(seq 0 42)
do
    CMD="$PYTHON mocap_single.py \
        --gpu 0 \
        --verbose \
        --seed 1234 \
        --method dopri5 \
        --shooting_model updown \
        --optimize_over_data_initial_conditions \
        --optimize_over_data_initial_conditions_type linear \
        --inflation_factor ${IF} \
        --nr_of_particles ${NP} \
        --pw ${PW} \
        --niters 200 \
        --subject_id ${subject} \
        --save_prefix pw_${PW}_if_${IF}_np_${NP}_subject_"
    echo $CMD
    $CMD
done
