#!/bin/bash

CUR_DATE=$(date +'%Y-%m-%d_%H-%M')
export OPENBLAS_NUM_THREADS=1
export JULIA_NUM_PRECOMPILE_TASKS=1
export JULIA_CPU_TARGET="generic"

#if .sh script is located in the ADAPT-GPT directory:
SCRIPT_DIR="$(dirname "$(realpath "$0")")"
#else - specify ADAPT-GPT directory below:
#SCRIPT_DIR="<YOUR_QAOA_GPT_DIRECTORY_HERE>"
COMPILE="yes" # yes or no

#######################################
# ADAPT parameters (adjust as needed) #
N_WORKERS=1 # number of concurrent ADAPT.jl instances 
OUTPUT_DIR="ADAPT.jl_results/test/$CUR_DATE"
GRAPHS_NUMBER=100
N_NODES=11
TRIALS_PER_GRAPH=5
ENERGY_TOL_FRAC="0.03"
MAX_PARAMS=50
WEIGHTED=true
NORMALIZE_WEIGHTS=false
GAMMA_0=$SCRIPT_DIR/"gamma0_grid.json" # or a number like GAMMA_0="0.1"
POOL_NAME="qaoa_double_pool" # or qaoa_mixer

#######################################

LOG_DIR=$OUTPUT_DIR/logs/$CUR_DATE
mkdir -p $LOG_DIR

###########################
# ADAPT Parallel executor # 
for ((i=0; i<N_WORKERS; i++))
do
    echo "Starting copy" $i " on node: " $HOSTNAME
    julia --compiled-modules=$COMPILE --project=$SCRIPT_DIR/ADAPT.jl $SCRIPT_DIR/adapt_maxcut_run_1thread.jl \
        --output-dir $OUTPUT_DIR \
        --graphs-number $GRAPHS_NUMBER \
        --n-nodes $N_NODES \
        --trials-per-graph $TRIALS_PER_GRAPH \
        --energy-tol-frac $ENERGY_TOL_FRAC \
        --max-params $MAX_PARAMS \
        --weighted $WEIGHTED \
        --run-qaoa true \
        --run-vqe false \
        --calc-h-eigen true \
        --run-diag-qaoa true \
        --normalize-weights $NORMALIZE_WEIGHTS \
        --g0 $GAMMA_0 \
        --op-pool $POOL_NAME \
        2>&1 | tee $LOG_DIR/worker_${HOSTNAME}_${i}.log \
        & 
done
wait