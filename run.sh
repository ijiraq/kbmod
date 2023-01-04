#!/bin/bash
if [ $# -lt 2 ]
then
   echo "USAGE: ${BASH_SOURCE[0]} python path/kbmod_run.py arg1 arg2 --options-for-script"
   echo ""
   echo "This script setsup the environment needed to run kbmod (based on the location of the script)"
   echo "and then executes the kbmod_run.py script with the correct environment settings"
   exit 255
fi

SRCDIR=$(dirname "${BASH_SOURCE[0]}")
SRCDIR=$(realpath ${SRCDIR})
echo "Running in ${SRCDIR}"
export PYTHONPATH="${SRCDIR}":"${SRCDIR}/analysis":"${SRCDIR}/search/pybinds":$PYTHONPATH

# Disable python multiprocessing
export NUMEXPR_MAX_THREADS=1
export OPENBLAS_NUM_THREADS="1"
export MKL_NUM_THREADS="1"
export MKL_DYNAMIC="FALSE"
export OMP_NUM_THREADS=1

export CUDADIR=/usr/local/cuda
export PATH=$PATH:$CUDADIR/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CUDADIR/lib64

$@
