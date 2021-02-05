#!/bin/sh
cd ..
cd ..
source ./msci/bin/activate
export LD_LIBRARY_PATH=/vols/cms/ktc17/cuda/lib64:$LD_LIBRARY_PATH
python ./NN.py 'rho_rho' ${SGE_TASK_ID}