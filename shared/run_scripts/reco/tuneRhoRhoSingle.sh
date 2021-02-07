#!/bin/sh
cd ..
cd ..
# source ./msci/bin/activate
# export LD_LIBRARY_PATH=/vols/cms/ktc17/cuda/lib64:$LD_LIBRARY_PATH
source /home/hep/shc3117/anaconda3/etc/profile.d/conda.sh
conda activate msci
python ./NN.py 'rho_rho' ${SGE_TASK_ID} -t