#!/bin/sh
cd ..
cd ..
# source ./msci/bin/activate
# export LD_LIBRARY_PATH=/vols/cms/ktc17/cuda/lib64:$
source /home/hep/shc3117/anaconda3/etc/profile.d/conda.sh
conda activate msci
python ./NN.py 'a1_a1' ${SGE_TASK_ID}