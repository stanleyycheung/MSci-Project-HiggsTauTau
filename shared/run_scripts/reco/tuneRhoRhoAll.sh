#!/bin/sh
cd ..
cd ..
source ./msci/bin/activate
export LD_LIBRARY_PATH=/vols/cms/ktc17/cuda/lib64:$LD_LIBRARY_PATH

for i in {1..6}
do
   python ./NN.py 'rho_rho' i -t
done
