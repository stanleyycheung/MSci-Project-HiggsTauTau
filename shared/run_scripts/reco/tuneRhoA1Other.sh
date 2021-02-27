cd ..
cd ..
source /home/hep/shc3117/anaconda3/etc/profile.d/conda.sh
conda activate msci
python ./NN.py 'rho_a1' ${SGE_TASK_ID} -t -tm 'random_sk'