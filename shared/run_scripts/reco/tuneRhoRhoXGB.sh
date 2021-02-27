cd ..
cd ..
source /home/hep/shc3117/anaconda3/etc/profile.d/conda.sh
conda activate msci
python ./xgb.py 'rho_rho' ${SGE_TASK_ID} -t