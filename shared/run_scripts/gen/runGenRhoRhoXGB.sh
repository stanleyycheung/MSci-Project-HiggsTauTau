cd ..
cd ..
source /home/hep/shc3117/anaconda3/etc/profile.d/conda.sh
conda activate msci
python ./xgboost.py 'rho_rho' ${SGE_TASK_ID} -g -r -hdf