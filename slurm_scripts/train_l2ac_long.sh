#!/bin/bash
#SBATCH --job-name=train_l2ac_50_e  	# create a short name for your job
#SBATCH --output=logs/%x-%j.out                 # output_file
#SBATCH --partition=general				# select partition
#SBATCH --qos=long						# select quality of service
#SBATCH --nodes=1                		# node count
#SBATCH --ntasks=1               		# total number of tasks across all nodes
#SBATCH --cpus-per-task=8        		# cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=20gb                		# total memory per node (4 GB per cpu-core is default)
#SBATCH --gres=gpu:8             		# number of gpus per node
#SBATCH --time=09:00:00          		# total run time limit (HH:MM:SS)
#SBATCH --mail-type=begin        		# send mail when job begins
#SBATCH --mail-type=end          		# send mail when job ends
#SBATCH --mail-type=fail         		# send mail if job fails
#SBATCH --mail-user=doterkuile@gmail.com

python_script=train_l2ac.py
config_file=L2AC_train.yaml

conda_env=l2acenv


module use /opt/insy/modulefiles

module purge


module load cuda/10.0
module load cudnn/10.0-7.6.0.64
module load miniconda
conda activate $conda_env



python $python_script $config_file
