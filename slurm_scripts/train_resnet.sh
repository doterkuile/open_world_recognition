#!/bin/bash
#SBATCH --job-name=train_resnet 	# create a short name for your job
#SBATCH --output=logs/%x-%j.out                 # output_file
#SBATCH --partition=general				# select partition
#SBATCH --qos=short						# select quality of service
#SBATCH --nodes=1                		# node count
#SBATCH --ntasks=1               		# total number of tasks across all nodes
#SBATCH --cpus-per-task=6        		# cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=9gb                		# total memory per node (4 GB per cpu-core is default)
#SBATCH --gres=gpu:2             		# number of gpus per node
#SBATCH --time=04:00:00          		# total run time limit (HH:MM:SS)
#SBATCH --mail-type=begin        		# send mail when job begins
#SBATCH --mail-type=end          		# send mail when job ends
#SBATCH --mail-type=fail         		# send mail if job fails
#SBATCH --mail-user=doterkuile@gmail.com




module use /opt/insy/modulefiles

module purge
module load cuda/10.0
module load cudnn/10.0-7.6.0.64
module load miniconda

# configuration variables
python_script=train_resnet.py
config_file=encoder_train.yaml
conda_env=l2acenv

file=config/$config_file



var_4=epochs
value_4=200

echo "$var_4 = ${value_4}"
sed -i "s/$var_4:.*/$var_4: ${value_4}/" $file

conda activate $conda_env

python $python_script $config_file

conda deactivate
