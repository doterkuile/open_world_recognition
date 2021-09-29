#!/bin/bash
#SBATCH --job-name=evaluate_l2ac.py  	# create a short name for your job
#SBATCH --output=logs/%x-%j.out                 # output_file
#SBATCH --partition=general				# select partition
#SBATCH --qos=short					# select quality of service
#SBATCH --nodes=1                		# node count
#SBATCH --ntasks=1               		# total number of tasks across all nodes
#SBATCH --cpus-per-task=4      		# cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=25gb                		# total memory per node (4 GB per cpu-core is default)
#SBATCH --gres=gpu:1             		# number of gpus per node
#SBATCH --time=04:00:00          		# total run time limit (HH:MM:SS)
#SBATCH --mail-type=begin        		# send mail when job begins
#SBATCH --mail-type=end          		# send mail when job ends
#SBATCH --mail-type=fail         		# send mail if job fails
#SBATCH --mail-user=doterkuile@gmail.com




module use /opt/insy/modulefiles

module purge
module load cuda/10.0
module load cudnn/10.0-7.6.0.64
module load miniconda/3.7

# configuration variables
python_script=evaluate_l2ac.py
base_config_file=config/L2AC_evaluate.yaml
conda_env=l2acenv

name_var=figure_title


conda activate $conda_env

name=$(cat $base_config_file | grep $name_var: | tail -n1 | awk '{print $2}')
# name=encoders_fn
# echo "Run $name"

config_file=results/${name}/${name}_config.yaml


mkdir -p results/${name}
cp -r $base_config_file $config_file
echo "copied config file to $config_file"


python $python_script $config_file


conda deactivate
