#!/bin/bash
#SBATCH --job-name=evaluate_object.py  	# create a short name for your job
#SBATCH --output=logs/%x-%j.out                 # output_file
#SBATCH --partition=general				# select partition
#SBATCH --qos=long				# select quality of service
#SBATCH --nodes=1                		# node count
#SBATCH --ntasks=1               		# total number of tasks across all nodes
#SBATCH --cpus-per-task=4      		# cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=6gb                		# total memory per node (4 GB per cpu-core is default)
#SBATCH --gres=gpu:1             		# number of gpus per node
#SBATCH --time=01:00:00          		# total run time limit (HH:MM:SS)
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

name_array=(models_0001 models_0019)
len=${#name_array[@]}

var_1=load_results
value_1=True

conda activate $conda_env


for ((i=0;i<$len; i++))

do
	echo "Run ${name_array[$i]}"

	config_file=results/${name_array[$i]}/${name_array[$i]}_config.yaml
	sed -i "s/$var_1:.*/$var_1: ${value_1}/" $config_file


	mkdir -p results/${name_array[$i]}
	# cp -r $base_config_file $config_file
	# echo "copied config file to $config_file"


	python $python_script $config_file

done


conda deactivate
