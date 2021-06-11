#!/bin/bash
#SBATCH --job-name=train_l2ac_top_n_1  	# create a short name for your job
#SBATCH --output=logs/%x-%j.out                 # output_file
#SBATCH --partition=general				# select partition
#SBATCH --qos=short						# select quality of service
#SBATCH --nodes=1                		# node count
#SBATCH --ntasks=1               		# total number of tasks across all nodes
#SBATCH --cpus-per-task=4        		# cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=10gb                		# total memory per node (4 GB per cpu-core is default)
#SBATCH --gres=gpu:2             		# number of gpus per node
#SBATCH --time=02:00:00          		# total run time limit (HH:MM:SS)
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
python_script=train_l2ac.py
config_file=L2AC_train.yaml
conda_env=l2acenv


# Loop variables
file=config/$config_file
var_1=name	
array_1=(0006 0007)
var_2=top_n
array_2=(1 2)
len=${#array_1[@]}

conda activate $conda_env


for ((i=0;i<$len; i++))
do
	echo "$var_1 = ${array_1[$i]}"
	echo "$var_2 = ${array_2[$i]}"
	sed -i "s/$var2:.*/$var2: ${array_2[$i]}/" $file
	sed -i "s/$var1:.*/$var1: ${array_1[$i]}/"  $file
	python $python_script $config_file

done
conda deactivate