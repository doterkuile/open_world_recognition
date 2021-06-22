#!/bin/bash
#SBATCH --job-name=train_data_setup   	# create a short name for your job
#SBATCH --output=logs/%x-%j.out                 # output_file
#SBATCH --partition=general				# select partition
#SBATCH --qos=short						# select quality of service
#SBATCH --nodes=1                		# node count
#SBATCH --ntasks=1               		# total number of tasks across all nodes
#SBATCH --cpus-per-task=1        		# cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=4096                		# total memory per node (4 GB per cpu-core is default)
#SBATCH --gres=gpu:1             		# number of gpus per node
#SBATCH --time=00:15:00          		# total run time limit (HH:MM:SS)
#SBATCH --mail-type=begin        		# send mail when job begins
#SBATCH --mail-type=end          		# send mail when job ends
#SBATCH --mail-type=fail         		# send mail if job fails
#SBATCH --mail-user=doterkuile@gmail.com

python_script=setup_train_data.py
config_file=CIFAR100_train.yaml

conda_env=l2acenv
package=open_world_recognition
data_path=/tudelft.net/staff_umbrella/openworldrecognition/

#mkdir -p /tmp/$USER && cp -a $data_path$package /tmp/$USER

module use /opt/insy/modulefiles
module purge

module load miniconda
conda activate $conda_env

module load cuda/10.0
module load cudnn/10.0-7.6.0.64

file=config/$config_file
var1=name
var2=top_n

# Loop variables
file=config/$config_file
var_1=top_n	
array_1=(1 2 9 9)
var_2=train_classes
array_2=(20 20 20 80)
var_3=test_classes
array_3=(20 20 20 80)
len=${#array_1[@]}


# conda activate $conda_env


for ((i=0;i<$len; i++))

do

	echo "$var_1 = ${array_1[$i]}"
	echo "$var_2 = ${array_2[$i]}"
    echo "$var_3 = ${array_3[$i]}"

	sed -i "s/$var_1:.*/$var_1: '${array_1[$i]}'/"  $file
	sed -i "s/$var_2:.*/$var_2: ${array_2[$i]}/" $file
	sed -i "s/$var_3:.*/$var_3: ${array_3[$i]}/" $file


	python $python_script $config_file
done


conda deactivate
