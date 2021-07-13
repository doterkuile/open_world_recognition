#!/bin/bash
#SBATCH --job-name=train_data_setup   	# create a short name for your job
#SBATCH --output=logs/%x-%j.out                 # output_file
#SBATCH --partition=general				# select partition
#SBATCH --qos=short						# select quality of service
#SBATCH --nodes=1                		# node count
#SBATCH --ntasks=1               		# total number of tasks across all nodes
#SBATCH --cpus-per-task=4        		# cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=20gb                		# total memory per node (4 GB per cpu-core is default)
#SBATCH --gres=gpu:1             		# number of gpus per node
#SBATCH --time=02:00:00          		# total run time limit (HH:MM:SS)
#SBATCH --mail-type=begin        		# send mail when job begins
#SBATCH --mail-type=end          		# send mail when job ends
#SBATCH --mail-type=fail         		# send mail if job fails
#SBATCH --mail-user=doterkuile@gmail.com


# configuration variables
python_script_1=setup_train_data.py
python_script_2=setup_test_data.py

config_file_base=TinyImageNet_train_base.yaml
config_file=TinyImageNet_train.yaml
conda_env=l2acenv

cp config/$base_config_file config/$config_file


module use /opt/insy/modulefiles
module purge

module load miniconda/3.7
conda activate $conda_env

module load cuda/10.0
module load cudnn/10.0-7.6.0.64


# Loop variables
file=config/$config_file
var_1=image_resize
array_1=(224 224 224)	
#array_1=(4 4 4 4)
var_2=unfreeze_layer
array_2=(62 62 62)
#array_2=(avgpool avgpool features _avg_pooling)
var_3=top_n
array_3=(1 4 9)
#array_3=(ResNet50 ResNet152 AlexNet EfficientNet)
len=${#array_1[@]}


# conda activate $conda_env


for ((i=0;i<$len; i++))

do

	echo "$var_1 = ${array_1[$i]}"
	echo "$var_2 = ${array_2[$i]}"
    echo "$var_3 = ${array_3[$i]}"

	sed -i "s/$var_1:.*/$var_1: ${array_1[$i]}/"  $file
	sed -i "s/$var_2:.*/$var_2: ${array_2[$i]}/" $file
	sed -i "s/$var_3:.*/$var_3: ${array_3[$i]}/" $file


	python $python_script_1 $config_file
	python $python_script_2 $config_file

done


conda deactivate
