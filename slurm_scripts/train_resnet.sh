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
module load miniconda/3.7

# configuration variables
python_script=train_resnet.py
config_file=encoder_train.yaml
conda_env=l2acenv

file=config/$config_file


# Loop variables
file=config/$config_file
var_1=name	
array_1=(e0001 e0002 e0001 e0001 e0040)
var_2=model_class
array_2=(resnet50 resnet50 resnet152 vgg11 vgg11)
var_3=pretrained 
array_3=(True False True True False)
len=${#array_1[@]}




var_4=epochs
value_4=5

conda activate $conda_env


for ((i=0;i<$len; i++))

do

        echo "$var_4 = ${value_4}"
        sed -i "s/$var_4:.*/$var_4: ${value_4}/" $file
	

	echo "$var_1 = ${array_1[$i]}"
	echo "$var_2 = ${array_2[$i]}"
        echo "$var_3 = ${array_3[$i]}"

	sed -i "s/$var_1:.*/$var_1: '${array_1[$i]}'/"  $file
	sed -i "s/$var_2:.*/$var_2: ${array_2[$i]}/" $file
	sed -i "s/$var_3:.*/$var_3: ${array_3[$i]}/" $file


	python $python_script $config_file

done

conda deactivate
