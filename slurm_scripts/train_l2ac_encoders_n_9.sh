#!/bin/bash
#SBATCH --job-name=train_l2ac_balanced_set  	# create a short name for your job
#SBATCH --output=logs/%x-%j.out                 # output_file
#SBATCH --partition=general				# select partition
#SBATCH --qos=long						# select quality of service
#SBATCH --nodes=1                		# node count
#SBATCH --ntasks=1               		# total number of tasks across all nodes
#SBATCH --cpus-per-task=2        		# cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=9gb                		# total memory per node (4 GB per cpu-core is default)
#SBATCH --gres=gpu:p100:1             		# number of gpus per node
#SBATCH --time=17:00:00          		# total run time limit (HH:MM:SS)
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
python_script=train_l2ac.py
base_config_file=L2AC_train_base.yaml
config_file=L2AC_train.yaml
conda_env=l2acenv

cp config/$base_config_file config/$config_file


# Loop variables
file=config/$config_file
var_1=name	
array_1=(0005 0006 0007 0008)
var_2=feature_layer
array_2=(avgpool avgpool features _avg_pooling)
var_3=encoder
array_3=(Resnet50 Resnet152 AlexNet EfficientNet)
var_4=top_n
array_4=(9 9 9 9)
len=${#array_1[@]}




var_e=epochs
value_e=200

conda activate $conda_env


for ((i=0;i<$len; i++))

do

        echo "$var_e = ${value_e}"
        sed -i "s/$var_e:.*/$var_e: ${value_e}/" $file
	

	echo "$var_1 = ${array_1[$i]}"
	echo "$var_2 = ${array_2[$i]}"
        echo "$var_3 = ${array_3[$i]}"
	echo "$var_4 = ${array_4[$i]}"

	sed -i "s/$var_1:.*/$var_1: '${array_1[$i]}'/"  $file
	sed -i "s/$var_2:.*/$var_2: ${array_2[$i]}/" $file
	sed -i "s/$var_3:.*/$var_3: ${array_3[$i]}/" $file
	sed -i "s/$var_4:.*/$var_4: ${array_4[$i]}/" $file


	python $python_script $config_file

done

conda deactivate
