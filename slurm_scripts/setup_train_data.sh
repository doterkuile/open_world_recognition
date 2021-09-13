#!/bin/bash
#SBATCH --job-name=train_data_setup   	# create a short name for your job
#SBATCH --output=logs/%x-%j.out                 # output_file
#SBATCH --partition=general				# select partition
#SBATCH --qos=short						# select quality of service
#SBATCH --nodes=1                		# node count
#SBATCH --ntasks=1               		# total number of tasks across all nodes
#SBATCH --cpus-per-task=4        		# cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=9gb                		# total memory per node (4 GB per cpu-core is default)
#SBATCH --gres=gpu:1             		# number of gpus per node
#SBATCH --time=4:00:00          		# total run time limit (HH:MM:SS)
#SBATCH --mail-type=begin        		# send mail when job begins
#SBATCH --mail-type=end          		# send mail when job ends
#SBATCH --mail-type=fail         		# send mail if job fails
#SBATCH --mail-user=doterkuile@gmail.com


# configuration variables
python_script_1=setup_train_data.py

base_config_file=TinyImageNet_train_base.yaml
conda_env=l2acenv


module use /opt/insy/modulefiles
module purge

module load miniconda/3.7
conda activate $conda_env

module load cuda/10.0
module load cudnn/10.0-7.6.0.64


# Loop variables
var_1=model_class
array_1=(ResNet152)
var_2=unfreeze_layer
array_2=(0)
var_3=top_n
array_3=(9)
var_4=feature_layer
array_4=(avgpool)
var_5=meta_trn
array_5=(80)
var_6=encoder_trn
array_6=(0)
var_7=meta_val
array_7=(20)
var_8=meta_tst
array_8=(80)
len=${#array_1[@]}


# conda activate $conda_env

var_n=name
array_n=(setup_data_rn152)


for ((i=0;i<$len; i++))

do
	config_file=output/${array_n[$i]}/${array_n[$i]}_config.yaml

	mkdir -p output/${array_n[$i]}
	cp -r config/$base_config_file $config_file

	echo "$var_1 = ${array_1[$i]}"
	echo "$var_2 = ${array_2[$i]}"
    echo "$var_3 = ${array_3[$i]}"
    echo "$var_4 = ${array_4[$i]}"
    echo "$var_5 = ${array_5[$i]}"
    echo "$var_6 = ${array_6[$i]}"
    echo "$var_7 = ${array_7[$i]}"
    echo "$var_8 = ${array_8[$i]}"



	sed -i "s/$var_1:.*/$var_1: ${array_1[$i]}/"  $config_file
	sed -i "s/$var_2:.*/$var_2: ${array_2[$i]}/" $config_file
	sed -i "s/$var_3:.*/$var_3: ${array_3[$i]}/" $config_file
	sed -i "s/$var_4:.*/$var_4: ${array_4[$i]}/" $config_file
    sed -i "s/$var_5:.*/$var_5: ${array_5[$i]}/" $config_file
    sed -i "s/$var_6:.*/$var_6: ${array_6[$i]}/" $config_file
    sed -i "s/$var_7:.*/$var_7: ${array_7[$i]}/" $config_file

    sed -i "s/$var_8:.*/$var_8: ${array_8[$i]}/" $config_file

	python $python_script_1 $config_file

done


conda deactivate
