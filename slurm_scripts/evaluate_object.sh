#!/bin/bash
#SBATCH --job-name=evaluate_object.py  	# create a short name for your job
#SBATCH --output=logs/%x-%j.out                 # output_file
#SBATCH --partition=general				# select partition
#SBATCH --qos=short					# select quality of service
#SBATCH --nodes=1                		# node count
#SBATCH --ntasks=1               		# total number of tasks across all nodes
#SBATCH --cpus-per-task=4      		# cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=12gb                		# total memory per node (4 GB per cpu-core is default)
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
python_script=evaluate_custom_object.py
base_config_file=config/new_object.yaml
conda_env=l2acenv

name_var=figure_title


conda activate $conda_env

var_1=experiment_name	
array_1=(l_c_e_0002)
var_2=memory_dataset
array_2=(webots_dataset_224)
var_3=input_dataset
array_3=(teapot_photos )

len=${#array_1[@]}

for ((i=0;i<$len; i++))

do

	# name=$(cat $base_config_file | grep $name_var: | tail -n1 | awk '{print $2}')
	# name=surface_plot_an_fn_cifar
	# echo "Run $name"



	config_file=results_teapot/${array_1[$i]}_${array_2[$i]}_${array_3[$i]}_config.yaml
	echo "New config file $config_file"

	mkdir -p results_teapot
	cp -r $base_config_file $config_file
	echo "copied config file to $config_file"

	echo "$var_1 = ${array_1[$i]}"
	echo "$var_2 = ${array_2[$i]}"
    echo "$var_3 = ${array_3[$i]}"

	sed -i "s/$var_1:.*/$var_1: '${array_1[$i]}'/"  $config_file
	sed -i "s/$var_2:.*/$var_2: ${array_2[$i]}/" $config_file
	sed -i "s/$var_3:.*/$var_3: ${array_3[$i]}/" $config_file

	python $python_script $config_file
done

conda deactivate
