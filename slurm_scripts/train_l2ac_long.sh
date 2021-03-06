#!/bin/bash
#SBATCH --job-name=train_l2ac	# create a short name for your job
#SBATCH --output=logs/%x-%j.out                 # output_file
#SBATCH --partition=general				# select partition
#SBATCH --qos=long						# select quality of service
#SBATCH --nodes=1                		# node count
#SBATCH --ntasks=1               		# total number of tasks across all nodes
#SBATCH --cpus-per-task=6        		# cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=14gb                		# total memory per node (4 GB per cpu-core is default)
#SBATCH --gres=gpu:2             		# number of gpus per node
#SBATCH --time=07:00:00          		# total run time limit (HH:MM:SS)
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
config_file=L2AC_train.yaml
conda_env=l2acenv


# Loop variables
var_1=name	
array_1=(0026)
var_2=batch_size
array_2=(10)
var_3=model_class
array_3=(L2AC)
#array_3=(L2AC L2AC_cosine L2AC_no_lstm L2AC_extended_similarity)
len=${#array_1[@]}




var_4=epochs
value_4=300

conda activate $conda_env


for ((i=0;i<$len; i++))

do
	config_file=output/${array_1[$i]}/${array_1[$i]}_config.yaml

	mkdir -p output/${array_1[$i]}
	cp -r config/$base_config_file $config_file

        echo "$var_4 = ${value_4}"
        sed -i "s/$var_4:.*/$var_4: ${value_4}/" $config_file
	

	echo "$var_1 = ${array_1[$i]}"
	echo "$var_2 = ${array_2[$i]}"
        echo "$var_3 = ${array_3[$i]}"

	sed -i "s/$var_1:.*/$var_1: '${array_1[$i]}'/"  $config_file
	sed -i "s/$var_2:.*/$var_2: ${array_2[$i]}/" $config_file
	sed -i "s/$var_3:.*/$var_3: ${array_3[$i]}/" $config_file


	python $python_script $config_file

done

conda deactivate
