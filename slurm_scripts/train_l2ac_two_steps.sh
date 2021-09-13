#!/bin/bash
#SBATCH --job-name=train_l2ac_two_step 	# create a short name for your job
#SBATCH --output=logs/%x-%j.out                 # output_file
#SBATCH --partition=general				# select partition
#SBATCH --qos=long						# select quality of service
#SBATCH --nodes=1                		# node count
#SBATCH --ntasks=1               		# total number of tasks across all nodes
#SBATCH --cpus-per-task=6       		# cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=15gb                		# total memory per node (4 GB per cpu-core is default)
#SBATCH --gres=gpu:1             		# number of gpus per node
#SBATCH --time=25:00:00          		# total run time limit (HH:MM:SS)
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
conda_env=l2acenv



# Loop variables
var_1=name	
array_1=(l_t_step_0001)
var_2=top_n
array_2=(9)
var_3=model_class
array_3=(L2AC_smaller_fc)
#array_3=(L2AC L2AC_cosine L2AC_no_lstm L2AC_extended_similarity L2AC_smaller_fc L2AC_abssub L2AC_concat)

var_4=encoder
array_4=(EfficientNet)
var_5=two_step_training
array_5=(True)
var_6=meta_trn
array_6=(80)
var_7=encoder_trn
array_7=(0)
var_8=unfreeze_layer
array_8=(0)
var_9=feature_layer
array_9=(_avg_pooling)
len=${#array_1[@]}




var_e=epochs
value_e=400

conda activate $conda_env


for ((i=0;i<$len; i++))

do
	config_file=output/${array_1[$i]}/${array_1[$i]}_config.yaml

	mkdir -p output/${array_1[$i]}
	cp -r config/$base_config_file $config_file

    echo "$var_e = ${value_e}"
    sed -i "s/$var_e:.*/$var_e: ${value_e}/" $config_file
	

	echo "$var_1 = ${array_1[$i]}"
	echo "$var_2 = ${array_2[$i]}"
    echo "$var_3 = ${array_3[$i]}"
    echo "$var_4 = ${array_4[$i]}"
    echo "$var_5 = ${array_5[$i]}"
    echo "$var_6 = ${array_6[$i]}"
    echo "$var_7 = ${array_7[$i]}"
    echo "$var_8 = ${array_8[$i]}"
    echo "$var_9 = ${array_9[$i]}"


	sed -i "s/$var_1:.*/$var_1: '${array_1[$i]}'/"  $config_file
	sed -i "s/$var_2:.*/$var_2: ${array_2[$i]}/" $config_file
	sed -i "s/$var_3:.*/$var_3: ${array_3[$i]}/" $config_file
	sed -i "s/$var_4:.*/$var_4: ${array_4[$i]}/" $config_file
	sed -i "s/$var_5:.*/$var_5: ${array_5[$i]}/" $config_file
	sed -i "s/$var_6:.*/$var_6: ${array_6[$i]}/" $config_file
	sed -i "s/$var_7:.*/$var_7: ${array_7[$i]}/" $config_file
	sed -i "s/$var_8:.*/$var_8: ${array_8[$i]}/" $config_file
	sed -i "s/$var_9:.*/$var_9: ${array_9[$i]}/" $config_file


	python $python_script $config_file

done

conda deactivate
