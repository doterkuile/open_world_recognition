#!/bin/bash
#SBATCH --job-name=example_script  	# create a short name for your job
#SBATCH --partition= general			# select partition
#SBATCH --qos= short					# select quality of service
#SBATCH --nodes=1                		# node count
#SBATCH --ntasks=1               		# total number of tasks across all nodes
#SBATCH --cpus-per-task=1        		# cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=4096                		# total memory per node (4 GB per cpu-core is default)
#SBATCH --gres=gpu:1             		# number of gpus per node
#SBATCH --time=00:20:00          		# total run time limit (HH:MM:SS)
#SBATCH --mail-type=begin        		# send mail when job begins
#SBATCH --mail-type=end          		# send mail when job ends
#SBATCH --mail-type=fail         		# send mail if job fails
#SBATCH --mail-user=doterkuile@gmail.com

python_script=
config_file=

conda_env=
package=
data_path=

mkdir -p /tmp/$USER && cp -a $data_path$package /tmp/$USER

module use /opt/insy/modulefiles
module purge

module load miniconda/3.7
conda activate $conda_env

python $python_script $config_file