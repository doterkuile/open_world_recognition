import torch
from open_world import OpenWorldUtils
import open_world.meta_learner.meta_learner_utils as meta_utils
import yaml
import numpy as np

ENABLE_TRAINING = True

def main():

	if not torch.cuda.is_available():
		print("Cuda device not available make sure CUDA has been installed")
		return
	torch.manual_seed(42)
	load_data = True
	config_file = 'config/L2AC_train.yaml'

	with open(config_file) as file:
		config = yaml.load(file, Loader=yaml.FullLoader)

	memory_path = config['dataset_path'] + '/memory.npz'

	# Parse config file
	(dataset, model, criterion, optimizer, epochs, batch_size, learning_rate) = OpenWorldUtils.parseConfigFile(config_file, ENABLE_TRAINING)

	# Setup dataset
	(train_data, test_data) = dataset.getData()
	classes = dataset.classes

	ncls = 9	# Top similar classes
	train_per_cls = 100 # Number of samples per class

	top_n = 5 # Top samples used for comparison
	class_set = classes

	num_train = 80 # Classes used for training
	num_valid = 20 # Classes used for validation

	data_path = config['dataset_path'] + "/train_idx.npz"
	data = np.load(data_path)

	train_X0 = np.repeat(data['train_X0'], ncls, axis=0)
	train_X1 = data['train_X1'][:, -ncls:, -top_n:].reshape(-1, top_n)
	train_Y = data['train_Y'][:, -ncls:].reshape(-1, )
	valid_X0 = np.repeat(data['valid_X0'], 2, axis=0)  # the validation data is balanced.
	valid_X1 = data['valid_X1'][:, -2:, -top_n:].reshape(-1, top_n)
	valid_Y = data['valid_Y'][:, -2:].reshape(-1, )
	data_rep = data['train_rep']

	input_feature = torch.tensor(data_rep[train_X0[0]]).cuda()
	comparison_features = torch.tensor(data_rep[train_X1[0]]).cuda()


	model.forward(input_feature, comparison_features)


	return





if __name__ == "__main__":
    main()
