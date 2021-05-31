import torch
import torch.utils.data as data_utils
from open_world import OpenWorldUtils
from open_world import ObjectDatasets
import open_world.meta_learner.meta_learner_utils as meta_utils
import yaml
import numpy as np
from torch.utils.data import DataLoader


ENABLE_TRAINING = True

def main():

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	torch.manual_seed(42)
	load_data = True
	config_file = 'config/L2AC_train.yaml'


	# Parse config file
	(dataset, model, criterion, optimizer, epochs, batch_size, learning_rate, config) = OpenWorldUtils.parseConfigFile(config_file, ENABLE_TRAINING)
	train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
	testdataset = ObjectDatasets.MetaDataset(config['dataset_path'], config['top_n'], config['top_k'], train=False)
	test_loader = DataLoader(testdataset, batch_size=batch_size, shuffle=True, pin_memory=True)

	meta_utils.trainMetaModel(model, train_loader, test_loader, epochs, criterion, optimizer)


	# Setup dataset
	# (train_data, test_data) = dataset.getData()
	# train_loader, test_loader = dataset.getDataloaders(batch_size)
	classes = dataset.classes
	# train = data_utils.TensorDataset()
	ncls = 9	# Top similar classes
	train_per_cls = 100 # Number of samples per class

	top_n = 5 # Top samples used for comparison
	class_set = classes

	num_train = 80 # Classes used for training
	num_valid = 20 # Classes used for validation




	return





if __name__ == "__main__":
    main()
