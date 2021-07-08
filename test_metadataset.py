import torch
import torch.utils.data as data_utils
from torch.utils.data import DataLoader

from open_world import OpenWorldUtils
from open_world import ObjectDatasets
import open_world.meta_learner.meta_learner_utils as meta_utils
import yaml
import numpy as np

ENABLE_TRAINING = True

def main():

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	torch.manual_seed(42)
	load_data = True
	config_file = 'config/L2AC_train.yaml'
	top_n = 6
	ncls = 10
	batch_size = 7

	with open(config_file) as file:
		config = yaml.load(file, Loader=yaml.FullLoader)

	data_path = f'datasets/{config["dataset_path"]}/train_idx.npz'
	metadataset = ObjectDatasets.MetaDataset(data_path, ncls, top_n)
	# metadataset.__getitem__(0)

	train_loader = DataLoader(metadataset, batch_size=batch_size, shuffle=True, pin_memory=True)
	(dataset, model, criterion, optimizer, epochs, batch_size, learning_rate) = OpenWorldUtils.parseConfigFile(config_file, ENABLE_TRAINING)

	for b, ([X0_train,X1_train], y_train) in enumerate(train_loader):
		break


	model.forward(X0_train.cuda(), X1_train.cuda())
	#
	# # Parse config file
	#
	# # Setup dataset
	# (train_data, test_data) = dataset.getData()
	# train_loader, test_loader = dataset.getDataloaders(batch_size)
	# classes = dataset.classes
	# # train = data_utils.TensorDataset()
	# ncls = 9	# Top similar classes
	# train_per_cls = 100 # Number of samples per class
	#
	# top_n = 5 # Top samples used for comparison
	# class_set = classes
	#
	# num_train = 80 # Classes used for training
	# num_valid = 20 # Classes used for validation
	#
	#
	#
	# data = np.load(data_path)
	#
	# train_X0 = np.repeat(data['train_X0'], ncls, axis=0)
	# train_X1 = data['train_X1'][:, -ncls:, -top_n:].reshape(-1, top_n)
	# train_Y = data['train_Y'][:, -ncls:].reshape(-1, )
	# valid_X0 = np.repeat(data['valid_X0'], 2, axis=0)  # the validation data is balanced.
	# valid_X1 = data['valid_X1'][:, -2:, -top_n:].reshape(-1, top_n)
	# valid_Y = data['valid_Y'][:, -2:].reshape(-1, )
	# data_rep = data['train_rep']
	#
	# input_feature = torch.tensor(data_rep[train_X0[0]]).cuda()
	# comparison_features = torch.tensor(data_rep[train_X1[0]]).cuda()
	#
	#
	# model.forward(input_feature, comparison_features)


	return





if __name__ == "__main__":
    main()
