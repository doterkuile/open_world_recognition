import torch
import torch.utils.data as data_utils
from open_world import OpenWorldUtils
from open_world import ObjectDatasets
import open_world.meta_learner.meta_learner_utils as meta_utils
import open_world.plot_utils as plot_utils
import yaml
import numpy as np
from torch.utils.data import DataLoader
import argparse


def main():
	# set random seed
	torch.manual_seed(42)

	# Main gpu checks
	multiple_gpu = True if torch.cuda.device_count() > 1 else False
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	if not torch.cuda.is_available():
		print("Cuda device not available make sure CUDA has been installed")
		return
	else:
		print(f'Running with {torch.cuda.device_count()} GPUs')
	# Get config file argument
	parser = argparse.ArgumentParser()
	parser.add_argument("config_file")
	args = parser.parse_args()
	config_file = args.config_file

	# Overwrite terminal argument if necessary
	# config_file = 'config/L2AC_train.yaml'

	# Parse config file
	(dataset, model, criterion, optimizer, epochs, batch_size, learning_rate, config) = OpenWorldUtils.parseConfigFile(
		config_file, device, multiple_gpu)

	train_classes = config['train_classes']
	train_samples_per_cls = config['train_samples_per_cls']

	train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True)

	test_dataset = ObjectDatasets.MetaDataset(config['dataset_path'], config['top_n'], config['top_k'],
											  train_classes, train_samples_per_cls,rain=False)
	test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)

	figure_path = config['figures_path'] + 'L2AC'
	(train_loss, test_loss, train_correct, test_correct) = meta_utils.trainMetaModel(model, train_loader, test_loader, epochs, criterion, optimizer, device)
	plot_utils.plot_losses(train_loss, test_loss, figure_path)
	return





if __name__ == "__main__":
    main()
