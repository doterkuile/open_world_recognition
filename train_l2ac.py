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
import os


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
	max_trn_batch = config['max_trn_batch']
	probability_treshold = config['probability_threshold']
	exp_name='no_name'
	try:
		exp_name = config['name']
	except KeyError:
		print(f'No exp name was given, continuing with no_name')



	train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True)

	test_dataset = ObjectDatasets.MetaDataset(config['dataset_path'], config['top_n'], config['top_k'],
											  train_classes, train_samples_per_cls, train=False)
	test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)

	figure_path = config['figures_path'] + exp_name
	(train_loss, test_loss, train_accs, test_accs) = meta_utils.trainMetaModel(model, train_loader, test_loader, epochs,
																			   criterion, optimizer, device, max_trn_batch, probability_treshold)
	plot_utils.plot_losses(train_loss, test_loss, figure_path)
	plot_utils.plot_accuracy(train_accs, test_accs, figure_path)

	if not os.path.exists(config['training_history_path']):
		os.makedirs(config['training_history_path'])
	results_path = config['training_history_path'] + '_' + exp_name + '_results.npz'
	np.savez(results_path, train_loss=train_loss, test_loss=test_loss, train_accs=train_accs, test_accs=test_accs)

	return





if __name__ == "__main__":
    main()
