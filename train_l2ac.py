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
import shutil


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
		print(f"Running with {torch.cuda.device_count()} GPUs")
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

	## Create new entry folder for results of experiment
	exp_name='no_name'
	try:
		exp_name = str(config['name'])
	except KeyError:
		print(f'No exp name was given, continuing with no_name')

	exp_folder = 'output/' + exp_name

	if not os.path.exists(exp_folder):
		os.makedirs(exp_folder)
	figure_path = exp_folder + '/' + exp_name
	results_path = exp_folder + '/' + exp_name + '_results.npz'
	model_path = exp_folder + '/' + exp_name + '_model.pt'
	config_save_path = exp_folder + '/' + exp_name + '_config.yaml'

	# Save config file in the exp directory
	shutil.copyfile('config/' + config_file, config_save_path)




	# Get hyperparameters
	train_classes = config['train_classes']
	train_samples_per_cls = config['train_samples_per_cls']
	probability_treshold = config['probability_threshold']



	train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True)

	test_dataset = ObjectDatasets.MetaDataset(config['dataset_path'], config['top_n'], config['top_k'],
											  train_classes, train_samples_per_cls, train=False)
	test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)

	(trn_metrics, tst_metrics) = meta_utils.trainMetaModel(model, train_loader, test_loader, epochs,
																			   criterion, optimizer, device, probability_treshold)

	# Train metrics
	trn_loss = trn_metrics['loss']
	trn_acc = trn_metrics['accuracy']
	trn_precision = trn_metrics['precision']
	trn_recall = trn_metrics['recall']
	trn_F1 = trn_metrics['F1']
	trn_mean_pred = trn_metrics['mean_pred']
	trn_mean_true = trn_metrics['mean_true']


	# Test metrics
	tst_loss = tst_metrics['loss']
	tst_acc = tst_metrics['accuracy']
	tst_precision = tst_metrics['precision']
	tst_recall = tst_metrics['recall']
	tst_F1 = tst_metrics['F1']
	tst_mean_pred = tst_metrics['mean_pred']
	tst_mean_true = tst_metrics['mean_true']


	# Plot metrics
	plot_utils.plot_losses(trn_loss, tst_loss, figure_path)
	plot_utils.plot_accuracy(trn_acc, tst_acc, figure_path)
	plot_utils.plot_precision(trn_precision, tst_precision, figure_path)
	plot_utils.plot_recall(trn_recall, tst_recall, figure_path)
	plot_utils.plot_F1(trn_F1, tst_F1, figure_path)
	plot_utils.plot_mean_prediction(trn_mean_pred, trn_mean_true, tst_mean_pred, tst_mean_true, figure_path)

	OpenWorldUtils.saveModel(model, model_path)

	np.savez(results_path, train_loss=trn_loss, test_loss=tst_loss, train_acc=trn_acc, test_acc=tst_acc,
			 train_precision=trn_precision, test_precision=tst_precision, train_recall=trn_recall,
			 test_recall=tst_recall, train_F1=trn_F1, test_F1=tst_F1)

	return





if __name__ == "__main__":
    main()
