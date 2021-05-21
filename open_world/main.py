from open_world import ObjectDatasets
from open_world import OpenWorldUtils
from open_world import RecognitionModels

import torch
import os
import torch.nn as nn
import time

ENABLE_TRAINING = False
SAVE_IMAGES = True


def main():

	if not torch.cuda.is_available():
		print("Cuda device not available make sure CUDA has been installed")
		return
	torch.manual_seed(42)

	# config_file_path = '../config/MNIST_fashion_test.yaml'
	# config_file_path = '../config/MNIST_test.yaml'
	config_file_path = '../config/CATDOG_test.yaml'

	# Parse config file
	(dataset, model, criterion, optimizer, config) = OpenWorldUtils.parseConfigFile(config_file_path, ENABLE_TRAINING)

	batch_size = config['batch_size']
	epochs = config['epochs']
	training_history_path = config['training_history_path']
	figure_path = config['figures_path']




	## Setup dataset
	(train_data, test_data) = dataset.getData()
	(train_loader, test_loader) = dataset.getDataloaders(batch_size)

	if ENABLE_TRAINING:
		(train_losses, test_losses, train_correct, test_correct) = OpenWorldUtils.trainModel(model, train_loader, test_loader, epochs, criterion, optimizer)
		OpenWorldUtils.saveTrainingLosses(train_losses, test_losses, train_correct, test_correct, training_history_path)
		OpenWorldUtils.saveModel(model, model.model_path)

	# n_training = len(train_loader.dataset)
	# n_test = len(test_loader.dataset)
	# OpenWorldUtils.plotLosses(training_history_path, n_training, n_test, figure_path)

	for i in range(0, 10):
		OpenWorldUtils.testModel(model, dataset)
		time.sleep(1)

	return





if __name__ == "__main__":
    main()