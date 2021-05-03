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

	# config_file = '../config/MNIST_fashion_test.yaml'
	# config_file = '../config/MNIST_test.yaml'
	config_file = '../config/CATDOG_test.yaml'

	# Parse config file
	(dataset, model, criterion, optimizer, epochs, batch_size, learning_rate) = OpenWorldUtils.parseConfigFile(config_file, ENABLE_TRAINING)


	## Setup dataset
	(train_data, test_data) = dataset.getData()
	(train_loader, test_loader) = dataset.getDataloaders(batch_size)

	if ENABLE_TRAINING:
		OpenWorldUtils.trainModel(model, train_loader, test_loader, epochs, criterion, optimizer)
		OpenWorldUtils.saveModel(model, model.model_path)

	for i in range(0, 10):
		OpenWorldUtils.testModel(model, dataset)
		time.sleep(1)

	return





if __name__ == "__main__":
    main()