from open_world import ObjectDatasets
from open_world import OpenWorldUtils
from open_world import RecognitionModels

import torch
import os
import torch.nn as nn

ENABLE_TRAINING = False
SAVE_IMAGES = True


def main():

	if not torch.cuda.is_available():
		print("Cuda device not available make sure CUDA has been installed")
		return
	torch.manual_seed(42)
	dataset_path = 'datasets/MNIST'
	model_path = '../networks/MNIST/test_model.pt'
	batch_size = 10
	learning_rate = 0.001
	epochs = 1


	## Setup dataset
	dataset = ObjectDatasets.MNISTDataset(dataset_path)
	(train_data, test_data) = dataset.getData()
	(train_loader, test_loader) = dataset.getDataloaders(batch_size)


	## Setup model
	model = RecognitionModels.MNISTNetwork().cuda()
	if not ENABLE_TRAINING:
		OpenWorldUtils.loadModel(model, model_path)

	criterion = nn.CrossEntropyLoss()
	optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
	if ENABLE_TRAINING:
		OpenWorldUtils.trainModel(model, train_loader, test_loader, epochs, criterion, optimizer)
		OpenWorldUtils.saveModel(model, model_path)
	OpenWorldUtils.testModel(model, test_data)

	return





if __name__ == "__main__":
    main()