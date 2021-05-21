from open_world import ObjectDatasets
from open_world import OpenWorldUtils
from open_world import RecognitionModels

import torch
import sklearn

import os
import torch.nn as nn
import time

import numpy as np

ENABLE_TRAINING = False
SAVE_IMAGES = True





def extract_features(train_data, model, load_data):

	classes = [train_data.class_to_idx[key] for key in train_data.class_to_idx.keys()]

	class_samples = {key: None for key in classes}

	train_rep, train_cls_rep = [], []
	if not load_data:
		with torch.no_grad():

			for cls in classes:
				class_samples[cls] = torch.stack([s[0] for s in train_data if s[1] == cls])


			for cls in classes:
				cls_rep = model.getFeatureExtractor(class_samples[cls].view(-1, 1, 28, 28).cuda()).cpu()
				train_rep.append(cls_rep)
				train_cls_rep.append(cls_rep.mean(dim=0))
			train_rep = torch.cat(train_rep)
			train_cls_rep = torch.stack(train_cls_rep)

			np.savez( "memory.npz",data_rep=train_rep, train_cls_rep=train_cls_rep)
	else:
		train_rep = np.load("memory.npz")['data_rep']
		train_cls_rep = np.load("memory.npz")['train_cls_rep']
	return (train_rep, train_cls_rep)

def data2np_train_idx_neg_cls(class_set, data_rep, data_cls_rep, classes, train_per_cls, top_k):
	X0, X1, Y = [], [], []
	base_cls_offset = classes.index(class_set[0])  # -> gives index of first class of interest
	for cls in class_set:
		tmp_X1 = []
		tmp_Y = []

		# index of class of interest
		ix = classes.index(cls)  # considers validation_set that not start from zero.
		cls_offset = ix * train_per_cls  # Some kind of offset?

		# find top_k non similar classes.
		rest_cls_idx = [classes.index(cls1) for cls1 in class_set if
						classes.index(cls1) != ix]  # Find all remaining classes

		# cosine similarity between train_per_class number of the class of interest and the mean feature vector of the rest of the classes
		# Finds the most similar classes based on the mean value
		sim = sklearn.metrics.pairwise.cosine_similarity(data_rep[cls_offset:cls_offset + train_per_cls],
														 data_cls_rep[rest_cls_idx])

		# Get indices of a sorted array
		sim_idx = sim.argsort(axis=1)
		# Add offset of the base class
		sim_idx += base_cls_offset
		# Plus one idx to correct for the removed class of interest
		sim_idx[sim_idx >= ix] += 1


		# Get thhe top classes from the sorted similarities
		sim_idx = sim_idx[:, -top_k:]

		# Loop over the k most similar classes based on the previous cosine similarity
		for kx in range(-top_k, 0):
			tmp_X1_batch = []
			# Train per class is still unclear. Total number of samples used per class?
			for jx in range(train_per_cls):
				# Same offset with unknown purpose
				cls1_offset = sim_idx[jx, kx] * train_per_cls
				# Find cosine similarity between the two offsets? Gives an array with size [1, train_per_cls]
				sim1 = sklearn.metrics.pairwise.cosine_similarity(data_rep[cls_offset + jx:cls_offset + jx + 1],
																	  data_rep[cls1_offset:cls1_offset + train_per_cls])
				# Sort indices and find most similar samples
				sim1_idx = sim1.argsort(axis=1)[:1, -(train_per_cls - 1):]

				sim1_idx += cls1_offset
				# Give size a second dimension, useful for vstack i think
				tmp_X1_batch.append(np.expand_dims(sim1_idx, 1))
			tmp_X1_batch = np.vstack(tmp_X1_batch)

			# Append indices and labels
			tmp_X1.append(tmp_X1_batch)
			tmp_Y.append(np.full((train_per_cls, 1), 0))

		# put sim in the last dim
		sim = sklearn.metrics.pairwise.cosine_similarity(
			data_rep[cls_offset:cls_offset + train_per_cls])  # Similarity between same class
		sim_idx = sim.argsort(axis=1)[:, :-1] + cls_offset  # add the offset to obtain the real offset in memory.
		# Append same class indices and labels to tmp_x1 and tmp_y
		tmp_X1.append(np.expand_dims(sim_idx, 1))
		tmp_Y.append(np.full((train_per_cls, 1), 1))

		# append all input samples indices
		X0.append(np.arange(cls_offset, cls_offset + train_per_cls).reshape(-1, 1))

		# make matrix with indices for all comparison samplles
		X1.append(np.concatenate(tmp_X1, 1))
		Y.append(np.concatenate(tmp_Y, axis=1))  # similar

	X0 = np.vstack(X0)
	X1 = np.vstack(X1)
	Y = np.concatenate(Y)
	shuffle_idx = np.random.permutation(X0.shape[0])
	return X0[shuffle_idx], X1[shuffle_idx], Y[shuffle_idx]


def main():

	if not torch.cuda.is_available():
		print("Cuda device not available make sure CUDA has been installed")
		return
	torch.manual_seed(42)
	load_data = True
	# # config_file = '../config/MNIST_fashion_test.yaml'
	config_file = '../config/MNIST_test.yaml'
	# config_file = '../config/CATDOG_test.yaml'
    #
	# # Parse config file
	(dataset, model, criterion, optimizer, epochs, batch_size, learning_rate) = OpenWorldUtils.parseConfigFile(config_file, ENABLE_TRAINING)

    #
    #
	# ## Setup dataset
	(train_data, test_data) = dataset.getData()
	(train_loader, test_loader) = dataset.getDataloaders(batch_size)
	classes = [cls.split(' -')[0] for cls in train_data.classes]


	(data_rep, data_cls_rep) = extract_features(train_data,model,load_data)

	top_k = 5	# Top similar classes
	train_per_cls = 100 # Number of samples per class

	class_set = classes

	num_train = 9 # Number of classes used for training
	num_valid = 1


	train_X0, train_X1, train_Y = data2np_train_idx_neg_cls(class_set[:num_train], data_rep, data_cls_rep, classes, train_per_cls, top_k)
	valid_X0, valid_X1, valid_Y = data2np_train_idx_neg_cls(class_set[-num_valid:], data_rep, data_cls_rep, classes, train_per_cls, top_k)

	# class_vectors = calculate_class_vector(features,classes, labels)
	# classes =
    #
	# if ENABLE_TRAINING:
	# 	OpenWorldUtils.trainModel(model, train_loader, test_loader, epochs, criterion, optimizer)
	# 	OpenWorldUtils.saveModel(model, model.model_path)
    #
	# for i in range(0, 10):
	# 	OpenWorldUtils.testModel(model, dataset)
	# 	time.sleep(1)

	return





if __name__ == "__main__":
    main()