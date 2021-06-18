import torch
from sklearn import metrics

import torch.utils.data as data_utils
from open_world import OpenWorldUtils
from open_world import ObjectDatasets
import open_world.meta_learner.meta_learner_utils as meta_utils
import open_world.plot_utils as plot_utils
import yaml
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import argparse
import os
import shutil
from torchvision.models import resnet50
from torchvision import datasets, transforms
import time
from tqdm import tqdm


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

	# Parse config file
	(dataset, model, criterion, optimizer, epochs, batch_size, learning_rate, config) = OpenWorldUtils.parseConfigFile(
		config_file, device, multiple_gpu)

	dataset_path = config['dataset_path']
	train_data = datasets.CIFAR100(root=dataset_path, train=True, download=True, transform=dataset.transform_train)
	test_data = datasets.CIFAR100(root=dataset_path, train=False, download=True, transform=dataset.transform_test)

	test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True, pin_memory=True)

	train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, pin_memory=True)

	trn_metrics, tst_metrics = trainMetaModel(model, train_loader, test_loader, epochs, criterion,optimizer,device)

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

	figure_path = 'figures/RESNET/resnet_'
	results_path = 'figures/RESNET/result.npz'
	# Plot metrics
	plot_utils.plot_losses(trn_loss, tst_loss, figure_path)
	plot_utils.plot_accuracy(trn_acc, tst_acc, figure_path)
	plot_utils.plot_precision(trn_precision, tst_precision, figure_path)
	plot_utils.plot_recall(trn_recall, tst_recall, figure_path)
	plot_utils.plot_F1(trn_F1, tst_F1, figure_path)
	plot_utils.plot_mean_prediction(trn_mean_pred, trn_mean_true, tst_mean_pred, tst_mean_true, figure_path)

	# OpenWorldUtils.saveModel(model, model_path)

	np.savez(results_path, train_loss=trn_loss, test_loss=tst_loss, train_acc=trn_acc, test_acc=tst_acc,
			 train_precision=trn_precision, test_precision=tst_precision, train_recall=trn_recall,
			 test_recall=tst_recall, train_F1=trn_F1, test_F1=tst_F1)

	return

def trainMetaModel(model, train_loader, test_loader, epochs, criterion, optimizer, device):
	start_time = time.time()

	trn_losses = []
	tst_losses = []
	trn_acc = []
	tst_acc = []
	trn_precision = []
	tst_precision = []
	trn_recall = []
	tst_recall = []
	trn_F1 = []
	tst_F1 = []
	trn_mean_pred = []
	trn_mean_true = []
	tst_mean_pred = []
	tst_mean_true = []

	for i in range(epochs):
		trn_corr = 0
		y_pred = []
		y_true = []

		# Run the training batches
		for b, (X0_train, y_train) in tqdm(enumerate(train_loader), total=int(len(train_loader.dataset)/train_loader.batch_size)):
			X0_train = X0_train.to(device)
			# X1_train = X1_train.to(device)
			y_train = y_train.view(-1, 1).to(device)

			# Limit the number of batches
			if b == (len(train_loader) - 2):
				break
			b += 1

			# Apply the model
			y_out = model(X0_train)
			y_out = y_out.sigmoid()
			y_out = F.log_softmax(y_out, dim=1)
			trn_loss = criterion(y_out, y_train.squeeze(1))

			# Tally the number of correct predictions
			predicted = torch.max(y_out.data, 1)[1]

			batch_corr = (predicted == y_train.squeeze()).sum()
			trn_corr += batch_corr

			y_pred.extend(predicted.cpu())
			y_true.extend(y_train.cpu())
			# Update parameters
			optimizer.zero_grad()
			trn_loss.backward()
			optimizer.step()
		# torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)

		# Print epoch results
		print(
			f'epoch: {i:2}  batch: {b:4} [{train_loader.batch_size * b:6}/{len(train_loader) * train_loader.batch_size}]'
			f'  loss: {trn_loss.item():10.8f} accuracy: {trn_corr.item() * 100 / (train_loader.batch_size * b):7.3f}%')

		# Training metrics
		y_pred = np.array(torch.stack(y_pred))
		y_true = np.array(torch.stack(y_true))

		trn_acc.append(metrics.accuracy_score(y_true, y_pred))
		trn_precision.append(metrics.precision_score(y_true=y_true, y_pred=y_pred, average='weighted', zero_division=0))
		trn_recall.append(metrics.recall_score(y_true, y_pred, average='weighted'))
		trn_F1.append(metrics.f1_score(y_true=y_true, y_pred=y_pred, average='weighted', zero_division=0))

		trn_mean_pred.append(y_pred.mean())
		trn_mean_true.append(y_true.mean())

		trn_losses.append(trn_loss.item())

		# Run the testing batches
		y_pred, y_true, tst_loss = validate_model(test_loader, model, criterion, device)

		tst_acc.append(metrics.accuracy_score(y_true, y_pred))
		tst_precision.append(metrics.precision_score(y_true=y_true, y_pred=y_pred, average='weighted', zero_division=0))
		tst_recall.append(metrics.recall_score(y_true, y_pred, average='weighted'))
		tst_F1.append(metrics.f1_score(y_true=y_true, y_pred=y_pred, average='weighted', zero_division=0))

		tst_mean_pred.append(y_pred.mean())
		tst_mean_true.append(y_true.mean())

		tst_losses.append(tst_loss.item())

	trn_metrics = {'loss': trn_losses,
				   'accuracy': trn_acc,
				   'precision': trn_precision,
				   'recall': trn_recall,
				   'F1': trn_F1,
				   'mean_pred': trn_mean_pred,
				   'mean_true': trn_mean_true}

	tst_metrics = {'loss': tst_losses,
				   'accuracy': tst_acc,
				   'precision': tst_precision,
				   'recall': tst_recall,
				   'F1': tst_F1,
				   'mean_pred': tst_mean_pred,
				   'mean_true': tst_mean_true}

	print(f'\nDuration: {time.time() - start_time:.0f} seconds')  # print the time elapsed

	return trn_metrics, tst_metrics

def validate_model(loader, model, criterion, device):
		num_correct = 0
		num_samples = 0

		# Set model to eval
		model.eval()

		y_true = []
		y_pred = []

		with torch.no_grad():
			for b, (X0_test, y_test) in enumerate(loader):

				X0_test = X0_test.to(device)
				y_test = y_test.view(-1, 1).to(device)

				if b == (len(loader) - 1):
					break

				# Apply the model
				y_val = model(X0_test)
				loss = criterion(y_val, y_test.squeeze(1))

				predicted = torch.max(y_val.data, 1)[1]
				y_pred.extend(predicted.cpu())
				y_true.extend(y_test.cpu())
				num_correct += (predicted == y_test.squeeze(1)).sum()
				num_samples += predicted.size(0)
				b += 1

		y_pred = np.array(torch.stack(y_pred))
		y_true = np.array(torch.stack(y_true))
		# Toggle model back to train
		model.train()
		test_acc = num_correct.item() * 100 / (num_samples)
		print(f'test accuracy: {num_correct.item() * 100 / (num_samples):7.3f}%')
		return y_pred, y_true, loss



if __name__ == "__main__":
    main()
