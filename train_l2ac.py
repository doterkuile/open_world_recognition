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
import matplotlib.pyplot as plt




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
	sim_path = exp_folder + '/' + exp_name + '_similarities.npz'
	model_path = exp_folder + '/' + exp_name + '_model.pt'
	config_save_path = exp_folder + '/' + exp_name + '_config.yaml'

	# Save config file in the exp directory
	shutil.copyfile('config/' + config_file, config_save_path)




	# Get hyperparameters
	train_classes = config['train_classes']
	train_samples_per_cls = config['train_samples_per_cls']
	probability_treshold = config['probability_threshold']
	create_similarity_gif = config['create_similarity_gif']




	train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True)

	test_dataset = ObjectDatasets.MetaDataset(config['dataset_path'], config['top_n'], config['top_k'],
											  train_classes, train_samples_per_cls, train=False)
	test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)

	gif_path = None

	if create_similarity_gif:
		gif_path = exp_folder + '/' + exp_name

	trn_metrics, trn_similarity_scores, tst_metrics, tst_similarity_scores = meta_utils.trainMetaModel(model, train_loader, test_loader, epochs,
																			   criterion, optimizer, device, probability_treshold, gif_path)

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


	# Plot similarities
	trn_final_same_cls = trn_similarity_scores['final_same_cls']
	trn_intermediate_same_cls = trn_similarity_scores['intermediate_same_cls']
	trn_final_diff_cls = trn_similarity_scores['final_diff_cls']
	trn_intermediate_diff_cls = trn_similarity_scores['intermediate_diff_cls']
	tst_final_same_cls = tst_similarity_scores['final_same_cls']
	tst_intermediate_same_cls = tst_similarity_scores['intermediate_same_cls']
	tst_final_diff_cls = tst_similarity_scores['final_diff_cls']
	tst_intermediate_diff_cls = tst_similarity_scores['intermediate_diff_cls']

	plot_utils.plot_intermediate_similarity(trn_intermediate_same_cls,trn_intermediate_diff_cls, tst_intermediate_same_cls, tst_intermediate_diff_cls, figure_path)
	plot_utils.plot_final_similarity(trn_final_same_cls,trn_final_diff_cls, tst_final_same_cls, tst_final_diff_cls, figure_path)

	trn_y_pred, trn_y_true, trn_losses, trn_sim_scores, trn_y_pred_raw = meta_utils.validate_model(train_loader, model,
																								 criterion, device,
																								 probability_treshold)
	tst_y_pred, tst_y_true, tst_losses, tst_sim_scores, tst_y_pred_raw = meta_utils.validate_model(test_loader, model,
																								 criterion, device,
																								 probability_treshold)

	trn_y_pred = np.array(torch.cat(trn_y_pred))
	trn_y_pred_raw = np.array(torch.cat(trn_y_pred_raw))
	trn_y_true = np.array(torch.cat(trn_y_true))

	trn_sim_scores = np.array(torch.cat(trn_sim_scores, dim=1).detach()).transpose(1, 0)

	tst_y_pred = np.array(torch.cat(tst_y_pred))
	tst_y_pred_raw = np.array(torch.cat(tst_y_pred_raw))
	tst_y_true = np.array(torch.cat(tst_y_true))
	tst_sim_scores = np.array(torch.cat(tst_sim_scores, dim=1).detach()).transpose(1, 0)

	fig_sim, axs_sim = plt.subplots(2, 1, figsize=(15, 10))
	fig_final, axs_final = plt.subplots(2, 1, figsize=(15, 10))



	title = 'Intermediate similarity score'
	plot_utils.plot_prob_density(fig_sim, axs_sim, trn_sim_scores, trn_y_true, tst_sim_scores, tst_y_true, title,
								 figure_path + '_intermediate_similarity')


	title = 'Final similarity score'
	plot_utils.plot_prob_density(fig_final, axs_final, trn_y_pred_raw, trn_y_true, tst_y_pred_raw, tst_y_true, title,
								 figure_path + '_final_similarity')


	OpenWorldUtils.saveModel(model, model_path)

	np.savez(results_path, train_loss=trn_loss, test_loss=tst_loss, train_acc=trn_acc, test_acc=tst_acc,
			 train_precision=trn_precision, test_precision=tst_precision, train_recall=trn_recall,
			 test_recall=tst_recall, train_F1=trn_F1, test_F1=tst_F1)

	np.savez(sim_path,
			 trn_final_same_cls=trn_similarity_scores['final_same_cls'],
			 trn_intermediate_same_cls=trn_similarity_scores['intermediate_same_cls'],
			 trn_final_diff_cls=trn_similarity_scores['final_diff_cls'],
			 trn_intermediate_diff_cls=trn_similarity_scores['intermediate_diff_cls'],
			 tst_final_same_cls=tst_similarity_scores['final_same_cls'],
			 tst_intermediate_same_cls=tst_similarity_scores['intermediate_same_cls'],
			 tst_final_diff_cls=tst_similarity_scores['final_diff_cls'],
			 tst_intermediate_diff_cls=tst_similarity_scores['intermediate_diff_cls'],
			 )


	return





if __name__ == "__main__":
    main()
