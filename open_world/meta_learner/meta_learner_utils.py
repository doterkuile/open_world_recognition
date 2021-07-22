from sklearn import metrics
import numpy as np
import torch
import torch.nn as nn
import time
from tqdm import tqdm
import math
from open_world import plot_utils
from IPython.display import HTML
from celluloid import Camera
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
import imageio
import copy


def trainMetaModel(model, train_loader, test_loader, epochs, criterion, test_criterion, optimizer, device,
                   probability_treshold, gif_path=None):
    trn_metrics_dict = {'loss': [],
                        'accuracy': [],
                        'precision': [],
                        'recall': [],
                        'F1': [],
                        'mean_pred': [],
                        'mean_true': []}

    tst_metrics_dict = {'loss': [],
                        'accuracy': [],
                        'precision': [],
                        'recall': [],
                        'F1': [],
                        'mean_pred': [],
                        'mean_true': []}

    trn_similarity_scores = {'final_same_cls': [],
                             'intermediate_same_cls': [],
                             'final_diff_cls': [],
                             'intermediate_diff_cls': [],
                             }
    tst_similarity_scores = {'final_same_cls': [],
                             'intermediate_same_cls': [],
                             'final_diff_cls': [],
                             'intermediate_diff_cls': [],
                             }

    fig_sim_list = []
    fig_final_list = []

    best_model = copy.deepcopy(model.state_dict())
    best_F1 = -2.0
    best_epoch = 0
    best_state = {'model': best_model,
                  'F1': best_F1,
                  'epoch': best_epoch, }

    # Calculate metrics before training has started
    trn_y_pred, trn_y_true, trn_loss, trn_sim_scores, trn_y_pred_raw = validate_model(train_loader, model,
                                                                                      criterion, device,
                                                                                      probability_treshold)
    calculate_metrics(trn_metrics_dict, trn_y_pred, trn_y_true, trn_loss)

    # Calculate metrics before training has started
    tst_y_pred, tst_y_true, tst_loss, tst_sim_scores, tst_y_pred_raw = validate_model(test_loader, model,
                                                                                      test_criterion, device,
                                                                                      probability_treshold)
    calculate_metrics(tst_metrics_dict, tst_y_pred, tst_y_true, tst_loss)


    for i in range(epochs):
        trn_corr = 0
        trn_y_pred = []
        trn_y_pred_raw = []
        trn_y_true = []
        trn_sim_scores = []
        trn_loss = []

        model.train()

        # Run the training batches
        for b, ((X0_train, X1_train), y_train, [X0_labels, X1_labels]) in tqdm(enumerate(train_loader),
                                                                               total=len(train_loader)):

            optimizer.zero_grad()

            X0_train = X0_train.to(device)
            X1_train = X1_train.to(device)
            y_train = y_train.view(-1, 1).to(device)

            # Limit the number of batches
            if b == (len(train_loader)):
                break
            b += 1

            # Apply the model
            y_out, sim_score = model(X0_train, X1_train)

            batch_loss = criterion(y_out, y_train)

            # Apply probability threshold
            predicted = y_out.detach().clone().sigmoid()

            predicted[predicted <= probability_treshold] = 0
            predicted[predicted > probability_treshold] = 1

            batch_corr = (predicted == y_train).sum()
            trn_corr += batch_corr

            trn_y_pred.extend(predicted.cpu())
            trn_y_true.extend(y_train.cpu())
            trn_y_pred_raw.extend(y_out.sigmoid().cpu())
            trn_sim_scores.extend(sim_score.cpu())

            # Update parameters
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            batch_loss.backward()
            optimizer.step()
            trn_loss.append(batch_loss.cpu().item())

        trn_loss = np.array(trn_loss).mean()

        # Print epoch results
        print(
            f'epoch: {i:2}  batch: {b:4} [{train_loader.batch_size * b:6}/{len(train_loader) * train_loader.batch_size}]'
            f'  loss: {trn_loss.item():10.8f} accuracy: {trn_corr.item() * 100 / (train_loader.batch_size * b):7.3f}%')

        # Training metrics
        trn_y_pred = np.array(torch.cat(trn_y_pred))
        trn_y_pred_raw = np.array(torch.cat(trn_y_pred_raw).detach())
        trn_y_true = np.array(torch.cat(trn_y_true))
        trn_sim_scores = np.array(torch.cat(trn_sim_scores, dim=1).detach()).transpose(1, 0)
        calculate_metrics(trn_metrics_dict, trn_y_pred, trn_y_true, trn_loss)

        # Run the testing batches
        tst_y_pred, tst_y_true, tst_loss, tst_sim_scores, tst_y_pred_raw = validate_model(test_loader, model,
                                                                                          test_criterion, device,
                                                                                          probability_treshold)
        calculate_metrics(tst_metrics_dict, tst_y_pred, tst_y_true, tst_loss)

        if tst_metrics_dict['F1'][-1] > best_F1:
            best_F1 = tst_metrics_dict['F1'][-1]
            best_epoch = i + 1
            best_model = copy.deepcopy(model.state_dict())
            best_state = {'model': best_model,
                          'F1': best_F1,
                          'epoch': best_epoch, }

        if gif_path is not None:
            fig_sim, axs_sim = plt.subplots(2, 1, figsize=(15, 10))
            fig_final, axs_final = plt.subplots(2, 1, figsize=(15, 10))
            title = f'Intermediate similarity score\n Epoch = {i + 1}'
            # Make gif of similarity function score
            plot_utils.plot_prob_density(fig_sim, axs_sim, trn_sim_scores, trn_y_true, tst_sim_scores, tst_y_true,
                                         title)
            fig_sim.savefig(f'{gif_path}/sim_{i}.png')
            plt.close(fig_sim)
            fig_sim_list.append(f'{gif_path}/sim_{i}.png')

            title = f'Final similarity score\n Epoch = {i + 1}'
            # Make gif of similarity function score
            plot_utils.plot_prob_density(fig_final, axs_final, trn_y_pred_raw, trn_y_true, tst_y_pred_raw, tst_y_true,
                                         title)
            fig_final.savefig(f'{gif_path}/final_{i}.png')
            fig_final_list.append(f'{gif_path}/final_{i}.png')
            plt.close(fig_final)

        validate_similarity_scores(trn_similarity_scores, model, train_loader, device)
        validate_similarity_scores(tst_similarity_scores, model, test_loader, device)

    if gif_path is not None:

        images_sim = []
        for filename in fig_sim_list:
            images_sim.append(imageio.imread(filename))
            os.remove(filename)

        imageio.mimsave(gif_path + '_intermediate_similarity.gif', images_sim, fps=2, loop=1)

        images_final = []
        for filename in fig_final_list:
            images_final.append(imageio.imread(filename))
            os.remove(filename)

        imageio.mimsave(gif_path + '_final_similarity.gif', images_final, fps=2, loop=1)

    return trn_metrics_dict, trn_similarity_scores, tst_metrics_dict, trn_similarity_scores, best_state


def validate_model(loader, model, criterion, device, probability_threshold):
    num_correct = 0
    num_samples = 0

    # Set model to eval
    model.eval()

    y_true = []
    y_pred = []
    y_pred_raw = []
    sim_scores = []
    tst_loss = []

    with torch.no_grad():
        for b, ([X0_test, X1_test], y_test, [X0_test_labels, X1_test_labels]) in enumerate(loader):

            X0_test = X0_test.to(device)
            X1_test = X1_test.to(device)
            y_test = y_test.view(-1, 1).to(device)

            if b == (len(loader)):
                break

            # Apply the model
            y_val, sim_score = model(X0_test, X1_test)

            batch_loss = criterion(y_val, y_test)
            tst_loss.append(batch_loss.cpu())

            predicted = y_val.sigmoid()
            predicted[predicted <= probability_threshold] = 0
            predicted[predicted > probability_threshold] = 1

            y_pred.extend(predicted.cpu())
            y_pred_raw.extend(y_val.sigmoid().cpu())
            y_true.extend(y_test.cpu())
            sim_scores.extend(sim_score.cpu())

            num_correct += (predicted == y_test).sum()
            num_samples += predicted.size(0)
            b += 1

    # Toggle model back to train
    model.train()
    tst_loss = np.array(torch.stack(tst_loss)).mean()
    test_acc = num_correct.item() * 100 / (num_samples)
    print(f'test accuracy: {test_acc:7.3f}%  test loss: {tst_loss:10.8f} ')

    y_pred = np.array(torch.cat(y_pred))
    y_pred_raw = np.array(torch.cat(y_pred_raw))
    y_true = np.array(torch.cat(y_true))
    sim_scores = np.array(torch.cat(sim_scores, dim=1).detach()).transpose(1, 0)
    return y_pred, y_true, tst_loss, sim_scores, y_pred_raw


def validate_similarity_scores(similarity_dict, model, data_loader, device):
    # Set model to eval
    model.eval()

    with torch.no_grad():
        for b, ([X0, X1], y_true, [X0_labels, X1_labels]) in enumerate(data_loader):
            break
        X0 = X0.to(device)
        X1 = X1.to(device)

        X0_extend = X0.repeat_interleave(X1.shape[1], dim=1)

        # Get final similarity score output for same identical sample
        y_out_same, sim_score = model(X0, X0_extend)
        final_same_cls = y_out_same.sigmoid().mean()

        # Get intermediate similarity score for same identical sample
        intermediate_same_cls = sim_score.mean()

        # make sure to select only x1 samples of a different class
        idx_diff_class = (y_true == 0).nonzero().squeeze()
        idx_same_class = (y_true == 1).nonzero().squeeze()

        # Get final similarity score output for different class sample
        y_out, sim_score = model(X0, X1)

        y_out_diff = sim_score[idx_diff_class].squeeze()
        intermediate_diff_cls = y_out_diff.mean()

        y_out_diff = y_out[idx_diff_class].squeeze()
        final_diff_cls = y_out_diff.mean()

        # Get intermediate similarity score for different class sample
        y_out_same = sim_score[idx_same_class].squeeze()
        intermediate_same_cls = y_out_same.mean()

        y_out_same = y_out[idx_same_class].squeeze().sigmoid()
        final_same_cls = y_out_same.mean()

    similarity_dict['final_same_cls'].append(final_same_cls.item())
    similarity_dict['intermediate_same_cls'].append(intermediate_same_cls.item())
    similarity_dict['final_diff_cls'].append(final_diff_cls.item())
    similarity_dict['intermediate_diff_cls'].append(intermediate_diff_cls.item())

    model.train()
    return


def calculate_metrics(metrics_dict, y_pred, y_true, loss):
    metrics_dict['accuracy'].append(metrics.accuracy_score(y_true, y_pred))
    metrics_dict['precision'].append(metrics.precision_score(y_true=y_true, y_pred=y_pred, zero_division=0))
    metrics_dict['recall'].append(metrics.recall_score(y_true, y_pred, zero_division=0))
    metrics_dict['F1'].append(metrics.f1_score(y_true=y_true, y_pred=y_pred, zero_division=0))

    metrics_dict['mean_pred'].append(y_pred.mean())
    metrics_dict['mean_true'].append(y_true.mean())

    metrics_dict['loss'].append(loss.item())

    return


def extract_features(data_loader, model, classes, device):
    class_samples = {key: [] for key in classes}
    train_rep, train_cls_rep, labels_rep = [], [], []

    model.eval()

    with torch.no_grad():
        for b, (x, y_train) in tqdm(enumerate(data_loader),
                                    total=int(len(data_loader.dataset) / data_loader.batch_size)):
            x = x.to(device)
            y_train = y_train.to(device)

            # Apply the model
            _, feature_vector = model(x)

            train_rep.append(feature_vector)
            labels_rep.append(y_train)

    train_rep = torch.cat(train_rep, dim=0)
    labels_rep = torch.cat(labels_rep, dim=0)

    sort_idx = []
    for cls in classes:
        train_cls_rep.append(train_rep[(labels_rep == cls).nonzero()].mean(dim=0))
        sort_idx.append((labels_rep == cls).nonzero())

    train_cls_rep = torch.cat(train_cls_rep, dim=0)
    sort_idx = torch.cat(sort_idx, dim=0).squeeze()

    # Sort labels and feature vectors
    labels_rep = labels_rep[sort_idx]
    train_rep = train_rep[sort_idx]

    return train_rep.detach().cpu(), train_cls_rep.detach().cpu(), labels_rep.detach().cpu()


def rank_samples_from_memory(class_set, data_rep, data_cls_rep, labels_rep, classes, train_samples_per_cls, top_n,
                             randomize_samples=True, same_class_reverse=False, balance_same_class_entries=False):
    X0, X1, Y = [], [], []
    base_cls_offset = class_set[0]  # classes.index(class_set[0])  # -> gives index of first class of interest
    for cls in tqdm(class_set):
        tmp_X1 = []
        tmp_Y = []

        # index of class of interest
        ix = classes.index(cls)  # considers validation_set that not start from zero.

        # Get index of first sample of class
        cls_offset = np.where(labels_rep == cls)[0].min()
        # cls_offset = ix * train_per_cls  # Some kind of offset?

        # find top_n similar classes.
        rest_cls_idx = [classes.index(cls1) for cls1 in class_set if
                        classes.index(cls1) != ix]  # Find all remaining classes

        # cosine similarity between train_per_class number of the class of interest and the mean feature vector of
        # the rest of the classes
        # Finds the most similar classes based on the mean value
        sim = metrics.pairwise.cosine_similarity(data_rep[cls_offset:cls_offset + train_samples_per_cls],
                                                 data_cls_rep[rest_cls_idx])
        # Get indices of a sorted array
        sim_idx = sim.argsort(axis=1)
        # Add offset of the base class
        sim_idx += base_cls_offset
        # Plus one idx to correct for the removed class of interest
        sim_idx[sim_idx >= cls] += 1

        # Get the top classes from the sorted similarities
        sim_idx = sim_idx[:, -top_n:]  # -> For each of the train_per_cls samples of a class.
        # Get the top_n similar classes

        # Loop over the k most similar classes based on the previous cosine similarity
        for kx in range(-top_n, 0):
            tmp_X1_batch = []
            for jx in range(train_samples_per_cls):
                # Same offset with unknown purpose
                cls1 = sim_idx[jx, kx]
                cls1_offset = np.where(labels_rep == cls1)[0].min()
                # Find cosine similarity between the two offsets? Gives an array with size [1, train_per_cls]
                sim1 = metrics.pairwise.cosine_similarity(data_rep[cls_offset + jx:cls_offset + jx + 1],
                                                          data_rep[cls1_offset:cls1_offset + train_samples_per_cls])
                # Sort indices and find most similar samples. Remove least similar example to make array of same
                # length as similarity of same class samples
                sim1_idx = sim1.argsort(axis=1)[:1, -(train_samples_per_cls - 1):]
                sim1_idx += cls1_offset
                # Give size a second dimension, useful for vstack i think
                tmp_X1_batch.append(np.expand_dims(sim1_idx, 1))
            tmp_X1_batch = np.vstack(tmp_X1_batch)

            # Append indices and labels
            tmp_X1.append(tmp_X1_batch)
            tmp_Y.append(np.full((train_samples_per_cls, 1), 0))

        # put same class in the last dim
        sim2 = metrics.pairwise.cosine_similarity(
            data_rep[cls_offset:cls_offset + train_samples_per_cls])  # Similarity between same class
        # Remove most similar sample as this is the same sample as input sample
        sim2_idx = sim2.argsort(axis=1)[:, :-1] + cls_offset  # add the offset to obtain the real offset in memory.
        # Append same class indices and labels to tmp_x1 and tmp_y
        tmp_X1.append(np.expand_dims(sim2_idx, 1))
        tmp_Y.append(np.full((train_samples_per_cls, 1), 1))

        # append all input samples indices
        X0.append(np.arange(cls_offset, cls_offset + train_samples_per_cls).reshape(-1, 1))

        # make matrix with indices for all comparison samplles
        X1.append(np.concatenate(tmp_X1, 1))
        Y.append(np.concatenate(tmp_Y, axis=1))  # similar

    X0 = np.vstack(X0)
    X1 = np.vstack(X1)
    Y = np.concatenate(Y)

    if randomize_samples:
        shuffle_idx = np.random.permutation(X0.shape[0])
        return X0[shuffle_idx], X1[shuffle_idx], Y[shuffle_idx]
    else:
        return X0, X1, Y
