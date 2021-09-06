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
import copy
import imageio


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

    ml_fig_list = []
    al_fig_list = []

    best_state = {'model': copy.deepcopy(model.state_dict()),
                  'F1': -1.0,
                  'epoch': 0, }

    # Calculate metrics before training has started
    trn_y_pred, trn_y_true, trn_loss, trn_ml_out, trn_y_pred_raw = validate_model(train_loader, model,
                                                                                  criterion, device,
                                                                                  probability_treshold)
    calculate_metrics(trn_metrics_dict, trn_y_pred, trn_y_true, trn_loss)

    # Calculate metrics before training has started
    tst_y_pred, tst_y_true, tst_loss, tst_ml_out, tst_y_pred_raw = validate_model(test_loader, model,
                                                                                  test_criterion, device,
                                                                                  probability_treshold)
    calculate_metrics(tst_metrics_dict, tst_y_pred, tst_y_true, tst_loss)

    for epoch in range(epochs):
        trn_y_pred = []
        trn_y_raw = []
        trn_y_true = []
        trn_ml_out = []
        trn_loss = []

        model.train()

        # Run the training batches
        for b, ((X0_train, X1_train), y_train, [X0_labels, X1_labels]) in tqdm(enumerate(train_loader),
                                                                               total=len(train_loader)):

            X0_train = X0_train.to(device)
            X1_train = X1_train.to(device)
            y_train = y_train.view(-1, 1).to(device)

            # Limit the number of batches
            if b == (len(train_loader)):
                break
            b += 1
            y_pred, matching_layer_output, batch_loss, predicted = train_batch_step(X0_train, X1_train, y_train,
                                                                                    model, criterion,
                                                                                    probability_treshold, optimizer)
            trn_y_pred.extend(predicted.cpu())
            trn_y_true.extend(y_train.cpu())
            trn_y_raw.extend(y_pred.cpu())
            trn_ml_out.extend(matching_layer_output.cpu())
            trn_loss.append(batch_loss.cpu().item())

        trn_loss = np.array(trn_loss).mean()

        # Training metrics
        trn_y_pred = np.array(torch.cat(trn_y_pred))
        trn_y_raw = np.array(torch.cat(trn_y_raw).detach())
        trn_y_true = np.array(torch.cat(trn_y_true))
        trn_ml_out = np.array(torch.cat(trn_ml_out, dim=1).detach()).transpose(1, 0)
        calculate_metrics(trn_metrics_dict, trn_y_pred, trn_y_true, trn_loss)

        trn_corr = (trn_y_true == trn_y_pred).nonzero()[0].shape[0]

        # Print epoch results
        print(
            f'epoch: {epoch:2}  batch: {b:4} [{train_loader.batch_size * b:6}/{len(train_loader.dataset)}]'
            f'  loss: {trn_loss:10.8f} accuracy: {trn_corr * 100 / len(train_loader.dataset):7.3f}%')

        # Run the testing batches
        tst_y_pred, tst_y_true, tst_loss, tst_ml_out, tst_y_raw = validate_model(test_loader, model,
                                                                                 test_criterion, device,
                                                                                 probability_treshold)
        calculate_metrics(tst_metrics_dict, tst_y_pred, tst_y_true, tst_loss)

        if tst_metrics_dict['F1'][-1] > best_state['F1']:
            best_state['F1'] = tst_metrics_dict['F1'][-1]
            best_state['epoch'] = epoch + 1
            best_state['model'] = copy.deepcopy(model.state_dict())

        if gif_path is not None:
            ml_image_name, al_image_name = plot_utils.create_gif_image(trn_ml_out, trn_y_true, tst_ml_out, tst_y_true,
                                                                       trn_y_raw, tst_y_raw, epoch, gif_path)
            ml_fig_list.append(ml_image_name)
            al_fig_list.append(al_image_name)



    if gif_path is not None:
        ml_gif_path = f"{gif_path}_matching_layer.gif"
        plot_utils.save_gif_file(ml_fig_list, ml_gif_path)
        al_gif_path = f"{gif_path}_final_similarity.gif"
        plot_utils.save_gif_file(al_fig_list, al_gif_path)

    return trn_metrics_dict, tst_metrics_dict, best_state


def train_batch_step(X0, X1, y_true, model, criterion, threshold, optimizer=None):
    # if optimizer is not None:

    # Apply the model
    y_pred, matching_layer_output = model(X0, X1)

    batch_loss = criterion(y_pred, y_true)

    # Apply probability threshold
    predicted = y_pred.detach().clone().sigmoid()

    predicted[predicted <= threshold] = 0
    predicted[predicted > threshold] = 1

    # Update parameters
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
    if optimizer is not None:
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()


    y_pred = y_pred.sigmoid()
    matching_layer_output = matching_layer_output.sigmoid()

    return y_pred, matching_layer_output, batch_loss, predicted


def validate_model(loader, model, criterion, device, probability_threshold):
    # Set model to eval
    model.eval()

    y_true = []
    y_pred = []
    y_pred_raw = []
    ml_out = []
    tst_loss = []
    with torch.no_grad():
        for b, ([X0, X1], y_test, [X0_test_labels, X1_test_labels]) in enumerate(loader):

            X0 = X0.to(device)
            X1 = X1.to(device)
            y_test = y_test.view(-1, 1).to(device)

            if b == len(loader):
                break
            y_out, matching_layer_output, batch_loss, predicted = train_batch_step(X0, X1, y_test,
                                                                                   model, criterion,
                                                                                   probability_threshold)
            y_pred.extend(predicted.cpu())
            y_pred_raw.extend(y_out.cpu())
            y_true.extend(y_test.cpu())
            ml_out.extend(matching_layer_output.cpu())
            tst_loss.append(batch_loss.cpu())
            b += 1

    # Toggle model back to train
    model.train()
    tst_loss = np.array(torch.stack(tst_loss)).mean()

    y_pred = np.array(torch.cat(y_pred))
    y_pred_raw = np.array(torch.cat(y_pred_raw))
    y_true = np.array(torch.cat(y_true))
    ml_out = np.array(torch.cat(ml_out, dim=1).detach()).transpose(1, 0)

    test_acc = (y_true == y_pred).nonzero()[0].shape[0] * 100 / (len(loader.dataset))
    print(f'test accuracy: {test_acc:7.3f}%  test loss: {tst_loss:10.8f} ')

    return y_pred, y_true, tst_loss, ml_out, y_pred_raw

def test_model(loader, model, criterion, device, probability_threshold):
    # Set model to eval
    model.eval()

    y_score = []
    memory_labels = []
    true_labels = []
    with torch.no_grad():
        for b, ([X0, X1], y_test, [X0_test_labels, X1_test_labels]) in enumerate(loader):

            X0 = X0.to(device)
            X1 = X1.to(device)
            y_test = y_test.view(-1, 1).to(device)

            y_out, matching_layer_output, batch_loss, predicted = train_batch_step(X0, X1, y_test,
                                                                                   model, criterion,
                                                                                   probability_threshold)
            y_score.append(y_out.cpu().reshape(-1))
            memory_labels.append(X1_test_labels[:,0])
            true_labels.append(X0_test_labels[0])

            b += 1

    # Toggle model back to train
    model.train()

    memory_labels = np.array(torch.stack(memory_labels))
    true_labels = np.array(torch.stack(true_labels))
    y_score = np.array(torch.stack(y_score))

    return y_score, memory_labels, true_labels




def trainMatchingLayer(model, train_loader, test_loader, epochs, criterion, test_criterion, optimizer, device,
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


    fig_sim_list = []
    fig_final_list = []

    best_model = copy.deepcopy(model.state_dict())
    best_F1 = -2.0
    best_epoch = 0
    best_state = {'model': best_model,
                  'F1': best_F1,
                  'epoch': best_epoch, }

    # Calculate metrics before training has started
    trn_y_pred, trn_y_true, trn_loss, trn_sim_scores, trn_y_pred_raw = validateMatchingLayer(train_loader, model,
                                                                                             criterion, device,
                                                                                             probability_treshold)
    calculate_metrics(trn_metrics_dict, trn_y_pred, trn_y_true, trn_loss)

    # Calculate metrics before training has started
    tst_y_pred, tst_y_true, tst_loss, tst_sim_scores, tst_y_pred_raw = validateMatchingLayer(test_loader, model,
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


            X0_train = X0_train.to(device)
            X1_train = X1_train.to(device)
            y_train = y_train.view(-1, 1).to(device)

            # y_train = torch.abs(y_train -1)
            y_train = y_train.repeat_interleave(X1_train.shape[1], dim=1)

            # Limit the number of batches
            if b == (len(train_loader)):
                break
            b += 1

            # Apply the model
            _, sim_score = model(X0_train, X1_train)
            y_train = y_train.view(-1, 1)
            sim_score = sim_score.view(-1, 1)

            batch_loss = criterion(sim_score, y_train)

            # Apply probability threshold
            predicted = sim_score.detach().clone().sigmoid()

            predicted[predicted <= probability_treshold] = 0
            predicted[predicted > probability_treshold] = 1

            batch_corr = (predicted == y_train).sum()
            trn_corr += batch_corr

            trn_y_pred.extend(predicted.cpu())
            trn_y_true.extend(y_train.cpu())
            trn_y_pred_raw.extend(sim_score.sigmoid().cpu())
            trn_sim_scores.extend(sim_score.cpu())

            # Update parameters
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

            trn_loss.append(batch_loss.cpu().item())

        trn_loss = np.array(trn_loss).mean()

        # Print epoch results
        print(
            f'epoch: {i:2}  batch: {b:4} [{train_loader.batch_size * b:6}/{len(train_loader) * train_loader.batch_size}]'
            f'  loss: {trn_loss.item():10.8f} accuracy: {trn_corr.item() * 100 / (len(train_loader.dataset) * X1_train.shape[1]):7.3f}%')

        # Training metrics
        trn_y_pred = np.array(torch.cat(trn_y_pred))
        trn_y_pred_raw = np.array(torch.cat(trn_y_pred_raw).detach())
        trn_y_true = np.array(torch.cat(trn_y_true))
        trn_sim_scores = np.array(torch.cat(trn_sim_scores, dim=0).detach())

        calculate_metrics(trn_metrics_dict, trn_y_pred, trn_y_true, trn_loss)

        # Run the testing batches
        tst_y_pred, tst_y_true, tst_loss, tst_sim_scores, tst_y_pred_raw = validateMatchingLayer(test_loader, model,
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
            fig_final, axs_final = plt.subplots(2, 1, figsize=(15, 10))
            title = f'Matching layer output\n Epoch = {i + 1}'
            fig_path = f'{gif_path}/matching_only_{i}.png'

            ml_trn_same_idx = (trn_y_true == 1).nonzero()[0].squeeze()
            ml_trn_diff_idx = (trn_y_true == 0).nonzero()[0].squeeze()
            ml_tst_same_idx = (tst_y_true == 1).nonzero()[0].squeeze()
            ml_tst_diff_idx = (tst_y_true == 0).nonzero()[0].squeeze()
            # Make gif of similarity function score
            plot_utils.plot_prob_density(fig_final, axs_final, trn_y_pred_raw, ml_trn_same_idx, ml_trn_diff_idx, tst_y_pred_raw, ml_tst_same_idx,
                                         ml_tst_diff_idx, title, fig_path)
            fig_final_list.append(fig_path)
            plt.close(fig_final)

    if gif_path is not None:

        images_final = []
        for filename in fig_final_list:
            images_final.append(imageio.imread(filename))
            os.remove(filename)

        imageio.mimsave(gif_path + '_matching_layer_only.gif', images_final, fps=2, loop=1)

    return trn_metrics_dict, tst_metrics_dict, best_state


def validateMatchingLayer(loader, model, criterion, device, probability_threshold):
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
            # y_test = torch.abs(y_test -1)

            y_test = y_test.repeat_interleave(X1_test.shape[1], dim=1)

            if b == (len(loader)):
                break

            # Apply the model
            _, sim_score = model(X0_test, X1_test)
            y_test = y_test.view(-1, 1)
            sim_score = sim_score.view(-1, 1)

            batch_loss = criterion(sim_score, y_test)
            tst_loss.append(batch_loss.cpu())

            predicted = sim_score.sigmoid()
            predicted[predicted <= probability_threshold] = 0
            predicted[predicted > probability_threshold] = 1

            y_pred.extend(predicted.cpu())
            y_pred_raw.extend(sim_score.sigmoid().cpu())
            y_true.extend(y_test.cpu())
            sim_scores.extend(sim_score.sigmoid().cpu())

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
    sim_scores = np.array(torch.cat(sim_scores, dim=0).detach())
    return y_pred, y_true, tst_loss, sim_scores, y_pred_raw


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
    data_rep, cls_rep, labels = [], [], []

    model.eval()

    with torch.no_grad():
        for b, (x, y_train) in tqdm(enumerate(data_loader),
                                    total=int(len(data_loader.dataset) / data_loader.batch_size)):
            x = x.to(device)
            y_train = y_train.to(device)

            # Apply the model
            _, feature_vector = model(x)

            data_rep.append(feature_vector)
            labels.append(y_train)

    data_rep = torch.cat(data_rep, dim=0)
    labels = torch.cat(labels, dim=0)

    class_samples = [cls for cls in classes if cls in labels]

    sort_idx = []
    for cls in class_samples:
        cls_rep.append(data_rep[(labels == cls).nonzero()].mean(dim=0))
        sort_idx.append((labels == cls).nonzero())

    cls_rep = torch.cat(cls_rep, dim=0)
    sort_idx = torch.cat(sort_idx, dim=0).squeeze()

    # Sort labels and feature vectors
    labels = labels[sort_idx]
    data_rep = data_rep[sort_idx]

    return data_rep.detach().cpu(), cls_rep.detach().cpu(), labels.detach().cpu()


def rank_input_to_memory(input_rep, input_labels, memory_rep, memory_labels, memory_cls_rep, input_sample_idx,
                         memory_sample_idx, partial_cls_set, complete_cls_set, top_n, randomize_samples=False):
    X0, X1, Y = [], [], []
    # Use first class of class_set as offset
    base_cls_offset = partial_cls_set[0]  # classes.index(class_set[0])  # -> gives index of first class of the list of c
    for cls in tqdm(partial_cls_set):
        tmp_X1 = []
        tmp_Y = []

        # index of class of interest
        ix = np.where(complete_cls_set == cls)[0].item()

        # Get index of first sample of class
        cls_offset = np.where(input_labels == cls)[0].min()
        # cls_offset = ix * train_per_cls  # Some kind of offset?

        # Select all remaining classes that are not equal to the class of interest
        rest_cls_idx = [np.where(complete_cls_set == cls1)[0].item() for cls1 in partial_cls_set if cls1 != cls]

        # cosine similarity between train_per_class number of the class of interest and the mean feature vector of
        # the rest of the classes
        # Finds the most similar classes based on the mean value
        # Cosine similarity between the input samples of cls of interest and the mean representation of
        # the remaining classes. Output is of size [input_samples_per_class, len(class_set) -1]
        cls_sample_idx = cls_offset + input_sample_idx
        sim = metrics.pairwise.cosine_similarity(input_rep[cls_sample_idx],
                                                memory_cls_rep[rest_cls_idx])

        # Get indices of a sorted array
        sim_idx = sim.argsort(axis=1)
        # Add offset of the base class
        sim_idx += base_cls_offset
        # Plus one idx to correct for the removed class of interest
        sim_idx[sim_idx >= cls] += 1

        # Get the top classes from the sorted similarities
        sim_idx = sim_idx[:, -top_n:]

        # Loop over the n most similar classes based on the previous cosine similarity
        for nx in range(-top_n, 0):
            tmp_X1_batch = []
            # Loop over the j input samples from the cls of interest
            for jx in range(input_sample_idx.shape[0]):
                # Select the nx th top n class for the jx th input sample of class of interest
                cls1 = sim_idx[jx, nx]
                # Find class offset in memory
                cls1_offset = np.where(memory_labels == cls1)[0].min()

                # Find cosine similarity between an input sample and all samples from same cls in memory
                # Output size = [1, memory_samples_per_class]
                memory_cls_sample_idx = cls1_offset + memory_sample_idx
                input_sample_offset = cls_offset + input_sample_idx.min()
                sim1 = metrics.pairwise.cosine_similarity(input_rep[input_sample_offset + jx].reshape(1, -1),
                                                          memory_rep[memory_cls_sample_idx])
                # Sort indices and find most similar samples. Remove least similar example to make array of same
                # length as similarity of same class samples
                sim1_idx = sim1.argsort(axis=1)[0, 1:].reshape(1,-1)
                sim1_idx += cls1_offset
                # Give size a second dimension for vstack
                tmp_X1_batch.append(np.expand_dims(sim1_idx, 1))

            tmp_X1_batch = np.vstack(tmp_X1_batch)

            # Append indices and labels
            tmp_X1.append(tmp_X1_batch)
            tmp_Y.append(np.full((input_sample_idx.shape[0], 1), 0))

        # put same class in the last dim
        memory_same_cls_sample_idx = np.where(memory_labels == cls)[0].min() + memory_sample_idx
        sim2 = metrics.pairwise.cosine_similarity(input_rep[cls_sample_idx],
            memory_rep[memory_same_cls_sample_idx])  # Similarity between same class

        # If memory != input, no duplicate samples, remove the least similar sample
        if sim2.max() < 1:

            sim2_idx = sim2.argsort(axis=1)[:, 1:] + cls_offset  # add the offset to obtain the real offset in memory.
        # If memory == input, remove duplicate samples
        else:
            # Remove most similar sample as this is the same sample as input sample
            sim2_idx = sim2.argsort(axis=1)[:,:-1] + cls_offset

        # Append same class indices and labels to tmp_x1 and tmp_y
        tmp_X1.append(np.expand_dims(sim2_idx,1))
        tmp_Y.append(np.full((input_sample_idx.shape[0], 1), 1))

        # append all input samples indices
        X0.append(cls_sample_idx.reshape(-1, 1))


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


def rank_test_data(input_rep, input_labels, memory_rep, memory_labels, memory_cls_rep, input_sample_idx,
                         memory_sample_idx, input_cls_set, memory_cls_set, complete_cls_set, top_n, randomize_samples=False):
    X0, X1, Y = [], [], []
    # Use first class of class_set as offset
    base_cls_offset = input_cls_set[0]  # classes.index(class_set[0])  # -> gives index of first class of the list of c
    for cls in tqdm(input_cls_set):
        tmp_X1 = []
        tmp_Y = []

        # index of class of interest
        ix = np.where(complete_cls_set == cls)[0].item()

        # Get index of first sample of class
        cls_offset = np.where(input_labels == cls)[0].min()
        # cls_offset = ix * train_per_cls  # Some kind of offset?

        # Select all remaining classes that are not equal to the class of interest
        rest_cls_idx = [np.where(complete_cls_set == cls1)[0].item() for cls1 in memory_cls_set]
        # cosine similarity between train_per_class number of the class of interest and the mean feature vector of
        # the rest of the classes
        # Finds the most similar classes based on the mean value
        # Cosine similarity between the input samples of cls of interest and the mean representation of
        # the remaining classes. Output is of size [input_samples_per_class, len(class_set) -1]
        cls_sample_idx = cls_offset + input_sample_idx
        sim = metrics.pairwise.cosine_similarity(input_rep[cls_sample_idx],
                                                memory_cls_rep[rest_cls_idx])

        # Get indices of a sorted array
        sim_idx = sim.argsort(axis=1)
        # Add offset of the base class
        sim_idx += base_cls_offset
        # Plus one idx to correct for the removed class of interest
        # sim_idx[sim_idx >= cls] += 1

        # Get the top classes from the sorted similarities
        sim_idx = sim_idx[:, -top_n:]

        # Loop over the n most similar classes based on the previous cosine similarity
        for nx in range(-top_n, 0):
            tmp_X1_batch = []
            tmp_Y_batch = []
            # Loop over the j input samples from the cls of interest
            for jx in range(input_sample_idx.shape[0]):
                # Select the nx th top n class for the jx th input sample of class of interest
                cls1 = sim_idx[jx, nx]
                # Find class offset in memory
                cls1_offset = np.where(memory_labels == cls1)[0].min()

                # Find cosine similarity between an input sample and all samples from same cls in memory
                # Output size = [1, memory_samples_per_class]
                memory_cls_sample_idx = cls1_offset + memory_sample_idx
                input_sample_offset = cls_offset + input_sample_idx.min()
                sim1 = metrics.pairwise.cosine_similarity(input_rep[input_sample_offset + jx].reshape(1, -1),
                                                          memory_rep[memory_cls_sample_idx])
                # Sort indices and find most similar samples. Remove least similar example to make array of same
                # length as similarity of same class samples
                sim1_idx = sim1.argsort(axis=1)[0, 1:].reshape(1,-1)
                sim1_idx += cls1_offset
                # Give size a second dimension for vstack
                tmp_X1_batch.append(np.expand_dims(sim1_idx, 1))
                tmp_Y_batch.append([1 if cls == cls1 else 0])

            tmp_X1_batch = np.vstack(tmp_X1_batch)
            tmp_Y_batch = np.vstack(tmp_Y_batch)
            # Append indices and labels
            tmp_X1.append(tmp_X1_batch)
            tmp_Y.append(tmp_Y_batch)

        # put same class in the last dim

        # append all input samples indices
        X0.append(cls_sample_idx.reshape(-1, 1))


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


def rank_samples_from_memory(class_set, data_rep, data_cls_rep, labels_rep, classes, train_samples_per_cls, top_n,
                             randomize_samples=False, same_class_reverse=False, balance_same_class_entries=False):
    X0, X1, Y = [], [], []
    base_cls_offset = class_set[0]  # classes.index(class_set[0])  # -> gives index of first class of interest
    label_offset = 500
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
