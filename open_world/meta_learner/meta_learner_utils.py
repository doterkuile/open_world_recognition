import sklearn
import numpy as np
import torch
import time
from tqdm import tqdm




def trainMetaModel(model, train_loader, test_loader, epochs, criterion, optimizer, device):
    start_time = time.time()

    trn_losses = []
    tst_losses = []
    trn_accs = []
    tst_accs = []

    max_trn_batch = 400
    max_tst_batch = 1200

    for i in range(epochs):
        trn_corr = 0
        tst_corr = 0

        # Run the training batches
        for b, ((X0_train, X1_train), y_train, [X0_labels, X1_labels]) in tqdm(enumerate(train_loader), total=int(len(train_loader.dataset)/train_loader.batch_size)):

            X0_train = X0_train.to(device)
            X1_train = X1_train.to(device)
            y_train = y_train.view(-1, 1).to(device)

            # Limit the number of batches
            if b == max_trn_batch:
                break
            b += 1

            # Apply the model
            y_pred = model(X0_train, X1_train)
            trn_loss = criterion(y_pred, y_train)

            # Tally the number of correct predictions
            predicted = y_pred.detach().clone()

            predicted[predicted <= 0.5] = 0
            predicted[predicted > 0.5] = 1

            batch_corr = (predicted == y_train).sum()
            trn_corr += batch_corr

            # Update parameters
            optimizer.zero_grad()
            model.reset_hidden()
            trn_loss.backward()
            optimizer.step()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)

            # Print interim results
            if b % 600 == 0:
                print(f'epoch: {i:2}  batch: {b:4} [{train_loader.batch_size * b:6}/{len(train_loader) * train_loader.batch_size}]'
                      f'  loss: {trn_loss.item():10.8f} accuracy: {trn_corr.item() * 100 / (train_loader.batch_size * b):7.3f}%')

        trn_acc = trn_corr.item() * 100 / (train_loader.batch_size * b)

        trn_losses.append(trn_loss.item())
        trn_accs.append(trn_acc)

        # Run the testing batches
        tst_corr, tst_loss = validate_model(test_loader, model, criterion, device)
        tst_acc = tst_corr.item() * 100 / (test_loader.batch_size * b)

        tst_losses.append(tst_loss.item())
        tst_accs.append(tst_acc)

    print(f'\nDuration: {time.time() - start_time:.0f} seconds')  # print the time elapsed

    return (trn_losses,tst_losses, trn_accs, tst_accs)


def validate_model(loader, model, criterion, device):
    num_correct = 0
    num_samples = 0

    # Set model to eval
    model.eval()

    y_true = []
    y_pred = []


    with torch.no_grad():
        for b, ([X0_test, X1_test], y_test, [X0_test_labels, X1_test_labels]) in enumerate(loader):

            X0_test = X0_test.to(device)
            X1_test = X1_test.to(device)
            y_test = y_test.view(-1,1).to(device)

            if b*loader.batch_size >= len(loader):
                break

            # Apply the model
            y_val = model(X0_test, X1_test)
            loss = criterion(y_val, y_test)

            # Tally the number of correct predictions

            predicted = y_val
            predicted[predicted <= 0.5] = 0
            predicted[predicted > 0.5] = 1
            y_pred.extend(predicted)
            y_true.extend(y_test)
            num_correct += (predicted == y_test).sum()
            num_samples += predicted.size(0)
            b += 1

    y_pred = torch.stack(y_pred)
    y_true = torch.stack(y_true)
    # Toggle model back to train
    model.train()
    print(f'test accuracy: {num_correct.item() * 100 / (num_samples):7.3f}%')
    return num_correct, loss

def extract_features(train_data, model, classes, memory_path, load_memory=False):

    class_samples = {key: [] for key in classes}
    train_rep, train_cls_rep, labels_rep= [], [], []

    if load_memory:
        train_rep = np.load(memory_path)['data_rep']
        train_cls_rep = np.load(memory_path)['train_cls_rep']
        labels_rep = np.load(memory_path)['labels_rep']

    else:
        with torch.no_grad():
            # Create dict of samples per class
            for sample, label in train_data:
                class_samples[label].append(sample)

            # Extract features of sample and mean feature vector per class
            for cls in tqdm(classes):
                # convert to tensor
                class_samples[cls] = torch.stack(class_samples[cls])

                # calculate features
                cls_rep = model(class_samples[cls].cuda())
                train_rep.append(cls_rep)
                labels_rep.append(cls *torch.ones(cls_rep.shape[0],1))

                # Mean feature vector per class
                train_cls_rep.append(cls_rep.mean(dim=0))

            # Convert to tensor
            train_rep = torch.cat(train_rep).cpu()
            train_cls_rep = torch.stack(train_cls_rep).cpu()
            labels_rep = torch.cat(labels_rep, dim=0).cpu()


    return train_rep, train_cls_rep, labels_rep


def rank_samples_from_memory(class_set, data_rep, data_cls_rep, labels_rep, classes, train_samples_per_cls, top_n, randomize_samples=True):

    X0, X1, Y = [], [], []
    base_cls_offset = classes.index(class_set[0])  # -> gives index of first class of interest
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
        sim = sklearn.metrics.pairwise.cosine_similarity(data_rep[cls_offset:cls_offset + train_samples_per_cls],
                                                         data_cls_rep[rest_cls_idx])
        # Get indices of a sorted array
        sim_idx = sim.argsort(axis=1)
        # Add offset of the base class
        sim_idx += base_cls_offset
        # Plus one idx to correct for the removed class of interest
        sim_idx[sim_idx >= ix] += 1

        # Get the top classes from the sorted similarities
        sim_idx = sim_idx[:, -top_n:] # -> For each of the train_per_cls samples of a class. Get the top_n
                                      # similar classes

        # Loop over the k most similar classes based on the previous cosine similarity
        for kx in range(-top_n, 0):
            tmp_X1_batch = []
            for jx in range(train_samples_per_cls):
                # Same offset with unknown purpose
                cls1 = sim_idx[jx, kx]
                cls1_offset = np.where(labels_rep == cls1)[0].min()
                # Find cosine similarity between the two offsets? Gives an array with size [1, train_per_cls]
                sim1 = sklearn.metrics.pairwise.cosine_similarity(data_rep[cls_offset + jx:cls_offset + jx + 1],
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

        # put sim in the last dim
        sim = sklearn.metrics.pairwise.cosine_similarity(
            data_rep[cls_offset:cls_offset + train_samples_per_cls])  # Similarity between same class
        # Remove most similar sample as this is the same sample as input sample
        sim_idx = sim.argsort(axis=1)[:, :-1] + cls_offset  # add the offset to obtain the real offset in memory.
        # Append same class indices and labels to tmp_x1 and tmp_y
        tmp_X1.append(np.expand_dims(sim_idx, 1))
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

