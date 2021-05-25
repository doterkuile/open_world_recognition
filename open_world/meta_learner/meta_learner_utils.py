import sklearn
import numpy as np
import torch


def extract_features(train_data, model, classes, memory_path, load_data=False):

    class_samples = {key: [] for key in classes}
    train_rep, train_cls_rep = [], []

    if load_data:
        train_rep = np.load(memory_path)['data_rep']
        train_cls_rep = np.load(memory_path)['train_cls_rep']

    else:
        with torch.no_grad():
            # Create dict of samples per class
            for sample, label in train_data:
                class_samples[label].append(sample)

            # Extract features of sample and mean feature vector per class
            for cls in classes:
                # convert to tensor
                class_samples[cls] = torch.stack(class_samples[cls])

                # calculate features
                cls_rep = model(class_samples[cls].cuda())
                train_rep.append(cls_rep)

                # Mean feature vector per class
                train_cls_rep.append(cls_rep.mean(dim=0))

            # Convert to tensor
            train_rep = torch.cat(train_rep).cpu()
            train_cls_rep = torch.stack(train_cls_rep).cpu()


    return train_rep, train_cls_rep


def rank_samples_from_memory(class_set, data_rep, data_cls_rep, classes, train_per_cls, top_k):

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

        # cosine similarity between train_per_class number of the class of interest and the mean feature vector of
        # the rest of the classes
        # Finds the most similar classes based on the mean value
        sim = sklearn.metrics.pairwise.cosine_similarity(data_rep[cls_offset:cls_offset + train_per_cls],
                                                         data_cls_rep[rest_cls_idx])
        # Get indices of a sorted array
        sim_idx = sim.argsort(axis=1)
        # Add offset of the base class
        sim_idx += base_cls_offset
        # Plus one idx to correct for the removed class of interest
        sim_idx[sim_idx >= ix] += 1

        # Get the top classes from the sorted similarities
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