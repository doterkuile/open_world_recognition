import torch
import torch.utils.data as data_utils
from open_world import OpenWorldUtils
from open_world import ObjectDatasets
from open_world import RecognitionModels
from open_world import loss_functions
import open_world.meta_learner.meta_learner_utils as meta_utils
import open_world.plot_utils as plot_utils
import yaml
import torchvision
from torchvision import transforms
import sklearn

import numpy as np
from torch.utils.data import DataLoader
import argparse
import os
import shutil
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

import time

def main():
    torch.manual_seed(42)

    # Main gpu checks
    multiple_gpu = True if torch.cuda.device_count() > 1 else False
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if not torch.cuda.is_available():
        print("Cuda device not available make sure CUDA has been installed")
        return
    else:
        print(f"Running with {torch.cuda.device_count()} GPUs")


    model, memory,image_dataset, encoder, object_data_folder, image_resize, top_n, top_k = parseConfigFile(device)

    transform_train = transforms.Compose([
        transforms.Resize((image_resize,image_resize)),
        transforms.ToTensor(),
    ])
    object_dataset = torchvision.datasets.ImageFolder(object_data_folder, transform=transform_train)

    train_data_loader = torch.utils.data.DataLoader(object_dataset, batch_size=100, shuffle=False)

    object_class = 0
    sample_class = 1
    sample_rep, class_rep, labels = meta_utils.extract_features(train_data_loader, encoder, [0,1], device)
    sample_rep = sample_rep.numpy()
    sample = sample_rep[-1].reshape(1,-1)
    labels = labels[0:-1]
    sample_rep = sample_rep[0:-1]
    new_class = max(list(set(memory.trn_true_labels))) + 1
    labels = labels.numpy() + new_class

    memory.trn_memory = np.concatenate([memory.trn_memory, sample_rep])
    memory.trn_true_labels = np.concatenate([memory.trn_true_labels, labels])

    # sample = getSample(object_dataset, encoder, device)
    top_classes, x1 = getSimilarClasses(sample, memory, memory.trn_true_labels,top_n, top_k)

    final_label = classifySample(sample, model, x1, top_classes, device)

    print(f"top classes are {top_classes}")
    if final_label < 0:
        print(f"The object cannot be identified")
    else:

        print(f"the final label is {final_label}.")

    sample_image = object_dataset[-1][0]
    showResults(sample_image, image_dataset, object_dataset, top_classes, final_label)

    return



def getSample(object_dataset, encoder, device):

    (sample, label)= object_dataset[-1]
    batch_shape = [1, sample.shape[0], sample.shape[1], sample.shape[2]]


    sample = sample.view(batch_shape).to(device)

    _, feature_vector = encoder(sample)

    return feature_vector.detach().cpu()

def getSimilarClasses(sample, memory, true_labels, top_n, top_k):

    trn_cls_rep = []
    sort_idx = []
    unique_class_list = np.array(list(set(true_labels)))
    for cls in unique_class_list:
        trn_cls_rep.append(memory.trn_memory[(true_labels == cls).nonzero()].mean(axis=0))
        sort_idx.append((true_labels == cls).nonzero())

    train_cls_rep = torch.tensor(trn_cls_rep)
    sort_idx = torch.tensor(sort_idx)

    sim = sklearn.metrics.pairwise.cosine_similarity(sample,
                                             train_cls_rep)
    sim_idx = sim.argsort(axis=1).squeeze()

    top_classes = unique_class_list[sim_idx[-top_n:]]

    x1 = []
    # Sort labels and feature vectors
    for cls in top_classes:
        class_vectors = memory.trn_memory[(true_labels == cls)]
        sim1 = sklearn.metrics.pairwise.cosine_similarity(sample,
                                             class_vectors)
        sim1_idx = sim1.argsort(axis=1).squeeze()
        x1.append(class_vectors[sim1_idx[-top_k:]])


    x1 = np.stack(x1)

    return top_classes, x1



def classifySample(sample, model, x1, top_classes, device):
    input_sample = torch.tensor(sample, device=device).reshape(1,1,-1)
    input_sample = input_sample.repeat_interleave(top_classes.shape[0],dim=0)
    x1 = torch.tensor(x1,device=device)
    model.eval()
    with torch.no_grad():

        y, matching_layer = model(input_sample, x1.to(device))
        predicted = y.detach().clone().sigmoid()

    if predicted.max() > 0.5:
        final_label = top_classes[predicted.argmax()]
    else:
        final_label = -1

    return final_label


def showResults(sample, old_image_dataset,new_image_dataset, top_classes, final_label):

    image_list = []
    label_list = []
    for cls in top_classes:

        if cls in old_image_dataset.class_idx['l2ac_train']:

            im ,  label = old_image_dataset.getImageFromClass(cls)

        else:
            im, label = new_image_dataset[0]
            im = np.transpose(im.numpy(), (1, 2, 0))

        image_list.append(im)
        label_list.append(im)
    sample = sample.numpy().transpose(1,2,0)
    image_list.append(sample)
    images = torch.tensor(np.stack(image_list).transpose(0,3,1,2))
    im_grid = make_grid(images, nrow=4)
    plt.figure(figsize=(10, 4))
    plt.imshow(np.transpose(im_grid.numpy(), (1, 2, 0)))
    plt.show()

    return


def parseConfigFile(device):
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file")
    args = parser.parse_args()
    evaluation_config_file = args.config_file

    with open(evaluation_config_file) as file:
        evaluation_config = yaml.load(file, Loader=yaml.FullLoader)


    experiment_name = str(evaluation_config['experiment_name'])
    exp_folder = f'output/{experiment_name}'
    trn_config_file = f'{exp_folder}/{experiment_name}_config.yaml'
    object_name = evaluation_config['object_name']
    object_data_folder = f"datasets/{object_name}/"
    with open(trn_config_file) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)


    same_class_reverse = config['same_class_reverse']
    same_class_extend_entries = config['same_class_extend_entries']

    batch_size = config['batch_size']

    # L2AC Parameters
    top_k = int(config['top_k'])
    top_n = int(config['top_n'])
    encoder_classes = config['class_ratio']['encoder_train']
    train_classes = config['class_ratio']['l2ac_train']
    test_classes = config['class_ratio']['l2ac_test']
    train_samples_per_cls = config['train_samples_per_cls']

    encoder = config['encoder']
    feature_layer = config['feature_layer']
    image_resize = config['image_resize']
    unfreeze_layer = config['unfreeze_layer']
    test_class_selection = config['test_class_selection']
    feature_scaling = config['feature_scaling']
    dataset_path = f"datasets/{config['dataset_path']}" + f'/{encoder}/{feature_layer}_{feature_scaling}_{image_resize}_{unfreeze_layer}_{train_classes}_{train_samples_per_cls}_{top_n}_{test_class_selection}.npz'
    dataset_class = config['dataset_class']
    enable_training = True
    meta_dataset = eval('ObjectDatasets.' + dataset_class)(dataset_path, top_n, top_k, train_classes, train_samples_per_cls
                                                      ,enable_training,  same_class_reverse, same_class_extend_entries)



    features_size = len(meta_dataset.trn_memory[0])


    model_path = 'output/' + str(config['name']) + '/' + str(config['name']) + '_model.pt'
    model_class = config['model_class']
    model = eval('RecognitionModels.' + model_class)(train_classes,features_size, batch_size, top_k).to(device)
    OpenWorldUtils.loadModel(model, model_path)
    num_classes = config['class_ratio']['encoder_train']
    pretrained = True
    figure_size = config['image_resize']
    encoder_class = config['encoder']
    encoder = eval('RecognitionModels.' + encoder_class)(model_class, num_classes, feature_layer, unfreeze_layer,
                                                     feature_scaling, pretrained).to(device)
    encoder_file_path = f'datasets/{config["dataset_path"]}/{encoder_class}/feature_encoder_{figure_size}_{unfreeze_layer}_{encoder_classes}.pt'
    encoder.load_state_dict(torch.load(encoder_file_path))

    class_ratio = config['class_ratio']

    train_phase = 'l2ac_train'
    image_dataset = eval('ObjectDatasets.' + config['dataset_path'] + "Dataset")('datasets/' + config['dataset_path'], class_ratio, train_phase, figure_size)


    return model, meta_dataset, image_dataset, encoder, object_data_folder, image_resize, top_n, top_k

if __name__ == "__main__":
    main()