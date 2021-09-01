import torch
from open_world import ObjectDatasets
from open_world import RecognitionModels
import open_world.meta_learner.meta_learner_utils as meta_utils
import yaml
import numpy as np
import argparse
import os


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
        print(f'Running with {torch.cuda.device_count()} GPUs')

    # Parse config file
    trn_dataset, val_dataset, tst_dataset, encoder, class_ratio, sample_ratio, top_n, randomize_samples, config = parseConfigFile(
        device, multiple_gpu)

    load_features = True
    feature_layer = config['feature_layer']
    encoder_class = config['model_class']
    dataset_path = f"datasets/{config['dataset_path']}/{encoder_class}"
    unfreeze_layer = config['unfreeze_layer']
    image_resize = config['image_resize']
    feature_scaling = config['feature_scaling']
    trn_classes = class_ratio['l2ac_train']
    trn_samples_per_cls = sample_ratio['l2ac_train']
    memory_path = f'{dataset_path}/{feature_layer}_{feature_scaling}_{image_resize}_{unfreeze_layer}_{trn_classes}_{trn_samples_per_cls}_{top_n}'
    feature_path = f'{dataset_path}/{feature_layer}_{feature_scaling}_{image_resize}_{unfreeze_layer}_features.npz'
    # If dataset folder does not exist make folder
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)


    complete_cls_idx = np.concatenate([trn_dataset.class_idx['l2ac_train'],
                                  trn_dataset.class_idx['l2ac_val'],
                                  trn_dataset.class_idx['l2ac_test']])

    if not load_features:
        trn_data_rep, trn_labels, trn_cls_rep = extract_all_features(trn_dataset, complete_cls_idx, encoder, device)
        val_data_rep, val_labels, val_cls_rep = extract_all_features(val_dataset, complete_cls_idx, encoder, device)
        tst_data_rep, tst_labels, tst_cls_rep = extract_all_features(tst_dataset, complete_cls_idx, encoder, device)

        data_rep = np.concatenate([trn_data_rep, val_data_rep, tst_data_rep], axis=0)
        labels = np.concatenate([trn_labels, val_labels, tst_labels], axis=0)
        cls_rep = np.concatenate([trn_cls_rep, val_cls_rep, tst_cls_rep], axis=0)
        np.savez(feature_path,
                 data_rep=data_rep, labels=labels,
                 cls_rep=cls_rep)
    else:
        data = np.load(feature_path)
        data_rep = data["data_rep"]
        labels = data['labels']
        cls_rep = data['cls_rep']



    trn_cls_idx = trn_dataset.class_idx['l2ac_train']
    val_cls_idx = trn_dataset.class_idx['l2ac_val']

    input_sample_idx = np.arange(0, 350)
    memory_sample_idx = np.arange(0, 350)
    X0_trn, X1_trn, Y_trn = meta_utils.rank_input_to_memory(data_rep, labels, data_rep, labels, cls_rep, input_sample_idx,
                                                            memory_sample_idx, trn_cls_idx, complete_cls_idx, top_n, randomize_samples)

    input_sample_idx = np.arange(350,450)
    X0_val, X1_val, Y_val = meta_utils.rank_input_to_memory(data_rep, labels, data_rep, labels, cls_rep, input_sample_idx,
                                                            memory_sample_idx, trn_cls_idx, complete_cls_idx, top_n, randomize_samples)


    print(f'Save results to {memory_path}_same_cls.npz')
    np.savez(f'{memory_path}_same_cls.npz',
             data_rep=data_rep, data_labels=labels, cls_rep=cls_rep,
             train_X0=X0_trn, train_X1=X1_trn, train_Y=Y_trn,
             valid_X0=X0_val, valid_X1=X1_val, valid_Y=Y_val)

    input_sample_idx = np.arange(0, 550)
    memory_sample_idx = np.arange(0, 550)
    X0_trn, X1_trn, Y_trn = meta_utils.rank_input_to_memory(data_rep, labels, data_rep, labels, cls_rep, input_sample_idx,
                                                            memory_sample_idx, trn_cls_idx, complete_cls_idx, top_n, randomize_samples)


    X0_val, X1_val, Y_val = meta_utils.rank_input_to_memory(data_rep, labels, data_rep, labels, cls_rep, input_sample_idx,
                                                            memory_sample_idx, val_cls_idx, complete_cls_idx, top_n, randomize_samples)

    print(f'Save results to {memory_path}_diff_cls.npz')
    np.savez(f'{memory_path}_diff_cls.npz',
             data_rep=data_rep, data_labels=labels, cls_rep=cls_rep,
             train_X0=X0_trn, train_X1=X1_trn, train_Y=Y_trn,
             valid_X0=X0_val, valid_X1=X1_val, valid_Y=Y_val)

    return


def sortData(data_rep, labels, classes):
    sort_idx = []
    for cls in classes:
        if cls in labels:
            sort_idx.append((labels == cls).nonzero()[0])

    sort_idx = np.concatenate(sort_idx, axis=0).squeeze()

    # Sort labels and feature vectors
    sorted_labels = labels[sort_idx]
    sorted_data_rep = data_rep[sort_idx,:]

    return sorted_data_rep, sorted_labels

def extract_all_features(dataset, class_set, encoder, device):
    batch_size = 100
    (trn_loader, val_loader, tst_loader) = dataset.getDataloaders(batch_size)

    trn_data_rep, trn_cls_rep, trn_labels = meta_utils.extract_features(trn_loader, encoder, class_set, device)
    val_data_rep, val_cls_rep, val_labels = meta_utils.extract_features(val_loader, encoder, class_set, device)
    tst_data_rep, tst_cls_rep, tst_labels = meta_utils.extract_features(tst_loader, encoder, class_set, device)

    data_rep = np.concatenate([trn_data_rep, val_data_rep, tst_data_rep], axis=0)
    labels = np.concatenate([trn_labels, val_labels, tst_labels], axis=0)

    data_rep, labels = sortData(data_rep, labels, class_set)

    return data_rep, labels, trn_cls_rep

def parseConfigFile(device, multiple_gpu):
    # Get config file argument
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file")
    args = parser.parse_args()
    config_file = args.config_file

    with open(config_file) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    # L2AC Parameters
    top_n = int(config['top_n'])
    train_samples_per_cls = config['train_samples_per_cls']  # Number of samples per class
    class_ratio = config['class_ratio']
    sample_ratio = config['sample_ratio']
    encoder_classes = class_ratio['encoder_train']
    train_classes = class_ratio['l2ac_train']  # Classes used for training
    test_classes = class_ratio['l2ac_test']  # Classes used for validation
    randomize_samples = config['randomize_samples']

    # Load dataset
    dataset_path = f"datasets/{config['dataset_path']}"
    dataset_class = config['dataset_class']
    figure_size = config['image_resize']
    unfreeze_layer = config['unfreeze_layer']

    train_phase = 'l2ac_train'
    train_dataset = eval('ObjectDatasets.' + dataset_class)(dataset_path, class_ratio, train_phase, figure_size)

    test_phase = 'l2ac_val'
    val_dataset = eval('ObjectDatasets.' + dataset_class)(dataset_path, class_ratio, test_phase, figure_size)

    test_phase = 'l2ac_test'
    test_dataset = eval('ObjectDatasets.' + dataset_class)(dataset_path, class_ratio, test_phase, figure_size)



    # Load model
    encoder_class = config['model_class']
    pretrained = config['pretrained']
    feature_layer = config['feature_layer']
    num_classes = config['class_ratio']['encoder_train']
    feature_scaling = config['feature_scaling']
    unfreeze_layer = config['unfreeze_layer']
    model = eval('RecognitionModels.' + encoder_class)(encoder_class, num_classes, feature_layer, unfreeze_layer, feature_scaling, pretrained).to(device)
    encoder_file_path = f'{dataset_path}/{config["model_class"]}/feature_encoder_{figure_size}_{unfreeze_layer}_{encoder_classes}.pt'

    model.load_state_dict(torch.load(encoder_file_path))

    return train_dataset, val_dataset, test_dataset, model, class_ratio, sample_ratio, top_n, randomize_samples, config


if __name__ == "__main__":
    main()
