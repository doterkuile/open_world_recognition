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

    # Setup dataset
    (trn_data, val_data_same_cls, tst_data_same_cls) = trn_dataset.getData()
    (val_data_diff_cls, _, _) = val_dataset.getData()
    (tst_data_diff_cls, _, _) = tst_dataset.getData()

    # setup dataloaders
    batch_size = 100
    (trn_loader, val_loader_same_cls, tst_loader_same_cls) = trn_dataset.getDataloaders(batch_size)
    (val_loader_diff_cls, _, _) = val_dataset.getDataloaders(batch_size)
    (tst_loader_diff_cls, tst_loader_input_1, tst_loader_input_2) = tst_dataset.getDataloaders(batch_size)

    all_cls_idx = np.concatenate([trn_dataset.class_idx['l2ac_train'],
                                  trn_dataset.class_idx['l2ac_val'],
                                  trn_dataset.class_idx['l2ac_test']])




    trn_cls_idx = np.array([trn_data.class_to_idx[key] for key in trn_data.class_to_idx.keys()])

    val_same_cls_idx = [val_data_same_cls.class_to_idx[key] for key in val_data_same_cls.class_to_idx.keys()]
    val_diff_cls_idx = [val_data_diff_cls.class_to_idx[key] for key in val_data_diff_cls.class_to_idx.keys()]
    tst_same_cls_idx = [tst_data_same_cls.class_to_idx[key] for key in tst_data_same_cls.class_to_idx.keys()]
    tst_diff_cls_idx = [tst_data_diff_cls.class_to_idx[key] for key in tst_data_diff_cls.class_to_idx.keys()]




    if not load_features:
        trn_data_rep, trn_labels, trn_cls_rep = extract_all_features(trn_dataset, all_cls_idx, encoder, device)
        val_data_rep, val_labels, val_cls_rep = extract_all_features(val_dataset, all_cls_idx, encoder, device)
        tst_data_rep, tst_labels, tst_cls_rep = extract_all_features(tst_dataset, all_cls_idx, encoder, device)

        data_rep = np.concatenate([trn_data_rep, val_data_rep, tst_data_rep], axis=0)
        labels = np.concatenate([trn_labels, val_labels, tst_labels], axis=0)

        np.savez(feature_path,
                 data_rep=data_rep, labels=labels,
                 trn_cls_rep=trn_cls_rep, val_cls_rep=val_cls_rep,
                 tst_cls_rep=tst_cls_rep)
    else:
        data = np.load(feature_path)
        data_rep = data["data_rep"]
        labels = data['labels']
        trn_cls_rep = data['trn_cls_rep']
        val_cls_rep = data['val_cls_rep']
        tst_cls_rep = data['tst_cls_rep']

    load_mutually_inclusive_data(trn_dataset, val_dataset, tst_dataset, encoder, device, top_n, memory_path,
                                 randomize_samples, data_rep, trn_cls_rep, labels)

    memory_sample_offset = 0
    input_sample_offset = 0
    memory_samples_per_cls = int(len(trn_data)/len(trn_data.classes))
    input_samples_per_cls = int(len(trn_data)/len(trn_data.classes))
    X0_trn, X1_trn, Y_trn = meta_utils.rank_input_to_memory(trn_cls_idx, trn_data_rep, trn_data_rep, trn_cls_rep, trn_labels,
                                                trn_labels, all_cls_idx, memory_sample_offset, memory_samples_per_cls,
                                                input_sample_offset, input_samples_per_cls, top_n, randomize_samples)


    memory_sample_offset = 0
    input_sample_offset = int(len(trn_data)/len(trn_data.classes))
    memory_samples_per_cls = int(len(trn_data)/len(trn_data.classes))
    input_samples_per_cls = int(len(val_data_same_cls)/len(val_data_same_cls.classes))
    X0_val, X1_val, Y_val = meta_utils.rank_input_to_memory(val_same_cls_idx, trn_data_rep, val_data_rep, trn_cls_rep, trn_labels,
                                                val_labels, all_cls_idx, memory_sample_offset, memory_samples_per_cls,
                                                input_sample_offset, input_samples_per_cls, 1, randomize_samples)

    memory_sample_offset = 0
    input_sample_offset = int(len(trn_data)/len(trn_data.classes)) + int(len(val_data_same_cls)/len(val_data_same_cls.classes))
    memory_samples_per_cls = int(len(trn_data)/len(trn_data.classes))
    input_samples_per_cls = int(len(tst_data_same_cls)/len(tst_data_same_cls.classes))
    X0_tst, X1_tst, Y_tst = meta_utils.rank_input_to_memory(tst_same_cls_idx, trn_data_rep, tst_data_rep, trn_cls_rep, trn_labels,
                                                tst_labels, all_cls_idx, memory_sample_offset, memory_samples_per_cls,
                                                input_sample_offset, input_samples_per_cls, top_n, randomize_samples)


    print(f'Save results to {memory_path}_same_cls.npz')
    np.savez(f'{memory_path}_same_cls.npz',
             data_rep=data_rep, data_labels=labels,
             train_X0=X0_trn, train_X1=X1_trn, train_Y=Y_trn,
             valid_X0=X0_val, valid_X1=X1_val, valid_Y=Y_val,
             test_X0=X0_tst, test_X1=X1_tst, test_Y=Y_tst)

    val_data_rep, val_cls_rep, val_labels = meta_utils.extract_features(val_loader_diff_cls, encoder, all_cls_idx, device)
    tst_data_rep, tst_cls_rep, tst_labels = meta_utils.extract_features(tst_loader_diff_cls, encoder, all_cls_idx, device)
    tst_input_rep_1, _, tst_input_labels_1 = meta_utils.extract_features(tst_loader_input_1, encoder, all_cls_idx, device)
    tst_input_rep_2, _, tst_input_labels_2 = meta_utils.extract_features(tst_loader_input_2, encoder, all_cls_idx, device)


    data_rep = np.concatenate([trn_data_rep, val_data_rep, tst_data_rep,tst_input_rep_1, tst_input_rep_2], axis=0)
    labels = np.concatenate([trn_labels, val_labels, tst_labels, tst_input_labels_1, tst_input_labels_2], axis=0)

    data_rep, labels = sortData(data_rep, labels, all_cls_idx)



    memory_sample_offset = 0
    input_sample_offset = 0
    memory_samples_per_cls = int(len(trn_data)/len(trn_data.classes))
    input_samples_per_cls = int(len(trn_data)/len(trn_data.classes))
    X0_val, X1_val, Y_val = meta_utils.rank_input_to_memory(val_diff_cls_idx, val_data_rep, val_data_rep, val_cls_rep, val_labels,
                                                val_labels, all_cls_idx, memory_sample_offset, memory_samples_per_cls,
                                                input_sample_offset, input_samples_per_cls, 1, randomize_samples)

    # TODO: Fix tst input and memory

    memory_sample_offset = 0
    memory_samples_per_cls = 500

    input_sample_offset = memory_samples_per_cls
    input_samples_per_cls = 50
    X0_tst, X1_tst, Y_tst = meta_utils.rank_input_to_memory(tst_diff_cls_idx, tst_data_rep, tst_data_rep, tst_cls_rep, tst_labels,
                                                tst_labels, all_cls_idx, memory_sample_offset, memory_samples_per_cls,
                                                input_sample_offset, input_samples_per_cls, top_n, randomize_samples)

    print(f'Save results to {memory_path}_diff_cls.npz')
    np.savez(f'{memory_path}_diff_cls.npz',
             data_rep=data_rep, data_labels=labels,
             train_X0=X0_trn, train_X1=X1_trn, train_Y=Y_trn,
             valid_X0=X0_val, valid_X1=X1_val, valid_Y=Y_val,
             test_X0=X0_tst, test_X1=X1_tst, test_Y=Y_tst)


    return


def dataToL2ACFormat(encoder, dataloader, cls_idx, samples_per_cls, top_n, device, randomize_samples):
    data_rep, cls_rep, labels_rep = meta_utils.extract_features(dataloader, encoder, cls_idx, device)

    X0, X1, Y = meta_utils.rank_samples_from_memory(cls_idx, data_rep, cls_rep, labels_rep, cls_idx, samples_per_cls,
                                                    top_n, randomize_samples)

    return data_rep, cls_rep, labels_rep, X0, X1, Y

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



def load_mutually_inclusive_data(trn_dataset, val_dataset, tst_dataset, encoder, device, top_n, memory_path, randomize_samples, data_rep, cls_rep, labels):
    # Setup dataset
    (trn_data, val_data, tst_data) = trn_dataset.getData()
    # (val_data_diff_cls, _, _) = val_dataset.getData()
    # (tst_data_diff_cls, _, _) = tst_dataset.getData()

    # setup dataloaders
    batch_size = 100
    (trn_loader, val_loader_same_cls, tst_loader_same_cls) = trn_dataset.getDataloaders(batch_size)
    # (val_loader_diff_cls, _, _) = val_dataset.getDataloaders(batch_size)
    # (tst_loader_diff_cls, tst_loader_input_1, tst_loader_input_2) = tst_dataset.getDataloaders(batch_size)




    all_cls_idx = np.concatenate([trn_dataset.class_idx['l2ac_train'],
                                  trn_dataset.class_idx['l2ac_val'],
                                  trn_dataset.class_idx['l2ac_test']])



    trn_cls_idx = np.array([trn_data.class_to_idx[key] for key in trn_data.class_to_idx.keys()])

    val_cls_idx = np.array([val_data.class_to_idx[key] for key in val_data.class_to_idx.keys()])
    # val_diff_cls_idx = [val_data_diff_cls.class_to_idx[key] for key in val_data_diff_cls.class_to_idx.keys()]
    tst_same_cls_idx = np.array([tst_data.class_to_idx[key] for key in tst_data.class_to_idx.keys()])
    # tst_diff_cls_idx = [tst_data_diff_cls.class_to_idx[key] for key in tst_data_diff_cls.class_to_idx.keys()]


    memory_sample_offset = 0
    input_sample_offset = 0
    memory_samples_per_cls = int(len(trn_data)/len(trn_data.classes))
    input_samples_per_cls = int(len(trn_data)/len(trn_data.classes))

    input_sample_idx = np.arange(0,350)
    memory_sample_idx = np.arange(0,350)

    X0_trn, X1_trn, Y_trn = meta_utils.rank_input_to_memory(data_rep, labels, data_rep, labels, cls_rep, input_sample_idx,
                                                            memory_sample_idx, trn_cls_idx, all_cls_idx, top_n, randomize_samples)

    input_sample_idx = np.arange(350,450)
    X0_val, X1_val, Y_val = meta_utils.rank_input_to_memory(data_rep, labels, data_rep, labels, cls_rep, input_sample_idx,
                                                            memory_sample_idx, trn_cls_idx, all_cls_idx, top_n, randomize_samples)


# TODO Check if tst ranking can be done now or later
    input_sample_idx = np.arange(450,550)

    X0_trn, X1_trn, Y_trn = meta_utils.rank_input_to_memory(data_rep, labels, data_rep, labels, cls_rep, input_sample_idx,
                                                            memory_sample_idx, trn_cls_idx, all_cls_idx, top_n, randomize_samples)



    memory_sample_offset = 0
    input_sample_offset = int(len(trn_data)/len(trn_data.classes))
    memory_samples_per_cls = int(len(trn_data)/len(trn_data.classes))
    input_samples_per_cls = int(len(val_data_same_cls)/len(val_data_same_cls.classes))
    X0_val, X1_val, Y_val = meta_utils.rank_input_to_memory(val_same_cls_idx, trn_data_rep, val_data_rep, trn_cls_rep, trn_labels,
                                                val_labels, all_cls_idx, memory_sample_offset, memory_samples_per_cls,
                                                input_sample_offset, input_samples_per_cls, 1, randomize_samples)

    memory_sample_offset = 0
    input_sample_offset = int(len(trn_data)/len(trn_data.classes)) + int(len(val_data_same_cls)/len(val_data_same_cls.classes))
    memory_samples_per_cls = int(len(trn_data)/len(trn_data.classes))
    input_samples_per_cls = int(len(tst_data_same_cls)/len(tst_data_same_cls.classes))
    X0_tst, X1_tst, Y_tst = meta_utils.rank_input_to_memory(tst_same_cls_idx, trn_data_rep, tst_data_rep, trn_cls_rep, trn_labels,
                                                tst_labels, all_cls_idx, memory_sample_offset, memory_samples_per_cls,
                                                input_sample_offset, input_samples_per_cls, top_n, randomize_samples)


    print(f'Save results to {memory_path}_same_cls.npz')
    np.savez(f'{memory_path}_same_cls.npz',
             data_rep=data_rep, data_labels=labels,
             train_X0=X0_trn, train_X1=X1_trn, train_Y=Y_trn,
             valid_X0=X0_val, valid_X1=X1_val, valid_Y=Y_val,
             test_X0=X0_tst, test_X1=X1_tst, test_Y=Y_tst)
    return


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
