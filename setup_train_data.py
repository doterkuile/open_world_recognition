import torch
from open_world import ObjectDatasets
from open_world import RecognitionModels
import open_world.meta_learner.meta_learner_utils as meta_utils
from open_world.ObjectDatasets import TrainPhase
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

    load_features = False
    feature_layer = config['feature_layer']
    encoder_class = config['model_class']
    dataset_path = f"datasets/{config['dataset_path']}/{encoder_class}"
    unfreeze_layer = config['unfreeze_layer']
    image_resize = config['image_resize']
    feature_scaling = config['feature_scaling']
    trn_classes = class_ratio[TrainPhase.META_TRN.value]
    trn_samples_per_cls = sample_ratio['l2ac_train_samples']
    memory_path = f'{dataset_path}/{feature_layer}_{feature_scaling}_{image_resize}_{unfreeze_layer}_{trn_classes}_{trn_samples_per_cls}_{top_n}'
    feature_path = f'{dataset_path}/{feature_layer}_{feature_scaling}_{image_resize}_{unfreeze_layer}_features.npz'
    # If dataset folder does not exist make folder
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)


    complete_cls_idx = np.concatenate([trn_dataset.class_idx[TrainPhase.META_TRN.value],
                                  trn_dataset.class_idx[TrainPhase.META_VAL.value],
                                  trn_dataset.class_idx[TrainPhase.META_TST.value]])

    if not load_features:
        print("Extracting features")
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

    ## Mutually exclusive classes variant uses different classes in trianing and validaton
    trn_cls_idx = trn_dataset.class_idx[TrainPhase.META_TRN.value]




    ## Non-mutually exclusive classes variant uses same classes, but has to dived the samples in train, validation and test
    input_sample_idx = np.arange(0, sample_ratio['l2ac_train_samples'])
    memory_sample_idx = np.arange(0, sample_ratio['l2ac_train_samples'])
    print("Rank training data, same class")

    X0_trn, X1_trn, Y_trn = meta_utils.rank_input_to_memory(data_rep, labels, data_rep, labels, cls_rep, input_sample_idx,
                                                            memory_sample_idx, trn_cls_idx, complete_cls_idx, top_n, randomize_samples)

    input_sample_idx = np.arange(sample_ratio['l2ac_train_samples'],sample_ratio['l2ac_train_samples'] + sample_ratio['l2ac_val_samples'])
    print("Rank validation data, same class")

    X0_val, X1_val, Y_val = meta_utils.rank_input_to_memory(data_rep, labels, data_rep, labels, cls_rep, input_sample_idx,
                                                            memory_sample_idx, trn_cls_idx, complete_cls_idx, 1, randomize_samples)


    tst_cls_selection = 'same_cls'
    input_classes, input_samples, memory_classes, memory_samples, complete_cls_set = getTestIdxSelection(
                                                                                                    tst_cls_selection,
                                                                                                    class_ratio,
                                                                                                    sample_ratio)

    X0_tst, X1_tst, Y_tst = meta_utils.rank_test_data(data_rep, labels, data_rep, labels, cls_rep, input_samples,
                                          memory_samples, input_classes, memory_classes, complete_cls_set,
                                          memory_classes.shape[0])




    print(f'Save results to {memory_path}_same_cls.npz')
    np.savez(f'{memory_path}_same_cls.npz',
             data_rep=data_rep, data_labels=labels, cls_rep=cls_rep,
             train_X0=X0_trn, train_X1=X1_trn, train_Y=Y_trn,
             valid_X0=X0_val, valid_X1=X1_val, valid_Y=Y_val,
             test_X0=X0_tst, test_X1=X1_tst, test_Y=Y_tst)

    ## Mutually exclusive classes variantsamples available per class
    input_sample_idx = np.arange(0, sample_ratio['l2ac_train_samples'])

    val_cls_idx = trn_dataset.class_idx[TrainPhase.META_VAL.value]

    print("Rank validation data, different class")
    X0_val, X1_val, Y_val = meta_utils.rank_input_to_memory(data_rep, labels, data_rep, labels, cls_rep, input_sample_idx,
                                                            memory_sample_idx, val_cls_idx, complete_cls_idx, 1, randomize_samples)

    tst_cls_selection = 'diff_cls'
    input_classes, input_samples, memory_classes, memory_samples, complete_cls_set = getTestIdxSelection(
                                                                                                    tst_cls_selection,
                                                                                                    class_ratio,
                                                                                                    sample_ratio)

    X0_tst, X1_tst, Y_tst = meta_utils.rank_test_data(data_rep, labels, data_rep, labels, cls_rep, input_samples,
                                          memory_samples, input_classes, memory_classes, complete_cls_set,
                                          memory_classes.shape[0])


    print(f'Save results to {memory_path}_diff_cls.npz')
    np.savez(f'{memory_path}_diff_cls.npz',
             data_rep=data_rep, data_labels=labels, cls_rep=cls_rep,
             train_X0=X0_trn, train_X1=X1_trn, train_Y=Y_trn,
             valid_X0=X0_val, valid_X1=X1_val, valid_Y=Y_val,
             test_X0=X0_tst, test_X1=X1_tst, test_Y=Y_tst)
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
    batch_size = 128
    (trn_loader, val_loader, tst_loader) = dataset.getDataloaders(batch_size)

    trn_data_rep, trn_cls_rep, trn_labels = meta_utils.extract_features(trn_loader, encoder, class_set, device)
    val_data_rep, val_cls_rep, val_labels = meta_utils.extract_features(val_loader, encoder, class_set, device)
    tst_data_rep, tst_cls_rep, tst_labels = meta_utils.extract_features(tst_loader, encoder, class_set, device)

    data_rep = np.concatenate([trn_data_rep, val_data_rep, tst_data_rep], axis=0)
    labels = np.concatenate([trn_labels, val_labels, tst_labels], axis=0)

    data_rep, labels = sortData(data_rep, labels, class_set)

    return data_rep, labels, trn_cls_rep

def getTestIdxSelection(tst_cls_selection, class_ratio, sample_ratio):
    if tst_cls_selection == 'same_cls':
        input_classes = np.arange(class_ratio[TrainPhase.ENCODER_TRN.value],
                                  class_ratio[TrainPhase.ENCODER_TRN.value] + class_ratio[TrainPhase.META_TRN.value] +
                                  class_ratio[TrainPhase.META_VAL.value] + class_ratio[TrainPhase.META_TST.value])

        input_samples = np.arange(sample_ratio['l2ac_train_samples'] + sample_ratio['l2ac_val_samples'],
                                  sample_ratio['l2ac_train_samples'] + sample_ratio['l2ac_val_samples'] +
                                  sample_ratio['l2ac_test_samples'])
        memory_classes = np.arange(class_ratio[TrainPhase.ENCODER_TRN.value],
                                   class_ratio[TrainPhase.ENCODER_TRN.value] + class_ratio[TrainPhase.META_TRN.value])
        memory_samples = np.arange(0, sample_ratio['l2ac_train_samples'])
    else:
        input_classes = np.arange(class_ratio[TrainPhase.ENCODER_TRN.value] + class_ratio[TrainPhase.META_TRN.value] +
                                  class_ratio[TrainPhase.META_VAL.value],
                                  class_ratio[TrainPhase.ENCODER_TRN.value] + class_ratio[TrainPhase.META_TRN.value] +
                                  class_ratio[TrainPhase.META_VAL.value] + class_ratio[TrainPhase.META_TST.value])

        input_samples = np.arange(sample_ratio['l2ac_train_samples'] + sample_ratio['l2ac_val_samples'],
                                  sample_ratio['l2ac_train_samples'] + sample_ratio['l2ac_val_samples'] +
                                  sample_ratio['l2ac_test_samples'])
        memory_classes = np.arange(class_ratio[TrainPhase.ENCODER_TRN.value] + class_ratio[TrainPhase.META_TRN.value] +
                                  class_ratio[TrainPhase.META_VAL.value],
                                  class_ratio[TrainPhase.ENCODER_TRN.value] + class_ratio[TrainPhase.META_TRN.value] +
                                  class_ratio[TrainPhase.META_VAL.value] + class_ratio[TrainPhase.META_TST.value])

        memory_samples = np.arange(0, sample_ratio['l2ac_train_samples'])

    complete_cls_set = np.arange(class_ratio[TrainPhase.ENCODER_TRN.value],
                                 class_ratio[TrainPhase.ENCODER_TRN.value] + class_ratio[TrainPhase.META_TRN.value] + class_ratio[TrainPhase.META_VAL.value] +
                                 class_ratio[TrainPhase.META_TST.value])

    return input_classes, input_samples, memory_classes, memory_samples, complete_cls_set




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
    encoder_classes = class_ratio[TrainPhase.ENCODER_TRN.value]
    train_classes = class_ratio[TrainPhase.META_TRN.value]  # Classes used for training
    test_classes = class_ratio[TrainPhase.META_TST.value]  # Classes used for validation
    randomize_samples = config['randomize_samples']

    # Load dataset
    dataset_path = f"datasets/{config['dataset_path']}"
    dataset_class = config['dataset_class']
    figure_size = config['image_resize']
    unfreeze_layer = config['unfreeze_layer']
    remove_teapot_cls = config['remove_teapot']

    train_phase = TrainPhase.META_TRN
    train_dataset = eval('ObjectDatasets.' + dataset_class)(dataset_path, class_ratio, train_phase, figure_size, remove_teapot_cls)

    test_phase = TrainPhase.META_VAL
    val_dataset = eval('ObjectDatasets.' + dataset_class)(dataset_path, class_ratio, test_phase, figure_size, remove_teapot_cls)

    test_phase = TrainPhase.META_TST
    test_dataset = eval('ObjectDatasets.' + dataset_class)(dataset_path, class_ratio, test_phase, figure_size, remove_teapot_cls)



    # Load model
    encoder_class = config['model_class']
    pretrained = config['pretrained']
    feature_layer = config['feature_layer']
    num_classes = config['class_ratio'][TrainPhase.ENCODER_TRN.value]
    feature_scaling = config['feature_scaling']
    unfreeze_layer = config['unfreeze_layer']
    model = eval('RecognitionModels.' + encoder_class)(encoder_class, num_classes, feature_layer, unfreeze_layer, feature_scaling, pretrained).to(device)
    encoder_file_path = f'{dataset_path}/{config["model_class"]}/feature_encoder_{figure_size}_{unfreeze_layer}_{encoder_classes}.pt'

    model.load_state_dict(torch.load(encoder_file_path))

    return train_dataset, val_dataset, test_dataset, model, class_ratio, sample_ratio, top_n, randomize_samples, config


if __name__ == "__main__":
    main()
