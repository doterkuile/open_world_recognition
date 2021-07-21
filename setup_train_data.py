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
    trn_dataset, tst_dataset, model, top_n, trn_classes, tst_classes, trn_samples_per_cls, randomize_samples, config = parseConfigFile(
        device, multiple_gpu)



    feature_layer = config['feature_layer']
    model_class = config['model_class']
    dataset_path = f"datasets/{config['dataset_path']}/{model_class}"
    unfreeze_layer = config['unfreeze_layer']
    image_resize = config['image_resize']
    feature_scaling = config['feature_scaling']
    memory_path = f'{dataset_path}/{feature_layer}_{feature_scaling}_{image_resize}_{unfreeze_layer}_{trn_classes}_{trn_samples_per_cls}_{top_n}'
    # If dataset folder does not exist make folder
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)

    # Setup dataset
    (trn_data, tst_data_same_cls) = trn_dataset.getData()
    (tst_data_diff_cls, _) = tst_dataset.getData()

    # setup dataloaders
    batch_size = 100
    (trn_loader, tst_loader_same_cls) = trn_dataset.getDataloaders(batch_size)
    (tst_loader_diff_cls, _) = tst_dataset.getDataloaders(batch_size)

    trn_cls_idx = [trn_data.class_to_idx[key] for key in trn_data.class_to_idx.keys()]
    tst_same_cls_idx = [tst_data_same_cls.class_to_idx[key] for key in tst_data_same_cls.class_to_idx.keys()]
    tst_diff_cls_idx = [tst_data_diff_cls.class_to_idx[key] for key in tst_data_diff_cls.class_to_idx.keys()]

    # Setup train data
    trn_data_rep, trn_cls_rep, trn_labels_rep, trn_X0, trn_X1, trn_Y = dataToL2ACFormat(model, trn_loader, trn_cls_idx,
                                                                                        trn_samples_per_cls, top_n,
                                                                                        device, randomize_samples)

    # Setup test data with different classes
    tst_samples_per_cls = int(len(tst_data_diff_cls) / len(tst_diff_cls_idx))
    tst_data_rep, tst_cls_rep, tst_labels_rep, tst_X0, tst_X1, tst_Y = dataToL2ACFormat(model, tst_loader_diff_cls,
                                                                                        tst_diff_cls_idx,
                                                                                        tst_samples_per_cls, 1,
                                                                                        device, randomize_samples)

    print(f'Save results to {memory_path}_diff_cls.npz')
    np.savez(f'{memory_path}_diff_cls.npz',
             train_rep=trn_data_rep, trn_labels_rep=trn_labels_rep,
             test_rep=tst_data_rep, tst_labels_rep=tst_labels_rep,
             train_X0=trn_X0, train_X1=trn_X1, train_Y=trn_Y,
             valid_X0=tst_X0, valid_X1=tst_X1, valid_Y=tst_Y)

    # Setup test data with same classes
    tst_samples_per_cls = int(len(tst_data_same_cls) / len(tst_same_cls_idx))
    tst_data_rep, tst_cls_rep, tst_labels_rep, tst_X0, tst_X1, tst_Y = dataToL2ACFormat(model, tst_loader_same_cls,
                                                                                        tst_same_cls_idx,
                                                                                        tst_samples_per_cls, 1,
                                                                                        device, randomize_samples)

    print(f'Save results to {memory_path}_same_cls.npz')
    np.savez(f'{memory_path}_same_cls.npz',
             train_rep=trn_data_rep, trn_labels_rep=trn_labels_rep,
             test_rep=tst_data_rep, tst_labels_rep=tst_labels_rep,
             train_X0=trn_X0, train_X1=trn_X1, train_Y=trn_Y,
             valid_X0=tst_X0, valid_X1=tst_X1, valid_Y=tst_Y)


    return


def dataToL2ACFormat(encoder, dataloader, cls_idx, samples_per_cls, top_n, device, randomize_samples):
    data_rep, cls_rep, labels_rep = meta_utils.extract_features(dataloader, encoder, cls_idx, device)

    X0, X1, Y = meta_utils.rank_samples_from_memory(cls_idx, data_rep, cls_rep, labels_rep, cls_idx, samples_per_cls,
                                                    top_n, randomize_samples)

    return data_rep, cls_rep, labels_rep, X0, X1, Y


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
    test_phase = 'l2ac_test'

    test_dataset = eval('ObjectDatasets.' + dataset_class)(dataset_path, class_ratio, test_phase, figure_size)
    # Load model
    model_class = config['model_class']
    pretrained = config['pretrained']
    feature_layer = config['feature_layer']
    num_classes = config['class_ratio']['encoder_train']
    feature_scaling = config['feature_scaling']
    unfreeze_layer = config['unfreeze_layer']
    model = eval('RecognitionModels.' + model_class)(model_class, num_classes, feature_layer, unfreeze_layer, feature_scaling, pretrained).to(device)
    encoder_file_path = f'{dataset_path}/{config["model_class"]}/feature_encoder_{figure_size}_{unfreeze_layer}.pt'

    model.load_state_dict(torch.load(encoder_file_path))

    return train_dataset, test_dataset, model, top_n, train_classes, test_classes, train_samples_per_cls, randomize_samples, config


if __name__ == "__main__":
    main()
