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

    train_phase = 'l2ac_test'


    # Parse config file
    dataset, model, top_n, train_classes, test_classes, train_samples_per_cls, randomize_samples, config = parseConfigFile(device, multiple_gpu, train_phase)



    # Setup dataset
    (test_data, _) = dataset.getData()



    feature_layer = config['feature_layer']
    model_class = config['model_class']
    dataset_path = f"datasets/{config['dataset_path']}/{model_class}"
    unfreeze_layer = config['unfreeze_layer']
    image_resize = config['image_resize']

    memory_path = f'{dataset_path}/{feature_layer}_{image_resize}_{unfreeze_layer}_{train_classes}_{train_samples_per_cls}_{top_n}_diff_cls'
    train_memory_path = f'{memory_path}_train.npz'

    # If dataset folder does not exist make folder
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)


    test_classes_idx = dataset.class_idx[train_phase]

    tst_data_rep, tst_data_cls_rep, tst_labels_rep = meta_utils.extract_features(test_data, model, test_classes_idx,
                                                                                 device)

    tst_samples_per_cls = int(len(test_data) / tst_data_cls_rep.shape[0])
    print(f'Rank validation samples with {test_classes} classes, {tst_samples_per_cls} samples per class')
    valid_X0, valid_X1, valid_Y = meta_utils.rank_samples_from_memory(test_classes_idx, tst_data_rep, tst_data_cls_rep,
                                                                      tst_labels_rep, test_classes_idx,
                                                                      tst_samples_per_cls, 1,
                                                                      randomize_samples)


    print(f'Save results to {memory_path}_test.npz')
    np.savez(f'{memory_path}_test.npz',
             test_rep=tst_data_rep, tst_labels_rep=tst_labels_rep,
             valid_X0=valid_X0, valid_X1=valid_X1, valid_Y=valid_Y)






    return memory_path

def combineTrainTest(memory_path):

    train_data = np.load(f'{memory_path}_train.npz')
    test_data = np.load(f'{memory_path}_test.npz')

    np.savez(f'{memory_path}.npz',
             train_rep=train_data['train_rep'], trn_labels_rep=train_data['trn_labels_rep'],  # including all validation examples.
             test_rep=test_data['test_rep'], tst_labels_rep=test_data['tst_labels_rep'],  # including all validation examples.
             train_X0=train_data['train_X0'], train_X1=train_data['train_X1'], train_Y=train_data['train_Y'],
             valid_X0=test_data['valid_X0'], valid_X1=test_data['valid_X1'], valid_Y=test_data['valid_Y'])




def parseConfigFile(device, multiple_gpu, train_phase):

    # Get config file argument
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file")
    args = parser.parse_args()
    config_file = args.config_file


    with open(config_file) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    # L2AC Parameters
    top_n = int(config['top_n'])
    train_samples_per_cls = config['train_samples_per_cls'] # Number of samples per class
    class_ratio = config['class_ratio']
    train_classes = class_ratio['l2ac_train'] # Classes used for training
    test_classes = class_ratio['l2ac_test'] # Classes used for validation
    randomize_samples = config['randomize_samples']

    # Load dataset
    dataset_path = f"datasets/{config['dataset_path']}"
    dataset_class = config['dataset_class']
    figure_size = config['image_resize']
    unfreeze_layer = config['unfreeze_layer']


    test_dataset = eval('ObjectDatasets.' + dataset_class)(dataset_path, class_ratio, train_phase, figure_size)

    # Load model
    model_class = config['model_class']
    pretrained = config['pretrained']
    feature_layer = config['feature_layer']
    num_classes = config['class_ratio']['encoder_train']
    model = eval('RecognitionModels.' + model_class)(model_class, num_classes, feature_layer, pretrained).to(device)
    encoder_file_path = f'{dataset_path}/{config["model_class"]}/feature_encoder_{figure_size}_{unfreeze_layer}.pt'

    model.load_state_dict(torch.load(encoder_file_path))

    return test_dataset, model, top_n, train_classes, test_classes, train_samples_per_cls, randomize_samples, config

if __name__ == "__main__":
    memory_path = main()
    combineTrainTest(memory_path)
