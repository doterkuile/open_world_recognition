import time
import torch
import numpy as np
import matplotlib.pyplot as plt
import yaml
import os
import argparse
import shutil
from open_world import loss_functions

from open_world import ObjectDatasets
from open_world import RecognitionModels
import torch.nn as nn




def parseConfigFile(device, multiple_gpu):

    # Parse input argument
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file")
    args = parser.parse_args()
    config_file = args.config_file

    # Open yaml file
    with open(config_file) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    # Save yaml file in output folder
    exp_name = str(config['name'])
    exp_folder = 'output/' + exp_name
    if not os.path.exists(exp_folder):
        os.makedirs(exp_folder)

    config_save_path = exp_folder + '/' + exp_name + '_config.yaml'
    try:
        shutil.copyfile(config_file, config_save_path)
    except shutil.SameFileError:
        pass

    # Training hyperparameters
    batch_size = config['batch_size']
    learning_rate = config['learning_rate']
    epochs = config['epochs']

    # L2AC Parameters
    top_k = int(config['top_k'])
    top_n = int(config['top_n'])
    train_classes = config['class_ratio']['l2ac_train']
    test_classes = config['class_ratio']['l2ac_test']
    train_samples = config['sample_ratio']['l2ac_train']

    # Dataset preparation parameters:
    same_class_reverse = config['same_class_reverse']
    same_class_extend_entries = config['same_class_extend_entries']

    # top_n classes + 1 (same class)
    weights = np.ones(batch_size-1) * 1/(top_n+1)
    weights = np.append(weights, (top_n)/(top_n+1))
    weights = torch.tensor(weights, dtype=torch.float).view(-1,1).to(device)

    # If same class extend entries is true then dataset is already balanced
    if not same_class_extend_entries:
        pos_weight = torch.tensor([top_n]).to(device).to(dtype=torch.float)
    else:
        pos_weight = torch.tensor(1.0).to(device).to(dtype=torch.float)


    ## Classes
    # Load dataset
    encoder = config['encoder']
    feature_layer = config['feature_layer']
    image_resize = config['image_resize']
    unfreeze_layer = config['unfreeze_layer']
    test_class_selection = config['test_class_selection']
    feature_scaling = config['feature_scaling']
    dataset_path = f"datasets/{config['dataset_path']}" + f'/{encoder}/{feature_layer}_{feature_scaling}_{image_resize}_{unfreeze_layer}_{train_classes}_{train_samples}_{top_n}_{test_class_selection}.npz'
    dataset_class = config['dataset_class']
    enable_training = True
    dataset = eval('ObjectDatasets.' + dataset_class)(dataset_path, top_n, top_k, train_classes, train_samples
                                                      ,enable_training,  same_class_reverse, same_class_extend_entries)

    features_size = len(dataset.memory[0])

    # Load model
 
    model_path = 'output/' + str(config['name']) + '/' + str(config['name']) + '_model.pt'
    model_class = config['model_class']
    model = eval('RecognitionModels.' + model_class)(train_classes,features_size, batch_size, top_k).to(device)

    # If multiple gpu's available
    # if multiple_gpu:
    #     print(f'The use of multiple gpus is enabled: using {torch.cuda.device_count()} gpus')
    #     batch_size = batch_size * torch.cuda.device_count()
    # model = nn.parallel.DistributedDataParallel(model)
    if not enable_training:
        print('Load model ' + model_path)
        loadModel(model, model_path)

    # criterion = eval('nn.' + config['criterion'])(pos_weight=pos_weight, reduction='mean')
    # criterion = eval('nn.' + config['criterion'])()
    criterion = eval(f'loss_functions.{config["criterion"]}')(pos_weight)
    optimizer = eval('torch.optim.' + config['optimizer'])(model.parameters(), lr=learning_rate, weight_decay=1e-5)

    # nn.BCEWithLogitsLoss(weight=)

    return dataset, model, criterion, optimizer, epochs, batch_size, learning_rate, config



def trainModel(model, train_loader, test_loader, epochs, criterion, optimizer):
    start_time = time.time()

    train_losses = []
    test_losses = []
    train_correct = []
    test_correct = []

    max_trn_batch = 800
    max_tst_batch = 300



    for i in range(epochs):
        trn_corr = 0
        tst_corr = 0

        # Run the training batches
        for b, (X_train, y_train) in enumerate(train_loader):

            # Limit the number of batches
            # if b == max_trn_batch:
            #     break
            b += 1
            # Apply the model
            y_pred = model(X_train.cuda())  # we don't flatten X-train here
            loss = criterion(y_pred, y_train.cuda())

            # Tally the number of correct predictions
            predicted = torch.max(y_pred.data, 1)[1]
            batch_corr = (predicted == y_train.cuda()).sum()
            trn_corr += batch_corr

            # Update parameters
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Print interim results
            if b % 600 == 0:
                print(f'epoch: {i:2}  batch: {b:4} [{10 * b:6}/60000]  loss: {loss.item():10.8f}  \
    accuracy: {trn_corr.item() * 100 / (10 * b):7.3f}%')

        train_losses.append(loss.cpu())
        train_correct.append(trn_corr.cpu())

        # Run the testing batches
        with torch.no_grad():
            for b, (X_test, y_test) in enumerate(test_loader):
                # Apply the model
                y_val = model(X_test.cuda())

                # Tally the number of correct predictions
                predicted = torch.max(y_val.data, 1)[1]
                tst_corr += (predicted == y_test.cuda()).sum()

        loss = criterion(y_val, y_test.cuda())
        test_losses.append(loss.cpu())
        test_correct.append(tst_corr.cpu())

    print(f'\nDuration: {time.time() - start_time:.0f} seconds')  # print the time elapsed

    return (train_losses, test_losses, train_correct, test_correct)



def saveTrainingLosses(train_losses, test_losses, train_correct, test_correct, file_path):
    np.savez(file_path, train_losses=train_losses, train_correct=train_correct, test_correct=test_correct, test_losses=test_losses)


def testModel(model, dataset):
    test_data = dataset.test_data
    class_labels = dataset.class_names
    x = np.random.randint(0, len(test_data))
    model.eval()

    with torch.no_grad():
        new_pred = model(test_data[x][0].view(dataset.image_shape).cuda()).argmax()
    print("Predicted value:", class_labels[new_pred.item()])


    im = dataset.getImage(x)


    plt.figure()
    plt.imshow(im, cmap="gist_yarg")
    plt.title(str(class_labels[new_pred.item()]))
    plt.show()

def saveModel(model, file_path):
    # if folder does not exist: make folder
    directory = '/'.join(file_path.split('/')[:-1])
    if not os.path.isdir(directory):
        os.mkdir(directory)
    torch.save(model.state_dict(), file_path)

def loadModel(model, file_path):
    try:
        model.load_state_dict(torch.load(file_path))
    except FileNotFoundError as e:
        print(e)
        print(f'Model at path {file_path} not found. Continuing with empty model')




