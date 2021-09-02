import torch
from sklearn import metrics

import torch.utils.data as data_utils
from open_world import OpenWorldUtils
from open_world import ObjectDatasets
from open_world import RecognitionModels
import open_world.meta_learner.meta_learner_utils as meta_utils
import open_world.plot_utils as plot_utils
import yaml
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import argparse
import os
import shutil
from torchvision.models import resnet50
from torchvision import datasets, transforms, models
import time
from tqdm import tqdm
import copy


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
        print(f"Running with {torch.cuda.device_count()} GPUs")

    # Parse config file
    (train_dataset, model, criterion, optimizer, epochs, batch_size, learning_rate,
     config) = parseConfigFile(
        device, multiple_gpu)

    exp_name = str(config['name'])
    exp_folder = 'output/' + exp_name
    figure_path = exp_folder + '/' + exp_name
    results_path = exp_folder + '/' + exp_name + '_results.npz'
    model_path = exp_folder + '/' + exp_name + '_model.pt'
    dataset_path = f'datasets/{config["dataset_path"]}'
    train_data, val_data, test_data = train_dataset.getData()
    encoder_train_classes = config['class_ratio']['encoder_train']
    figure_size = config['image_resize']
    unfreeze_layer = config['unfreeze_layer']
    encoder_file_path = f'{dataset_path}/{config["model_class"]}/feature_encoder_{figure_size}_{unfreeze_layer}_{encoder_train_classes}.pt'

    if encoder_train_classes == 0:
        OpenWorldUtils.saveModel(model,encoder_file_path)
        print("No finetuning of encoder, saving pretrained model only")
        return

    # train_data = datasets.CIFAR100(root=dataset_path, train=True, download=True, transform=train_dataset.transform_train)
    # test_data = datasets.CIFAR100(root=dataset_path, train=False, download=True, transform=train_dataset.transform_test)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, pin_memory=True)#, num_workers=4)
    test_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True, pin_memory=True)#, num_workers=4)


    trn_metrics, tst_metrics, best_state = trainMetaModel(model, train_loader, test_loader, epochs, criterion,optimizer,device)

    model.load_state_dict(best_state['model'])

    # Train metrics
    trn_loss = trn_metrics['loss']
    trn_acc = trn_metrics['accuracy']
    trn_precision = trn_metrics['precision']
    trn_recall = trn_metrics['recall']
    trn_F1 = trn_metrics['F1']
    trn_mean_pred = trn_metrics['mean_pred']
    trn_mean_true = trn_metrics['mean_true']


    # Test metrics
    tst_loss = tst_metrics['loss']
    tst_acc = tst_metrics['accuracy']
    tst_precision = tst_metrics['precision']
    tst_recall = tst_metrics['recall']
    tst_F1 = tst_metrics['F1']
    tst_mean_pred = tst_metrics['mean_pred']
    tst_mean_true = tst_metrics['mean_true']

    # figure_path = 'figures/RESNET/resnet_'
    # results_path = 'figures/RESNET/result.npz'
    # Plot metrics
    plot_utils.plot_losses(trn_loss, tst_loss, figure_path)
    plot_utils.plot_accuracy(trn_acc, tst_acc, figure_path)
    plot_utils.plot_precision(trn_precision, tst_precision, figure_path)
    plot_utils.plot_recall(trn_recall, tst_recall, figure_path)
    plot_utils.plot_F1(trn_F1, tst_F1, figure_path)
    plot_utils.plot_mean_prediction(trn_mean_pred, trn_mean_true, tst_mean_pred, tst_mean_true, figure_path)

    # Save in output folder
    OpenWorldUtils.saveModel(model, model_path)
    # Save encoder in datasetfolder
    figure_size = config['image_resize']
    unfreeze_layer = config['unfreeze_layer']
    encoder_file_path = f'{dataset_path}/{config["model_class"]}/feature_encoder_{figure_size}_{unfreeze_layer}_{encoder_train_classes}.pt'
    OpenWorldUtils.saveModel(model,encoder_file_path)
    torch.save(best_state, f'{exp_folder}/{exp_name}_best_state.pth')

    np.savez(results_path, train_loss=trn_loss, test_loss=tst_loss, train_acc=trn_acc, test_acc=tst_acc,
             train_precision=trn_precision, test_precision=tst_precision, train_recall=trn_recall,
             test_recall=tst_recall, train_F1=trn_F1, test_F1=tst_F1)

    return

def trainMetaModel(model, train_loader, test_loader, epochs, criterion, optimizer, device):
    start_time = time.time()

    trn_losses = []
    tst_losses = []
    trn_acc = []
    tst_acc = []
    trn_precision = []
    tst_precision = []
    trn_recall = []
    tst_recall = []
    trn_F1 = []
    tst_F1 = []
    trn_mean_pred = []
    trn_mean_true = []
    tst_mean_pred = []
    tst_mean_true = []


    model.train()

    best_model = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_epoch = 0
    best_state = {'model': best_model,
                  'acc': best_acc,
                  'epoch': best_epoch, }


    for i in range(epochs):
        trn_corr = 0
        trn_loss = []
        y_pred = []
        y_true = []

        # Run the training batches
        for b, (X0_train, y_train) in tqdm(enumerate(train_loader), total=int(len(train_loader.dataset)/train_loader.batch_size)):

            optimizer.zero_grad()
            X0_train = X0_train.to(device)
            y_train = y_train.to(device)

            # Limit the number of batches
            if b == (len(train_loader)):
                break
            b += 1

            # Apply the model
            y_out, feature_layer = model(X0_train)
            batch_loss = criterion(y_out, y_train)
            y_out = F.log_softmax(y_out, dim=1)

            # Tally the number of correct predictions
            predicted = torch.max(y_out.data, 1)[1]

            batch_corr = (predicted == y_train).sum()
            trn_corr += batch_corr

            y_pred.extend(predicted.cpu())
            y_true.extend(y_train.cpu())
            # Update parameters
            batch_loss.backward()
            optimizer.step()
            trn_loss.append(batch_loss.detach().cpu())

        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)

        trn_loss = torch.stack(trn_loss).numpy().mean()
        # Print epoch results
        print(
            f'epoch: {i:2}  batch: {b:4} [{train_loader.batch_size * b:6}/{len(train_loader) * train_loader.batch_size}]'
            f'  loss: {trn_loss.item():10.8f} accuracy: {trn_corr.item() * 100 / (train_loader.batch_size * b):7.3f}%')

        # Training metrics
        y_pred = torch.stack(y_pred).numpy()
        y_true = torch.stack(y_true).numpy()
        # trn_losses = np.array(trn_losses).mean()

        trn_acc.append(metrics.accuracy_score(y_true, y_pred))
        trn_precision.append(metrics.precision_score(y_true=y_true, y_pred=y_pred, average='weighted', zero_division=0))
        trn_recall.append(metrics.recall_score(y_true, y_pred, average='weighted'))
        trn_F1.append(metrics.f1_score(y_true=y_true, y_pred=y_pred, average='weighted', zero_division=0))

        trn_mean_pred.append(y_pred.mean())
        trn_mean_true.append(y_true.mean())

        trn_losses.append(trn_loss)

        # Run the testing batches
        y_pred, y_true, tst_loss = validate_model(test_loader, model, criterion, device)

        tst_acc.append(metrics.accuracy_score(y_true, y_pred))
        tst_precision.append(metrics.precision_score(y_true=y_true, y_pred=y_pred, average='weighted', zero_division=0))
        tst_recall.append(metrics.recall_score(y_true, y_pred, average='weighted'))
        tst_F1.append(metrics.f1_score(y_true=y_true, y_pred=y_pred, average='weighted', zero_division=0))

        if tst_acc[-1] > best_acc:
            best_acc = tst_acc[-1]
            best_epoch = i + 1
            best_model = copy.deepcopy(model.state_dict())
            best_state = {'model': best_model,
                          'acc': best_acc,
                          'epoch': best_epoch,}

        tst_mean_pred.append(y_pred.mean())
        tst_mean_true.append(y_true.mean())

        tst_losses.append(tst_loss.item())

    trn_metrics = {'loss': trn_losses,
                   'accuracy': trn_acc,
                   'precision': trn_precision,
                   'recall': trn_recall,
                   'F1': trn_F1,
                   'mean_pred': trn_mean_pred,
                   'mean_true': trn_mean_true}

    tst_metrics = {'loss': tst_losses,
                   'accuracy': tst_acc,
                   'precision': tst_precision,
                   'recall': tst_recall,
                   'F1': tst_F1,
                   'mean_pred': tst_mean_pred,
                   'mean_true': tst_mean_true}

    print(f'\nDuration: {time.time() - start_time:.0f} seconds')  # print the time elapsed

    return trn_metrics, tst_metrics, best_state

def validate_model(loader, model, criterion, device):
        num_correct = 0
        num_samples = 0

        # Set model to eval
        model.eval()

        y_true = []
        y_pred = []
        losses = []

        with torch.no_grad():
            for b, (X0_test, y_test) in enumerate(loader):

                X0_test = X0_test.to(device)
                y_test = y_test.to(device)

                # if b == (len(loader) - 1):
                #     break

                # Apply the model
                y_val, feature_layer = model(X0_test)
                loss = criterion(y_val, y_test)
                losses.append(loss.cpu())

                predicted = torch.max(y_val.data, 1)[1]
                y_pred.extend(predicted.cpu())
                y_true.extend(y_test.cpu())
                num_correct += (predicted == y_test).sum()
                num_samples += predicted.size(0)
                b += 1

        losses = np.array(torch.stack(losses)).mean()
        y_pred = np.array(torch.stack(y_pred))
        y_true = np.array(torch.stack(y_true))
        # Toggle model back to train
        model.train()
        test_acc = num_correct.item() * 100 / (num_samples)
        print(f'test accuracy: {num_correct.item() * 100 / (num_samples):7.3f}%  test loss: {losses:10.8f} ')
        return y_pred, y_true, losses



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
    class_ratio = config['class_ratio']
    train_phase = 'encoder_train'
    dataset_class = config['dataset_class']
    dataset_path = f'datasets' \
                   f'/{config["dataset_path"]}'
    image_resize = config['image_resize']
    feature_layer = config['feature_layer']

    dataset = eval('ObjectDatasets.' + dataset_class)(dataset_path,class_ratio,train_phase, image_resize)

    # Load model
    model_class = config['model_class']
    pretrained = config['pretrained']
    unfreeze_layer = config['unfreeze_layer']
    feature_scaling = config['feature_scaling']
    model = eval('RecognitionModels.' + model_class)(model_class, class_ratio[train_phase], feature_layer, unfreeze_layer, feature_scaling, pretrained)

    model.to(device)

    criterion = eval('nn.' + config['criterion'])(reduction='mean')
    optimizer = eval('torch.optim.' + config['optimizer'])(model.parameters(), lr=learning_rate, momentum=0.9)

    return dataset, model, criterion, optimizer, epochs, batch_size, learning_rate, config


if __name__ == "__main__":
    main()
