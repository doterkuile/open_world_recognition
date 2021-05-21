import time
import torch
import numpy as np
import matplotlib.pyplot as plt
import yaml
import os

from open_world import ObjectDatasets
from open_world import RecognitionModels
import torch.nn as nn




def parseConfigFile(config_file, enable_training):

    with open(config_file) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    # Load dataset
    dataset_path = config['dataset_path']
    dataset_class = config['dataset_class']
    dataset = eval('ObjectDatasets.' + dataset_class)(dataset_path)


    # Load model
    model_path = config['model_path']
    model_class = config['model_class']
    model = eval('RecognitionModels.' + model_class)(model_path).cuda()
    if not enable_training:
        print('Load model ' + model_path)
        loadModel(model, model_path)

    # Training parameters
    batch_size = config['batch_size']
    learning_rate = config['learning_rate']
    epochs = config['epochs']
    criterion = eval('nn.' + config['criterion'])()
    optimizer = eval('torch.optim.' + config['optimizer'])(model.parameters(), lr=config['learning_rate'])

    return (dataset, model, criterion, optimizer, epochs, batch_size, learning_rate)




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
    model.load_state_dict(torch.load(file_path))

def plotLosses(trainig_file_path, n_training, n_test, figure_path):

    data = np.load(trainig_file_path + '.npz')
    train_losses = data['train_losses']
    test_losses = data['test_losses']
    train_correct = data['train_correct']
    test_correct = data['test_correct']
    fig = plt.figure()
    plt.plot(train_losses, label='training loss')
    plt.plot(test_losses, label='validation loss')
    plt.title('Loss at the end of each epoch')
    plt.legend();
    plt.show()
    fig.savefig(figure_path + 'losses')

    fig = plt.figure()
    plt.plot([float(t) / float(n_training)*100 for t in train_correct], label='training accuracy')
    plt.plot([float(t) / float(n_test)*100 for t in test_correct], label='validation accuracy')
    plt.title('Accuracy at the end of each epoch')
    plt.legend();
    plt.show()
    fig.savefig(figure_path + 'accuracy')



    return


