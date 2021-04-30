import time
import torch
import numpy as np
import matplotlib.pyplot as plt


def trainModel(model, train_loader, test_loader, epochs, criterion, optimizer):
    start_time = time.time()

    train_losses = []
    test_losses = []
    train_correct = []
    test_correct = []

    for i in range(epochs):
        trn_corr = 0
        tst_corr = 0

        # Run the training batches
        for b, (X_train, y_train) in enumerate(train_loader):
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

        train_losses.append(loss)
        train_correct.append(trn_corr)

        # Run the testing batches
        with torch.no_grad():
            for b, (X_test, y_test) in enumerate(test_loader):
                # Apply the model
                y_val = model(X_test.cuda())

                # Tally the number of correct predictions
                predicted = torch.max(y_val.data, 1)[1]
                tst_corr += (predicted == y_test.cuda()).sum()

        loss = criterion(y_val, y_test.cuda())
        test_losses.append(loss)
        test_correct.append(tst_corr)

    print(f'\nDuration: {time.time() - start_time:.0f} seconds')  # print the time elapsed

def testModel(model, test_data):
    x = np.random.randint(0, len(test_data))

    model.eval()
    with torch.no_grad():
        new_pred = model(test_data[x][0].view(1, 1, 28, 28).cuda()).argmax()
    print("Predicted value:", new_pred.item())

    
    plt.figure(figsize=(1, 1))
    plt.imshow(test_data[x][0].reshape((28, 28)), cmap="gist_yarg")
    plt.title(str(new_pred.item()))
    plt.show()

def saveModel(model, file_path):
    torch.save(model.state_dict(), file_path)

def loadModel(model, file_path):
    model.load_state_dict(torch.load(file_path))