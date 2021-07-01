# originally taken from:
# https://github.com/tjmoon0104/pytorch-tiny-imagenet
# and modified (replaced logging script and added best
# model saving code)

import torch, time, copy, sys, os
import numpy as np
from sklearn import metrics

import open_world.meta_learner.meta_learner_utils as meta_utils

def train_model(output_path, model, dataloaders, \
            dataset_sizes, criterion, optimizer, \
            i_p, model_dir, filename, num_epochs=5, \
            scheduler=None):

    trn_metrics_dict = {'loss': [],
                    'accuracy': [],
                    'precision': [],
                    'recall': [],
                    'F1': [],
                    'mean_pred': [],
                    'mean_true': []}

    tst_metrics_dict = {'loss': [],
                    'accuracy': [],
                    'precision': [],
                    'recall': [],
                    'F1': [],
                    'mean_pred': [],
                    'mean_true': []}

    metrics_dict = {'train': trn_metrics_dict,
               'val': tst_metrics_dict}



    if not os.path.exists('models/'+str(output_path)):
        os.makedirs('models/'+str(output_path))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best = 0
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 10)



        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                if scheduler != None:
                    scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode
            trn_loss = []
            y_pred = []
            y_true = []

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for i,(inputs, labels) in enumerate(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs, feature = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                y_pred.extend(preds.cpu())
                y_true.extend(labels.cpu())

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                print("\x1b[2K\rIteration: {}/{}, Loss: {:.2f}".format(i+1, len(dataloaders[phase]), loss.item() * inputs.size(0)), end="")

#                 print( (i+1)*100. / len(dataloaders[phase]), "% Complete" )
                sys.stdout.flush()

            # Training metrics
            y_pred = np.array(torch.stack(y_pred))
            y_true = np.array(torch.stack(y_true))
            epoch_loss = running_loss / dataset_sizes[phase]

            metrics_dict[phase]['accuracy'].append(metrics.accuracy_score(y_true, y_pred))
            metrics_dict[phase]['precision'].append(metrics.precision_score(y_true=y_true, y_pred=y_pred, average='weighted',zero_division=0))
            metrics_dict[phase]['recall'].append(metrics.recall_score(y_true, y_pred, average='weighted',zero_division=0))
            metrics_dict[phase]['F1'].append(metrics.f1_score(y_true=y_true, y_pred=y_pred, average='weighted',zero_division=0))

            metrics_dict[phase]['mean_pred'].append(y_pred.mean())
            metrics_dict[phase]['mean_true'].append(y_true.mean())

            metrics_dict[phase]['loss'].append(loss.item())




            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            if phase == 'train':
                avg_loss = epoch_loss
                t_acc = epoch_acc
            else:
                val_loss = epoch_loss
                val_acc = epoch_acc

#             print('{} Loss: {:.4f} Acc: {:.4f}'.format(
#                 phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best = epoch + 1
                best_model_wts = copy.deepcopy(model.state_dict())
                print('\nBest Model Found... Saving!')
                state = {
                    'model': model.state_dict(),
                    'acc': val_acc,
                    'epoch': epoch,
                }
                if not os.path.isdir(model_dir):
                    os.mkdir(model_dir)
                torch.save(state, model_dir + '/' + str(i_p) + '_best_ckpt.pth')

        with open(filename, 'a') as f:
            f.write(f'RSH_p {i_p:.3f} epochs {epoch:03} Train_Accuracy {t_acc:.3f} Train_Loss {avg_loss:.3f} Validation_Accuracy {val_acc:.3f} Validation_Loss {val_loss:.3f} completed on {time.strftime("%Y%m%d-%H%M%S")}' + "\n")

        print('\nTrain Loss: {:.4f} Acc: {:.4f}'.format(avg_loss, t_acc))
        print(  'Val Loss: {:.4f} Acc: {:.4f}'.format(val_loss, val_acc))
        print()
        # torch.save(model.state_dict(), './models/' + str(output_path) + '/model_{}_epoch.pt'.format(epoch+1))
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best Validation Accuracy: {}, Epoch: {}'.format(best_acc, best))

    return metrics_dict['train'], metrics_dict['val']
