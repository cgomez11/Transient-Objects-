import os
import copy
import time
import tqdm
import argparse
import warnings
import numpy as np

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.utils import data
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torchvision import datasets, transforms, models
from torch.optim.lr_scheduler import LambdaLR, StepLR, MultiStepLR, ExponentialLR, ReduceLROnPlateau

warnings.filterwarnings("ignore")
#import architecture
import densenet_2D

def makedir(outdir):
    if not os.path.exists(outdir):
        os.makedirs(outdir)

parser = argparse.ArgumentParser(description='inputs to train')
parser.add_argument('--data_path', type=str, default='', metavar='N',
                    help='path where data is')
parser.add_argument('--path_to_save', type=str, default='', metavar='N',
                    help='path to save stats and models')
parser.add_argument('--config', type=str, default='random_unbalanced',
                    help='data partition configuration')
parser.add_argument('--resume', type=str, default='',
                    help='path to resume training when interrupted')
parser.add_argument('--batch_size', type=int, default=4, help='define batch size')
parser.add_argument('--gpuNum', type=str, default='3', help='define gpu number')
parser.add_argument('--nChannels', type=int, default=3, help='number of inputs channels')
parser.add_argument('--nClasses', type=int, default=6, help='number of classes')
parser.add_argument('--nEpochs', type=int, default=50, help='number of epochs to train models')
parser.add_argument('--LR', type=float, default=1e-1, help='define learning rate')
parser.add_argument('--GR', type=int, default=12, help='define learning rate')
parser.add_argument('--depth', type=int, default=30, help='define learning rate')
parser.add_argument('--FT', type=bool, default=False, help='train from zero or ft')
parser.add_argument('--pretrained_DN', type=str, default='', help='path to load pretrained DN')
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpuNum
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# define model
model = densenet_2D.DenseNet(growthRate = args.GR, depth = args.depth, reduction = 0.5,
                            bottleneck = True, nClasses = args.nClasses, inputChannels = args.nChannels)
model_dict = model.state_dict()
# to finetune a model
if args.FT:
    checkpoint_DN = torch.load(args.pretrained_DN)
    old_state_dict = checkpoint_DN['state_dict']
    # do not copy the final linear layer
    pretrained_dict = {k: v for k, v in old_state_dict.items()
                                   if k in model_dict and not k.startswith('fc')}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    #model.load_state_dict(old_state_dict)
# to retrain all the network
for param in model.parameters():
    param.requires_grad = True
model.to(device)

# to resume training 
if args.resume != '': # For training from a previously saved state
    checkpoint = torch.load(args.resume)
    init_epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['state_dict'])
    optimizer = optim.SGD(model.parameters(), lr = args.LR, momentum = 0.9, weight_decay = 1e-4)
    optimizer.load_state_dict(checkpoint['optimizer'])
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.cuda()
else:
    init_epoch = 0
    optimizer = optim.SGD(model.parameters(), lr = args.LR, momentum = 0.9, weight_decay = 1e-4)

# Define loss function
criterion = nn.CrossEntropyLoss()
scheduler = ReduceLROnPlateau(optimizer, 'min', patience=20)

# Load data
path_to_data = args.data_path
save_path = args.path_to_save
makedir(save_path)

specs = ('python main_train_DN.py --data_path {} --path_to_save {} --config {} --resume {}  '
         '--batch_size {} --nChannels {} --nClasses {} --LR {} --GR {} --depth {} '
         '--FT {} --pretrained_DN {} \n'.format(
            args.data_path, args.path_to_save, args.config, args.resume,
            args.batch_size, args.nChannels, args.nClasses, args.LR, 
            args.GR, args.depth, args.FT, args.pretrained_DN))

with open(args.path_to_save + 'train_test_stats.txt','a') as f:
                f.write(specs)

classes_names = ['BZ','AGN', 'CV','OTHER','SN','NON']

class Dataset(data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, list_IDs, labels, transform=None):
        'Initialization'
        self.labels = labels
        self.list_IDs = list_IDs
        self.transform = transform

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        ID = self.list_IDs[index]
        # Load data and get label
        y = self.labels[ID]
        X = np.load(path_to_data + classes_names[y] + '/'+ID + '.npy')
        if self.transform:
            X = self.transform(X)
        return X, y

path_to_dicts = '/home/cgomez11/Astronomy/configurations/'
partition = np.load(path_to_dicts + 'partitions_' + args.config + '.npy').item()
labels = np.load(path_to_dicts + 'labels_' + args.config + '.npy').item()

# define dataloader
image_datasets = {x: Dataset(partition[x], labels, transform = transforms.Compose([transforms.ToTensor()])) for x in ['train', 'validation']} 
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size = args.batch_size, shuffle = True, num_workers = 4) for x in ['train', 'validation']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'validation']}
print('training dataset size:', dataset_sizes['train'])
print('Validation dataset size:', dataset_sizes['validation'])
print('Done creating dataloader \n')

def train_model(model, optimizer, scheduler, num_epochs=50):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in tqdm.trange(init_epoch,num_epochs, desc = 'Epoch'):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        list_labels = []
        arr_output = np.empty((0,args.nClasses))

        # Each epoch has a training and validation phase
        for phase in ['train', 'validation']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in tqdm.tqdm(dataloaders[phase], total=len(dataloaders['train']), desc='Batch'):
                inputs = inputs.to(device)
                inputs = inputs.float()
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()
                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    out_scores = model(inputs) #the outputs are scores
                    # apply SM to get class probabilities
                    sm = nn.Softmax(dim = 1)
                    output_sm = sm(out_scores)
                    _, preds = torch.max(output_sm, 1)

                    loss = criterion(out_scores, labels)

                    if phase =='validation':
                        # save labels and probabilities to calculate metrics
                        list_labels = list_labels + list(labels.data.cpu().numpy())
                        arr_output = np.concatenate((arr_output, output_sm), axis = 0)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                # statistics
                running_loss += loss.item() * inputs.size(0) # scale loss with the batch_size
                running_corrects += torch.sum(preds == labels.data)
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            if phase == 'validation':
                scheduler.step(epoch_loss)
                line_to_save = 'Epoch {}: {} Loss: {:.4f} Acc: {:.4f}\n'.format(epoch, 'test', epoch_loss, epoch_acc)
            else: 
                for g in optimizer.param_groups:
                    lr_to_save = g['lr']
                line_to_save = 'Epoch {}: {} Loss: {:.4f} Acc: {:.4f}\n LR: {:.6f}'.format(epoch, 'train', epoch_loss, epoch_acc, lr_to_save)

            with open(args.path_to_save + 'train_test_stats.txt','a') as f:
                f.write(line_to_save)
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'validation' and epoch_acc > best_acc: #save the current best model
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                the_epoch = epoch
                filename = args.path_to_save + 'complete_model'+ '_BEST.pth'
                state = {'epoch': the_epoch , 'state_dict': model.state_dict(),
                     'optimizer': optimizer.state_dict() }
                torch.save(state, filename)
                final_outputs = arr_output # class probabilities for all evaluation examples
                final_labels = np.array(list_labels) # class labels for all evaluation examples
            if phase =='train': # save the model at each epoch
                the_epoch = epoch
                filename = args.path_to_save + 'complete_model_'+ str(the_epoch) +'.pth'
                state = {'epoch': the_epoch , 'state_dict': model.state_dict(),
                     'optimizer': optimizer.state_dict() }
                torch.save(state, filename)


    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, final_labels, final_outputs

best_model, arr_labels, arr_o = train_model(model, optimizer, scheduler, num_epochs=args.nEpochs)

torch.save(best_model.state_dict(), args.path_to_save + 'best_model_all.pth')
np.save(args.path_to_save + 'labels_best_model_all.npy', arr_labels)
np.save(args.path_to_save + 'preds_best_model_all.npy', arr_o)