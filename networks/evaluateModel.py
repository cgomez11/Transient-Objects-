import os
import pdb
import tqdm
import glob
import time
import shutil
import argparse
import numpy as np 
import os.path as osp
import scipy.io as sio

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.utils import data
import torch.nn.functional as F
from torch.autograd import Variable
import torch.utils.data as data_utils
from torchvision import datasets, transforms, models
from torch.optim.lr_scheduler import LambdaLR, StepLR, MultiStepLR, ExponentialLR, ReduceLROnPlateau

import densenet_2D

parser = argparse.ArgumentParser(description='inputs to train')
parser.add_argument('--data_path', type=str, default='', metavar='N',
                    help='path where data is')
parser.add_argument('--path_to_save', type=str, default='', metavar='N',
                    help='path to save stats and best model')
parser.add_argument('--config', type=str, default='random_unbalanced',
                    help='data partition configuration')
parser.add_argument('--resume', type=str, default='',
                    help='path to model to continue training')
parser.add_argument('--gpuNum', type=str, default='2', help='define gpu number')
parser.add_argument('--batch_size', type=int, default=4, help='define batch size')
parser.add_argument('--nChannels', type=int, default=3, help='number of inputs channels')
parser.add_argument('--nClasses', type=int, default=6, help='number of classes')
parser.add_argument('--GR', type=int, default=12, help='define learning rate')
parser.add_argument('--depth', type=int, default=30, help='define learning rate')
parser.add_argument('--validate', type=str, default='test', help='define the set to eval: train or test')
args = parser.parse_args()

def makedir(outdir):
    if not os.path.exists(outdir):
        os.makedirs(outdir)

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpuNum
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = densenet_2D.DenseNet(growthRate = args.GR, depth = args.depth, reduction = 0.5,
                            bottleneck = True, nClasses = args.nClasses, inputChannels = args.nChannels)

model = model.to(device)
model.eval() # set to evaluation mode

if args.resume != '': # For loading the model to eval
    checkpoint = torch.load(args.resume)
    the_epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['state_dict'])

# Load data
classes_names = ['BZ', 'AGN', 'CV','OTHER','SN','NON']
path_to_data = args.data_path
save_path = args.path_to_save
makedir(save_path)

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
        y = self.labels[ID]
        # Load data and get label
        X = np.load(path_to_data + classes_names[y] + '/'+ID + '.npy')#+ '.png')
        if self.transform:
            X = self.transform(X)
        return X, y

path_to_dicts = '/media/user_home2/cgomez11/Astronomy/Networks/ImageGeneration/configurations/'
partition = np.load(path_to_dicts + 'partitions_' + args.config + '.npy').item()
labels = np.load(path_to_dicts + 'labels_' + args.config + '.npy').item()

# Define dataloaders
image_datasets = {x: Dataset(partition[x], labels, transform = transforms.Compose([transforms.ToTensor() ])) for x in ['train', 'validation']} 
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=args.batch_size, shuffle = True, num_workers = 4) for x in ['train', 'validation']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'validation']}

# to save predictions, labels, and probabilities
list_preds = []
list_labels = []
arr_output = np.empty((0,args.nClasses))

for inputs, labels in tqdm.tqdm(dataloaders[args.validate], total = len(dataloaders[args.validate]), desc = 'Batch'):
    inputs = inputs.to(device)
    inputs = inputs.float()
    labels = labels.to(device)

    output_scores = model(inputs) 
    sm = nn.Softmax(dim=1)
    output_sm = sm(output_scores)

    _, preds = torch.max(output_sm, 1)
    list_labels = list_labels + list(labels.data.cpu().numpy())
    list_preds = list_preds + list(preds.data.cpu().numpy())
    arr_output = np.concatenate((arr_output, output_sm.detach().cpu().numpy()),axis=0)

np.save(args.path_to_save + args.validate +'_' + str(the_epoch) +'_targets.npy', np.array(list_labels))
np.save(args.path_to_save + args.validate + '_'+ str(the_epoch) +'_outputs.npy', arr_output)