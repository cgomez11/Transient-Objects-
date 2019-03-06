import torch
import pandas as pd
import glob
from torchvision import datasets, transforms, models
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.optim.lr_scheduler import LambdaLR, StepLR, MultiStepLR, ExponentialLR, ReduceLROnPlateau
from torch.nn.utils.rnn import pad_sequence
import torchvision
from torch.utils import data
from skimage import io, transform
import numpy as np
import copy
import time
import cv2
import os
import torch.nn.functional as F
import tqdm
import pdb
import argparse
#python train_DenseNet_zero.py --data_path '/media/SSD4/Astronomy/ImageComposition/dict_Dataset_9channels_png/' 
#--path_to_save '/media/SSD4/Astronomy/models/densenetOwn/9channel_png_aug/' 
#--config '9channel_png_aug' --gpuNum '3' --batch_size 8
#--nChannels 9 --nClasses 6
def makedir(outdir):
    if not os.path.exists(outdir):
        os.makedirs(outdir)

import densenet_2D #lo load complete pretrained model 
import densenet_2D_NOFC #to load DN without FC
import DensenetGRU_ownBatch
parser = argparse.ArgumentParser(description='inputs to train')
parser.add_argument('--data_path', type=str, default='', metavar='N',
                    help='path where data is')
parser.add_argument('--path_to_save', type=str, default='', metavar='N',
                    help='path to save stats and best model')
parser.add_argument('--config', type=str, default='random_unbalanced',
                    help='data partition configuration')
parser.add_argument('--resume', type=str, default='',
                    help='path to resume training')
parser.add_argument('--batch_size', type=int, default=4, help='define batch size')
parser.add_argument('--own_batch_size', type=int, default=4, help='define own batch size')
parser.add_argument('--gpuNum', type=str, default='3', help='define gpu number')
parser.add_argument('--nChannels', type=int, default=3, help='number of inputs channels')
parser.add_argument('--nClasses', type=int, default=6, help='number of classes')
parser.add_argument('--LR', type=float, default=1e-1, help='define learning rate')
parser.add_argument('--GR', type=int, default=12, help='define learning rate')
parser.add_argument('--depth', type=int, default=30, help='define learning rate')
parser.add_argument('--hidden', type=int, default=128, help='define hidden state size')
parser.add_argument('--layers', type=int, default=2, help='define lstm layers')
parser.add_argument('--path_pretrained', type=str, default=2, help='path to load pretrained densenet')
parser.add_argument('--features_path', type=str, default='', help='path to load features')
parser.add_argument('--augm', type=int, default=55, help='augmentation factor of training instances')
parser.add_argument('--dir', type=bool, default=False, help='number of dirs')
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpuNum
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#define pretrained model to remove the extra fc layer: NO se cargan pesos
"""
pretrained_model = densenet_2D.DenseNet(growthRate=args.GR, depth=args.depth, reduction=0.5,
                           bottleneck=True, nClasses=args.nClasses, inputChannels=args.nChannels)
pretrained_model = pretrained_model.to(device)
old_dict = pretrained_model.state_dict() 
"""
ckc_point = torch.load(args.path_pretrained)
old_dict = ckc_point['state_dict']

densenet_fts = densenet_2D_NOFC.DenseNet(growthRate=args.GR, depth=args.depth, reduction=0.5,
                           bottleneck=True, nClasses=args.nClasses, inputChannels=args.nChannels)
densenet_fts = densenet_fts.to(device)
densenet_fts_dict = densenet_fts.state_dict()

pretrained_dict = {k: v for k, v in old_dict.items() if k in densenet_fts_dict}
densenet_fts_dict.update(pretrained_dict)
densenet_fts.load_state_dict(densenet_fts_dict) 

for param in densenet_fts.parameters():
    param.requires_grad = False #requires_grad es el flag para saber si calcula gradientes o no
if args.dir == True:
    n_dir = 2
else:
    n_dir =1
model = DensenetGRU_ownBatch.DN_RNN(pre_model= densenet_fts,gr=args.GR,depth=args.depth,nChannels=args.nChannels,
    hidden_size=args.hidden,num_layers=args.layers,num_classes=args.nClasses, device=device, bidir=args.dir, num_dir=n_dir)

model = model.to(device)

def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp

n_params = get_n_params(model)
if args.resume != '': # For training from a previously saved state
    load_model = True
    #model.load_state_dict(torch.load(args.finetune))
    checkpoint = torch.load(args.resume)
    init_epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['state_dict'])
    optimizer = optim.SGD(model.parameters(), lr=args.LR, momentum=0.9, weight_decay=1e-4)
    #optimizer =optim.Adam(model.parameters(), lr=args.LR, weight_decay=1e-4)
    optimizer.load_state_dict(checkpoint['optimizer'])
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.cuda()
    res_flag = 1
    #LOSS????
    # freeze layers
    #for param in model.parameters():
    #    param.requires_grad = False
else:
    init_epoch = 0
    optimizer = optim.SGD(model.parameters(), lr=args.LR, momentum=0.9, weight_decay=1e-4)
    #optimizer =optim.Adam(model.parameters(), lr=args.LR, weight_decay=1e-4)
    #model.apply(weights_init)

#weights = [0.99, 0.94, 0.84, 0.81, 0.78,0.64]
#class_weights = torch.FloatTensor(weights).cuda()
#criterion = nn.NLLLoss()
criterion = nn.CrossEntropyLoss(reduction='sum')
scheduler = ReduceLROnPlateau(optimizer, 'min',patience=50)

path_to_DF_objs = '/media/user_home2/cgomez11/Astronomy/CRTS/scripts/MoreObs/newDF/NEWcompleteDF_'

class Dataset(data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, list_IDs, labels,  which_set, transform=None):
        'Initialization'
        self.labels = labels
        self.list_IDs = list_IDs
        self.transform = transform
        self.which_set = which_set

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        ID = self.list_IDs[index]
        ID = ID[0:23]
        # Load data and get label
        y = self.labels[ID]
        obj_type = classes_names[y]
        fts = torch.load(path_to_fts + classes_names[y] + '/' + ID + '.pt')

        if self.which_set =='train':
            fts = fts.view(-1,int(fts.shape[0]/args.augm),fts.shape[1])
        elif self.which_set =='validation':
            fts = fts.unsqueeze(dim=0)
       
        return fts, y

path_to_fts = args.features_path
classes_names = ['BZ', 'AGN', 'CV','OTHER','SN']
path_to_df = '/media/user_home2/cgomez11/Astronomy/CRTS/scripts/MoreObs/newDF/NEWcompleteDF_'
path_to_ims = '/media/SSD3/Astronomy/CompleteSeason/'
path_to_extra_ims = '/media/user_home2/cgomez11/Astronomy/Networks/Data/ExtraIms/'

path_to_dicts = '/media/user_home2/cgomez11/Astronomy/Networks/ImageGeneration/configurations/'
partition = np.load(path_to_dicts + 'partitions_' + args.config + '.npy').item()
labels = np.load(path_to_dicts + 'labels_' + args.config + '.npy').item()

#count = [16, 157,413,505,564,916] #desbalance original
count = [916,916,916,916,916,916]
def make_weights_balanced(count, images, n_class):
    weight_per_class = [0.] * n_class
    N = float(sum(count))
    for i in range(n_class):                                                                       
        weight_per_class[i] = N/float(count[i])
    weights = [0] * len(images)
    for idx, val in enumerate(images):                                          
        weights[idx] = weight_per_class[labels[val]]
    return weights
weights = make_weights_balanced(count, partition['train'], args.nClasses)
#weights = 1 - np.array([16/total_train,157/total_train,413/total_train,505/total_train,564/total_train,916/total_train])
weights = torch.FloatTensor(weights).cuda()
sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights), replacement=False)

save_path = args.path_to_save
makedir(save_path)

image_datasets = {x: Dataset(partition[x], labels, which_set=x,transform=transforms.Compose([transforms.ToTensor()])) for x in ['train', 'validation']} 
#dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=args.batch_size, shuffle=False, num_workers=4,sampler=sampler) for x in ['train', 'validation']}
dataloaders = {}
dataloaders = {'train':  torch.utils.data.DataLoader(image_datasets['train'], batch_size=args.batch_size, shuffle=False, num_workers=1,sampler=sampler), 'validation':  torch.utils.data.DataLoader(image_datasets['validation'], batch_size=args.batch_size, num_workers=4, shuffle=True)}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'validation']}
print('training dataset size:', dataset_sizes['train'])
print('Validation dataset size:', dataset_sizes['validation'])
print('Done creating dataloader \n')

line_to_save = 'Model params {}, Training size {}, val size {}, batch size {}, Hidden {}, layers {} \n'.format(n_params,dataset_sizes['train'], dataset_sizes['validation'], args.batch_size, args.hidden, args.layers)
with open(args.path_to_save+'train_test_stats.txt','a') as f:
    f.write(line_to_save)

def train_model(model, optimizer, scheduler, num_epochs=50):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    #for epoch in range(num_epochs):
    for epoch in tqdm.trange(init_epoch,num_epochs,desc='Epoch'):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        list_labels_train = []
        arr_output_train = np.empty((0,args.nClasses))
        list_labels = []
        arr_output = np.empty((0,args.nClasses))

        # Each epoch has a training and validation phase
        for phase in ['train', 'validation']:
            if phase=='train':
                aug = 55
            elif phase =='validation':
                aug = 1
        #for phase in ['validation']:
            if phase == 'train':
                #scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            avg_loss = 0
            len_seqs = 0
            flag_begin = True
            # Iterate over data.
            #for inputs, labels in dataloaders[phase]:
            for batch_idx, (fts, labels) in tqdm.tqdm(enumerate(dataloaders[phase]), total=len(dataloaders['train']), desc='Batch'):
                #print('shape inputs padded', inputs.shape)
                #pdb.set_trace()
                #packed_data = packed.to(device)
                labels = labels.to(device)
                fts = fts.to(device)
                fts = fts.squeeze(dim=0) #remove batch 1 dimension
                labels = labels.repeat(fts.shape[0])

                #if flag_begin==True:
                #    lens_batch =0
                #lens_batch += fts.shape[0]
                #flag_begin = False

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    #output_scores = model(packed_data,inputs.shape[0]) #outputs salen de fc: son scores
                    output_scores = model(fts)
                    sm = nn.Softmax(dim=1)
                    output_sm = sm(output_scores)
                    _, preds = torch.max(output_sm, 1)
                    #for each instance 
                    loss = criterion(output_scores, labels)/(args.own_batch_size*aug) #sum over 55
                    if phase =='train':
                        loss.backward()
                    #loss_mini_batch +=  loss.data[0]
                    #loss = F.nll_loss(outputs, labels)
                    if phase =='validation': # se acumula en cada batch
                        #list_labels = list_labels + list(labels.data.cpu().numpy()) #before when batch_size was the batch
                        list_labels = list_labels + list(labels.data.cpu().numpy())
                        #list_preds = list_preds + list(preds.data.cpu().numpy())
                        #list_labels.append(labels.data.cpu().numpy())
                        arr_output = np.concatenate((arr_output, output_sm),axis=0) #before
                        #arr_output = np.concatenate((arr_output, torch.cat(l_output_sm_batch)), axis=0)

                    # backward + optimize only if in training phase
                    if phase == 'train' and (batch_idx%args.own_batch_size==0) and (batch_idx!=0): #and (flag_begin!=True):
                        #for p in model.parameters():
                        #    pdb.set_trace()
                        #    p.grad /= lens_batch
                        optimizer.step()
                        optimizer.zero_grad() #do multiple backwards without resetting the gradients: accumulating
                        #list_labels_train = list_labels_train + list(labels.data.cpu().numpy())
                        #arr_output_train = np.concatenate((arr_output_train, output_sm.detach().cpu().numpy()),axis=0)

                # statistics
                #running_loss += loss.item() * inputs.size(0) #le estoy pasando loss del batch
                #running_loss += loss.item() * args.own_batch_size
                #sigo iterando en el batch
                running_loss += loss.item() #acumular sum of loss over the iterations
                len_seqs += fts.shape[0] #acumular longitudes
                #len_seqs += fts.shape[0] #before 
                #print('loss',running_loss)
                running_corrects += torch.sum(preds == labels.data)
                #running_corrects += torch.sum(preds == labels.data)
      
            epoch_loss = running_loss / len_seqs
            #print('compare losses', epoch_loss_before, epoch_loss)
            #print('size',dataset_sizes[phase])
            epoch_acc = running_corrects.double() / len_seqs
            if phase == 'validation':
                scheduler.step(epoch_loss)
                line_to_save = 'Epoch {}: {} Loss: {:.4f} Acc: {:.4f}\n'.format(epoch, 'test', epoch_loss, epoch_acc)
            else: 
                for g in optimizer.param_groups:
                    lr_to_save = g['lr']
                line_to_save = 'Epoch {}: {} Loss: {:.4f} Acc: {:.4f} LR: {:.4f} \n'.format(epoch, 'train', epoch_loss, epoch_acc, lr_to_save)
            #print('{} Loss: {:.4f} Acc: {:.4f}'.format(
            #    phase, epoch_loss, epoch_acc))
            with open(args.path_to_save+'train_test_stats.txt','a') as f:
                f.write(line_to_save)
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'validation' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                the_epoch = epoch
                filename = args.path_to_save + 'complete_model'+ '_BEST.pth'
                state = {'epoch': the_epoch , 'state_dict': model.state_dict(),
                     'optimizer': optimizer.state_dict() }
                torch.save(state, filename)

                #ir guardando best model
                #final_list_preds = list_preds
                #final_list_labels = list_labels
                final_outputs = arr_output
                final_labels = np.array(list_labels)
            if phase =='train':
                the_epoch = epoch
                filename = args.path_to_save + 'complete_model_'+ str(the_epoch) +'.pth'
                state = {'epoch': the_epoch , 'state_dict': model.state_dict(),
                     'optimizer': optimizer.state_dict() }
                torch.save(state, filename)
            #elif phase =='train':
            #    np.save(args.path_to_save + 'train_labels_' + str(epoch) + '_model.npy', list_labels_train)
            #    np.save(args.path_to_save + 'train_preds_' + str(epoch) + '_model.npy',arr_output_train)

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, final_labels, final_outputs

best_model, arr_labels, arr_o = train_model(model, optimizer,scheduler, num_epochs=200)
#path_to_save = '/media/SSD4/Astronomy/models/densenet121/nonorm_pngIms_balanced/'
torch.save(best_model.state_dict(), args.path_to_save + 'best_model_all.pth')
np.save(args.path_to_save + 'labels_best_model_all.npy', arr_labels)
np.save(args.path_to_save + 'preds_best_model_all.npy', arr_o)
