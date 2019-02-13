import torch
from torchvision import datasets, transforms, models
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.optim.lr_scheduler import LambdaLR, StepLR, MultiStepLR, ExponentialLR, ReduceLROnPlateau
import torch
import torchvision
from torch.utils import data
from skimage import io, transform
from functionAugData import generateInstancesSeq
import numpy as np
import copy
import pandas as pd
import time
import glob
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
#import DensenetGRU_all #before
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
parser.add_argument('--gpuNum', type=str, default='3', help='define gpu number')
parser.add_argument('--nChannels', type=int, default=3, help='number of inputs channels')
parser.add_argument('--nClasses', type=int, default=6, help='number of classes')
parser.add_argument('--LR', type=float, default=1e-1, help='define learning rate')
parser.add_argument('--GR', type=int, default=12, help='define learning rate')
parser.add_argument('--depth', type=int, default=30, help='define learning rate')
parser.add_argument('--hidden', type=int, default=128, help='define hidden state size')
parser.add_argument('--layers', type=int, default=2, help='define lstm layers')
parser.add_argument('--path_pretrained', type=str, default=2, help='path to load pretrained densenet')
parser.add_argument('--path_complete_model', type=str, default='', help='path to load complete model')
parser.add_argument('--own_batch_size', type=int, default=4, help='define own batch size')
parser.add_argument('--dir', type=bool, default=False, help='number of dirs')
parser.add_argument('--which_set', type=str, default='validation', help='define the set for forward')
parser.add_argument('--augm', type=int, default=10, help='factor to augment instances')
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpuNum
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#for the pretrained model
pretrained_model = densenet_2D.DenseNet(growthRate=args.GR, depth=args.depth, reduction=0.5,
                           bottleneck=True, nClasses=args.nClasses, inputChannels=args.nChannels)
pretrained_model = pretrained_model.to(device)

checkpoint = torch.load(args.path_pretrained)
init_epoch = checkpoint['epoch']
pretrained_model.load_state_dict(checkpoint['state_dict'])
old_dict = pretrained_model.state_dict() 


densenet_fts = densenet_2D_NOFC.DenseNet(growthRate=args.GR, depth=args.depth, reduction=0.5,
                           bottleneck=True, nClasses=args.nClasses, inputChannels=args.nChannels)
densenet_fts = densenet_fts.to(device)
densenet_fts_dict = densenet_fts.state_dict()

pretrained_dict = {k: v for k, v in old_dict.items() if k in densenet_fts_dict}
densenet_fts_dict.update(pretrained_dict)
densenet_fts.load_state_dict(pretrained_dict) #ya quedo con los pesos pre-entrenados sin la capa fc
#toca congelar esos pesos, que solo aprenda el LSTM

for param in densenet_fts.parameters():
    param.requires_grad = True #requires_grad es el flag para saber si calcula gradientes o no

if args.dir == True:
    n_dir = 2
else:
    n_dir =1
#no necesito pretrained weights de DN, se cargan con toda la red
#model = DensenetGRU_all.DN_RNN(gr=args.GR,depth=args.depth,nChannels=args.nChannels,
#    hidden_size=args.hidden,num_layers=args.layers,num_classes=args.nClasses, device=device, bidir=args.dir, num_dir=n_dir)
model = DensenetGRU_ownBatch.DN_RNN(pre_model= densenet_fts,gr=args.GR,depth=args.depth,nChannels=args.nChannels,
    hidden_size=args.hidden,num_layers=args.layers,num_classes=args.nClasses, device=device, bidir=args.dir, num_dir=n_dir)

model = model.to(device)
checkpoint = torch.load(args.path_complete_model)
model.load_state_dict(checkpoint['state_dict'])

#get the number of parameters
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

criterion = nn.CrossEntropyLoss()
scheduler = ReduceLROnPlateau(optimizer, 'min',patience=50)

path_to_dicts = '/media/user_home2/cgomez11/Astronomy/Networks/ImageGeneration/configurations/'
partition = np.load(path_to_dicts + 'partitions_' + args.config + '.npy').item()
labels = np.load(path_to_dicts + 'labels_' + args.config + '.npy').item()

count = [916,916,916,916,916,916] #desbalance original
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

path_to_data = args.data_path
classes_names = ['BZ', 'AGN', 'CV','OTHER','SN']
path_to_df = '/media/user_home2/cgomez11/Astronomy/CRTS/scripts/MoreObs/newDF/NEWcompleteDF_'
path_to_ims = '/media/SSD3/Astronomy/CompleteSeason/'
path_to_extra_ims = '/media/user_home2/cgomez11/Astronomy/Networks/Data/ExtraIms/'

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
#to load one sequence for one instance  
"""
  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        ID = self.list_IDs[index]
        # Load data and get label
        y = self.labels[ID]
        df_obj = pd.read_pickle(path_to_df + classes_names[y] + 'Season.pkl')
        sorted_months = list(df_obj[df_obj['CRTS ID']==ID]['Sorted_months'].values[0])
        l_images_dates = []
        path_to_ims = '/media/SSD4/Astronomy/CompleteSeason/'
        if os.path.exists(path_to_extra_ims + classes_names[y] + '/' + ID):
            path_to_ims = path_to_extra_ims
        else:
            path_to_ims = path_to_ims
        for dates in sorted_months:
            list_ims = glob.glob(path_to_ims + classes_names[y] + '/'+ ID + '/*' + dates  + '*.npy')
            ims_to_append = np.load(list_ims[0])
            l_images_dates.append(np.expand_dims(ims_to_append, axis=2))
        concat_seq = np.concatenate(l_images_dates, axis=2)
        X = concat_seq.astype('float64')
        if X.shape[2]<=3:#replicate instance
            X = np.concatenate((X,X), axis=2)
        elif X.shape[2] >=30:
            X = X[:,:,0:30]
        if self.transform:
            X = self.transform(X)
        return X, y
"""
#to load more than one sequence for each instance 
  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        ID = self.list_IDs[index]
        # Load data and get label
        y = self.labels[ID]
        X = generateInstancesSeq(ID, classes_names[y], self.f_aug, self.which_set) 
        #y = [y]*self.f_aug
        #X = np.array(X)
        X = X.astype('float64') #64x64xCxf_aug - transpose and to_tensor?
        #print('loaded images size', X.shape)
        if X.shape[2]<=3:#replicate instance
            X = np.concatenate((X,X), axis=2)
        #fts = torch.load(path_to_fts+classes_names[y] + '/' + ID + '.pt')
        #y = torch.from_numpy(y)
        if self.transform:
            X = self.transform(X)
        #print(X.shape)
        return X, y, ID

#load dataset
save_path = args.path_to_save
makedir(save_path)
#means, stds = findStats(args.nChannels,path_to_data,partition)
#print('Done finding means', means, stds)
#means = [0.485, 0.456, 0.406]
#stds = [0.229, 0.224, 0.225]
#image_datasets = {x: Dataset(partition[x], labels, transform=transforms.Compose([ToTensor(0,0) ])) for x in ['train', 'validation']} 
image_datasets = {x: Dataset(partition[x], labels, transform=transforms.Compose([transforms.ToTensor()]), f_aug=args.augm, which_set=x) for x in ['train', 'validation']} 
dataloaders = {'train':  torch.utils.data.DataLoader(image_datasets['train'], batch_size=args.batch_size, shuffle=False, num_workers=4, sampler=sampler), 'validation':  torch.utils.data.DataLoader(image_datasets['validation'], batch_size=args.batch_size, num_workers=4, shuffle=True)}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'validation']}
print('training dataset size:', dataset_sizes['train'])
print('Validation dataset size:', dataset_sizes['validation'])
print('Done creating dataloader \n')

line_to_save = 'Model params {}, Training size {}, val size {}, batch size {}, Hidden {}, layers {} \n'.format(n_params,dataset_sizes['train'], dataset_sizes['validation'], args.batch_size, args.hidden, args.layers)
with open(args.path_to_save+'train_test_stats.txt','a') as f:
    f.write(line_to_save)

#def train_model(model, criterion, optimizer, scheduler, num_epochs=50):
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
        #for phase in ['validation']:
            if phase == 'train':
                #scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            #for inputs, labels in dataloaders[phase]:
            for batch_idx, (inputs, labels) in tqdm.tqdm(enumerate(dataloaders[phase]), total=len(dataloaders['train']), desc='Batch'):
                #print(type(labels))
                inputs = inputs.to(device)
                #print('shape inputs', inputs.shape)
                inputs = inputs.float()
                bs, ncrops, c, h, w = inputs.size()
                inputs = inputs.view(-1, c, h, w) #ixCx64x64
                
                for cc in range(inputs.size(1)-2): 
                    out_t =inputs[:,cc:cc+3,:,:]
                    #l_outs.append(out_t.unsqueeze(0))
                    l_outs.append(out_t)
                out_c = torch.cat(l_outs, 0)  #seq_lenx3xhxw
                #apply DN to the repeated sequences combined 
                out_model = densenet_fts(out_c) #(new_seq_len*aug)xfts
                #then adjust according to the augmentation 
                fts = out_model.view(-1,int(out_model.shape[0] / args.augm),out_model.shape[1]) #aug(batch)xseq_lenxfts_size
                labels = labels.to(device)
                #fts = fts.to(device)

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    output_scores = model(fts) #outputs salen de fc: son scores
                    sm = nn.Softmax(dim=1)
                    output_sm = sm(output_scores)
                    _, preds = torch.max(output_sm, 1)
                    #print(outputs)
                    #print(labels)

                    loss = criterion(output_scores, labels)/ (args.own_batch_size * args.augm) #scale the loss 
                    if phase =='train':
                        loss.backward()
                    #print(loss)
                    #loss = F.nll_loss(outputs, labels)
                    if phase =='validation':
                        list_labels = list_labels + list(labels.data.cpu().numpy())
                        #list_preds = list_preds + list(preds.data.cpu().numpy())
                        #list_labels.append(labels.data.cpu().numpy())
                        arr_output = np.concatenate((arr_output, output_sm),axis=0)

                    # backward + optimize only if in training phase
                    if phase == 'train' and (batch_idx%args.own_batch_size==0) and (batch_idx!=0):
                        optimizer.step()
                        optimizer.zero_grad() #zero the parameter gradient
                        #save train metrics 
                        lab_to_save = copy.deepcopy(labels)
                        out_to_save = copy.deepcpy(output_sm)
                        list_labels_train = list_labels_train + list(lab_to_save.data.cpu().numpy())
                        arr_output_train = np.concatenate((arr_output_train, out_to_save.detach().cpu().numpy()),axis=0)

                # statistics
                running_loss += loss.item() #acumular loss individual
                #update average loss
                if batch_idx==0:
                    avg_loss = loss.item()
                else:
                    avg_loss = (avg_loss*(batch_idx) + loss.item())/(batch_idx+1)
                #print('loss',running_loss)
                running_corrects += torch.sum(preds == labels.data)
            epoch_loss = avg_loss
            #print('size',dataset_sizes[phase])
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
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
                np.save(args.path_to_save + 'train_'+ str(the_epoch) +'_outputs.npy', arr_output_train)
                np.save(args.path_to_save + 'train_'+ str(the_epoch) +'_targets.npy', np.array(list_labels_train))
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

best_model, arr_labels, arr_o = train_model(model, optimizer,scheduler, num_epochs=100)
#path_to_save = '/media/SSD4/Astronomy/models/densenet121/nonorm_pngIms_balanced/'
torch.save(best_model.state_dict(), args.path_to_save + 'best_model_all.pth')
np.save(args.path_to_save + 'labels_best_model_all.npy', arr_labels)
np.save(args.path_to_save + 'preds_best_model_all.npy', arr_o)
