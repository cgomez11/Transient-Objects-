import torch
from torchvision import datasets, transforms, models
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.optim.lr_scheduler import LambdaLR, StepLR, MultiStepLR, ExponentialLR, ReduceLROnPlateau
import torchvision
from torch.utils import data
from skimage import io, transform
from functionAugData_ALLTnT import generateInstancesSeq
import numpy as np
import copy
import pandas as pd
import time
import glob
#import cv2
import os
import torch.nn.functional as F
import tqdm
import pdb
import argparse

def makedir(outdir):
    if not os.path.exists(outdir):
        os.makedirs(outdir)
from calculate_metrics import metrics
#import DensenetGRU_all #before
import DensenetGRU_all_add as DensenetGRU_all
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
parser.add_argument('--weights', type=str, default='', help='weights to load')
parser.add_argument('--replace', type=bool, default=False, help='replace instances or not in sampler')
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpuNum
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if args.dir == True:
    n_dir = 2
else:
    n_dir =1
print('number of reps', args.augm)

model = DensenetGRU_all.DN_RNN(gr=args.GR, depth=args.depth, nChannels=args.nChannels, hidden_size=args.hidden, num_layers=args.layers, num_classes=args.nClasses,bidir=args.dir, num_dir=n_dir, device=device)
model = model.to(device)
model_dict = model.state_dict()


if args.path_complete_model != '':
    checkpoint = torch.load(args.path_complete_model)

if args.weights =='all':
    #load all the weights to the net
    model.load_state_dict(checkpoint['state_dict'])
    print('all weights  loaded')
elif args.weights =='cnn':
    #load only the weights of cnn
    old_dict = checkpoint['state_dict']
    cmp_dict = {k: v for k, v in old_dict.items() if k in model_dict and k.startswith('pre_model')}
    model_dict.update(cmp_dict)
    model.load_state_dict(model_dict)

for param in model.pre_model.parameters():
    param.requires_grad = True

for params in model.gru.parameters():
    params.requires_grad = False
for params in model.fc.parameters():
    params.requires_grad = False

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
print('number of parameters', n_params)

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
ww = 1- (np.array([157, 413, 505, 564, 916, 9168])/11723)
class_weights = torch.FloatTensor(ww).cuda()
criterion = nn.CrossEntropyLoss(reduction='sum', weight = class_weights)
#scheduler = ReduceLROnPlateau(optimizer, 'min',patience=50)
scheduler = StepLR(optimizer, step_size=30, gamma=0.1)

path_to_dicts =  '/home/cgomez11/Astronomy/Networks/ImageGeneration/configurations/'
partition = np.load(path_to_dicts + 'partitions_' + args.config + '.npy').item()
#partition = {'train': partition['train'][0:10], 'validation': partition['validation'][0:10]}
labels = np.load(path_to_dicts + 'labels_' + args.config + '.npy').item()

#count = [157, 413, 505,564,916,9168] #desbalance original
count = [4584]*6
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
sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights), replacement = args.replace)

path_to_data = args.data_path
classes_names = ['BZ', 'AGN', 'CV','OTHER','SN','NON']
path_to_ims = '/home/cgomez11/Astronomy/Networks/Data/CompleteSeason/'
path_to_extra_ims = '/home/cgomez11/Astronomy/Networks/Data/ExtraIms/'
fix_len = 19
class Dataset(data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, list_IDs,labels,f_aug, which_set, transform=None):
        'Initialization'
        self.labels = labels
        self.list_IDs = list_IDs
        self.f_aug = f_aug
        self.which_set = which_set
        self.transform = transform

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        ID = self.list_IDs[index]
        #if obj_type != 'NON':
        #    ID = ID[0:23]
        #modificar ID
        y = self.labels[ID]
        obj_type = classes_names[y]
        if obj_type != 'NON':
            ID = ID[0:23]
        X = generateInstancesSeq(ID, obj_type, self.f_aug, self.which_set) #HxWxCxr=1
        half_fix = int(fix_len // 2)
        if X.shape[2] >= fix_len and X.shape[2]%2 ==0: #even len
            half_seq = int(X.shape[2] /2)
            X = X[:,:,half_seq - half_fix:half_seq+half_fix + 1,:]
        elif X.shape[2] >= fix_len and X.shape[2]%2 != 0: 
            half_seq = int(X.shape[2] //2) + 1
            X = X[:,:,half_seq - half_fix-1:half_seq+half_fix,:]

        #if X.shape[2] >=fix_len:
        #    X = X[:,:,0:fix_len,:]
        X = X.astype('float64') #64x64xCxf_aug
        first_date = np.expand_dims(X[:,:,0,:], axis=2)
        last_date = np.expand_dims(X[:,:,-1,:], axis=2)
        X_fix = np.zeros((64,64,fix_len,self.f_aug))
        X_len = X.shape[2]
        filling_dates = fix_len - X_len
        rep_factor = filling_dates // 2
        if filling_dates == 0:
            X_fix = X
        elif filling_dates == 1:
            X_fix[:,:,0:X_len,:] = X
            X_fix[:,:,-1,:] = X[:,:,-1,:]
        elif filling_dates % 2 == 0:
            X_fix[:,:,0:rep_factor,:] = np.repeat(first_date, rep_factor, axis=2)
            X_fix[:,:,rep_factor:rep_factor + X_len,:] = X
            X_fix[:,:,rep_factor + X_len:,:] = np.repeat(last_date, rep_factor, axis=2)
        elif filling_dates % 2 != 0:
            X_fix[:,:,0:rep_factor,:] = np.repeat(first_date, rep_factor, axis=2)
            X_fix[:,:,rep_factor:rep_factor + X_len,:] = X
            X_fix[:,:,rep_factor + X_len:,:] = np.repeat(last_date, rep_factor+1, axis=2)
        
        if self.transform:
            X_fix = self.transform(X_fix)
        #fts = torch.load(path_to_fts+classes_names[y] + '/' + ID + '.pt')
        return X_fix, y, ID  

class ToTensor(object):
    def __call__(self, X):
        X = X.transpose((3,2,0,1)) #ixCx64x64
        return torch.from_numpy(X)

#load dataset
save_path = args.path_to_save
makedir(save_path)
#means, stds = findStats(args.nChannels,path_to_data,partition)
#print('Done finding means', means, stds)
#means = [0.485, 0.456, 0.406]
#stds = [0.229, 0.224, 0.225]
#image_datasets = {x: Dataset(partition[x], labels, transform=transforms.Compose([ToTensor(0,0) ])) for x in ['train', 'validation']} 
image_datasets = {x: Dataset(partition[x], labels, transform=transforms.Compose([ToTensor()]), f_aug=args.augm, which_set=x) for x in ['train', 'validation']} 
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
        scheduler.step()

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
            len_seqs = 0

            # Iterate over data.
            #for inputs, labels in dataloaders[phase]:
            for batch_idx, (inputs, labels, ID) in tqdm.tqdm(enumerate(dataloaders[phase]), total=len(dataloaders['train']), desc='Batch'):
                #print(ID)
                labels = labels.to(device)
                inputs = inputs.to(device) #bsxixCx64x64
                inputs = inputs.float()
                bs, ncrops, c, h, w = inputs.size()
                inputs = inputs.view(-1, c, h, w) #ixCx64x64
                #pdb.set_trace()
                # forward
                # track history if only in train
                optimizer.zero_grad() 
                with torch.set_grad_enabled(phase == 'train'):
                    output_scores = model(inputs)
                    sm = nn.Softmax(dim=1)
                    output_sm = sm(output_scores)
                    _, preds = torch.max(output_sm, 1)
                    #print(outputs)
                    #print(labels)

                    loss = criterion(output_scores, labels)/ (args.own_batch_size * args.augm) #scale the loss 
                    if phase =='train':
                        loss.backward()
                        optimizer.step()
                    #print(loss)
                    #loss = F.nll_loss(outputs, labels)
                    if phase =='validation':
                        list_labels = list_labels + list(labels.data.cpu().numpy())
                        #list_preds = list_preds + list(preds.data.cpu().numpy())
                        #list_labels.append(labels.data.cpu().numpy())
                        arr_output = np.concatenate((arr_output, output_sm.cpu()),axis=0)

                    del output_sm
                    torch.cuda.empty_cache()

                # statistics
                running_loss += criterion(output_scores, labels)#acumular loss individual
                del output_scores
                len_seqs += args.augm
                #print('loss',running_loss)
                running_corrects += torch.sum(preds == labels.data)
            epoch_loss = running_loss / len_seqs
            #print('size',dataset_sizes[phase])
            epoch_acc = running_corrects.double() / len_seqs
            if phase == 'validation':
                #scheduler.step(epoch_loss)
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

                final_outputs = arr_output
                final_labels = np.array(list_labels)
            if phase =='validation':
                final_outputs = arr_output
                final_labels = np.array(list_labels)
                metrics_each_class, avg_metrics = metrics(epoch, args.nClasses, final_labels, final_outputs)
                file_name = args.path_to_save + '/' + phase + '_'+str(the_epoch)+'_metrics_per_class.txt'
                np.savetxt(file_name, metrics_each_class, fmt=['%u','%.4e', '%.4e', '%.4e'], header = 'C Fmeasu Precision Recall')
                print(avg_metrics)
                with open(file_name,'a') as f:
                    f.write(avg_metrics)

            if phase =='train':
                the_epoch = epoch
                filename = args.path_to_save + 'complete_model_'+ str(the_epoch) +'.pth'
                state = {'epoch': the_epoch , 'state_dict': model.state_dict(),
                     'optimizer': optimizer.state_dict() }
                torch.save(state, filename)
                #np.save(args.path_to_save + 'train_'+ str(the_epoch) +'_outputs.npy', arr_output_train)
                #np.save(args.path_to_save + 'train_'+ str(the_epoch) +'_targets.npy', np.array(list_labels_train))
                #metrics_each_class, avg_metrics = metrics(args.nClasses, np.array(list_labels_train), arr_output_train)
                #file_name = args.path_to_save + '/' + phase + '_'+str(the_epoch)+'_metrics_per_class.txt'
                #np.savetxt(file_name, metrics_each_class, fmt=['%u','%.4e', '%.4e', '%.4e'], header = 'C Fmeasu Precision Recall')
                #with open(filename,'a') as f:
                #    f.write(avg_metrics)
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
