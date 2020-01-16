import torch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from  torch.autograd import Variable
import torch.optim as optim
from torchvision import datasets, transforms
import torchvision.models as models
#import lenstronomy.Util.image_util as image_util
import os, sys
import h5py
import pandas as pd
import numpy as np
import scipy.ndimage
from scipy.ndimage import gaussian_filter
from scipy.ndimage import rotate
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import fbeta_score
from tqdm import tqdm
import gc
import astropy
from astropy.io import fits
from astropy.table import Table


root_folder = "/home/zjin16/Strong_Lens_Finder/data/Public/"
loaded_model_path = './saved_model/2019-12-27im_mag_eff_resnet18.mdl'



if os.path.exists(loaded_model_path):
    net = torch.load(loaded_model_path)
    print('loaded mdl！')
else:
    print('No model to load. Should stop!')

print(os.getcwd())

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
data_transform = transforms.Compose([
            transforms.ToTensor(), # scale to [0,1] and convert to tensor
            normalize,
            ])
target_transform = torch.Tensor

##
glo_batch_size = 50
#test_num_batch = 5
##
y_true=np.zeros(glo_batch_size*test_num_batch)
y_pred=np.zeros(glo_batch_size*test_num_batch)
threshold=0.5
beta=np.sqrt(0.001)

class LensDataset(Dataset): # torch.utils.data.Dataset
    def __init__(self, root_dir, train=True, transform=None, target_transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.target_transform = target_transform
        self.train = train# training set or test set


        if self.train:
            self.path = root_dir#os.path.join(self.root_dir, self.train_folder)
            self.df = pd.read_csv(self.path + 'train.csv')

        else:
            self.path = root_dir#os.path.join(self.root_dir, self.test_folder)
            self.df = pd.read_csv(self.path + 'val.csv')

    def __getitem__(self, index):

        #print(self.df['ID'])
        ID = self.df['ID'].iloc[[index]]
        n_source_im = self.df['n_source_im'].iloc[[index]]
        mag_eff = self.df['mag_eff'].iloc[[index]]
        #print(mag_eff.values)
        #print(mag_eff.values.shape)
        if np.isnan(mag_eff.values[0])==True:
            mag_eff.values[0] = 0.0
        if np.isnan(n_source_im.values[0])==True:
            n_source_im.values[0] = 0.0
        #print('ground truth:(n_source_im, mag_eff)=',n_source_im.values[0],mag_eff.values[0])
        channel_names = ['EUC_H', 'EUC_J', 'EUC_Y', 'EUC_VIS']
        y=np.array([n_source_im.values[0], mag_eff.values[0]])
        # filepath = "/media/joshua/HDD_fun2/Public/EUC_Y/imageEUC_Y-" + str(ID.values[0]) + ".fits"
        # lens_data = fits.open(filepath)
        # img = lens_data[0].data
        image = np.zeros((4, 224, 224))

        try:
            for i, channel in enumerate(channel_names):

                filepath = self.path + channel + "/image" + channel + "-" + str(ID.values[0]) + ".fits"
                lens_data = fits.open(filepath)
                img = lens_data[0].data
                img *= 10e8
                img_channel_0 = scipy.ndimage.zoom(img, 224/img.shape[0], order=1)
                image[i, :, :] += img_channel_0
        except:
            print("error", ID)
            pass





        # if self.transform is not None:
        #     image = self.transform(image)

        return image, ID.values[0],y

    def __len__(self):
        return self.df.shape[0]


test_loader = torch.utils.data.DataLoader(LensDataset(root_folder, train=False, transform=data_transform, target_transform=target_transform),
            batch_size = glo_batch_size, shuffle = True)

net.cuda()




for batch_idx, (data, ID, y) in enumerate(test_loader):
    data, target = data.float(), y.float()
    data, target = Variable(data).cuda(), Variable(target).cuda()

    #print("n_source", n_sources)



    #img = scipy.ndimage.zoom(img, 1.4, order=1)
    #img = scipy.ndimage.zoom(img, 1.4, order=1)
    output = net(data)
    #output = F.sigmoid(output)

    #print("prediction:(n_source_im, mag_eff)",output.data.cpu().numpy()) #(1,2) shape when batch size=1,training variable=2
    for i in range(output.data.cpu().numpy().shape[0]):
        truth=target.data.cpu().numpy()[i]
        pred=output.data.cpu().numpy()[i]
        ##(n_source_im , mag_eff)
        if truth[0] > 0:
            if truth[1] > 1.6:
                criteria_tru=1.0
            if truth[1] < 1.0:
                criteria_tru=0.0
            if (truth[1]>=1.0) and (truth[1]<=1.6):
                criteria_tru=(truth[1]-1.0)*(5.0/3.0)
        if truth[0]<= 0:
            criteria_tru = 0.0

        if pred[0] > 0:
            if pred[1] > 1.6:
                criteria_pred=1.0
            if pred[1] < 1.0:
                criteria_pred=0.0
            if (pred[1]>=1.0) and (pred[1]<=1.6):
                criteria_pred=(pred[1]-1.0)*(5.0/3.0)
        if pred[0]<= 0:
            criteria_pred = 0.0
        print('groundtruth:(n_source_im, mag_eff)=',truth[0],truth[1])
        print('prediction :(n_source_im, mag_eff)=',pred[0],pred[1])
        print('criteria(ground truth,prediction)=',criteria_tru,criteria_pred)

        if criteria_tru>=threshold:
            y_true[int(batch_idx*glo_batch_size+i)]=1
        if criteria_pred>=threshold:
            y_pred[int(batch_idx*glo_batch_size+i)]=1

    '''
    if output.data.cpu().numpy()[0][0] < 0.7:
        image = np.zeros((4, 224, 224))
        channel_names = ['EUC_H', 'EUC_J', 'EUC_Y', 'EUC_VIS']
        for i, channel in enumerate(channel_names):
            filepath = root_folder + channel + "/image" + channel + "-" + str(ID.numpy()[0]) + ".fits"
            lens_data = fits.open(filepath)
            img = lens_data[0].data
            img_channel_0 = scipy.ndimage.zoom(img, 224/img.shape[0], order=1)
            image[i, :, :] += img_channel_0
            plt.subplot(1, 4, i+ 1)
            plt.imshow(image[i, :, :])
            plt.title(channel + str(n_sources.numpy()[0]))
        plt.show()
    '''
    print("______")

    #print("flux tpye (prediction):", pred_flux_type)

    '''
    if batch_idx >= (test_num_batch-1):
        break
    '''

print('fbeta_score=',(y_true, y_pred, average='micro', beta=beta))
