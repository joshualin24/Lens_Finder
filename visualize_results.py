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
from tqdm import tqdm
import gc
import astropy
from astropy.io import fits
from astropy.table import Table


from sklearn.metrics import fbeta_score

root_folder = "/home/zjin16/Strong_Lens_Finder/data/Public/"
loaded_model_path = './saved_model/2020-02-04im_mag_eff_pix_einradius_resnet18.mdl'



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
test_num_batch = 400
##
y_true=np.zeros((6,glo_batch_size*test_num_batch))
y_pred=np.zeros((6,glo_batch_size*test_num_batch))

threshold=np.array([0.80,0.90,0.99,0.80,0.90,0.99])
#beta=np.sqrt(0.001)

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
        n_pix_source = self.df['n_pix_source'].iloc[[index]]

        ein_radius = self.df['ein_area'].iloc[[index]]
        #print(mag_eff.values)
        #print(mag_eff.values.shape)
        if np.isnan(mag_eff.values[0])==True:
            mag_eff.values[0] = 0.0
        if np.isnan(n_source_im.values[0])==True:
            n_source_im.values[0] = 0.0
        if np.isnan(n_pix_source.values[0])==True:
            n_pix_source.values[0] = 0.0
        if np.isnan(ein_radius.values[0])==True:
            ein_radius.values[0] = 0.0

        ein_radius.values[0] = np.sqrt(ein_radius.values[0]/np.pi)*1.0e6

        #print('ground truth:(n_source_im, mag_eff)=',n_source_im.values[0],mag_eff.values[0])
        channel_names = ['EUC_H', 'EUC_J', 'EUC_Y', 'EUC_VIS']
        y=np.array([n_source_im.values[0], mag_eff.values[0],n_pix_source.values[0],ein_radius.values[0]])
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

criteria_target_list = []
criteria_output_list = []


for batch_idx, (data, ID, y) in enumerate(test_loader):
    data, target = data.float(), y.float()
    data, target = Variable(data).cuda(), Variable(target).cuda()

    #print("n_source", n_sources)



    #img = scipy.ndimage.zoom(img, 1.4, order=1)
    #img = scipy.ndimage.zoom(img, 1.4, order=1)
    output = net(data)
    #output = F.sigmoid(output)

    #print("prediction:(n_source_im, mag_eff)",output.data.cpu().numpy()) #(1,2) shape when batch size=1,training variable=2
    for i in range(glo_batch_size):
        truth=target.data.cpu().numpy()[i]
        pred=output.data.cpu().numpy()[i]
        ##(n_source_im , mag_eff)
        '''
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
        print('~~~~~criteria~~~~~(ground truth,prediction)=',criteria_tru,criteria_pred)
        '''

        #'''
        if truth[0] > 0:
            if truth[1] > 1.6:
                if truth[2] > 20:
                    criteria_tru=1.0
                if truth[2]<= 20:
                    criteria_tru=0.0

            if truth[1] < 1.0:
                criteria_tru=0.0
            if (truth[1]>=1.0) and (truth[1]<=1.6):
                if truth[2] > 20:
                    criteria_tru=(truth[1]-1.0)*(5.0/3.0)
                if truth[2]<= 20:
                    criteria_tru=0.0

        if truth[0]<= 0:
            criteria_tru = 0.0

        if pred[0] > 0:
            if pred[1] > 1.6:
                if pred[2] > 20:
                    criteria_pred=1.0
                if pred[2]<= 20:
                    criteria_pred=0.0

            if pred[1] < 1.0:
                criteria_pred=0.0
            if (pred[1]>=1.0) and (pred[1]<=1.6):
                if pred[2] > 20:
                    criteria_pred=(pred[1]-1.0)*(5.0/3.0)
                if pred[2]<= 20:
                    criteria_pred=0.0

        if pred[0]<= 0:
            criteria_pred = 0.0

        print('groundtruth:(n_source_im, mag_eff, n_pix_source,e_radius)=',truth[0],truth[1],truth[2],truth[3])
        print('prediction :(n_source_im, mag_eff, n_pix_source,e_radius)=',pred[0],pred[1],pred[2],pred[3])
        print('~~~~~criteria~~~~~(ground truth,prediction)=',criteria_tru,criteria_pred)
        #'''

        #'''
        for j in range(6):
            if criteria_tru>=threshold[j]:
                y_true[j,int(batch_idx*glo_batch_size+i)]=1
            if criteria_pred>=threshold[j]:
                y_pred[j,int(batch_idx*glo_batch_size+i)]=1

        #'''

        ##criteria_target_list.append(criteria_tru)
        ##criteria_output_list.append(criteria_pred)

        #print("______")

'''
def fbeta_score(y_true, y_pred, beta, threshold, eps=1e-9):
    beta2 = beta**2

    y_pred = torch.ge(y_pred.float(), threshold).float()
    y_true = y_true.float()

    true_positive = (y_pred * y_true).sum(dim=1)
    precision = true_positive.div(y_pred.sum(dim=1).add(eps))
    recall = true_positive.div(y_true.sum(dim=1).add(eps))

    return torch.mean(
        (precision*recall).
        div(precision.mul(beta2) + recall + eps).
        mul(1 + beta2))


py_pred = torch.from_numpy(np.array([criteria_output_list]))
py_true = torch.from_numpy(np.array([criteria_target_list]))



print('fbeta_score=',fbeta_score(py_true, py_pred,beta=0.001,threshold=0.80),'beta,threshold=0.001,0.80')
print('fbeta_score=',fbeta_score(py_true, py_pred,beta=0.001,threshold=0.90),'beta,threshold=0.001,0.90')
print('fbeta_score=',fbeta_score(py_true, py_pred,beta=0.001,threshold=0.99),'beta,threshold=0.001,0.99')

print('fbeta_score=',fbeta_score(py_true, py_pred,beta=0.01,threshold=0.80),'beta,threshold=0.01,0.80')
print('fbeta_score=',fbeta_score(py_true, py_pred,beta=0.01,threshold=0.90),'beta,threshold=0.01,0.90')
print('fbeta_score=',fbeta_score(py_true, py_pred,beta=0.01,threshold=0.99),'beta,threshold=0.01,0.99')
'''

print('fbeta_score=',fbeta_score(y_true[0],y_pred[0],average='macro',beta=0.001),';beta,threshold=0.001,0.80')
print('fbeta_score=',fbeta_score(y_true[1],y_pred[1],average='macro',beta=0.001),';beta,threshold=0.001,0.90')
print('fbeta_score=',fbeta_score(y_true[2],y_pred[2],average='macro',beta=0.001),';beta,threshold=0.001,0.99')

print('fbeta_score=',fbeta_score(y_true[3],y_pred[3],average='macro',beta=0.01),';beta,threshold=0.01,0.80')
print('fbeta_score=',fbeta_score(y_true[4],y_pred[4],average='macro',beta=0.01),';beta,threshold=0.01,0.90')
print('fbeta_score=',fbeta_score(y_true[5],y_pred[5],average='macro',beta=0.01),';beta,threshold=0.01,0.99')

print(y_true.shape)
print(y_pred.shape)
print(y_true)
print(y_pred)
