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
loaded_model_path = './saved_model/2020-02-06einradius_resnet18_lastest.mdl'



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

        ID = self.df['ID'].iloc[[index]]

        ein_radius = self.df['ein_area'].iloc[[index]]
        if np.isnan(ein_radius.values[0])==True:
            ein_radius.values[0] = 0.0

        ein_radius.values[0] = np.sqrt(ein_radius.values[0]/np.pi)*1.0e6  #rad*1.0e6

        #print('ground truth:(n_source_im, mag_eff)=',n_source_im.values[0],mag_eff.values[0])
        channel_names = ['EUC_H', 'EUC_J', 'EUC_Y', 'EUC_VIS']
        y=np.array([ein_radius.values[0]])
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

truth_array = np.zeros(20000)
pred_array = np.zeros(20000)


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

        truth=truth/1.0e6*(180/np.pi)*3600
        pred=pred/1.0e6*(180/np.pi)*3600

        truth_array[int(batch_idx*glo_batch_size+i)]=truth
        pred_array[int(batch_idx*glo_batch_size+i)]=pred

        print('groundtruth:(e_radius)=',truth)
        print('prediction :(e_radius)=',pred)

        print("______")

np.save('/home/zjin16/Lens_Finder/er_truth.npy',truth_array)
np.save('/home/zjin16/Lens_Finder/er_pred.npy',pred_array)


plt.scatter(truth_array,pred_array,s=0.5,c='b')
plt.xlabel('ground truth')
plt.ylabel('prediction')
plt.title('Einstein radius prediction vs ground truth')
plt.plot((0,5,8),(0,5,8),ls=':',c='r')

plt.savefig('/home/zjin16/Lens_Finder/einstein_radius.png')
