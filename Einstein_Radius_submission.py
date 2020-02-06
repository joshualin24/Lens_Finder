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
import astropy
from astropy.io import fits
from astropy.table import Table
import pandas as pd
import csv
import numpy as np
import matplotlib.pyplot as plt
from astropy.coordinates import search_around_sky, SkyCoord
from astropy import units as u
from sklearn.model_selection import train_test_split
from shutil import copyfile
import os, sys
import gc
import datetime


root_folder = "/media/joshua/HDD_fun2/Lens_finder_test/Public/"
loaded_model_path = './saved_model/2020-02-04Einstein_Radius_resnet18.mdl'
# files = os.listdir(EHT_test_path)
#loaded_model_path = './saved_model/flux_resnet18.mdl'


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

### score_list for full train data
df_score = pd.read_csv("./2020-02-02Lens_finding_challenge_submission.csv")



### set the threshold to train Einstein radius
threshold = 0.85
df_Einstein_selected = df_score[df_score['score'] > threshold]
print(df_Einstein_selected.shape)

ID_list = []
Einstein_Radius_list = []
count = 0
for ID in df_Einstein_selected['ID']:
    count += 1
    #print("score", df_Einstein_selected[df_Einstein_selected['ID']==ID].score.values[0])
    image = np.zeros((4, 224, 224))
    channel_names = ['EUC_H', 'EUC_J', 'EUC_Y', 'EUC_VIS']
    #plt.figure(figsize=(20, 5))
    for i, channel in enumerate(channel_names):
        filepath = root_folder + channel + "/image" + channel + "-" + str(ID) + ".fits"
        lens_data = fits.open(filepath)
        img = lens_data[0].data
        img_channel_0 = scipy.ndimage.zoom(img, 224/img.shape[0], order=1)
        image[i, :, :] += img_channel_0
    image *= 10e8
    blind_image = torch.from_numpy(image).float().cuda().unsqueeze(0)

    ### flux output
    blind_output = net(blind_image)
    #print("blind_output", blind_output.data.cpu().numpy()[0][0])
    ID_list.append(ID)
    Einstein_Radius_list.append(blind_output.data.cpu().numpy()[0][0]/ (np.pi**0.5) )
    if count % 200 == 0:
        print("count", count)
    #     break
    gc.collect()


df_Einstein = pd.DataFrame()
df_Einstein['ID'] = ID_list
df_Einstein['Einstein_Radius'] = Einstein_Radius_list
print(df_Einstein.head())


datetime_today = str(datetime.date.today())
df_Einstein.to_csv(datetime_today +  "Einstein_Radius_submission.csv")
