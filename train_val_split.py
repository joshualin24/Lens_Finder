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



#root_folder = "./v0812_redshift_aug_Full_train/"
root_folder = "/media/joshua/HDD_fun2/Public/"

#train_folder = "train/"
#test_folder = "test/"
#full_data = pd.read_csv(root_folder + "image_catalog2.0train.csv" , skiprows=28)
full_data = pd.read_csv(root_folder + "image_catalog2.0train_v2.csv", header = 0)
#full_data.to_csv("./Full_train/clean_full_data.csv")
#print(full_data)

# if not os.path.exists(root_folder + train_folder):
#     os.mkdir(root_folder + train_folder)
#
# if not os.path.exists(root_folder + test_folder):
#     os.mkdir(root_folder + test_folder)


train, val =  train_test_split(full_data, test_size=0.2, random_state=42)

print(full_data.tail(), full_data.shape)
print(train.head())
print(val.shape)
#
train.to_csv(root_folder + "train.csv")
val.to_csv(root_folder + "val.csv")
