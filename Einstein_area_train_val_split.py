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


root_folder = "/media/joshua/HDD_fun2/Public/"


### score_list for full train data
df_score = pd.read_csv("./2020-01-30Lens_finding_challenge_submission.csv")

### read all the info provided
full_train_data = pd.read_csv(root_folder + "image_catalog2.0train_v2.csv", header = 0)

### merge the data
merge_data = pd.concat([df_score, full_train_data.reindex(df_score.index)], axis=1)

### select useful info to save
df_Einstein = merge_data[['ID', 'score', 'ein_area']]

### set the threshold to train Einstein radius
threshold = 0.8
df_Einstein_standard = df_Einstein[df_Einstein['score'] > threshold]
print(df_Einstein_standard.shape)



train, val =  train_test_split(df_Einstein_standard, test_size=0.2, random_state=42)

#print(full_data.tail(), full_data.shape)
print(train.head())
print(val.shape)
#
train.to_csv(root_folder + "Einstein_area_train.csv")
val.to_csv(root_folder + "Einstein_area_val.csv")
