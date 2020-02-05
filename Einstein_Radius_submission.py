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


root_folder = "/media/joshua/HDD_fun2/Lens_finder_test/Public/"


### score_list for full train data
df_score = pd.read_csv("./2020-02-02Lens_finding_challenge_submission.csv")



### set the threshold to train Einstein radius
threshold = 0.85
df_Einstein_selected = df_score[df_score['score'] > threshold]
print(df_Einstein_selected.shape)
