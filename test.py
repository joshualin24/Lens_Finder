# -*- coding: utf-8 -*-
"""
Created on Thu Dec 26 12:46:07 2019

@author: zehaojin
"""
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

'''
"/home/zjin16/Strong_Lens_Finder/data/Public/"
pd.read_csv(self.path + 'train.csv')





a=np.load('void_project/data/count_graph.npy')
plt.scatter(a[0],a[1],s=0.5,c='b')
plt.xlabel('ground truth')
plt.ylabel('prediction')
plt.title('void counts prediction vs ground truth')
plt.plot((220,230,240,250,260),(220,230,240,250,260),ls=':',c='r')

plt.savefig('void_project/result/counts.png')
'''


truth=np.load('/home/zjin16/Lens_Finder/er_truth.npy')
pred=np.load('/home/zjin16/Lens_Finder/er_pred.npy')


diff = np.array(truth) - np.array(pred)
print(np.std(diff))
