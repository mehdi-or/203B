#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from scipy.stats import invgamma, gengamma, norm
import matplotlib.pyplot as plt
#%reset -f

#%% Importing data
def is_float(string):
    """ True if given string is float else False"""
    try:
        return float(string)
    except ValueError:
        return False

data = []
with open('/home/sorouji/Desktop/UCI/Winter2020/203B/Test2/Test/test2.dat', 'r') as f:
    d = f.readlines()
    for i in d:
        k = i.rstrip().split(" ")
        data.append([float(i) if is_float(i) else i for i in k]) 

data = np.array(data, dtype='O')
data2= pd.DataFrame(data[1:,:], columns=['sub', 'cond','rt'])

#%% Part I and II
#data2['sub'=1]
joda=data2[(data2['sub']==1) & (data2['cond']==2)]

person_effect_mean = []
for subject in np.unique(data[1:,0]):
    mean1 = data2[(data2['sub']==subject) & (data2['cond']==1)].iloc[:,2].mean()
    mean2 = data2[(data2['sub']==subject) & (data2['cond']==2)].iloc[:,2].mean()
    person_effect_mean.append(mean2-mean1)
person_effect_mean_sorted = np.sort(person_effect_mean)
plt.plot(person_effect_mean_sorted)

#no effect thershold
zero_line = np.zeros((len(np.unique(data[1:,0])),1) )
plt.plot(zero_line,'-.')

#Average effect Thershold
ave = np.mean(person_effect_mean)
stand = np.std(person_effect_mean)
ave_line = ave*np.ones((len(np.unique(data[1:,0])),1) )
plt.plot(ave_line,'-.')

plt.xlabel('Participant')
plt.ylabel('effect')

plt.gca().legend(('effect','No_effect_thresh','average_effect_thresh='+str(np.round(ave*1000,1))+'ms'))

plt.figure(2)
plt.hist(person_effect_mean,50)

#drawing effect PDF
x_min, x_max = [-0.12, 0.12]
x = np.linspace(x_min, x_max, 100)
y_eff = norm.pdf(x,ave,stand)
plt.plot(x, y_eff)

#drawing no efffect PDF
y_noeff = norm.pdf(x,0,stand)
plt.plot(x, y_noeff)
plt.gca().legend(('effec_fitted','no_effect', 'data'))
plt.xlabel('effect (s)')
plt.ylabel('number of people')
#%% Part III
mu=np.random.normal(0.5,0.5,100000)
plt.figure(3)
plt.hist(mu, 100)
shape = 1
scale = 0.05
samp_invgamma = 1/(np.random.gamma(shape, 1/scale,1000))
