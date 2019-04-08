
import os
import cv2
from scipy.stats import moment
import pandas as pd
import numpy as np
import collections
import argparse
import csv
import shutil
import uuid
import sys
sys.path.append("..")
from Ricci import *
from Ricci_moment import *
import matplotlib
from PIL import Image
import matplotlib.pyplot as plt


DIR = 'CAMELYON17/'

Moment1 = []
Moment2 = []
Moment3 = []
Moment4 = []

all = 0
for i in os.listdir(DIR):
    if os.path.isfile(os.path.join(DIR,i)) and i.startswith('n'):
        if(all == 2):
            break
        all+=1
    
        M1, M2, M3, M4 = Ricci_moment_g(str(DIR + i), 'Result_for_report/', 'N')
        Moment1.append(M1)
        Moment2.append(M2)
        Moment3.append(M3)
        Moment4.append(M4)

AVR_Moment1 = sum(Moment1) / float(len(Moment1))
AVR_Moment2 = sum(Moment2) / float(len(Moment2))
AVR_Moment3 = sum(Moment3) / float(len(Moment3))
AVR_Moment4 = sum(Moment4) / float(len(Moment4))

plt.figure(figsize=(10,10))

plt.subplot(4,1,1), plt.title('Moment 1')
plt.hist(Moment1, 100, label='AVR of Moment 1 = '"%.0f" % AVR_Moment1)
plt.legend(loc='best')
         
plt.subplot(4,1,2), plt.title('Moment 2')
plt.hist(Moment2, 100, label='AVR of Moment 2 = '"%.0f" % AVR_Moment2)
plt.legend(loc='best')

plt.subplot(4,1,3), plt.title('Moment 3')
plt.hist(Moment3, 100, label='AVR of Moment 3 = '"%.0f" % AVR_Moment3)
plt.legend(loc='best')
         
plt.subplot(4,1,4), plt.title('Moment 4')
plt.hist(Moment4, 100, label='AVR for Moment 4 = '"%.0f" % AVR_Moment4)
plt.legend(loc='best')

plt.subplots_adjust(hspace = 0.8)

plt.savefig('Result/T_G/Result of all test/Histogram_for_all_moments.png')

D_Moment1 = pd.DataFrame({'Moment 1': Moment1})
D_Moment2 = pd.DataFrame({'Moment 2': Moment2})
D_Moment3 = pd.DataFrame({'Moment 3': Moment3})
D_Moment4 = pd.DataFrame({'Moment 4': Moment4})

DF_M = pd.concat([D_Moment1, D_Moment2, D_Moment3, D_Moment4], axis = 1)

DF_M.to_csv('Result/T_G/Result of all test/List_of_Moments.csv', sep = ';')
