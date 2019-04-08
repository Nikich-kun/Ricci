import os
import cv2
from scipy.stats import moment
import numpy as np
import collections
import argparse
import csv
import shutil
import sys
sys.path.append("..")
from Ricci import *
import matplotlib
from PIL import Image
import matplotlib.pyplot as plt

###########| Normal img with Geometry weight |#####################

DER  = '/Users/nikita/Desktop/CAMELYON17/'
img_n = cv2.imread(str(DER +'n_b4_223_362_.png'))

R_ng = Ricci_img_g(img_n, k_size = 1)

width_ng, height_ng = R_ng.shape
R_new_ng = np.reshape(R_ng,(width_ng * height_ng))

M_new_ng1 = moment(R_new_ng, moment = 1)
M_new_ng2 = moment(R_new_ng, moment = 2)
M_new_ng3 = moment(R_new_ng, moment = 3)
M_new_ng4 = moment(R_new_ng, moment = 4)

plt.figure(figsize=(10,10))
plt.subplot(3,3,1),plt.title('Org'),plt.imshow(img_n)
plt.subplot(3,3,3),plt.title('Ricci for Normal with Geometry weight '),plt.imshow(R_ng)

plt.subplot(3,3,5),plt.title('Hist for Normal with Geometry weight'),plt.ylim(ymax=9000),plt.xlim(-100, 900)
plt.hist(R_new_ng, 100)

plt.text(150, 8000, 'Moment 1 = '"%.0f" % M_new_ng1)
plt.text(150, 6000, 'Moment 2 = '"%.0f" % M_new_ng2)
plt.text(150, 4000, 'Moment 3 = '"%.0f" % M_new_ng3)
plt.text(150, 2000, 'Moment 4 = '"%.0f" % M_new_ng4)

plt.savefig('Normal_img_with_Geometry_weight.png')

###########| Tumor img with Geometry weight |#####################

img_t = cv2.imread(str(DER +'t_b4_251_215_.png'))

R_tg = Ricci_img_g(img_t, k_size = 1)

width_tg, height_tg = R_tg.shape
R_new_tg = np.reshape(R_tg,(width_tg * height_tg))

M_new_tg1 = moment(R_new_tg, moment = 1)
M_new_tg2 = moment(R_new_tg, moment = 2)
M_new_tg3 = moment(R_new_tg, moment = 3)
M_new_tg4 = moment(R_new_tg, moment = 4)

plt.figure(figsize=(10,10))
plt.subplot(3,3,1),plt.title('Org'),plt.imshow(img_t)
plt.subplot(3,3,3),plt.title('Ricci for Tumor with Geometry weight'),plt.imshow(R_tg)
plt.subplot(3,3,5),plt.title('Hist for Tormal with Geometry weight'),plt.ylim(ymax=9000), plt.xlim(-100, 900)
plt.hist(R_new_tg, 100)

plt.text(150, 8000, 'Moment 1 = '"%.0f" % M_new_tg1)
plt.text(150, 6000, 'Moment 2 = '"%.0f" % M_new_tg2)
plt.text(150, 4000, 'Moment 3 = '"%.0f" % M_new_tg3)
plt.text(150, 2000, 'Moment 4 = '"%.0f" % M_new_tg4)

plt.savefig('Tumor_img_with_Geometry_weight.png')

###########| Normal img with Combinatorial weight |#####################

R_nc = Ricci_img_c(img_n)

width_nc, height_nc = R_nc.shape
R_new_nc = np.reshape(R_nc,(width_nc * height_nc))

M_new_nc1 = moment(R_new_nc, moment = 1)
M_new_nc2 = moment(R_new_nc, moment = 2)
M_new_nc3 = moment(R_new_nc, moment = 3)
M_new_nc4 = moment(R_new_nc, moment = 4)

plt.figure(figsize=(10,10))
plt.subplot(3,3,1),plt.title('Org'),plt.imshow(img_n)
plt.subplot(3,3,3),plt.title('Ricci for Normal with Combinatorial weight'),plt.imshow(R_nc)
plt.subplot(3,3,5),plt.title('Hist for Normal with Combinatorial weight'),plt.ylim(ymax=9000), plt.xlim(-100, 900)
plt.hist(R_new_nc, 100)

plt.text(150, 8000, 'Moment 1 = '"%.0f" % M_new_nc1)
plt.text(150, 6000, 'Moment 2 = '"%.0f" % M_new_nc2)
plt.text(150, 4000, 'Moment 3 = '"%.0f" % M_new_nc3)
plt.text(150, 2000, 'Moment 4 = '"%.0f" % M_new_nc4)

plt.savefig('Normal_img_with_Combinatorial_weight.png')

###########| Tumor img with Combinatorial weight |#####################

R_tc = Ricci_img_c(img_t)

width_tc, height_tc = R_tc.shape
R_new_tc = np.reshape(R_tc,(width_tc * height_tc))

M_new_tc1 = moment(R_new_tc, moment = 1)
M_new_tc2 = moment(R_new_tc, moment = 2)
M_new_tc3 = moment(R_new_tc, moment = 3)
M_new_tc4 = moment(R_new_tc, moment = 4)

plt.figure(figsize=(10,10))
plt.subplot(3,3,1),plt.title('Org'),plt.imshow(img_t)
plt.subplot(3,3,3),plt.title('Ricci for Tumor with Combinatorial weight'),plt.imshow(R_tc)
plt.subplot(3,3,5),plt.title('Hist for Tumor with Combinatorial weight'),plt.ylim(ymax=9000), plt.xlim(-100, 900)
plt.hist(R_new_tc, 100)

plt.text(150, 8000, 'Moment 1 = '"%.0f" % M_new_tc1)
plt.text(150, 6000, 'Moment 2 = '"%.0f" % M_new_tc2)
plt.text(150, 4000, 'Moment 3 = '"%.0f" % M_new_tc3)
plt.text(150, 2000, 'Moment 4 = '"%.0f" % M_new_tc4)

plt.savefig('Tumor_img_with_Combinatorial_weight.png')
