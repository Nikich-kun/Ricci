import os
import cv2
from scipy.stats import moment
import numpy as np
import collections
import argparse
import csv
import shutil
import uuid
import sys
sys.path.append("..")
from Ricci import *
import matplotlib
from PIL import Image
import matplotlib.pyplot as plt

def Ricci_moment_g(img, out_dir, type):
    
    img = cv2.imread(img)
    R_img = Ricci_img_g(img, k_size = 1)

    width, height  = R_img.shape
    R_reshape_img = np.reshape(R_img,(width * height))

    Moment1 = moment(R_reshape_img, moment = 1)
    Moment2 = moment(R_reshape_img, moment = 2)
    Moment3 = moment(R_reshape_img, moment = 3)
    Moment4 = moment(R_reshape_img, moment = 4)

    plt.figure()
    plt.title('Org'),plt.imshow(img)
    plt.savefig(str(out_dir + type +'-ORG'+ '-' + 'GEOM' + '-' + str(uuid.uuid4()) + '.png'))
    
    plt.figure()
    plt.title('Ricci image'),plt.imshow(R_img)
    plt.savefig(str(out_dir + type + '-RIMG'+'-' + 'GEOM' + '-' + str(uuid.uuid4()) + '.png'))
    
    plt.figure()
    plt.subplot(3,3,5),plt.title('Hist for Normal with Geometry weight'),plt.ylim(ymax=9000),plt.xlim(-100, 900)
    plt.hist(R_reshape_img, 100)
    
    for area in ['Moment 1 = '"%.2f" % Moment1, 'Moment 2 = '"%.2f" % Moment2, 'Moment 3 = '"%.2f" % Moment3, 'Moment 4 = '"%.2f" % Moment4]:
        plt.scatter([], [], label=str(area))
    
    plt.legend( bbox_to_anchor=(1.05, 1), loc=2,title='Set of Moments')

    plt.savefig(str(out_dir + type + '-HIST'+'-' + 'GEOM' + '-' + str(uuid.uuid4()) + '.png'))
    
    plt.show()

    return Moment1, Moment2, Moment3, Moment4

def Ricci_moment_c(img, out_dir, type):
    
    img = cv2.imread(img)
    
    R_img = Ricci_img_c(img)
    
    width, height  = R_img.shape
    R_reshape_img = np.reshape(R_img,(width * height))
    
    Moment1 = moment(R_reshape_img, moment = 1)
    Moment2 = moment(R_reshape_img, moment = 2)
    Moment3 = moment(R_reshape_img, moment = 3)
    Moment4 = moment(R_reshape_img, moment = 4)
    
    plt.figure(figsize=(30,30))
    plt.title('Org'),plt.imshow(img)
    plt.savefig(str(out_dir +'-ORG'+ type + '-' + 'COMB' + '-' + str(uuid.uuid4()) + '.png'))
    
    plt.figure(figsize=(30,30))
    plt.title('Ricci image'),plt.imshow(R_img)
    plt.savefig(str(out_dir + type +'-RIMG'+ '-' + 'COMB' + '-' + str(uuid.uuid4()) + '.png'))
    
    plt.figure(figsize=(30,30))
    plt.title('Hist for Normal with Geometry weight'),plt.ylim(ymax=9000),plt.xlim(-100, 900)
    plt.hist(R_reshape_img, 100)
    
    for area in ['Moment 1 = '"%.2f" % Moment1, 'Moment 2 = '"%.2f" % Moment2, 'Moment 3 = '"%.2f" % Moment3, 'Moment 4 = '"%.2f" % Moment4]:
        plt.scatter([], [], label=str(area))
    
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, title='Set of Moments')
    
    plt.savefig(str(out_dir + type + '-HIST'+'-' + 'COMB' + '-' + str(uuid.uuid4()) + '.png'))
    
    plt.close()
    
    return Moment1, Moment2, Moment3, Moment4
