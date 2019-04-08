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


DIR = '/Users/nikita/Desktop/Ricci_Project/Ricci_gist_results/data_c/'


M1, M2, M3, M4 = Ricci_moment_c(str(DIR + 'C37.tif'), 'Result_data_c/', 'T')



