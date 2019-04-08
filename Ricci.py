from sklearn import preprocessing
import matplotlib.pyplot as plt
import numpy as np
import cv2

def img_double(img):

	min_val = np.min(img.ravel())
	max_val = np.max(img.ravel())
	out = (img.astype('float') - min_val)/(max_val - min_val)
	return out

def Ricci_img_g(Img, k_size):
    
    assert(k_size == 1 or k_size == 3 or k_size == 5); # Check error for Sobel kernel
    width, height, c = Img.shape # Main sizes for image
    
    # Check image chanel
    if c > 1:
        Img = 0.2126 * Img[:,:,0] + 0.7152 * Img[:,:,1] + 0.0722 * Img[:,:,2] # If image chenel is not gray scale then will change
    else:
        Img = Img # Save if every things norm
    
    # Calculate Beta
    BetaX = width/ 255.0
    BetaY = height/ 255.0

    Io = Img[2:Img.shape[0] - 2, 2:Img.shape[1] - 2]
    Iu = Img[1:Img.shape[0] - 3, 2:Img.shape[1] - 2]
    Il = Img[2:Img.shape[0] - 2, 1:Img.shape[1] - 3]
    Ir = Img[2:Img.shape[0] - 2, 3:Img.shape[1] - 1]
    Ip = Img[2:Img.shape[0] - 2, 4:Img.shape[1]]
    Id = Img[3:Img.shape[0] - 1, 2:Img.shape[1] - 2]
    Ik = Img[4:Img.shape[0], 2:Img.shape[1] - 2]
    
    dX = cv2.Sobel(Id, cv2.CV_64F, 1,0, k_size)
    dY = cv2.Sobel(Ir, cv2.CV_64F, 0,1, k_size)
    dX1 = cv2.Sobel(Ik, cv2.CV_64F, 1,0, k_size)
    dY1 = cv2.Sobel(Ip, cv2.CV_64F, 0,1, k_size)
    dX2 = cv2.Sobel(Iu, cv2.CV_64F, 1,0, k_size)
    dY2 = cv2.Sobel(Il, cv2.CV_64F, 0,1, k_size)
    
    WeX = np.sqrt(BetaX + np.multiply(dX, dX))
    WeY = np.sqrt(BetaY + np.multiply(dY, dY))
    WeX1 = np.sqrt(BetaX + np.multiply(dX1, dX1))
    WeY1 = np.sqrt(BetaY + np.multiply(dY1, dY1))
    WeX2 = np.sqrt(BetaX + np.multiply(dX2, dX2))
    WeY2 = np.sqrt(BetaY + np.multiply(dY2, dY2))
    
    wc1 = Io
    wc5 = Iu
    wc3 = Il
    wc4 = Ir
    wc2 = Id
    
    Rx = np.multiply(WeX,((WeX/wc1) +(WeX/wc2)-((np.sqrt(np.multiply(WeX, WeX1))/wc1) + (np.sqrt(np.multiply(WeX, WeX2))/wc2))))
    Ry = np.multiply(WeX,((WeY/wc1) +(WeY/wc4)-((np.sqrt(np.multiply(WeY, WeY1))/wc1) + (np.sqrt(np.multiply(WeY, WeY2))/wc4))))
    
    R = Rx + Ry

    D1x=(WeX/wc1) -(WeX/wc2)
    D1y=(WeX/wc1) - (WeY/wc4);
    D1 = D1x + D1y

    D2x = WeX/np.sqrt(wc1*wc2)
    D2y = WeY/np.sqrt(wc1*wc4)
    D2 = D2x + D2y;

    F2 = (WeX1/wc1) + (WeX2/wc1) + (WeY1/wc1) + (WeY2/wc1) - (WeX1/np.sqrt(wc1*wc2) + WeX2/np.sqrt(wc1*wc5) + WeY1/np.sqrt(wc1*wc4) + WeY2/np.sqrt(wc1*wc3))
    
    Ricci = R

    
    return Ricci

def Ricci_img_c(Img):
    
    
    width, height, c = Img.shape # Main sizes for image
    
    # Check image chanel
    if c > 1:
        Img = 0.2126 * Img[:,:,0] + 0.7152 * Img[:,:,1] + 0.0722 * Img[:,:,2] # If image chenel is not gray scale then will change
    else:
        Img = Img # Save if every things norm

    # Calculate Beta
    BetaX = width/ 255.0
    BetaY = height/ 255.0
    
    Io = Img[2:Img.shape[0] - 2, 2:Img.shape[1] - 2]
    Iu = Img[1:Img.shape[0] - 3, 2:Img.shape[1] - 2]
    Il = Img[2:Img.shape[0] - 2, 1:Img.shape[1] - 3]
    Ir = Img[2:Img.shape[0] - 2, 3:Img.shape[1] - 1]
    Ip = Img[2:Img.shape[0] - 2, 4:Img.shape[1]]
    Id = Img[3:Img.shape[0] - 1, 2:Img.shape[1] - 2]
    Ik = Img[4:Img.shape[0], 2:Img.shape[1] - 2]

    
    WeX = (Id + Io)/2
    WeY = (Ir + Io)/2
    WeX1 = (Ik + Id)/2
    WeY1 = (Ip + Ir)/2
    WeX2 = (Io + Iu)/2
    WeY2 = (Io + Il)/2

    wc1 = Io
    wc5 = Iu
    wc3 = Il
    wc4 = Ir
    wc2 = Id
    
    Rx = np.multiply(WeX,((WeX/wc1) +(WeX/wc2)-((np.sqrt(np.multiply(WeX, WeX1))/wc1) + (np.sqrt(np.multiply(WeX, WeX2))/wc2))))
    Ry = np.multiply(WeX,((WeY/wc1) +(WeY/wc4)-((np.sqrt(np.multiply(WeY, WeY1))/wc1) + (np.sqrt(np.multiply(WeY, WeY2))/wc4))))

    R = Rx + Ry

    D1x=(WeX/wc1) -(WeX/wc2)
    D1y=(WeX/wc1) - (WeY/wc4);
    D1 = D1x + D1y
    
    D2x = WeX/np.sqrt(wc1*wc2)
    D2y = WeY/np.sqrt(wc1*wc4)
    D2 = D2x + D2y;
    
    F2 = (WeX1/wc1) + (WeX2/wc1) + (WeY1/wc1) + (WeY2/wc1) - (WeX1/np.sqrt(wc1*wc2) + WeX2/np.sqrt(wc1*wc5) + WeY1/np.sqrt(wc1*wc4) + WeY2/np.sqrt(wc1*wc3))

    Ricci = R
    
    
    return Ricci

def WeightingMethods(img, ksize):
    
    assert(ksize == 1 or ksize == 3 or ksize == 5); # Check error for Sobel kernel
    
    width, height, c = img.shape # Main sizes for image
    
    # Check image chanel
    if c > 1:
        im = 0.2126 * img[:,:,0] + 0.7152 * img[:,:,1] + 0.0722 * img[:,:,2] # If image chenel is not gray scale then will change
    else:
        im = img # Save if every things norm

    # Calculate Beta
    BetaX = width/ 255.0
    BetaY = height/ 255.0
    
    # First order derivatives as Sobel. Used OpenCV library
    dX = cv2.Sobel(im, cv2.CV_64F, 1,0, ksize)
    dY = cv2.Sobel(im, cv2.CV_64F, 0,1, ksize)
    
    # Calculate Geometric Weights
    WeX = np.sqrt(BetaX + dX * dX)
    WeY = np.sqrt(BetaY + dY * dY)
    
    WeX_std = (WeX - np.min(WeX))/(np.max(WeX) - np.min(WeX))
    WeY_std = (WeY - np.min(WeY))/(np.max(WeY) - np.min(WeY))
    
    WeX_scl = WeX_std * 255
    WeY_scl = WeY_std * 255

    G = np.multiply(WeX, WeY)
    G_std = np.multiply(WeX_std, WeY_std)
    
    return WeX, WeY
