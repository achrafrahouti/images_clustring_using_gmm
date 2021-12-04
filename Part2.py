# -*- coding: utf-8 -*-
"""
Created on Mon Nov 29 13:09:19 2021

@author: Rahouti
"""

from sklearn.mixture import GaussianMixture
clf = GaussianMixture(n_components =2,covariance_type='diag',init_params='kmeans')
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import glob
class GMM():
    
    def __init__(self):
        path_TEST = "./DataToPredict/*.jpg"
        path_train = "./DataToLearn/*.jpg"
        self.list_image=[]
        self.Pretretement_images(path_train)
        self.Apprentissage(clf)
        self.classification(path_TEST)
    def Pretretement(self,path_img):
            # lire image 
            img=plt.imread(path_img)  
            #redimentionemment
            img = cv.resize(img,(60,40))
            #binarisation
            (thresh, im_bw) = cv.threshold(img, 128, 1, cv.THRESH_BINARY )
            img=np.array(im_bw)
            # Flatten image              
            img=img.flatten()
            return img
    def Pretretement_images(self,path):
        path = glob.glob(path)
        for path_img in path:
            self.list_image.append(self.Pretretement(path_img)) 

    def Apprentissage(self,clf):
        clf.fit(self.list_image)
    def classification(self,path):
        path = glob.glob(path)
        plt.figure(figsize=(60,40))
        k=1
        for path_img in path:  
            img=self.Pretretement(path_img)
            image=img.reshape(1,-1)
            img = plt.imread(path_img) 
            plt.subplot(2,5,k)
            plt.imshow(img,cmap='gray')
            class_ = clf.predict(image)
            plt.title("class"+str(class_),fontsize=60)
            k+=1

app=GMM()
