import os
import torch
import torchvision
import numpy as np
import scipy.misc as m
import skimage.color as skcolor
import skimage.util as skutl
from skimage import filters
from netframework.dataloaders.imageutl import *

from torch.autograd import Variable

from torch.utils import data
import random

class cdataset(data.Dataset):
    def __init__(self, root, ext='jpg', lext='gif', load_weight=True,ifolder='image',lfolder='label',wfolder='', transform_param=None, n_classes=2):
        self.root = root #root path
        self.n_classes = n_classes #number of classes
        self.transform_param=transform_param #transforms
        self.dataprov = dataProvide( self.root, fn_image=ifolder,ext=ext, fn_label=lfolder, lext=lext)

        self.load_weight=load_weight
        if self.load_weight==True:
            self.weightprov = matProvide(self.root, fn_image=ifolder,ext=ext, fn_weight=wfolder, wext='mat')

    def __len__(self):
        return self.dataprov.num

    def __getitem__(self, index):
        np.random.seed( random.randint(0, 2**32))

        img = self.dataprov.getimage(index)
        lbl = self.dataprov.getlabel()
        
        if self.load_weight==True:
            wht = self.weightprov.getweight(index)
        else:
             wht = lbl

        if img.ndim==2:
            img=np.repeat(img[:,:,np.newaxis],3,axis=2)
        if lbl.ndim==3:
            lbl=np.squeeze(lbl[:,:,0],axis=2)
        if wht.ndim==3:
            wht=np.squeeze(wht[:,:,0],axis=2)

        sample = {'image': img, 'label': lbl, 'weight': wht}

        if self.transform_param is not None:
            sample = self.transform_param(sample)

        sample['idx']=index
        return sample

def warp_Variable(sample,device):
    images, labels,weight = sample['image'], sample['label'],sample['weight']
    images=images.to(device)
    labels=labels.to(device)
    weight=weight.to(device)
    
    sample = {'image': images, 'label':labels, 'weight':weight, 'idx': sample['idx']}
    return sample
