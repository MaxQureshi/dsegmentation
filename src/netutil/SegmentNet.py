import torch
from netframework.netutil.NetFramework import NetFramework
import torch.nn.functional as F
from scipy.misc import imsave
import os
import numpy as np
from scipy.io import savemat, loadmat

class SegmentNet(NetFramework):
    def __init__(self,default_path):
        NetFramework.__init__(self,default_path)

    def crop(self,w,h,target):
        nt,ht,wt= target.size()
        offset_w,offset_h=(wt-w) // 2 ,(ht-h) // 2
        if offset_w>0 and offset_h>0:
            target=target[:,offset_h:-offset_h,offset_w:-offset_w]
        
        return target

    def valid_visualization(self,current_epoch,index=0,save=False):  
        index=0
        with torch.no_grad():

            sample=self.testdataset[ index ]
            sample['image'].unsqueeze_(0)
            sample['label'].unsqueeze_(0)
            sample['weight'].unsqueeze_(0)

            sample=self.warp_var_mod.warp_Variable(sample,self.device)
            images=sample['image']
            labels=sample['label']
            weights=sample['weight']

            outputs = self.net(images)       
            prob=F.softmax(outputs,dim=1)
            prob=prob.detach()[0]
            _,maxprob=torch.max(prob,0)       

            if self.visdom==True:
                self.visheatmap.show('Binary Segmentation',maxprob.cpu().numpy(),colormap='Jet',scale=0.5)
                self.visheatmap.show('Output Map 0',prob.cpu()[0].numpy(),colormap='Jet',scale=0.5)
                self.visheatmap.show('Output Map 1',prob.cpu()[1].numpy(),colormap='Jet',scale=0.5)
                if prob.size(0)>2:
                    self.visheatmap.show('Output Map 2',prob.cpu()[2].numpy(),colormap='Jet',scale=0.5)
                #if current_epoch==self.init_epoch:
                    labels=self.crop(prob.size(1),prob.size(2),labels)
                    weights=self.crop(prob.size(1),prob.size(2),weights)
                    images=self.crop(prob.size(1),prob.size(2),images[0])
                    self.visheatmap.show('Label',labels.detach().cpu()[0].numpy(),colormap='Jet',scale=0.5)
                    self.visheatmap.show('Weight map',weights.detach().cpu()[0].numpy(),colormap='Jet',scale=0.5)
                    self.visheatmap.show('Image',images.detach().cpu().numpy()[0],colormap='Greys',scale=0.5)

            if save==True:
                if prob.size(0)>2:
                    imsave(os.path.join(self.folders['images_path'],'image-{:d}-{:03d}'.format(index+1,current_epoch) +'.png'), prob.cpu().numpy().transpose((1,2,0))*255)
                else:
                    imsave(os.path.join(self.folders['images_path'],'image-{:d}-{:03d}'.format(index+1,current_epoch) +'.png'), prob[1].cpu().numpy()*255)

        return 1
