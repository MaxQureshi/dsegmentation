import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

# from  visdom import Visdom
# from netframework.utils import graphics as gph

#########################################################################################################    
class WeightedCrossEntropy2d(nn.Module):

    def __init__(self, power=2):
        super(WeightedCrossEntropy2d, self).__init__()
        self.power = power

    def crop(self,w,h,target):
        nt,ht,wt= target.size()
        offset_w,offset_h=(wt-w) // 2 ,(ht-h) // 2
        if offset_w>0 and offset_h>0:
            target=target[:,offset_h:-offset_h,offset_w:-offset_w].clone()
        
        return target

    def to_one_hot(self,target,size):
        n, c, h, w = size

        ymask = torch.FloatTensor(size).zero_()
        new_target=torch.LongTensor(n,1,h,w)
        if target.is_cuda:
            ymask=ymask.cuda(target.get_device())
            new_target=new_target.cuda(target.get_device())

        new_target[:,0,:,:]=torch.clamp(target.detach(),0,c-1)
        ymask.scatter_(1, new_target , 1.0)
        
        return torch.autograd.Variable(ymask)


    def forward(self, input, target,weight=None):
        n, c, h, w = input.size()
        log_p = F.log_softmax(input,dim=1)

        target=self.crop(w,h,target)
        ymask=self.to_one_hot(target,log_p.size())

        if weight is not None:
            weight=self.crop(w,h,weight)
            for classes in range(c):
                ymask[:,classes,:,:]= ymask[:,classes,:,:].clone() * (weight ** self.power)

        logpy = (log_p * ymask).sum(1)
        loss = -(logpy).mean()

        return loss
#########################################################################################################
class WeightedFocalLoss2d(nn.Module):
    def __init__(self, gamma=2, power=1):
        super(WeightedFocalLoss2d, self).__init__()
        self.gamma = gamma
        self.power = power

    def crop(self,w,h,target):
        nt,ht,wt= target.size()
        offset_w,offset_h=(wt-w) // 2 ,(ht-h) // 2
        if offset_w>0 and offset_h>0:
            target=target[:,offset_h:-offset_h,offset_w:-offset_w]
        
        return target

    def to_one_hot(self,target,size):
        n, c, h, w = size

        ymask = torch.FloatTensor(size).zero_()
        new_target=torch.LongTensor(n,1,h,w)
        if target.is_cuda:
            ymask=ymask.cuda(target.get_device())
            new_target=new_target.cuda(target.get_device())

        new_target[:,0,:,:]=torch.clamp(target.detach(),0,c-1)
        ymask.scatter_(1, new_target , 1.0)
        
        return torch.autograd.Variable(ymask)

    def forward(self, input, target, weight=None):
        n, c, h, w = input.size()
        log_p = F.log_softmax(input,dim=1)

        target=self.crop(w,h,target)
        ymask=self.to_one_hot(target,log_p.size())

        if weight is not None:
            weight=self.crop(w,h,weight)
            for classes in range(c):
                ymask[:,classes,:,:]=ymask[:,classes,:,:]*(weight ** self.power)

        
        dweight= (1 - F.softmax(input,dim=1)) ** self.gamma
        logpy = (log_p * ymask * dweight).sum(1)
        loss = -(logpy).mean()

        return loss
#########################################################################################################
class Accuracy(nn.Module):
    def __init__(self):
        super(Accuracy, self).__init__()
        pass

    def crop(self,w,h,target):
        nt,ht,wt= target.size()
        offset_w,offset_h=(wt-w) // 2 ,(ht-h) // 2
        if offset_w>0 and offset_h>0:
            target=target[:,offset_h:-offset_h,offset_w:-offset_w]
        
        return target
        
    def forward(self,input,target):
        n, c, h, w = input.size()
        nt,ht,wt= target.size()

        target=self.crop(w,h,target)

        prob=F.softmax(input,dim=1)
        prob=prob.detach()

        _,maxprob=torch.max(prob,1)

        correct=0
        for cl in range(c):
            correct+= (((maxprob.eq(cl) + target.detach().eq(cl)).eq(2)).view(-1).float().sum(0) +1) / (target.detach().eq(cl).view(-1).float().sum(0) + 1)

        correct=correct/c
        res=correct.mul_(100.0)

        return res
#########################################################################################################
class mAP(nn.Module):
    def __init__(self):
        super(mAP, self).__init__()
        pass

    # def prob2seg(self,output,thresh,threshdiv,cellclass,diviclass,diststeps=3000):

    #     if output.shape[2]>1:
    #         cellsprob=output[:,:,cellclass]
    #     else:
    #         cellsprob=output

    #     prediction=np.array(cellsprob>thresh).astype(int)
    #     predictionlb=label(prediction,connectivity=1)        

    #     if output.shape[2]>2:
    #         divis=output[:,:,diviclass]
    #         divis=np.array(divis>=threshdiv).astype(int)
    #         prediction[divis.astype(bool)]=0
    #         prediction=remove_small_objects(prediction.astype(bool),50,connectivity=1).astype(int)
    #         prediction=binary_fill_holes(prediction).astype(int)
    #         predictionlbnew=label(prediction,connectivity=1)

    #         ccomp=np.array( (prediction+divis)>0 ).astype(int)
    #         ccomp=label(ccomp,connectivity=1)

    #         numcc=np.max(ccomp)
    #         numcell=np.max(predictionlbnew)

    #         for cp in range(1,numcc+1):
    #             o1=np.array((ccomp==cp) & (prediction==1)).astype(int)
    #             x1,y1=np.where(o1==1)
    #             if x1.size>0:
    #                 o2=np.array((ccomp==cp) & (divis==1)).astype(int)
    #                 x2,y2=np.where(o2==1)

    #                 if x2.size>0:
    #                     for a in range(0,x2.size,diststeps):
    #                         minl=np.min(np.array([diststeps+a-1,x2.size-1])) +1
    #                         dis=cdist(np.vstack((x1,y1)).transpose(), np.vstack((x2[a:minl],y2[a:minl])).transpose() )
    #                         ind=np.argmin(dis,axis=0)
    #                         predictionlbnew[x2[a:minl],y2[a:minl]]=predictionlbnew[x1[ind],y1[ind]]
    #             else:
    #                 predictionlbnew[np.array((ccomp==cp) & (divis==1))]=numcell+1
    #                 predictionlbnew[np.array((divis==1))]=numcell+1
    #                 numcell+=1

    #         predictionlb=predictionlbnew

    #     cellscount=np.max(predictionlb)
    #     for i in range(1,cellscount+1):
    #         cell= np.array( (predictionlb ==i) ).astype(int)
    #         cell= binary_fill_holes(cell)
    #         predictionlb[cell]=i

    #     return predictionlb

    def crop(self,w,h,target):
        nt,ht,wt= target.size()
        offset_w,offset_h=(wt-w) // 2 ,(ht-h) // 2
        if offset_w>0 and offset_h>0:
            target=target[:,offset_h:-offset_h,offset_w:-offset_w]
        
        return target

    def jaccard(self,x,y):
        z=x.eq(1).float() + y.eq(1).float()
        intersection = (z.eq(2).float()).sum()
        union = (z.ge(1).float()).sum()

        # z=(x==1).astype('float') + (y==1).astype('float')
        # intersection = (z==2).astype('float').sum()
        # union = (z>=1).astype('float').sum()

        if (intersection.item()==0) and (union.item()==0):
            iou=torch.ones(1)
        elif union.item()==0:
            iou=torch.zeros(1)
        else:
            iou=intersection/union
        return iou

    def forward(self,input,target):
        n, c, h, w = input.size()
        nt,ht,wt= target.size()

        target=self.crop(w,h,target)

        prob=F.softmax(input,dim=1)
        prob=prob.detach()

        _,maxprob=torch.max(prob,1)

        iou=self.jaccard(target,maxprob)

        return iou
        
    # def forward(self,input,target):
    #     n, c, h, w = input.size()
    #     nt,ht,wt= target.size()

    #     target=self.crop(w,h,target)

    #     prob=F.softmax(input,dim=1)
    #     prob=prob.detach()

    #     prob=prob[0].detach().cpu().numpy().transpose((1,2,0))
    #     seg=prob2seg(output=prob,thresh=0.5,threshdiv=0.05,cellclass=1,diviclass=2,diststeps=3000)
    #     region=label2region(seg)

    #     gt= (target==1).astype('uint8')
    #     gtmask=label(gt,connectivity=1)
    #     numccgt=np.max(gtmask)

    #     ccomp=label(region,connectivity=1)
    #     numcc=np.max(ccomp)

    #     for i in range(numcc):
    #         for j in range(numccgt):
    #             pcell=(ccomp==i).astype('uint8')
    #             gtcell=(gtmask==i).astype('uint8')

    #     return res
#########################################################################################################
