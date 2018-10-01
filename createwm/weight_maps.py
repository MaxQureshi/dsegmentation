import numpy as np
from scipy import ndimage
from skimage.morphology import convex_hull_image, thin
from scipy.spatial.distance import cdist

# class balance weight map
def balancewm(mask):

    wc = np.empty(mask.shape)
    classes = np.unique(mask)
    freq = [ 1.0 / np.sum(mask==i) for i in classes ]
    freq /= max(freq)


    for i in range(len(classes)):
        wc[mask == classes[i]] = freq[i]

    return wc 

# unet weight map
def unetwm(mask,w0,sigma):
    mask = mask.astype('float')
    wc = balancewm(mask)

    cells,cellscount = ndimage.measurements.label(mask==1)
    # maps = np.zeros((mask.shape[0],mask.shape[1],cellscount))
    d1 = np.ones_like(mask)*np.Infinity
    d2 = np.ones_like(mask)*np.Infinity
    for ci in range(1,cellscount+1):
        dstranf = ndimage.distance_transform_edt(cells!=ci)
        d1 = np.amin( np.concatenate((dstranf[:,:,np.newaxis],d1[:,:,np.newaxis]),axis=2),axis=2)
        ind = np.argmin( np.concatenate((dstranf[:,:,np.newaxis],d1[:,:,np.newaxis]),axis=2),axis=2)
        dstranf[ind==0]=np.Infinity
        if cellscount>1:
            d2 = np.amin( np.concatenate((dstranf[:,:,np.newaxis],d2[:,:,np.newaxis]),axis=2),axis=2)
        else:
            d2 = d1.copy()

    # maps = np.sort(maps,axis=2)
    # d1 = maps[:,:,0]
    # if cellscount>1:
    #     d2 = maps[:,:,1]
    # else:
    #     d2 = d1
    uwm = 1 + wc + (mask==0).astype('float')*w0*np.exp( (-(d1+d2)**2) / (2*sigma)).astype('float')
    
    return uwm

def distranfwm(mask,beta):
    mask = mask.astype('float')
    wc = balancewm(mask)

    dwm = ndimage.distance_transform_edt(mask!=1)
    dwm[dwm>beta] = beta
    dwm= wc + (1.0 - dwm/beta) +1
    
    return dwm

def shapeawewm(mask,sigma):
    mask = mask.astype('float')
    wc = balancewm(mask)
    binimage=(mask==1).astype('float')
    diststeps=10000
    
    cells,cellscount = ndimage.measurements.label(binimage)
    chull = np.zeros_like(mask)
    # convex hull of each object
    for ci in range(1,cellscount+1):
        I = (cells==ci).astype('float')
        R = convex_hull_image(I) - I
        R = ndimage.binary_opening(R,structure=np.ones((3,3))).astype('float')
        R = ndimage.binary_dilation(R,structure=np.ones((3,3))).astype('float')
        chull += R

    # distance transform to object skeleton
    skcells=thin(binimage)
    dtcells=ndimage.distance_transform_edt(skcells!=1)
    border=binimage-ndimage.binary_erosion(input=(binimage),structure=np.ones((3,3)),iterations=1).astype('float')
    tau=np.max(dtcells[border==1])+0.1
    dtcells=np.abs(1-dtcells*border/tau)*border

    # distance transform to convex hull skeleton
    skchull=thin(chull)
    dtchull=ndimage.distance_transform_edt(skchull!=1)
    border=chull-ndimage.binary_erosion(input=(chull),structure=np.ones((3,3)),iterations=1).astype('float')
    dtchull=np.abs(1-dtchull*border/tau)*border

    # maximum border
    saw=np.concatenate((dtcells[:,:,np.newaxis],dtchull[:,:,np.newaxis]),2)
    saw = np.max(saw,2)
    saw /= np.max(saw)

    # propagate contour values inside the objects
    prop=binimage+chull
    prop[prop>1]=1
    prop=ndimage.binary_erosion(input=(prop),structure=np.ones((3,3)),iterations=1).astype('float')
    current_saw=saw

    for i in range(20):
        tprop=ndimage.binary_erosion(input=(prop),structure=np.ones((3,3)),iterations=1).astype('float')
        border=prop-tprop
        prop=tprop

        x1,y1 = np.where(border!=0)
        x2,y2 = np.where(current_saw!=0)

        if x1.size==0 or x2.size==0: break

        tsaw=np.zeros_like(saw)
        for a in range(0,x1.size,diststeps):
            minl=np.min(np.array([diststeps+a-1,x1.size-1])) +1
            dis=cdist(np.vstack((x2,y2)).transpose(), np.vstack((x1[a:minl],y1[a:minl])).transpose())
            ind=np.argmin(dis,axis=0)
            tsaw[x1[a:minl],y1[a:minl]]=current_saw[x2[ind],y2[ind]]

        saw=np.concatenate((saw[:,:,np.newaxis],tsaw[:,:,np.newaxis]),2)
        saw = np.max(saw,2)
        saw=ndimage.filters.gaussian_filter(saw,sigma)
        saw/=np.max(saw)
        current_saw=saw*(border!=0).astype('float')

    saw = saw + wc +1
    return saw
