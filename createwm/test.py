#%%
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

from imageutl import imageProvide
from weight_maps import balancewm, unetwm, distranfwm, shapeawewm
from scipy.io import savemat
import os

def myplot(image):
    fig, ax=plt.subplots()
    plt3=ax.matshow(image,cmap=plt.get_cmap('jet'))
    fig.colorbar(plt3)
    plt.show()

path_images='../../data/cambia_wells/labels2c/'

if not os.path.isdir(path_images+'../bwm/'): os.mkdir(path_images+'../bwm/')
if not os.path.isdir(path_images+'../unet2/'): os.mkdir(path_images+'../unet2/')
if not os.path.isdir(path_images+'../unet3/'): os.mkdir(path_images+'../unet3/')
if not os.path.isdir(path_images+'../dwm/'): os.mkdir(path_images+'../dwm/')
if not os.path.isdir(path_images+'../saw/'): os.mkdir(path_images+'../saw/')


data = imageProvide(path=path_images,ext='png')

#%%
for i in range(len(data.files)):
    print('\n Image '+str(i))
    image=data.getimage(i)
    image=image[:,:,2]
    image[image==14]=2
    image[image==161]=1
    image[image==255]=0

    print('Balance WM 2 classes')
    bwm=balancewm(image==1)

    print('Unet WM 2 classes')
    u2wm=unetwm(image==1,10,25)

    print('Unet WM 3 classes')
    u3wm=unetwm(image,10,25)

    print('Distance Transform WM 3 classes')
    dwm=distranfwm(image,beta=30)

    print('Shape Aware WM 3 classes')
    saw=shapeawewm(image,sigma=1)

    savemat(path_images+'../bwm/'+ (data.files[i]).replace('png','mat'),{'weight':bwm} )
    savemat(path_images+'../unet2/'+ (data.files[i]).replace('png','mat'),{'weight':u2wm} )
    savemat(path_images+'../unet3/'+ (data.files[i]).replace('png','mat'),{'weight':u3wm} )
    savemat(path_images+'../dwm/'+ (data.files[i]).replace('png','mat'),{'weight':dwm} )
    savemat(path_images+'../saw/'+ (data.files[i]).replace('png','mat'),{'weight':saw} )