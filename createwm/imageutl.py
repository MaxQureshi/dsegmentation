import os
import numpy as np
import PIL.Image
import scipy.misc
import scipy.io

#########################################################################################################
class imageProvide(object):
    '''
    Management the image resources  
    '''

    def __init__(self, path, ext='jpg', fn_image=''):
        
        if os.path.isdir(path) is not True:
            raise ValueError('Path {} is not directory'.format(path))
        
        self.fn_image = fn_image;
        self.path = path;
        self.pathimage = os.path.join(path, fn_image);

        self.files = [ f for f in sorted(os.listdir(self.pathimage)) if f.split('.')[-1] == ext ];
        self.num = len(self.files);
        
        self.ext = ext;
        self.index = 0;

    def getimage(self, i):
        '''
        Get image i
        '''
        #check index
        if i<0 and i>self.num: raise ValueError('Index outside range');
        self.index = i;
        pathname = os.path.join(self.path,self.fn_image,self.files[i]);        
        return np.array(self._loadimage(pathname));

    def getindex(self, i):
        if i<0 and i>self.num: raise ValueError('Index outside range');
        self.index = i;

    def next(self):
        '''
        Get next image
        '''
        i = self.index;        
        pathname = os.path.join(self.pathimage, self.files[i]); 
        im = self._loadimage(pathname);
        self.index = (i + 1) % self.num;
        return np.array(im);

    def getimagename(self):
        '''
        Get current image name
        '''
        return self.files[self.index];

    def isempty(self): return self.num == 0;

    def _loadimage(self, pathname):
        '''
        Load image using pathname
        '''

        if os.path.exists(pathname):
            try:
                image = PIL.Image.open(pathname)
                image.load()
            except IOError as e:
                raise ValueError('IOError: Trying to load "%s": %s' % (pathname, e.message) ) 
        else:
            raise ValueError('"%s" not found' % pathname)


        if image.mode in ['L', 'RGB']:
            # No conversion necessary
            return image
        elif image.mode in ['1']:
            # Easy conversion to L
            return image.convert('L')
        elif image.mode in ['LA']:
            # Deal with transparencies
            new = PIL.Image.new('L', image.size, 255)
            new.paste(image, mask=image.convert('RGBA'))
            return new
        elif image.mode in ['CMYK', 'YCbCr']:
            # Easy conversion to RGB
            return image.convert('RGB')
        elif image.mode in ['P', 'RGBA']:
            # Deal with transparencies
            new = PIL.Image.new('RGB', image.size, (255, 255, 255))
            new.paste(image, mask=image.convert('RGBA'))
            return new
        else:
            raise ValueError('Image mode "%s" not supported' % image.mode);
        
        return image;
#########################################################################################################
class dataProvide(imageProvide):
    '''
    Management dataset <images, labes>
    '''
    def __init__(self, path, ext='jpg', fn_image='images', fn_label='labels', posfix='', lext='jpg'):
        super(dataProvide, self).__init__(path, ext, fn_image );
        self.fn_label = fn_label;
        self.posfix = posfix;
        self.lext = lext;
                
    def getlabel(self):
        '''
        Get current label
        '''
        i = self.index;
        name = self.files[i].split('.');
        pathname = os.path.join(self.path,self.fn_label,'{}{}.{}'.format(name[0],self.posfix, self.lext) );        
        label = np.array(self._loadimage(pathname));
        if label.ndim == 3: label = label[:,:,2];
        return label;
#########################################################################################################
class matProvide(imageProvide):
    '''
    Management dataset <images, weight>
    '''
    def __init__(self, path, ext='jpg', fn_image='images', fn_weight='weight', posfix='', wext='mat'):
        super(matProvide, self).__init__(path, ext, fn_image );
        self.fn_weight = fn_weight;
        self.posfix = posfix;
        self.wext = wext;
                
    def getweight(self,i):
        '''
        Get current weight
        '''
        if i<0 and i>self.num: raise ValueError('Index outside range');
        self.index = i;
        name = self.files[i].split('.');
        pathname = os.path.join(self.path,self.fn_weight,'{}{}.{}'.format(name[0],self.posfix, self.wext) );        
        weight = scipy.io.loadmat(pathname)['weight'];
        if weight.ndim == 3: weight = weight[:,:,0];
        return weight;
#########################################################################################################