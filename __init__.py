import numpy as np
import numpy as np
import scipy as sp
import scipy.signal
import scipy.ndimage
import types  
import multiprocessing    
import multiprocessing.pool

class MocorOptimizer:
    def __init__(self,guys,clipin,tqdmnotebook=False,nthreads=8):
        self.guys=guys
        self.clipin=clipin
        self.tqdmnotebook=tqdmnotebook
        self.nthreads=nthreads

    def initialize(self):
        '''
        Initializes a reasonable guess for the motion correction of guys
        '''
        self.status='initializing'
        sizes=np.array([x.shape for x in self.guys])
        shp=np.min(sizes,axis=0)
        sl=(slice(0,shp[0]),slice(0,shp[1]))
        self.template = np.mean([x[sl] for x in self.guys],axis=0)
        self._get_offsets()

    def _get_offsets(self):
        binsize=50
        if self.tqdmnotebook:
            import tqdm
            r=lambda x: tqdm.tqdm_notebook(x,total=len(self.guys)//binsize)
        else:
            r=lambda x:x

        offsets=np.zeros((len(self.guys),2),dtype=np.int)
        for i,v in r(get_offsets_multithreaded_iterator(
                        self.guys,
                        self.template,
                        self.clipin,
                        nthreads=self.nthreads,
                        binsize=binsize)):
            offsets[i]=v
        self.offsets=offsets

    def update(self):
        '''
        Performs a round of optimization to improve the template and offsets
        '''
        self.keepgoing=True

        # perform motion correction based on current offset
        tseries2,temppos=apply_a_mocor(self.guys,self.offsets)

        # get a new template out of that, set that to our new template
        self.template = np.mean(tseries2,axis=0)

        # compute the new offsets for that template
        self._get_offsets()

def get_offsets_multithreaded_iterator(guys,template,clipin,nthreads=8,binsize=50):
    '''
    Find a location for the template in each guy, sliding the template around by at most 'clipin'
    '''

    n=len(guys)
    clipin=int(clipin)
    temp=template[clipin:-clipin,clipin:-clipin]
    def go(sl):
        lst=[]
        for i in sl:
            img=guys[i]
            assert (np.array(temp.shape) < np.array(img.shape)).all()
            rez=normcor(img,temp)
            lst.append(np.unravel_index(np.argmax(rez),rez.shape))
        return sl,np.array(lst)-clipin

    bins=np.r_[0:len(guys):binsize,len(guys)]
    todo=[range(bins[i],bins[i+1]) for i in range(len(bins)-1)]
    with multiprocessing.pool.ThreadPool(nthreads) as ex:
        yield from ex.imap_unordered(go,todo)  

def apply_a_mocor(guys,gamma):
    '''
    Input:
    - guys are a list of images, possibly with different sizes
    - gamma indicates the upper-left corner of where a reference template should be placed in each image

    Output:
    - tseries2 - A regular Nt x Nrow x Ncol array
    - temppos  - the canonical location of the reference template for each t

    In the resulting tseries2, we should have that

       tseries2[t,temppos[0],temppos[1]] 

    corresponds to the upper-left corner of the reference template
    for every t.  

    '''

    temppos=np.min(gamma,axis=0) 
    sizes=np.array([x.shape for x in guys])

    # tseries[t,0,0] = guys[t,startsites[t][0],startsites[t][1]]
    startsites=gamma-temppos

    # need 
    # tseries[t,Nrows-1,Ncols-1] = guys[t,startsites[t][0]+Nrows-1,startsites[t][1]+Ncols-1]
    # so need startsites+[Nrows,Ncols] <= sizes
    # so need [Nrows,Ncols] <= sizes - startsites
    targsh = np.min(sizes - startsites,axis=0)

    # do the mocor
    n=len(guys)
    tseries2=np.zeros((n,targsh[0],targsh[1]),dtype=guys[0].dtype)

    for i in range(n):
        tseries2[i]=guys[i][
            startsites[i,0]:startsites[i,0]+targsh[0],
            startsites[i,1]:startsites[i,1]+targsh[1]]
             
    return tseries2,temppos


###################################################
###################################################
###################################################
###################################################

def unif_filt1_valid(A,m):
    assert m>1
    if m%2==0:
        return sp.ndimage.uniform_filter1d(A,m,axis=0)[m//2:A.shape[0]-m//2+1]*m
    else:
        return sp.ndimage.uniform_filter1d(A,m,axis=0)[m//2:A.shape[0]-m//2]*m
    
def unif_filt2_valid(A,m1,m2):
    B=unif_filt1_valid(A,m1)
    B=unif_filt1_valid(B.T,m2).T
    return B

def normcor(X,Y):
    '''
    X is a big 2-array
    Y is a smaller 2-array
    
    For each valid placement of Y inside X, we want to compute
    the normalized correlation between Y and the X pixels where Y is placed
    '''

    X=np.require(X,dtype=np.float)
    Y=np.require(Y,dtype=np.float)
    assert len(X.shape)==2
    assert len(Y.shape)==2

    assert X.shape[0]>=Y.shape[0]
    assert X.shape[1]>=Y.shape[1]
    
    # normalize Y
    Y=Y-np.mean(Y)
    Y=Y/np.linalg.norm(Y)
    
    # get the number of pix
    NN=np.prod(Y.shape)
    
    # for each valid placement rgn, get sum x_i
    # sums=sp.signal.fftconvolve(X,np.ones(Y.shape),'valid')
    sums=unif_filt2_valid(X,*Y.shape)
    
    # for each valid placement rgn, get sum x_i**2
    # sums2=sp.signal.fftconvolve(X**2,np.ones(Y.shape),'valid')
    sums2=unif_filt2_valid(X**2,*Y.shape)
    
    # for each valid placement rgn, get sum x_i*y_i
    dots=sp.signal.fftconvolve(X,Y[::-1,::-1],mode='valid')
    # dots=sp.signal.convolve(X,Y[::-1,::-1],mode='valid')
    
    # for each valid placement rgn, get vr_i
    '''
    Want: 
    
      sum (x_i - rgnmean)**2
    = sum x_i**2 + rgnmean**2 - 2*x_i*rgnmean
    = sums2 + NN*(rgnmean**2) - 2*sums*rgnmean
    = sums2 + (sums**2)/NN - 2*sums*sums/NN
    = sums2 - sums*sums/NN
    '''
    normsq = sums2 - (sums**2)/NN
    norms = np.sqrt(normsq)
    
    # for each valid placement rgn, get |X-Y|
    '''
    Want
      sum y_i * (x_i - rgnmean)/norms
    = (sum y_i * x_i)/norms - sumsy*rgnmean/norms
    = (sum y_i * x_i)/norms - sumsy*rgnmean/norms
    '''
    cos = dots /norms
    
    return cos

def localprops(A,s1):
    '''
    Let p denote a position in image A.  We compute

    E[A_[p+Z]],std(A_[p+Z])

    where Z is a gaussian.

    We compute this for every position in the image.
    '''
    means=sp.ndimage.gaussian_filter(A,s1)
    meansq=sp.ndimage.gaussian_filter(A**2,s1)
    return means,np.sqrt(meansq-means**2)

def rectangle_intersection(st1,en1,st2,en2):
    st=np.array([st1,st2]).max(axis=0)
    en=np.array([en1,en2]).min(axis=0)

    assert en.shape==st.shape

    if (en<=st).any():
        return False,None,None
    else:
        return True,st,en


def rect2slice(st,en,integerify=True):
    if integerify:
        return tuple([slice(int(np.ceil(a)),int(np.floor(b))) for (a,b) in zip(st,en)])
    else:
        return tuple([slice(a,b) for (a,b) in zip(st,en)])

def slice2rect(*slices):
    st=[]
    en=[]

    for x in slices:
        st.append(x.start)
        en.append(x.stop)

    return st,en

def sliceit(F,st1,st2,en2,fill_value=0.0):
    '''
    Given a tensor F indicating the value of a function from st1 to st1+F.shape[0]

    find it's slice from st2 to en2, filling with "fill_value" if necessary.
    '''

    st1=np.require(st1)
    st2=np.require(st2)
    en1=st1+np.r_[F.shape]
    en2=np.require(en2)

    gsize=en2-st2
    assert gsize.dtype==int
    assert F.shape==tuple(en1-st1)

    qualia,st,en = rectangle_intersection(st1,en1,st2,en2)

    if qualia:
        if fill_value is None:
            if (en-st!=st2-en2).any():
                raise Exception("Need a fill value for %s %s %s %s"%(st1,en1,st2,en2))
            rez = np.empty(gsize)
            rez[rect2slice(st-st2,en-st2)] = F[rect2slice(st-st1,en-st1)]
            return rez
        else:
            rez = np.ones(gsize)*fill_value
            rez[rect2slice(st-st2,en-st2)] = F[rect2slice(st-st1,en-st1)]
            return rez

    else:
        if fill_value is None:
            raise Exception("Need a fill value for %s %s %s %s"%(st1,en1,st2,en2))
        else:
            return np.ones(gsize)*fill_value
   