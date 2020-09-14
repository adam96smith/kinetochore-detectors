import numpy as np
import random as rd
import scipy.stats as stats
from skimage.transform import resize
from scipy.ndimage import convolve
from scipy.signal import fftconvolve
import time
import cv2


def normalize(arr):
    """ Normalize the image to be between 0 and 1 """ 
    arr = (arr - np.min(arr))/(np.max(arr) - np.min(arr)+1e-10) 
    return arr


def random_kernel_3d(n):

    x = np.linspace(-5,5,n)

    x1 = np.exp(-(x)**2/8)
    x2 = np.exp(-(x)**2/8)
    x3 = np.exp(-(x)**2/8)
    x4 = np.exp(-(x+1.5*np.random.rand()+1.5)**2/3)
    x5 = np.exp(-(x+1.5*np.random.rand()+1.5)**2/8)

    X1,Y1,Z1 = np.meshgrid(x1,x2,x3)
    V1,V2,V3 = np.meshgrid(x4,x4,x5)

    center = (n//2,n//2)

    angles = 360*np.random.rand(3)
    scale = 1.0

    # Perform rotation
    M = cv2.getRotationMatrix2D(center, angles[0], scale) #Transformation matrix
    V1 = cv2.warpAffine(V1, M, (n,n))

    M = cv2.getRotationMatrix2D(center, angles[1], scale)
    V2 = cv2.warpAffine(V2, M, (n,n))

    M = cv2.getRotationMatrix2D(center, angles[2], scale)
    V3 = cv2.warpAffine(V3, M, (n,n))

    kern = (X1*Y1*Z1)-(V1*V2*V3)
    for i in range(0,n):
        for j in range(0,n):
            for l in range(0,n):
                if kern[i,j,l] < 0:
                    kern[i,j,l] = 0
    kern = normalize(kern)
        
    # Shift kernel so max is in the middle
    max_ind = np.unravel_index(np.argmax(kern, axis=None), kern.shape)
    shift = tuple(np.subtract((n//2,n//2,n//2),max_ind))
    
    kern = np.roll(kern,shift[0],axis=0)
    kern = np.roll(kern,shift[1],axis=1)
    out = np.roll(kern,shift[2],axis=2)
    
    intensity = max(0.4,min(0.15+np.random.gamma(8, scale = .05),1.0))
    
    out = intensity*out

    return out, intensity


def noise(n):
    
    x = np.linspace(-5,5,n)
    x1 = 5*stats.norm.pdf(x,0,2)
    x2 = 5*stats.norm.pdf(x,0,2)
    x3 = 5*stats.norm.pdf(x,0,1.2)
    
    n1 = np.random.rand(n)
    n2 = np.random.rand(n)
    n3 = np.random.rand(n)
    
    
    X1,X2,X3 = np.meshgrid(x1,x2,x3)
    N1,N2,N3 = np.meshgrid(n1,n2,n3)
    
    out = normalize((X1*X2*X3)*(N1*N2*N3))
    
    return out


def distance_check(arr):

    # Array is (n,3)

    n = len(arr[:,0])
    
    X = np.zeros([n,n])
    Y = np.zeros([n,n])
    Z = np.zeros([n,n])
    
    for i in range(0,n):
        for j in range(0,n):
            X[i,j] = (arr[i,0] - arr[j,0])**2
            Y[i,j] = (arr[i,1] - arr[j,1])**2
            Z[i,j] = (arr[i,2] - arr[j,2])**2
            
    M = np.sqrt(X+Y+Z)
    for i in range(0,n):
        M[i,i] = 10000
    
    return np.min(M)


def partner_pos(arr,minDistance,xyz_res,fail_count,fail_threshold):

    # arr (n,3)
    
    lim = minDistance
    
    n = len(arr[:,0])
    out = np.ones([n,3])
    
    positions = np.ones([2*n,3])
    positions[0:n,:] = arr[:,:]
    
    counter = 0
    
    while counter < n:
        
        (x1,y1,z1) = arr[counter,:]
        x = [x1,y1,z1]
        r = np.random.rand(3)
        partner_position = np.zeros(3)
        for i in range(0,3):
            if r[i] > 0.5:
                partner_position[i] = max(min(x[i]+int(lim+0.5*np.random.randn()),xyz_res),0)
            else:
                partner_position[i] = max(min(x[i]-int(lim+0.5*np.random.randn()),xyz_res),0)
                
        distances = np.sqrt((positions[:,0]-partner_position[0])**2 + 
                            (positions[:,1]-partner_position[1])**2 +
                            (positions[:,2]-partner_position[2])**2)
        
        if np.min(distances) >= minDistance:
            out[counter,:] = partner_position
            positions[n+counter,:] = partner_position
            counter += 1
        else:
            fail_count += 1
            if fail_count > fail_threshold:
                return out , fail_count
    
    return out, fail_count


class Distribution(object):
    """
    draws samples from a one dimensional probability distribution,
    by means of inversion of a discrete inverstion of a cumulative density function

    the pdf can be sorted first to prevent numerical error in the cumulative sum
    this is set as default; for big density functions with high contrast,
    it is absolutely necessary, and for small density functions,
    the overhead is minimal

    a call to this distibution object returns indices into density array
    """
    def __init__(self, pdf, sort = True, interpolation = True, transform = lambda x: x):
        self.shape          = pdf.shape
        self.pdf            = pdf.ravel()
        self.sort           = sort
        self.interpolation  = interpolation
        self.transform      = transform

        #a pdf can not be negative
        assert(np.all(pdf>=0))

        #sort the pdf by magnitude
        if self.sort:
            self.sortindex = np.argsort(self.pdf, axis=None)
            self.pdf = self.pdf[self.sortindex]
        #construct the cumulative distribution function
        self.cdf = np.cumsum(self.pdf)
    @property
    def ndim(self):
        return len(self.shape)
    @property
    def sum(self):
        """cached sum of all pdf values; the pdf need not sum to one, and is imlpicitly normalized"""
        return self.cdf[-1]
    def __call__(self, N):
        """draw """
        #pick numbers which are uniformly random over the cumulative distribution function
        choice = np.random.uniform(high = self.sum, size = N)
        #find the indices corresponding to this point on the CDF
        index = np.searchsorted(self.cdf, choice)
        #if necessary, map the indices back to their original ordering
        if self.sort:
            index = self.sortindex[index]
        #map back to multi-dimensional indexing
        index = np.unravel_index(index, self.shape)
        index = np.vstack(index)
        #is this a discrete or piecewise continuous distribution?
        if self.interpolation:
            index = index + np.random.uniform(size=index.shape)
        return self.transform(index)


def pinky_plus(t,T,T_change,xy_res):
    h = 20
    k = 2
    out = []
    if t < T_change:
        x = np.linspace(-10,10,(xy_res+1)//8)
        p = np.exp(-(x**2)/30)
        tmp = p[:,None]*p[None,:]
        pdf = normalize(tmp)
        dist = Distribution(pdf)
        out.append(dist(k).mean(axis=1))

    else:
        t_max = T-T_change
        tt = t-T_change
        x = np.linspace(-10-h*(tt/t_max),10,(xy_res+1)//8)
        y = np.linspace(-10,10+h*(tt/t_max),(xy_res+1)//8)
        tmp = np.zeros([len(x),len(x),2])
        p1 = np.exp(-(x**2)/30)
        p2 = np.exp(-(y**2)/30)
        tmp[:,:,0] = p1[:,None]*p1[None,:]
        tmp[:,:,1] = p2[:,None]*p2[None,:]
        pdf = normalize(np.max(tmp,axis=2))
        dist1 = Distribution(tmp[:,:,0])
        dist2 = Distribution(tmp[:,:,1])
        out.append(dist1(k).mean(axis=1))
        out.append(dist2(k).mean(axis=1))
        
    return out, pdf

def stepper(arr,t,T,T_change,state,xyz_res):
    
    if state == 0: #pre anaphase
        
        pos, _ = pinky_plus(t,T,T_change,xyz_res)
        a = 8*pos[0][0]
        b = 8*pos[0][1]
        c = xyz_res/2+(xyz_res/5)*np.random.randn()
        
    if state == 1: #anaphase, spot 1
        
        pos, _ = pinky_plus(t,T,T_change,xyz_res)
        a = 8*pos[0][0]
        b = 8*pos[0][1]
        c = xyz_res/2+(xyz_res/5)*np.random.randn()
        
    if state == 2: #anaphase, spot 2
        
        pos, _ = pinky_plus(t,T,T_change,xyz_res)
        a = 8*pos[1][0]
        b = 8*pos[1][1]
        c = xyz_res/2+(xyz_res/5)*np.random.randn()

        
    del_x = a-arr[0]
    del_y = b-arr[1]
    del_z = c-arr[2]
        
    x_pos = max(min(abs(arr[0]+int(.1*np.random.rand()*del_x)),xyz_res),1)
    y_pos = max(min(abs(arr[1]+int(.1*np.random.rand()*del_y)),xyz_res),1)
    z_pos = max(min(abs(arr[2]+int(.1*np.random.rand()*del_z)),xyz_res),1)

    # returns new positions of spot based on a step by pinky_plus
    
    return x_pos, y_pos, z_pos


def time_step_1(arr,minDistance,t,T,T_change,xyz_res,fail_count,fail_threshold):
    
    n = len(arr[:,0])
    
    out = np.ones([n,3])
    
    counter = 0
    while counter < n:
        x_new, y_new, z_new = stepper(arr[counter,:],t,T,T_change,0,xyz_res)
        
        distances = np.sqrt((out[:,0]-x_new)**2 +
                           (out[:,1]-y_new)**2 +
                           (out[:,2]-z_new)**2)
        
        min_distance = np.min(distances)
        if min_distance >= minDistance:
            out[counter,:] = [x_new,y_new,z_new]
            counter += 1
        else:
            fail_count += 1
            if fail_count > fail_threshold:
                return out , fail_count
    
    return out, fail_count


def time_step_2(arr,minDistance,t,T,T_change,xyz_res,fail_count,fail_threshold):
    
    pairs = len(arr[:,0])
    
    out = np.ones([pairs,2,3])
    
    out_reshaped = np.zeros([2*pairs, 3])
    
    counter = 0
    while counter < pairs:
        x_new, y_new, z_new = stepper(arr[counter,0,:],t,T,T_change,1,xyz_res)
        
        distances = np.sqrt((out_reshaped[:,0]-x_new)**2 +
                            (out_reshaped[:,1]-y_new)**2 +
                            (out_reshaped[:,2]-z_new)**2)
        min_distance = np.min(distances)
        if min_distance >= minDistance:
            out_reshaped[counter,:] = [x_new, y_new, z_new]
            counter += 1
        else:
            fail_count += 1
            if fail_count > fail_threshold:
                return out , fail_count
            
    counter = 0
    while counter < pairs:
        x_new, y_new, z_new = stepper(arr[counter,1,:],t,T,T_change,2,xyz_res)
        
        distances = np.sqrt((out_reshaped[:,0]-x_new)**2 +
                            (out_reshaped[:,1]-y_new)**2 +
                            (out_reshaped[:,2]-z_new)**2)
        min_distance = np.min(distances)
        if min_distance >= minDistance:
            out_reshaped[pairs+counter,:] = [x_new, y_new, z_new]
            counter += 1
        else:
            fail_count += 1
            if fail_count > fail_threshold:
                return out , fail_count
            
    out[:,0,:]  = out_reshaped[0:pairs,:]
    out[:,1,:]  = out_reshaped[pairs::,:]
            
    return out, fail_count


def cell_body(t,T,T_change,xy_res,z_res):
    
    _, dist = pinky_plus(t,T,T_change,xy_res)
    
    tmp = resize(dist,(xy_res+1,xy_res+1))
    
    out = np.zeros([xy_res+1,xy_res+1,z_res+1])
    x = np.linspace(-5,5,z_res+1)
    f = 5*stats.norm.pdf(x,0,2)
    
    for i in range(0,z_res+1):
        out[:,:,i] = f[i]*tmp
        
    return out


def determine_points(stack,minDistance,T,T_change,xyz_res):
    
    fail_threshold = 1*(10**5)
    n = stack.shape[0]
    
    # Check the points have sufficient spacing for optimisation
    min_distance = 0
    # min_distance = distance_check(stack[:,1,:,1])
    while min_distance < 15:
        stack[:,0,:,0] = np.round(xyz_res*np.random.rand(n,3))
        min_distance = distance_check(stack[:,0,:,0])
    fail_count = 0
    stack[:,1,:,0], fail_count = partner_pos(stack[:,0,:,0],minDistance//2,xyz_res,fail_count,fail_threshold)
    
    
    # Determine all the positions over all time
    for t in range(1,T_change):
        stack[:,0,:,t], fail_count = time_step_1(stack[:,0,:,t-1],minDistance,t,T,T_change,xyz_res,fail_count,fail_threshold)
        if fail_count > fail_threshold:
            return stack , 'failed'
        stack[:,1,:,t], fail_count = partner_pos(stack[:,0,:,t],minDistance//2,xyz_res,fail_count,fail_threshold)
        if fail_count > fail_threshold:
            return stack , 'failed'

    for t in range(T_change,T):
        stack[:,:,:,t], fail_count = time_step_2(stack[:,:,:,t-1],minDistance,t,T,T_change,xyz_res,fail_count,fail_threshold)
        if fail_count > fail_threshold:
            return stack , 'failed'

    return stack, 'success'


def movie_maker(SNR,movie_length):
    
    ttt = time.time()
    
    # Parameters
    N = 46 #Number of pairs
    minDistance = 8

    xyz_res = 255
    
    T = 50
    T_change = min(max(round(T/2+3*np.random.randn()),2),T) # split at t= ~50-60
    
    # Initialise particle positions
    stack = np.zeros([N,2,3,T])
    
    status = 'failed'
    while status == 'failed':
        # Set positions for all points every time step
        tmp, status = determine_points(stack,minDistance,T,T_change,xyz_res)
        
    print('Positions complete...')
    stack = tmp        
    
    # Constuct the Movie
    out = np.zeros([xyz_res+1,2*(xyz_res+1),80,movie_length]).astype(np.uint8)

    # Determine the PSFs
    
    n = 2*(4 + round(3*np.random.rand()))+1 # radius (odd 9-15) of psf filter
    tn = n-2 # size of targets
    
    psf = np.zeros([46,2,n,n,n])
    ind1 = np.linspace(0,45,46).astype(np.uint8)
    rd.shuffle(ind1)
    target_psf = np.zeros([46,2,tn,tn,tn])
    
########################################################################################################################################  
    intensities = []
    for j in range(0,46):
        k, intensity = random_kernel_3d(n)
        intensities.append(intensity)
        y = np.zeros([tn,tn,tn])
        tmp = np.zeros([tn,tn,n])
        for i in range(0,n):
            tmp[:,:,i] = cv2.resize(k[:,:,i],(tn,tn), cv2.INTER_NEAREST)
        for i in range(0,tn):
            y[i,:,:] = cv2.resize(tmp[i,:,:],(tn,tn), cv2.INTER_NEAREST)

        psf[j,0,:,:,:] = k
        psf[ind1[j],1,:,:,:] = k
        target_psf[j,0,:,:,:] = normalize(y)
        target_psf[ind1[j],1,:,:,:] = normalize(y)
    
    #Average Signal Intensity
    I = np.mean(intensities)
#     print('Mean Signal: ', I)
    
########################################################################################################################################    
    
    start_time = 15 + int((T-movie_length-15)*np.random.rand())
    
    b = n//2     # half the box width
    bt = tn//2   # half the box width for the target spots

    #Add the spots in manually rather than through convolution for efficiency
    for t in range(0,movie_length):
        hyperstack = np.zeros([xyz_res+1,xyz_res+1,xyz_res+1]).astype(np.float32)
        GT = np.zeros([xyz_res+1,xyz_res+1,xyz_res+1]).astype(np.float32)
        
########################################################################################################################################

        
        for i in range(0,N):
            x2,x1,x3 = int(stack[i,0,0,start_time+t]),int(stack[i,0,1,start_time+t]),int(stack[i,0,2,start_time+t])
            y2,y1,y3 = int(stack[i,1,0,start_time+t]),int(stack[i,1,1,start_time+t]),int(stack[i,1,2,start_time+t])
            
            hyperstack[x1-b:x1+b+1,x2-b:x2+b+1,x3-b:x3+b+1] = np.maximum(psf[i,0,:,:,:],hyperstack[x1-b:x1+b+1,x2-b:x2+b+1,x3-b:x3+b+1])
            hyperstack[y1-b:y1+b+1,y2-b:y2+b+1,y3-b:y3+b+1] = np.maximum(psf[i,1,:,:,:],hyperstack[y1-b:y1+b+1,y2-b:y2+b+1,y3-b:y3+b+1])

            GT[x1-bt:x1+bt+1,x2-bt:x2+bt+1,x3-bt:x3+bt+1] = np.maximum(target_psf[i,0,:,:,:],
                                                                       GT[x1-bt:x1+bt+1,x2-bt:x2+bt+1,x3-bt:x3+bt+1])
            GT[y1-bt:y1+bt+1,y2-bt:y2+bt+1,y3-bt:y3+bt+1] = np.maximum(target_psf[i,1,:,:,:],
                                                                       GT[y1-bt:y1+bt+1,y2-bt:y2+bt+1,y3-bt:y3+bt+1])


#         cell = (cell_body(t,T,T_change,xyz_res,79) > 0.2) # The minimum spot intensity
        cell = normalize(np.sqrt(cell_body(t,T,T_change,xyz_res,79)))
##########################################################################################################################################        
        #Box for the noise
        nd = 15
#         d = nd // 2 

        # Add Background noise spots
        background = np.zeros([xyz_res+1,xyz_res+1,80])
        for j in range(0,200000):
            rx = int(xyz_res*np.random.rand())
            ry = int(xyz_res*np.random.rand())
            rz = int(79*np.random.rand())
            
            background[rx,ry,rz] = max(cell[rx,ry,rz],0.3+0.5*np.random.rand())

        noise_spot = noise(nd)

        background = fftconvolve(background,noise_spot,mode='same')
##########################################################################################################################################
        
        output = (255*np.concatenate((normalize(hyperstack),GT),axis=1)).astype(np.uint8)

        for i in range(0,xyz_res):
            out[i,:,:,t] = cv2.resize(output[i,:,:], (80,2*(xyz_res+1)), cv2.INTER_LINEAR)
          
        tmp = (255*I*normalize(background)/SNR).astype(np.uint8)
            
        out[:,0:xyz_res+1,:,t] = np.maximum(tmp,out[:,0:xyz_res+1,:,t])

    loc = np.zeros([92,3,movie_length])
    
    loc[0:N,:,:] = stack[:,0,:,start_time:start_time+movie_length]
    loc[N::,:,:] = stack[:,1,:,start_time:start_time+movie_length]
    loc[:,2,:] = 80/256*loc[:,2,:]
    
    print('Movie Generated at: ', 1/round((time.time()-ttt)/movie_length,3), ' fps')
    
    return out, loc