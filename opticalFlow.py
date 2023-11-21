from __future__ import division
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.filters import convolve as filter2
#from scipy.ndimage import imread
#
#from pyOpticalFlow import getimgfiles

FILTER = 7
QUIVER = 5

def compareGraphs(u,v,Inew,scale=3):
    """
    makes quiver
    """
    ax = plt.figure().gca()
    ax.imshow(Inew,cmap = 'gray')
    # plt.scatter(POI[:,0,1],POI[:,0,0])
    for i in range(0,len(u),QUIVER):
        for j in range(0,len(v),QUIVER):
            ax.arrow(j,i, v[i,j]*scale, u[i,j]*scale, color='red')

	# plt.arrow(POI[:,0,0],POI[:,0,1],0,-5)

    #plt.draw(); plt.pause(0.01)

def HS(im1, im2, alpha, Niter):
    """
    im1: image at t=0
    im2: image at t=1
    alpha: regularization constant
    Niter: number of iteration
    """

	#set up initial velocities
    uInitial = np.zeros([im1.shape[0],im1.shape[1]])
    vInitial = np.zeros([im1.shape[0],im1.shape[1]])

	# Set initial value for the flow vectors
    U = uInitial
    V = vInitial

	# Estimate derivatives
    [fx, fy, ft] = computeDerivatives(im1, im2)

	# Averaging kernel
    kernel=np.array([[1/12, 1/6, 1/12],
                      [1/6,    0, 1/6],
                      [1/12, 1/6, 1/12]],float)

	# Iteration to reduce error
    for _ in range(Niter):
        #%% Compute local averages of the flow vectors
        uAvg = filter2(U,kernel)
        vAvg = filter2(V,kernel)
        #%% common part of update step
        der = (fx*uAvg + fy*vAvg + ft) / (alpha**2 + fx**2 + fy**2)
        #%% iterative step
        U = uAvg - fx * der
        V = vAvg - fy * der

    return U,V



def computeDerivatives(im1, im2):
#%% build kernels for calculating derivatives
    kernelX = np.array([[-1, 1],
                         [-1, 1]]) * .25 #kernel for computing d/dx
    kernelY = np.array([[-1,-1],
                         [ 1, 1]]) * .25 #kernel for computing d/dy
    kernelT = np.ones((2,2))*.25

    fx = filter2(im1,kernelX) + filter2(im2,kernelX)
    fy = filter2(im1,kernelY) + filter2(im2,kernelY)

    #ft = im2 - im1
    ft = filter2(im1,kernelT) + filter2(im2,-kernelT)

    return fx,fy,ft

def rlof(im0, im1):
    im0 = (im0)
    im1 = (im1)

    param = cv.optflow.RLOFOpticalFlowParameter_create()
    param.setSupportRegionType(cv.optflow.SR_FIXED)  # SR_CROSS works for RGB only
    param.setSolverType(cv.optflow.ST_BILINEAR)  # ST_STANDART does not work
    param.setUseInitialFlow(True)
    param.setUseGlobalMotionPrior(True)
    param.setUseMEstimator(False)  # if set true, it does not work properly
    param.setMaxIteration(30)
    param.setNormSigma0(3.2)
    param.setNormSigma1(7.0)
    param.setLargeWinSize(21)
    param.setSmallWinSize(9)
    param.setMaxLevel(9)
    param.setMinEigenValue(0.0001)
    param.setCrossSegmentationThreshold(25)
    param.setGlobalMotionRansacThreshold(10.0)
    param.setUseIlluminationModel(True)

    flow = cv.optflow.calcOpticalFlowDenseRLOF(im0, im1, None, rlofParam=param, interp_type=cv.optflow.INTERP_GEO)

    return flow