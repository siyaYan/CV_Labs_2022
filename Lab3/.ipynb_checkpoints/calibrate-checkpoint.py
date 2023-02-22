# -*- coding: utf-8 -*-
# CLAB3 
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from vgg_KR_from_P import vgg_KR_from_P as vgg

###################################################################
# read images and set points
I = Image.open('stereo2012a.jpg');
im = np.asarray(I)
    
I2a = Image.open('Left.jpg');
I2b = Image.open('Right.jpg');
i2aw,i2ah=I2a.size
i2bw,i2bh=I2b.size

# change to float32
# print(X[0][0].astype(np.float32))

# after getting 6 points in 2d coordinates where are at[(7,7,0),(14,7,0),(0,7,7),(0,14,7),(7,0,7),(7,0,14)] in 3d coordinates
XYZ=np.asarray([[7,7,0],[14,7,0],[0,7,7],[0,14,7],[7,0,7],[7,0,14]])
# 2*6
uv=np.asarray([[357.78571428571433, 300.12337662337654], 
               [400.6428571428572, 308.6948051948051], 
               [281.42207792207796, 311.0324675324675], 
               [273.62987012987014, 262.72077922077915], 
               [331.2922077922078, 367.91558441558436], 
               [297.00649350649354, 391.2922077922077]])

newUV=np.ones((uv.shape[1]+1,uv.shape[0]))    #3*6
newXYZ=np.ones((XYZ.shape[1]+1,XYZ.shape[0])) #4*6

#####################################################################
# get points function: 
# change (x,y) list objects to array[[]] objects
def getpoint(i):
    plt.imshow(i)
    newuv = plt.ginput(6) # Graphical user interface to get 6 points    
    plt.close()
    UV=np.zeros((uv.shape[0],uv.shape[1]))
    for i in range(UV.shape[0]):
        UV[i]=[newuv[i][0],newuv[i][1]]   
        
    UV=np.float32(UV)
    return UV


#####################################################################

def calibrate(im, XYZ, uv):    
    # step1:change to homogeneous coordinates
    newUV[0:2,:]=[uv[:,0],uv[:,1]]
    newXYZ[0:3,:]=[XYZ[:,0],XYZ[:,1],XYZ[:,2]]
    
    # step2: normalise two pointsets
    [x,T1]=normalise2d(newUV)
    [X,T2]=normalise3d(newXYZ)  
    # test normalisation func
    # print(np.linalg.inv(T2)@X)

    # step3: calculate A metric 
    # Ai=((-X,-Y,-Z,-1,0,0,0,0,xX,xY,xZ,x),(0,0,0,0,-X,-Y,-Z,-1,yX,yY,yZ,y))  
    # 6 points is 6*(2*12) matrix
    A=[]
    for i in range(x.shape[1]):
        A.append([-X[0][i],-X[1][i],-X[2][i],-1,0,0,0,0,x[0][i]*X[0][i],x[0][i]*X[1][i],x[0][i]*X[2][i],x[0][i]])
        A.append([0,0,0,0,-X[0][i],-X[1][i],-X[2][i],-1,x[1][i]*X[0][i],x[1][i]*X[1][i],x[1][i]*X[2][i],x[1][i]])
        
    # step4: calculate V   
    [U,D,V]=np.linalg.svd(A)
    
    # step5: get last colum of V   
    p=np.reshape(V[-1,:],(3,4))

    # step6: denormalise P    
    P=np.linalg.inv(T1)@p@T2
    C = P
    # temp=C@newXYZ
    # temp[:]/=temp[2]
    # print('after',newUV,temp)
    
    return C


# normalise 2d points,input points array(homogeneous)
# return:normalised points, normalisation matrix
def normalise2d(points):
    #calculate the mean point
    c=points.mean(1)
    newP=np.ones(points.shape)
    # calcualte the distance bewteen points to mean point
    newP[0,:]=points[0,:]-c[0]
    newP[1,:]=points[1,:]-c[1]
    # calculate the mean sqrt error of distance
    meandist=np.mean(np.sqrt(newP[0,:]**2+newP[1,:]**2))
    scale=np.sqrt(2)/meandist
    
    # 3*3 normalise metrix
    T=np.asarray([[scale,0,-scale*c[0]],[0,scale,-scale*c[1]],[0, 0, 1]])
    
    # normalization:3*6 metrix
    newpts=np.dot(T,points)
    return newpts,T


# normalise 3d points,input points array(homogeneous)
# return:normalised points, normalisation matrix
def normalise3d(points):
    #calculate the mean point
    c=points.mean(1)
    newP=np.ones(points.shape)
    # calcualte the distance bewteen points to mean point
    newP[0,:]=points[0,:]-c[0]
    newP[1,:]=points[1,:]-c[1]
    newP[2,:]=points[2,:]-c[2]
    # calculate the mean sqrt error of distance
    meandist=np.mean(np.sqrt(newP[0,:]**2+newP[1,:]**2+newP[2,:]**2))
    scale=np.sqrt(3)/meandist
    
    # 4*4 normalise metrix
    T=np.asarray([[scale,0,0,-scale*c[0]],[0,scale,0,-scale*c[1]],[0,0,scale,-scale*c[2]],[0, 0,0, 1]])
    
    # normalization:4*6 metrix
    newpts=np.dot(T,points)    
    return newpts,T


# display points and lines in image for task1
# calculate the mean sqrt error
def DisplayAndMse(C):
    #step1: plot the image and the original picked points
    plt.imshow(I)
    plt.scatter(newUV[0], newUV[1], facecolors='none', edgecolors='b',s=28)  
    #step2: use calibrate matrix to calulate the 2d position from 3d coordinates
    newXYZ2d=C@newXYZ
    # change homogeneous coordinates back to Cartesian coordinates
    XYZ2d=newXYZ2d[:,:]/newXYZ2d[-1,:]
    #plot the correspond points
    plt.scatter(XYZ2d[0], XYZ2d[1], facecolors='none', edgecolors='r',s=20)  
    #step3: calculate the mean sqrt error
    mse = np.mean((newUV[:2,:] - XYZ2d[:2,:])**2);
    print('the mse between UV and projected XYZ is: ',mse)
    #step4: calculate the vanishing points
    # calculate vanish point1   
    temp1=uv[0]-uv[1]
    m1=temp1[1]/temp1[0]
    c1=uv[1,1]-m1*uv[1,0]
    temp2=uv[2]-uv[3]
    m2=temp2[1]/temp2[0]
    c2=uv[3,1]-m2*uv[3,0]
    x0=(c1-c2)/(m2-m1)
    y0=m1*x0+c1   
    # calculate vanish point2   
    temp3=uv[4]-uv[5]
    m3=temp3[1]/temp3[0]
    c3=uv[5,1]-m3*uv[5,0]
    x1=(c3-c1)/(m1-m3)
    y1=m3*x1+c3  
    # calculate vanish point3   
    x2=(c2-c3)/(m3-m2)
    y2=m2*x2+c2
    #step5: connect the vanishing points and original points to draw lines    
    # obtain the 3d original point in 2d Cartesian coordinates     
    original2d=C@np.asarray([[0],[0],[0],[1]])
    original2d=original2d[:,:]/original2d[-1,:]    
    # plot the three lines from original point to vanishing points
    plt.plot(np.asarray([x0,original2d[0]]),np.asarray([y0,original2d[1]]), linestyle='-',color='g')
    plt.plot(np.asarray([x1,original2d[0]]),np.asarray([y1,original2d[1]]), linestyle='-',color='g')
    plt.plot(np.asarray([x2,original2d[0]]),np.asarray([y2,original2d[1]]), linestyle='-',color='g')   
    plt.show()
    return mse

def resizeImg(i):
    width, height = i.size
    # print(width,height)
    newsize = (int(width/2), int(height/2))
    i = i.resize(newsize)
    resizedUV=getpoint(i)
    print('After resized, get points: ',resizedUV) 
    
    return i,resizedUV

###########################################################################
'''
%% TASK 1: CALIBRATE
%
% Function to perform camera calibration
%
% Usage:   calibrate(image, XYZ, uv)
%          return C
%   Where:   image - is the image of the calibration target.
%            XYZ - is a N x 3 array of  XYZ coordinates
%                  of the calibration target points. 
%            uv  - is a N x 2 array of the image coordinates
%                  of the calibration target points.
%            K   - is the 3 x 4 camera calibration matrix.
%  The variable N should be an integer greater than or equal to 6.
%
%  This function plots the uv coordinates onto the image of the calibration
%  target. 
%
%  It also projects the XYZ coordinates back into image coordinates using
%  the calibration matrix and plots these points too as 
%  a visual check on the accuracy of the calibration process.
%
%  Lines from the origin to the vanishing points in the X, Y and Z
%  directions are overlaid on the image. 
%
%  The mean squared error between the positions of the uv coordinates 
%  and the projected XYZ coordinates is also reported.
%
%  The function should also report the error in satisfying the 
%  camera calibration matrix constraints.
% 
% Xiran Yan(u7167582), 3/5/22 
'''

############################################################################
#normalise function by width and height, input P and P_points(homogeneous)
#return:normalised points1,normalisation1,normalised points2,normalisation2
def normalise(uv1,uv2):
    
    T1_=np.asarray([
        [i2aw+i2ah,0,i2aw/2],
        [0, i2aw+i2ah,i2ah/2],
        [0, 0, 1]
    ])
    T2_=np.asarray([
        [i2bw+i2bh,0,i2bw/2],
        [0, i2bw+i2bh,i2bh/2],
        [0, 0, 1]
    ])
    T1=np.linalg.inv(T1_)
    T2=np.linalg.inv(T2_)
    
    return np.dot(T1,uv1),T1,np.dot(T2,uv2),T2


def homography(u2Trans, v2Trans, uBase, vBase):
    # step1:change 6*2 matrix to homogeneous coordinates 6*3 matrix
    uv1=np.ones((uBase.shape[0],3))
    uv2=np.ones((u2Trans.shape[0],3))
    for i in range(uBase.shape[0]):
        uv1[i]=[uBase[i], vBase[i], 1]
        uv2[i]=[u2Trans[i], v2Trans[i], 1]
    
    # step2 change to 3*6 metrix
    x1=uv1.T
    x2=uv2.T

    # step3 normalise by width and height    
    # [x1,T1,x2,T2]=normalise(uv1,uv2)
    # step4: calculate  A metric 
    # Ai=((-x1,-y1,-1,0,0,0,x2x1,x2y1,x2),(0,0,0,-x1,-y1,-1,y2x1,y2y1,y2))
    # 6*(2*9)
    A=[]
    for i in range(uv1.shape[0]):
        A.append([-x1[0][i],-x1[1][i],-1,0,0,0,x2[0][i]*x1[0][i],x2[0][i]*x1[1][i],x2[0][i]])
        A.append([0,0,0,-x1[0][i],-x1[1][i],-1,x2[1][i]*x1[0][i],x2[1][i]*x1[1][i],x2[1][i]])
                 
    # step5: calculate V   
    [U,D,V]=np.linalg.svd(A)
    # print(A)
    
    # step6: get last col of V, and change to 3*3  
    p=np.reshape(V[-1,:],(3,3))
    # step7: denormalise P    
    # P=np.matmul(np.linalg.inv(T2), np.matmul(p,T1)) 
    P=p
    H = P
    
    return H 
    # test function and diff
    # print('original: ',uv1,uv2)
    # temp=H@x1
    # temp[:]/=temp[2]
    # print('after',x2,temp)
############################################################################

'''
%% TASK 2: 
% Computes the homography H applying the Direct Linear Transformation 
% The transformation is such that 
% p = np.matmul(H, p.T), i.e.,
% (uBase, vBase, 1).T = np.matmul(H, (u2Trans , v2Trans, 1).T)
% Note: we assume (a, b, c) => np.concatenate((a, b, c), axis), be careful when 
% deal the value of axis 
%
% INPUTS: 
% u2Trans, v2Trans - vectors with coordinates u and v of the transformed image point (p') 
% uBase, vBase - vectors with coordinates u and v of the original base image point p  
% 
% OUTPUT 
% H - a 3x3 Homography matrix  
% 
% Xiran Yan(u7167582), 3/5/22 
'''


############################################################################
def rq(A):
    # RQ factorisation

    [q,r] = np.linalg.qr(A.T)   # numpy has QR decomposition, here we can do it 
                                # with Q: orthonormal and R: upper triangle. Apply QR
                                # for the A-transpose, then A = (qr).T = r.T@q.T = RQ
    R = r.T
    Q = q.T
    return R,Q



##########################################################################
# run the requirements for task1 
def task1():
    # step1 
    # resized images and get input point
    i,resizedUV=resizeImg(I)
    
    # step2    
    # calculate the original size C 
    C=calibrate(im,XYZ,uv)
    print('The calibrate matrix is:',C)

    # display points, lines, mse and KRt
    mse=DisplayAndMse(C)
    K,R,t=vgg(C)
    

    # K[R|−Rt]  P=K[R|trequirement]. 
    
    tr=-1*np.matmul(R,t)
    print('K matrix is:',K)
    print('R matrix is:',R,R.shape)
    print('t matrix is:',tr)


    plt.close()

    # step3   
    # calculate the resized image C and krt
    C_r=calibrate(im,XYZ,resizedUV)
    print('The resized calibrate matrix is:',C_r)
    
    K_r,R_r,t_r=vgg(C_r)
    tr_r=-1*np.matmul(R_r,t_r)
    
    print('K_resized matrix is:',K)
    print('R_resized matrix is:',R)
    print('t_resized matrix is:',tr_r)

# run the requirements for task2    
def task2():
    # step1 get points from 2 images     
    uv1=getpoint(I2a)
    uv2=getpoint(I2b)
    print(uv1,uv2)

    # step2 plot points into images    
    f, index =plt.subplots(1,2,figsize=(10,10))
    index[0].imshow(I2a)
    index[0].scatter(uv1[:,0], uv1[:,1], facecolors='none', edgecolors='b',s=20)
    index[0].set_title("original image1 with blue circle")

    index[1].imshow(I2b)
    index[1].scatter(uv2[:,0], uv2[:,1], facecolors='none', edgecolors='r',s=20)
    index[1].set_title("target image with red circle")
    
    plt.show()
    
#     # step3 calculate Homography
    H=homography(uv2[:,0], uv2[:,1],uv1[:,0],uv1[:,1])
    print('Homography matrix is: ',H)
    # h=cv2.getPerspectiveTransform(uv2,uv1)
    # print(h)
    img=cv2.imread('Left.jpg')
    out=cv2.warpPerspective(img,H,(i2bw,i2bh)) 
    # print(I2a,out)
    plt.imshow(out)
    plt.title('wrap image')
    plt.show()
    
    
    
###############################################################################

task1()
task2()