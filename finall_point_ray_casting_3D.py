
from math import sqrt, acos, atan,tan
import glm
import numpy as np
import pandas as pd
from scipy.ndimage.interpolation import shift
# from glm import Mat4x4,Vec4
# from PIL import Image
# from glm.detail.type_vec3 import Vec3

# import argparse
# We also represent coordinates in 3D space as Vectors.
class Vector2D(object):
    '''A vector in 3D space. Coordinate system is cartesian.'''

    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y
        #self.z = z

    def __str__(self):
        '''Return a printable representation of a vector showing its three coordinates.'''
        return 'Vector(' + ','.join([str(self.x), str(self.y)]) + ')'

    def magnitude(self):
        '''Return the magnitude (length) of this vector.'''
        return sqrt(self.x ** 2 + self.y ** 2 )

    def __mul__(self, scalar):
        '''Multiply this vector by a scalar, returning a new vector.'''
        return Vector2D(self.x * scalar, self.y * scalar)

    def __div__(self, scalar):
        '''Divide this vector by a scalar returning a new vector.'''
        return Vector2D(self.x / scalar, self.y / scalar)

    def normalise(self):
        '''Return a normalised version of this vector, such that its magnitude is 1.'''
        magnitude = self.magnitude()
        if magnitude == 0:
            # Somehow we have a degenerate vector.
            return self
        else:
            return self / self.magnitude()

    def angle(self, other):
        '''Return the angle in radians between this vector and another vector.'''
        dp = self.dot_product(other)
        return acos(dp / self.magnitude() * other.magnitude())

    def dot_product(self, other):
        '''Return the dot product of this vector and another vector.'''
        return self.x * other.x + self.y * other.y

    def add_vector(self, other):
        '''Add this vector to another vector, returning a new vector as the result.'''
        return Vector2D(self.x + other.x, self.y + other.y)

    def sub_vector(self, other):
        '''Subtract another vector from this vector, returning a new vector as the result.'''
        return Vector2D(self.x - other.x, self.y - other.y)

    def negate_vector(self):
        '''Negate all the components of this vector, returning a new vector as the result.'''
        return Vector2D(-self.x, -self.y)

    def reflect_about_vector(self, other):
        '''Reflect this vector about another vector, returning a new vector as the result.'''
        normal = other.normalise()
        dp = self.dot_product(other)
        return self.sub_vector(normal * (2 * dp))
    #//////////////////////defining sobel function to obtain normal of a pixel to use it in light function//////////////////////////////////////
    def sobel(self,width,height,depth):
      self.x=(self.x+(1./width))-(self.x-(1./width))
      self.y=(self.y+(1./height))-(self.y-(1./height))
      #self.z=(self.z+(1./depth))-(self.z-(1./depth))
      return self
    #//////////////////////////////////////implementation of cross section function//////////////////////////////////////

    # def crossProduct(self,a,b):
    #   self.x=a.y*b.z-a.z*b.y
    #   self.y=a.z*b.x-a.x*b.z
    #   self.z=a.x*b.y-a.y*b.x

      #//    return (a.y*b.z-a.z*b.y,a.z*b.x-a.x*b.z,a.x*b.y-a.y*b.x);
      # return self


#degree to gradian conversion
def deg2rad (degrees):
    return degrees * 4.0 * atan (1.0) / 180.0
# ///////////////////////
def max1 (a,b):
    if a>b:
        return a
    else:
        return b
#// or: return comp(a,b)?b:a; for version (2)


def Cross(a,b):
      x=a.y*b.z-a.z*b.y
      y=a.z*b.x-a.x*b.z
      z=a.x*b.y-a.y*b.x

      # //    return (a.y*b.z-a.z*b.y,a.z*b.x-a.x*b.z,a.x*b.y-a.y*b.x);
      return glm.vec3(x,y,z)

def magnitude(a):
        '''Return the magnitude (length) of this vector.'''
        return sqrt(a.x ** 2 + a.y ** 2+a.z**2 )
def magnitude2(a):
        '''Return the magnitude (length) of this vector.'''
        b=np.transpose(a)
        return np.sqrt(b[0] ** 2 + b[1] ** 2+b[2]**2 )
##################################################################################################################

def draw_grid2(width,height,depth,cameraEye,volume_centre,cameraUp):

    ###############Forward Axis(Z Axis)#################
    direction_vector=glm.normalize(glm.vec3((cameraEye)-(volume_centre)))
    # mag=magnitude(Vec3((cameraEye)-(volume_centre)))
    #alpha/2=15 degree which is equal to 0.261799 radian
    # length=mag*tan(0.261799)

    # try:
    ################Right Vector(X Axis)#####################
    orange_vector=glm.normalize(Cross((direction_vector),(cameraUp)))
    ###################Up Vector(Y Axis)#######################
    blue_vector=glm.normalize(Cross((orange_vector),(direction_vector)))
    # except:
    #     print(0)
    #     orange_vector=Vec3(0,0,1)
    #     blue_vector=Vec3(1,1,0)



    # ww1=width/blue_vector[0]
    # hh1=height/blue_vector[1]
    # dd1=depth/blue_vector[2]
    ww1=width
    hh1=height
    dd1=depth
    # if ww1>hh1 and ww1>dd1:
    #     length=ww1
    # if hh1>ww1 and hh1>dd1:
    #     length=hh1
    # if dd1>ww1 and dd1>hh1:
    #     length=dd1
    length=max(ww1,dd1,hh1)


    grid_points1=np.zeros((abs((int)(length)),abs((int)(length)),3))
    grid_points2=np.zeros((abs((int)(length)),abs((int)(length)),3))
    h=k=0
    for i in range(-abs((int)(length/2)),abs((int)(length/2))):

        for j in range(-abs((int)(length/2)),abs((int)(length/2))):
            a=volume_centre+i*orange_vector+j*blue_vector
            grid_points2[h,k]=[a.x,a.y,a.z]
            k=k+1
        k=0
        h=h+1
    k=h=0
    for i in range(-abs((int)(length/2)),abs((int)(length/2))):

        for j in range(-abs((int)(length/2)),abs((int)(length/2))):
            a=cameraEye+i*orange_vector+j*blue_vector
            grid_points1[h,k]=[a.x,a.y,a.z]
            k=k+1
        k=0
        h=h+1
    return grid_points1,grid_points2


def draw_grid3(width,height,depth,cameraEye,volume_centre,cameraUp):

    ###############Forward Axis(Z Axis)#################
    direction_vector=glm.normalize(glm.vec3((cameraEye)-(volume_centre)))
    # mag=magnitude(Vec3((cameraEye)-(volume_centre)))
    #alpha/2=15 degree which is equal to 0.261799 radian
    # length=mag*tan(0.261799)

    # try:
    ################Right Vector(X Axis)#####################
    orange_vector=glm.normalize(Cross((direction_vector),(cameraUp)))
    ###################Up Vector(Y Axis)#######################
    blue_vector=glm.normalize(Cross((orange_vector),(direction_vector)))
    # except:
    #     print(0)
    #     orange_vector=Vec3(0,0,1)
    #     blue_vector=Vec3(1,1,0)



    # ww1=width/blue_vector[0]
    # hh1=height/blue_vector[1]
    # dd1=depth/blue_vector[2]
    ww1=width
    hh1=height
    dd1=depth
    if ww1>hh1 and ww1>dd1:
        length=ww1
    if hh1>ww1 and hh1>dd1:
        length=hh1
    if dd1>ww1 and dd1>hh1:
        length=dd1


    grid_points1=np.zeros((abs((int)(length)),abs((int)(length)),3))
    grid_points2=np.zeros((abs((int)(length)),abs((int)(length)),3))
    h=k=0
    for i in range(-abs((int)(length/2)),abs((int)(length/2))):

        for j in range(-abs((int)(length/2)),abs((int)(length/2))):
            a=volume_centre+i*orange_vector+j*blue_vector
            grid_points2[h,k]=[a.x,a.y,a.z]
            k=k+1
        k=0
        h=h+1
    k=h=0
    for i in range(-abs((int)(length/2)),abs((int)(length/2))):

        for j in range(-abs((int)(length/2)),abs((int)(length/2))):
            a=cameraEye+i*orange_vector+j*blue_vector
            grid_points1[h,k]=[a.x,a.y,a.z]
            k=k+1
        k=0
        h=h+1

    cameraEye1=cameraEye+2*orange_vector+0*blue_vector
    cameraEye2=cameraEye+0*orange_vector+2*blue_vector
    cameraEye3=cameraEye+0*orange_vector-2*blue_vector
    cameraEye4=cameraEye-2*orange_vector+0*blue_vector
    return grid_points1,grid_points2,cameraEye1,cameraEye2,cameraEye3,cameraEye4







def main3_editted(f_data,width,height,depth,cameraEye,volume_centre):
    # from tempfile import TemporaryFile
    # outfile = TemporaryFile()
    cameraEye=glm.vec3(cameraEye)
    rayymax=-1000
    rayymaxy=-1000
    rayymaxz=-1000
    raymax=-1000
    raymaxy=-1000
    raymaxz=-1000

    rayyminx=1000
    rayyminy=1000
    rayyminz=1000
    rayminx=1000
    rayminy=1000
    rayminz=1000

    cameraUp=glm.vec3(0,1,0)
    # volume_centre=glm.vec3(int(width/2),int(height/2),int(depth/2))
    volume_centre=glm.vec3(int(volume_centre[0]),int(volume_centre[1]),int(volume_centre[2]))

    grid1,grid2=draw_grid2(width,height,depth,cameraEye,volume_centre,cameraUp)




    stepSize = 1#0.01

    # RAYY1=np.zeros(grid2.shape,500)
    # RAYY2=np.zeros(grid2.shape,500)
    # RAYY3=np.zeros(grid2.shape,500)

    RAY1=np.zeros(grid2.shape)
    RAY2=np.zeros(grid2.shape)
    RAY3=np.zeros(grid2.shape)
    pixel_values=np.zeros((grid2.shape[0],grid2.shape[1],1000))
    t_list=np.empty((grid2.shape[0],grid2.shape[1],1000),dtype=object)
    for i in range(0,grid2.shape[0],1):
      # print(i)

      for j in range(0,grid2.shape[1],1):
        # print(i,j)
        n2=0
        temp=0
        # pixelPosition1=Vec3(grid1[i,j,0],grid1[i,j,1],grid1[i,j,2])
        pixelPosition2=glm.vec3(grid2[i,j,0],grid2[i,j,1],grid2[i,j,2])
        ray = glm.normalize(pixelPosition2 - cameraEye)
        RAY1[i][j]=[ray.x]
        RAY2[i][j]=[ray.y]
        RAY3[i][j]=[ray.z]
        if ray.x>raymax:
                raymax=ray.x
        if ray.y>raymaxy:
                raymaxy=ray.y
        if ray.z>raymaxz:
                raymaxz=ray.z
        if ray.x<rayminx:
                rayminx=ray.x
        if ray.y<rayminy:
                rayminy=ray.y
        if ray.z<rayminz:
                rayminz=ray.z



        n=((magnitude((pixelPosition2 - cameraEye))-(max(width,height,depth)/2)))
        vv=n

        # n=0
        # nn=(n+max(width,height,depth))/stepSize
        nn=(n+max(width,height,depth))
        # nn=(1000+max(width,height,depth))
        while(n<nn):
            rayy= glm.vec3(cameraEye + ray * stepSize * (float)(n))
            if rayy.x>rayymax:
                rayymax=rayy.x
            if rayy.y>rayymaxy:
                rayymaxy=rayy.y
            if rayy.z>rayymaxz:
                rayymaxz=rayy.z

            if rayy.x<rayyminx:
                rayyminx=rayy.x
            if rayy.y<rayyminy:
                rayyminy=rayy.y
            if rayy.z<rayyminz:
                rayyminz=rayy.z
            rayy_n_1= glm.vec3(cameraEye + ray * stepSize * (float)(n-1))
            while (int)(rayy.x)==(int)(rayy_n_1.x) and (int)(rayy.y)==(int)(rayy_n_1.y) and (int)(rayy.z)==(int)(rayy_n_1.z):
                n=n+1
                rayy= glm.vec3(cameraEye + ray * stepSize * (float)(n))


            if(n<nn):
                # print(i,j)

                if(rayy.x<width and rayy.x>=0 and rayy.y<height and rayy.y>=0 and rayy.z<depth and rayy.z>=0 and rayy_n_1.x<width and rayy_n_1.x>=0 and rayy_n_1.y<height and rayy_n_1.y>=0 and rayy_n_1.z<depth and rayy_n_1.z>=0):
                    # print(i,j)
                    # RAYY1[i][j][vv-n]=[rayy.x]
                    # RAYY2[i][j][vv-n]=[rayy.y]
                    # RAYY3[i][j][vv-n]=[rayy.z]
                    temp=f_data[(int)((rayy_n_1.x))][(int)((rayy_n_1.y))][(int)((rayy_n_1.z))]
                    # pixel_values[(int)((rayy.x))][(int)((rayy.y))][(int)((rayy.z))]=f_data[(int)((rayy.x))][(int)((rayy.y))][(int)(rayy.z)]


                    if (f_data[(int)(rayy.x)][(int)(rayy.y)][(int)(rayy.z)]==255 and temp==0):
                                # print(i,j)
                                t_list[i][j][n2]=(["in",(int)((rayy.x)),(int)((rayy.y)),(int)((rayy.z))])
                                # print(t_list[i][j][n2])
                                n2=n2+1



                    elif(f_data[(int)(rayy.x)][(int)((rayy.y))][(int)((rayy.z))]==0 and temp==255):
                                # print(i,j)
                                t_list[i][j][n2]=(["out",(int)((rayy.x)),(int)((rayy.y)),(int)((rayy.z))])
                                # print(t_list[i][j][n2])
                                n2=n2+1

                n=n+1


    f_data=[]
    print("rayymax")
    print(rayymax)
    print(rayymaxy)
    print(rayymaxz)

    print(rayyminx)
    print(rayyminy)
    print(rayyminz)

    print(raymax)
    print(raymaxy)
    print(raymaxz)

    print(rayminx)
    print(rayminy)
    print(rayminz)

    # np.save("outfile1.npy",RAYY1)
    # np.save("outfile2.npy",RAYY2)
    # np.save("outfile3.npy",RAYY3)

    np.save("outray1.npy",RAY1)
    np.save("outray2.npy",RAY2)
    np.save("outray3.npy",RAY3)
    return(t_list),f_data,grid1,grid2




def main3(f_data,width,height,depth,cameraEye,volume_centre):

    cameraEye=glm.vec3(cameraEye)

    cameraUp=glm.vec3(0,1,0)
    # volume_centre=glm.vec3(int(width/2),int(height/2),int(depth/2))
    volume_centre=glm.vec3(int(volume_centre[0]),int(volume_centre[1]),int(volume_centre[2]))

    grid1,grid2=draw_grid2(width,height,depth,cameraEye,volume_centre,cameraUp)



    stepSize = 1#0.01

    rayy_n_1_L=[]
    # pixel_values=np.zeros((grid2.shape[0],grid2.shape[1],1000))
    t_list=np.empty((grid2.shape[0],grid2.shape[1],1000),dtype=object)
    # t_list=[]
    # f_data2=dict(np.ndenumerate(f_data))
    for ii in range(0,grid2.shape[0]-1,1):
      # print(ii)

      for j in range(0,grid2.shape[1]-1,1):

                # print(i,j)
        n2=0
        temp=0
        # pixelPosition1=Vec3(grid1[i,j,0],grid1[i,j,1],grid1[i,j,2])
        pixelPosition2=glm.vec3(grid2[ii,j,0],grid2[ii,j,1],grid2[ii,j,2])
        ray = glm.normalize(pixelPosition2 - cameraEye)

        n=((magnitude((pixelPosition2 - cameraEye))-(max(width,height,depth)/2)))

        # n=0
        # nn=(n+max(width,height,depth))/stepSize
        nn=(n+max(width,height,depth))
        # nn=(1000+max(width,height,depth))
        # rayy=np.zeros((1,int(nn-n+1)))
        rayy=np.zeros((1,int(nn)-int(n)),dtype=glm.vec3)
        rayy_n_1=np.zeros((1,int(nn)-int(n)+1),dtype=glm.vec3)
        n=int(n)

        nn=int(nn)
######################################################################################################
        # for i in range(int(n),int(nn)):
        #
        #     s=(glm.vec3((cameraEye + ray * stepSize * (float)(i))))
        #
        #     rayy[0,i-n]=[s.x,s.y,s.z]
#########################################################################################################
        # rayy=cameraEye+ray*stepSize*(range(int(n),int(nn)))
    ##########################################################################################
        a1=np.zeros((3,int(nn)-int(n)))
        b1=np.zeros((3,int(nn)-int(n)))
        c1=np.zeros((3,int(nn)-int(n)))
        cam=np.zeros((3,int(nn)-int(n)))

        a1[:][0]=((int(nn)-int(n)))*[ray.x]
        a1[:][1]=((int(nn)-int(n)))*[ray.y]
        a1[:][2]=((int(nn)-int(n)))*[ray.z]
        cam[:][0]=((int(nn)-int(n)))*[cameraEye.x]
        cam[:][1]=((int(nn)-int(n)))*[cameraEye.y]
        cam[:][2]=((int(nn)-int(n)))*[cameraEye.z]
        b1[:][0]=(((int(nn)-int(n)))*[stepSize])
        b1[:][1]=(((int(nn)-int(n)))*[stepSize])
        b1[:][2]=(((int(nn)-int(n)))*[stepSize])
        c1[:][0]=np.asarray(range(int(n),int(nn)))
        c1[:][1]=np.asarray(range(int(n),int(nn)))
        c1[:][2]=np.asarray(range(int(n),int(nn)))

        # rayy=np.transpose(cam)+np.dot(np.dot(np.transpose(np.asarray(a1)),(np.asarray(b1))),np.transpose(np.asarray(c1)))
        rayy=np.asarray(cam)+np.multiply(np.multiply((np.asarray(a1)),(np.asarray(b1))),np.asarray(c1))
                ##############################################################################
        rayy=[[np.where((X[0]<width and X[0]>=0 and X[1]<height and X[1]>=0 and X[2]<depth and X[2]>=0),X,None)] for X in rayy]
        z=[(np.where([(x[0]!=(None)) for x in rayy]))]
        z=np.unique(z[0][0])

        if z.__len__()!=0:
            rayy=[rayy[l][:] for l in z]
        else:
            break


        rayy_n_1[0,0]=0
        rayy_n_1[0,1:rayy.__len__()+1]=rayy[:]

        rayy_L=[f_data[int(k[0][0])][int(k[0][1])][int(k[0][2])] for k in rayy]
        rayy_n_1_L=[0]+rayy_L
        # print(rayy_n_1_L)

        # rayy_n_1_L.append(0)
        # rayy_n_1_L=rayy_n_1_L+rayy_L
        # rayy_n_1_L.pop()
        #
        rayy_test=[a_i - b_i for a_i, b_i in zip(rayy_L, rayy_n_1_L)]
        in_list=np.where(np.asarray(rayy_test)==255)
        out_list=np.where(np.asarray(rayy_test)==-255)
        if in_list[0].__len__()!=0:
            n=0
            for z in in_list[0]:
            # t_list.append=(["in",(int)((i)),(int)((j)),(int)(z)])
                t_list[ii][j][n2]=((["in",(int)((rayy[z][0][0])),(int)((rayy[z][0][1])),(int)((rayy[z][0][2]))]))
                n2=n2+1
        if out_list[0].__len__()!=0:
            for z in out_list[0]:
                t_list[ii][j][n2]=(["out",(int)((rayy[z][0][0])),(int)((rayy[z][0][1])),(int)((rayy[z][0][2]))])
                n2=n2+1
    f_data=[]
    return(t_list),f_data,grid1,grid2


def main3_extended(f_data,width,height,depth,cameraEye,volume_centre):

    cameraEye=glm.vec3(cameraEye)

    cameraUp=glm.vec3(0,1,0)
    # volume_centre=glm.vec3(int(width/2),int(height/2),int(depth/2))
    volume_centre=glm.vec3(int(volume_centre[0]),int(volume_centre[1]),int(volume_centre[2]))

    grid1,grid2=draw_grid2(width,height,depth,cameraEye,volume_centre,cameraUp)



    stepSize = 1#0.01


    pixelPositionss2=(grid2)

    # ray = glm.normalize(pixelPositionss2 - [cameraEye.x,cameraEye.y,cameraEye.z])
    a_ray = (pixelPositionss2 - [cameraEye[0],cameraEye[1],cameraEye[2]])
    a_ray=np.transpose(a_ray)
    ray=np.true_divide(a_ray,np.sqrt((np.power(a_ray[0],2)+np.power(a_ray[1],2)+np.power(a_ray[2],2))))
    # ray=np.transpose(ray)
    # print("ray maxs and min")
    # print(ray.max())
    # print(ray.min())
    # del(a_ray)

    n=(np.sqrt((np.power(a_ray[0],2)+np.power(a_ray[1],2)+np.power(a_ray[2],2))))-(max(width,height,depth)/2)
    # ray=ray.transpose()
    del(a_ray)
    del(pixelPositionss2)
    # n=0
    # nn=(n+max(width,height,depth))/stepSize
    nn=(int(np.round(n.max()))+max(width,height,depth))

    n=int(np.round(n.min()))
    # n=1
    # nn=1000
    nn=(nn)
    print("print n")
    print(n)
    print(nn)
    # ray[0]=np.transpose(np.load("outfile1.npy",allow_pickle=True))[0]
    # ray[1]=np.transpose(np.load("outfile2.npy",allow_pickle=True))[1]
    # ray[2]=np.transpose(np.load("outfile3.npy",allow_pickle=True))[2]
##########################################################################################


    a1=np.zeros((grid2.shape[0],grid2.shape[1],int(nn)-int(n),grid2.shape[2]))
    b1=np.ones((grid2.shape[0],grid2.shape[1],int(nn)-int(n),grid2.shape[2]))
    c1=np.zeros((grid2.shape[0],grid2.shape[1],int(nn)-int(n),grid2.shape[2]))
    cam=np.zeros((grid2.shape[0],grid2.shape[1],int(nn)-int(n),grid2.shape[2]))

    a1=np.transpose(a1)
    cam=np.transpose(cam)
    b1=np.transpose(b1)
    c1=np.transpose(c1)
    a1[0]=[(ray)[0]]
    a1[1]=[(ray)[1]]
    a1[2]=[(ray)[2]]
    cam[0]=[cameraEye.x]
    cam[1]=[cameraEye.y]
    cam[2]=[cameraEye.z]
    # reverse_count=np.asarray(range((n),(nn)))
    # reverse_count=reverse_count[nn:n:-1]
    c1[0]=np.transpose(grid2.shape[0]*[grid2.shape[1]*[np.asarray(range((n),(nn)))]])
    c1[1]=np.transpose(grid2.shape[0]*[grid2.shape[1]*[np.asarray(range((n),(nn)))]])
    c1[2]=np.transpose(grid2.shape[0]*[grid2.shape[1]*[np.asarray(range((n),(nn)))]])
    # rayy=np.asarray(cam)+np.multiply(np.multiply((np.asarray(a1)),(np.asarray(b1))),([np.asarray(range(int(n),int(nn)))]))
    rayy=np.asarray(cam)+np.multiply(np.multiply((np.asarray(a1)),(np.asarray(b1))),np.asarray(c1))
    rayy=np.around(rayy)
    print("print maxs")
    print(rayy[0].max())
    print(rayy[1].max())
    print(rayy[2].max())
    del(a1)
    del(b1)
    del(c1)
    del(cam)
            ##############################################################################
##########################################################################################
    a1=np.zeros((grid2.shape[0],grid2.shape[1],int(nn)-int(n),grid2.shape[2]))
    b1=np.ones((grid2.shape[0],grid2.shape[1],int(nn)-int(n),grid2.shape[2]))
    c1=np.zeros((grid2.shape[0],grid2.shape[1],int(nn)-int(n),grid2.shape[2]))
    cam=np.zeros((grid2.shape[0],grid2.shape[1],int(nn)-int(n),grid2.shape[2]))

    a1=np.transpose(a1)
    cam=np.transpose(cam)
    b1=np.transpose(b1)
    c1=np.transpose(c1)
    a1[0]=[(ray)[0]]
    a1[1]=[(ray)[1]]
    a1[2]=[(ray)[2]]
    cam[0]=[cameraEye.x]
    cam[1]=[cameraEye.y]
    cam[2]=[cameraEye.z]

    c1[0]=np.transpose(grid2.shape[0]*[grid2.shape[1]*[np.asarray(range((n-1),(nn-1)))]])
    c1[1]=np.transpose(grid2.shape[0]*[grid2.shape[1]*[np.asarray(range((n-1),(nn-1)))]])
    c1[2]=np.transpose(grid2.shape[0]*[grid2.shape[1]*[np.asarray(range((n-1),(nn-1)))]])
    del(ray)

    rayy_n_1=np.asarray(cam)+np.multiply(np.multiply((np.asarray(a1)),(np.asarray(b1))),np.asarray(c1))
    rayy_n_1=np.around(rayy_n_1)
    del(a1)
    del(b1)
    del(c1)
    del(cam)
            ##############################################################################

    rayy[0][np.where(rayy[0]<0)]=None
    rayy[1][np.where(rayy[1]<0)]=None
    rayy[2][np.where(rayy[2]<0)]=None
    rayy[0][np.where(rayy[0]>width-2)]=None


    rayy[1][np.where(rayy[1]>height-2)]=None

    rayy[2][np.where(rayy[2]>depth-2)]=None

    rayy[0][np.where(np.logical_or(np.logical_or([(np.isnan(rayy[1]))],[np.isnan(rayy[0])]) , [np.isnan(rayy[2])]))[1:4]]=None
    rayy[1][np.where(np.logical_or(np.logical_or([(np.isnan(rayy[1]))],[np.isnan(rayy[0])]) , [np.isnan(rayy[2])]))[1:4]]=None
    rayy[2][np.where(np.logical_or(np.logical_or([(np.isnan(rayy[1]))],[np.isnan(rayy[0])]) , [np.isnan(rayy[2])]))[1:4]]=None

    # a=rayy[0][np.where(np.logical_and([(~np.isnan(rayy_n_1[0]))],[(~np.isnan(rayy[0]))])[0])]
    # a2=rayy[1][np.where(np.logical_and([(~np.isnan(rayy_n_1[1]))],[(~np.isnan(rayy[1]))])[0])]
    # a3=rayy[2][np.where(np.logical_and([(~np.isnan(rayy_n_1[2]))],[(~np.isnan(rayy[2]))])[0])]

    # a=rayy[0][np.where(np.logical_and([(~np.isnan(rayy_n_1[0]))],[(~np.isnan(rayy[0]))]))[1],np.where(np.logical_and([(~np.isnan(rayy_n_1[0]))],[(~np.isnan(rayy[0]))]))[2],np.where(np.logical_and([(~np.isnan(rayy_n_1[0]))],[(~np.isnan(rayy[0]))]))[3]]
    # a2=rayy[1][np.where(np.logical_and([(~np.isnan(rayy_n_1[0]))],[(~np.isnan(rayy[0]))]))[1],np.where(np.logical_and([(~np.isnan(rayy_n_1[0]))],[(~np.isnan(rayy[0]))]))[2],np.where(np.logical_and([(~np.isnan(rayy_n_1[0]))],[(~np.isnan(rayy[0]))]))[3]]
    # a3=rayy[2][np.where(np.logical_and([(~np.isnan(rayy_n_1[0]))],[(~np.isnan(rayy[0]))]))[1],np.where(np.logical_and([(~np.isnan(rayy_n_1[0]))],[(~np.isnan(rayy[0]))]))[2],np.where(np.logical_and([(~np.isnan(rayy_n_1[0]))],[(~np.isnan(rayy[0]))]))[3]]

    #
    a=rayy[0][np.where(~(np.isnan(rayy[0])))]
    # a=[int(j) for j in a]
    a2=rayy[1][np.where(~(np.isnan(rayy[0])))]
    # a2=[int(j) for j in a2]

    a3=rayy[2][np.where(~(np.isnan(rayy[0])))]
    # a3=[int(j) for j in a3]


###############################################################################################

    # rayy_n_1[0][np.where(rayy_n_1[0]<0)]=None
    # rayy_n_1[1][np.where(rayy_n_1[1]<0)]=None
    # rayy_n_1[2][np.where(rayy_n_1[2]<0)]=None
    # # rayy_n_1[0][np.where(rayy_n_1[0]>width-1)]=None
    # rayy_n_1[0][np.where(rayy_n_1[0]>width-1)]=None
    #
    #
    # # rayy_n_1[1][np.where(rayy_n_1[1]>height-1)]=None
    # rayy_n_1[1][np.where(rayy_n_1[1]>height-1)]=None
    #
    # # rayy_n_1[2][np.where(rayy_n_1[2]>depth-1)]=None
    # rayy_n_1[2][np.where(rayy_n_1[2]>depth-1)]=None
    #
    # # rayy_n_1[np.where(np.logical_or(np.logical_or([(np.isnan(rayy_n_1[1]))],[np.isnan(rayy_n_1[0])]) , [np.isnan(rayy_n_1[2])]))]=None
    # rayy_n_1[0][np.where(np.logical_or(np.logical_or([(np.isnan(rayy_n_1[1]))],[np.isnan(rayy_n_1[0])]) , [np.isnan(rayy_n_1[2])]))[1:4]]=None
    # rayy_n_1[1][np.where(np.logical_or(np.logical_or([(np.isnan(rayy_n_1[1]))],[np.isnan(rayy_n_1[0])]) , [np.isnan(rayy_n_1[2])]))[1:4]]=None
    # rayy_n_1[2][np.where(np.logical_or(np.logical_or([(np.isnan(rayy_n_1[1]))],[np.isnan(rayy_n_1[0])]) , [np.isnan(rayy_n_1[2])]))[1:4]]=None
    #
    # # an=rayy_n_1[0][np.where(np.logical_and([(~np.isnan(rayy_n_1[0]))],[(~np.isnan(rayy[0]))])[0])]
    # # a2n=rayy_n_1[1][np.where(np.logical_and([(~np.isnan(rayy_n_1[1]))],[(~np.isnan(rayy[1]))])[0])]
    # # a3n=rayy_n_1[2][np.where(np.logical_and([(~np.isnan(rayy_n_1[2]))],[(~np.isnan(rayy[2]))])[0])]
    #
    # # an=rayy_n_1[0][np.where(np.logical_and([(~np.isnan(rayy_n_1[0]))],[(~np.isnan(rayy[0]))]))[1],np.where(np.logical_and([(~np.isnan(rayy_n_1[0]))],[(~np.isnan(rayy[0]))]))[2],np.where(np.logical_and([(~np.isnan(rayy_n_1[0]))],[(~np.isnan(rayy[0]))]))[3]]
    # # a2n=rayy_n_1[1][np.where(np.logical_and([(~np.isnan(rayy_n_1[0]))],[(~np.isnan(rayy[0]))]))[1],np.where(np.logical_and([(~np.isnan(rayy_n_1[0]))],[(~np.isnan(rayy[0]))]))[2],np.where(np.logical_and([(~np.isnan(rayy_n_1[0]))],[(~np.isnan(rayy[0]))]))[3]]
    # # a3n=rayy_n_1[2][np.where(np.logical_and([(~np.isnan(rayy_n_1[0]))],[(~np.isnan(rayy[0]))]))[1],np.where(np.logical_and([(~np.isnan(rayy_n_1[0]))],[(~np.isnan(rayy[0]))]))[2],np.where(np.logical_and([(~np.isnan(rayy_n_1[0]))],[(~np.isnan(rayy[0]))]))[3]]

    an=rayy_n_1[0][np.where(~(np.isnan(rayy[0])))]
    an[np.where(np.isnan(an))]=0
    # an=[int(j) for j in an]
    a2n=rayy_n_1[1][np.where(~(np.isnan(rayy[0])))]
    # a2n=[int(j) for j in a2n]
    a2n[np.where(np.isnan(a2n))]=0
    a3n=rayy_n_1[2][np.where(~(np.isnan(rayy[0])))]
    a3n[np.where(np.isnan(a3n))]=0
    # a3n=[int(j) for j in a3n]

    ########################################################################################
    # an=np.where(~np.isnan(rayy_n_1))[1]
    # a2n=np.where(~np.isnan(rayy_n_1))[2]
    # a3n=np.where(~np.isnan(rayy_n_1))[3]
    # a=np.where(~np.isnan(rayy))[1]
    # a2=np.where(~np.isnan(rayy))[2]
    # a3=np.where(~np.isnan(rayy))[3]
    rayy_L=np.zeros(f_data.shape)
    a=np.asarray(a,dtype=int)
    a2=np.asarray(a2,dtype=int)
    a3=np.asarray(a3,dtype=int)
    an=np.asarray(an,dtype=int)
    a2n=np.asarray(a2n,dtype=int)
    a3n=np.asarray(a3n,dtype=int)
    # print(a)
    # print(an)

    rayy_L[[a],[a2],[a3]]=f_data[[a],[a2],[a3]]
    rayy_n_1_L=np.zeros(f_data.shape)
    # rayy_n_1_L2[[a-1],[a2-1],[a3-1]]=f_data[[a-1],[a2-1],[a3-1]]
    del(rayy)

    # rayy_n_1_L=np.zeros(f_data.shape)
    rayy_n_1_L[[an],[a2n],[a3n]]=f_data[[an],[a2n],[a3n]]

    # rayy_L=np.zeros(f_data.shape)
    # rayy_L[[a],[a2],[a3]]=f_data[[a],[a2],[a3]]
    # rayy_n_1_L2=np.zeros((np.asarray(f_data).shape))
    # rayy_n_1_L2[:,:,1:]=rayy_L[:,:,0:-1]
    # rayy_test=rayy_L-rayy_n_1_L2
    rayy_test=rayy_L-rayy_n_1_L
    in_list=np.where(np.asarray(rayy_test)==255)

    out_list=np.where(np.asarray(rayy_test)==-255)
    t_list=np.zeros(f_data.shape)
    t_list=np.asarray(t_list,dtype=object)
    # t_list=np.asarray(t_list,dtype=object)
    try:
        if in_list[0].__len__()!=0:

            in_list2=np.zeros((4,np.transpose(in_list).__len__()),dtype=object)
            # in_list=(in_list)
            in_list2[0]=np.transpose(in_list).__len__()*["in"]
            in_list2[1:4]=(in_list)
        df1 = pd.DataFrame(data=in_list2[0])
        df2 = pd.DataFrame(data=in_list2[1])
        df3 = pd.DataFrame(data=in_list2[2])
        df4 = pd.DataFrame(data=in_list2[3])
        # dfs=df1[0].str.cat(df2[0].to_string(),sep=" ")df1[0].str.cat(df2[0].to_string(),sep=" ")
        dfs=df1[0].str.cat((df2[0].apply(str).str.cat((df3[0].apply(str).str.cat(df4[0].apply(str),sep=" ")),sep=" ")),sep=" ")

        # dfs = pd.concat([df1, df2,df3,df4], axis=1)
        in_list2=pd.DataFrame(dfs).to_numpy()

        t_list[np.where(np.asarray(rayy_test)==255)]=np.transpose(in_list2)
    except:
        in_list2=[]
    try:
        if out_list[0].__len__()!=0:

            out_list2=np.zeros((4,np.transpose(out_list).__len__()),dtype=object)
            out_list2[0]=np.transpose(out_list).__len__()*["out"]
            out_list2[1:4]=(out_list)

        df1 = pd.DataFrame(data=out_list2[0])
        df2 = pd.DataFrame(data=out_list2[1])
        df3 = pd.DataFrame(data=out_list2[2])
        df4 = pd.DataFrame(data=out_list2[3])
        # dfs = pd.concat([df1, df2,df3,df4], axis=1)
        dfs=df1[0].str.cat((df2[0].apply(str).str.cat((df3[0].apply(str).str.cat(df4[0].apply(str),sep=" ")),sep=" ")),sep=" ")
        # dfs = pd.concat([df1, df2,df3,df4], axis=1)
        out_list2=pd.DataFrame(dfs).to_numpy()
        t_list[np.where(np.asarray(rayy_test)==-255)]=np.transpose(out_list2)
    except:
        out_list2=[]




    return(t_list),f_data,grid1,grid2


def main3_2(f_data,width,height,depth,cameraEye):

    cameraEye=glm.vec3(cameraEye)

    cameraUp=glm.vec3(0,1,0)
    volume_centre=glm.vec3(int(width/2),int(height/2),int(depth/2))

    grid1,grid2,cameraEye2,cameraEye3,cameraEye4,cameraEye5=draw_grid3(width,height,depth,cameraEye,volume_centre,cameraUp)
    length=int(grid1.__len__())
    print("cameraEyes:")
    print(cameraEye)
    print(cameraEye2)
    print(cameraEye3)
    print(cameraEye4)
    print(cameraEye5)





    stepSize = 1#0.01
    w11=min(length,width)
    h11=min(length,height)
    d11=min(length,depth)


    pixel_values=np.zeros((grid2.shape[0],grid2.shape[1],1000))
    t_list=np.empty((grid2.shape[0],grid2.shape[1],1000),dtype=object)
    t_list1=np.empty((grid2.shape[0],grid2.shape[1],1000),dtype=object)

    print("view point #1")
    for i in range(0,grid2.shape[0],10):
      print(i)

      for j in range(0,grid2.shape[1],10):
        # print(i,j)
        n2=0
        temp=0
        # pixelPosition1=Vec3(grid1[i,j,0],grid1[i,j,1],grid1[i,j,2])
        pixelPosition2=glm.vec3(grid2[i,j,0],grid2[i,j,1],grid2[i,j,2])
        ray = glm.normalize(pixelPosition2 - cameraEye)

        n=((magnitude((pixelPosition2 - cameraEye))-(max(width,height,depth)/2)))

        # n=0
        # nn=(n+max(width,height,depth))/stepSize
        nn=(n+max(width,height,depth))
        # nn=(1000+max(width,height,depth))
        while(n<nn):
            rayy= glm.vec3(cameraEye + ray * stepSize * (float)(n))
            rayy_n_1= glm.vec3(cameraEye + ray * stepSize * (float)(n-1))
            while (int)(rayy.x)==(int)(rayy_n_1.x) and (int)(rayy.y)==(int)(rayy_n_1.y) and (int)(rayy.z)==(int)(rayy_n_1.z):
                n=n+1
                rayy= glm.vec3(cameraEye + ray * stepSize * (float)(n))


            if(n<nn):
                # print(i,j)

                if(rayy.x<w11 and rayy.x>=0 and rayy.y<h11 and rayy.y>=0 and rayy.z<d11 and rayy.z>=0 and rayy_n_1.x<w11 and rayy_n_1.x>=0 and rayy_n_1.y<h11 and rayy_n_1.y>=0 and rayy_n_1.z<d11 and rayy_n_1.z>=0):
                    # print(i,j)
                    temp=pixel_values[(int)((rayy_n_1.x))][(int)((rayy_n_1.y))][(int)((rayy_n_1.z))]
                    pixel_values[(int)((rayy.x))][(int)((rayy.y))][(int)((rayy.z))]=f_data[(int)((rayy.x))][(int)((rayy.y))][(int)(rayy.z)]


                    if (pixel_values[(int)(rayy.x)][(int)(rayy.y)][(int)(rayy.z)]==255 and temp==0):
                                # print(i,j)
                                t_list1[i][j][n2]=(["in",(int)((rayy.x)),(int)((rayy.y)),(int)((rayy.z))])
                                # print(t_list[i][j][n2])
                                n2=n2+1



                    elif(pixel_values[(int)(rayy.x)][(int)((rayy.y))][(int)((rayy.z))]==0 and temp==255):
                                # print(i,j)
                                t_list1[i][j][n2]=(["out",(int)((rayy.x)),(int)((rayy.y)),(int)((rayy.z))])
                                # print(t_list[i][j][n2])
                                n2=n2+1

                n=n+1
    del(grid1)
    del(grid2)
    del(pixel_values)
    print("view point #2")
    cameraEye2=(cameraEye2[0],cameraEye2[1],cameraEye2[2])
    cameraEye2=(glm.vec3(cameraEye2))
    grid1,grid2=draw_grid2(width,height,depth,cameraEye2,volume_centre,cameraUp)
    length=grid1.__len__()
    pixel_values=np.zeros((grid2.shape[0],grid2.shape[1],1000))
    t_list2=np.empty((grid2.shape[0],grid2.shape[1],1000),dtype=object)
    length=int(length)
    w11=min(length,width)
    h11=min(length,height)
    d11=min(length,depth)
    for i in range(0,grid2.shape[0],10):
      print(i)

      for j in range(0,grid2.shape[1],10):
        # print(i,j)
        n2=0
        temp=0
        # pixelPosition1=Vec3(grid1[i,j,0],grid1[i,j,1],grid1[i,j,2])
        pixelPosition2=glm.vec3(grid2[i,j,0],grid2[i,j,1],grid2[i,j,2])
        ray = glm.normalize(pixelPosition2 - cameraEye2)

        n=((magnitude((pixelPosition2 - cameraEye2))-(max(width,height,depth)/2)))

        # n=0
        # nn=(n+max(width,height,depth))/stepSize
        nn=(n+max(width,height,depth))
        # nn=(1000+max(width,height,depth))
        while(n<nn):
            rayy= glm.vec3(cameraEye2 + ray * stepSize * (float)(n))
            rayy_n_1= glm.vec3(cameraEye2 + ray * stepSize * (float)(n-1))
            while (int)(rayy.x)==(int)(rayy_n_1.x) and (int)(rayy.y)==(int)(rayy_n_1.y) and (int)(rayy.z)==(int)(rayy_n_1.z):
                n=n+1
                rayy= glm.vec3(cameraEye2 + ray * stepSize * (float)(n))


            if(n<nn):
                # print(i,j)

                if(rayy.x<w11 and rayy.x>=0 and rayy.y<h11 and rayy.y>=0 and rayy.z<d11 and rayy.z>=0 and rayy_n_1.x<w11 and rayy_n_1.x>=0 and rayy_n_1.y<h11 and rayy_n_1.y>=0 and rayy_n_1.z<d11 and rayy_n_1.z>=0):
                    # print(i,j)
                    temp=pixel_values[(int)((rayy_n_1.x))][(int)((rayy_n_1.y))][(int)((rayy_n_1.z))]
                    pixel_values[(int)((rayy.x))][(int)((rayy.y))][(int)((rayy.z))]=f_data[(int)((rayy.x))][(int)((rayy.y))][(int)(rayy.z)]


                    if (pixel_values[(int)(rayy.x)][(int)(rayy.y)][(int)(rayy.z)]==255 and temp==0):
                                # print(i,j)
                                t_list2[i][j][n2]=(["in",(int)((rayy.x)),(int)((rayy.y)),(int)((rayy.z))])
                                # print(t_list[i][j][n2])
                                n2=n2+1



                    elif(pixel_values[(int)(rayy.x)][(int)((rayy.y))][(int)((rayy.z))]==0 and temp==255):
                                # print(i,j)
                                t_list2[i][j][n2]=(["out",(int)((rayy.x)),(int)((rayy.y)),(int)((rayy.z))])
                                # print(t_list[i][j][n2])
                                n2=n2+1

                n=n+1
    del(grid1)
    del(grid2)
    del(pixel_values)
    print("view point #3")
    cameraEye3=(cameraEye3[0],cameraEye3[1],cameraEye3[2])
    cameraEye3=(glm.vec3(cameraEye3))
    grid1,grid2=draw_grid2(width,height,depth,cameraEye3,volume_centre,cameraUp)
    length=grid1.__len__()
    pixel_values=np.zeros((grid2.shape[0],grid2.shape[1],1000))
    t_list3=np.empty((grid2.shape[0],grid2.shape[1],1000),dtype=object)
    length=int(length)
    w11=min(length,width)
    h11=min(length,height)
    d11=min(length,depth)
    for i in range(0,grid2.shape[0],10):
      print(i)

      for j in range(0,grid2.shape[1],10):
        # print(i,j)
        n2=0
        temp=0
        # pixelPosition1=Vec3(grid1[i,j,0],grid1[i,j,1],grid1[i,j,2])
        pixelPosition2=glm.vec3(grid2[i,j,0],grid2[i,j,1],grid2[i,j,2])
        ray = glm.normalize(pixelPosition2 - cameraEye3)

        n=((magnitude((pixelPosition2 - cameraEye3))-(max(width,height,depth)/2)))

        # n=0
        # nn=(n+max(width,height,depth))/stepSize
        nn=(n+max(width,height,depth))
        # nn=(1000+max(width,height,depth))
        while(n<nn):
            rayy= glm.vec3(cameraEye3 + ray * stepSize * (float)(n))
            rayy_n_1= glm.vec3(cameraEye3 + ray * stepSize * (float)(n-1))
            while (int)(rayy.x)==(int)(rayy_n_1.x) and (int)(rayy.y)==(int)(rayy_n_1.y) and (int)(rayy.z)==(int)(rayy_n_1.z):
                n=n+1
                rayy= glm.vec3(cameraEye3 + ray * stepSize * (float)(n))


            if(n<nn):
                # print(i,j)

                if(rayy.x<w11 and rayy.x>=0 and rayy.y<h11 and rayy.y>=0 and rayy.z<d11 and rayy.z>=0 and rayy_n_1.x<w11 and rayy_n_1.x>=0 and rayy_n_1.y<h11 and rayy_n_1.y>=0 and rayy_n_1.z<d11 and rayy_n_1.z>=0):
                    # print(i,j)
                    temp=pixel_values[(int)((rayy_n_1.x))][(int)((rayy_n_1.y))][(int)((rayy_n_1.z))]
                    pixel_values[(int)((rayy.x))][(int)((rayy.y))][(int)((rayy.z))]=f_data[(int)((rayy.x))][(int)((rayy.y))][(int)(rayy.z)]


                    if (pixel_values[(int)(rayy.x)][(int)(rayy.y)][(int)(rayy.z)]==255 and temp==0):
                                # print(i,j)
                                t_list3[i][j][n2]=(["in",(int)((rayy.x)),(int)((rayy.y)),(int)((rayy.z))])
                                # print(t_list[i][j][n2])
                                n2=n2+1



                    elif(pixel_values[(int)(rayy.x)][(int)((rayy.y))][(int)((rayy.z))]==0 and temp==255):
                                # print(i,j)
                                t_list3[i][j][n2]=(["out",(int)((rayy.x)),(int)((rayy.y)),(int)((rayy.z))])
                                # print(t_list[i][j][n2])
                                n2=n2+1

                n=n+1

    del(grid1)
    del(grid2)
    del(pixel_values)
    print("view point #4")
    cameraEye4=(cameraEye4[0],cameraEye4[1],cameraEye4[2])
    cameraEye4=(glm.vec3(cameraEye4))
    grid1,grid2=draw_grid2(width,height,depth,cameraEye4,volume_centre,cameraUp)
    length=grid1.__len__()
    pixel_values=np.zeros((grid2.shape[0],grid2.shape[1],1000))
    t_list4=np.empty((grid2.shape[0],grid2.shape[1],1000),dtype=object)
    length=int(length)
    w11=min(length,width)
    h11=min(length,height)
    d11=min(length,depth)
    for i in range(0,grid2.shape[0],10):
      print(i)

      for j in range(0,grid2.shape[1],10):
        # print(i,j)
        n2=0
        temp=0
        # pixelPosition1=Vec3(grid1[i,j,0],grid1[i,j,1],grid1[i,j,2])
        pixelPosition2=glm.vec3(grid2[i,j,0],grid2[i,j,1],grid2[i,j,2])
        ray = glm.normalize(pixelPosition2 - cameraEye4)

        n=((magnitude((pixelPosition2 - cameraEye4))-(max(width,height,depth)/2)))

        # n=0
        # nn=(n+max(width,height,depth))/stepSize
        nn=(n+max(width,height,depth))
        # nn=(1000+max(width,height,depth))
        while(n<nn):
            rayy= glm.vec3(cameraEye4 + ray * stepSize * (float)(n))
            rayy_n_1= glm.vec3(cameraEye4 + ray * stepSize * (float)(n-1))
            while (int)(rayy.x)==(int)(rayy_n_1.x) and (int)(rayy.y)==(int)(rayy_n_1.y) and (int)(rayy.z)==(int)(rayy_n_1.z):
                n=n+1
                rayy= glm.vec3(cameraEye4 + ray * stepSize * (float)(n))


            if(n<nn):
                # print(i,j)

                if(rayy.x<w11 and rayy.x>=0 and rayy.y<h11 and rayy.y>=0 and rayy.z<d11 and rayy.z>=0 and rayy_n_1.x<w11 and rayy_n_1.x>=0 and rayy_n_1.y<h11 and rayy_n_1.y>=0 and rayy_n_1.z<d11 and rayy_n_1.z>=0):
                    # print(i,j)
                    temp=pixel_values[(int)((rayy_n_1.x))][(int)((rayy_n_1.y))][(int)((rayy_n_1.z))]
                    pixel_values[(int)((rayy.x))][(int)((rayy.y))][(int)((rayy.z))]=f_data[(int)((rayy.x))][(int)((rayy.y))][(int)(rayy.z)]


                    if (pixel_values[(int)(rayy.x)][(int)(rayy.y)][(int)(rayy.z)]==255 and temp==0):
                                # print(i,j)
                                t_list4[i][j][n2]=(["in",(int)((rayy.x)),(int)((rayy.y)),(int)((rayy.z))])
                                # print(t_list[i][j][n2])
                                n2=n2+1



                    elif(pixel_values[(int)(rayy.x)][(int)((rayy.y))][(int)((rayy.z))]==0 and temp==255):
                                # print(i,j)
                                t_list4[i][j][n2]=(["out",(int)((rayy.x)),(int)((rayy.y)),(int)((rayy.z))])
                                # print(t_list[i][j][n2])
                                n2=n2+1

                n=n+1


    del(grid1)
    del(grid2)
    del(pixel_values)
    print("view point #5")
    cameraEye5=(cameraEye5[0],cameraEye5[1],cameraEye5[2])
    cameraEye5=(glm.vec3(cameraEye5))
    grid1,grid2=draw_grid2(width,height,depth,cameraEye5,volume_centre,cameraUp)
    length=grid1.__len__()
    pixel_values=np.zeros((grid2.shape[0],grid2.shape[1],1000))
    t_list5=np.empty((grid2.shape[0],grid2.shape[1],1000),dtype=object)
    length=int(length)
    w11=min(length,width)
    h11=min(length,height)
    d11=min(length,depth)
    for i in range(0,grid2.shape[0],10):
      print(i)

      for j in range(0,grid2.shape[1],10):
        # print(i,j)
        n2=0
        temp=0
        # pixelPosition1=Vec3(grid1[i,j,0],grid1[i,j,1],grid1[i,j,2])
        pixelPosition2=glm.vec3(grid2[i,j,0],grid2[i,j,1],grid2[i,j,2])
        ray = glm.normalize(pixelPosition2 - cameraEye5)

        n=((magnitude((pixelPosition2 - cameraEye5))-(max(width,height,depth)/2)))

        # n=0
        # nn=(n+max(width,height,depth))/stepSize
        nn=(n+max(width,height,depth))
        # nn=(1000+max(width,height,depth))
        while(n<nn):
            rayy= glm.vec3(cameraEye5 + ray * stepSize * (float)(n))
            rayy_n_1= glm.vec3(cameraEye5 + ray * stepSize * (float)(n-1))
            while (int)(rayy.x)==(int)(rayy_n_1.x) and (int)(rayy.y)==(int)(rayy_n_1.y) and (int)(rayy.z)==(int)(rayy_n_1.z):
                n=n+1
                rayy= glm.vec3(cameraEye5 + ray * stepSize * (float)(n))


            if(n<nn):
                # print(i,j)

                if(rayy.x<w11 and rayy.x>=0 and rayy.y<h11 and rayy.y>=0 and rayy.z<d11 and rayy.z>=0 and rayy_n_1.x<w11 and rayy_n_1.x>=0 and rayy_n_1.y<h11 and rayy_n_1.y>=0 and rayy_n_1.z<d11 and rayy_n_1.z>=0):
                    # print(i,j)
                    temp=pixel_values[(int)((rayy_n_1.x))][(int)((rayy_n_1.y))][(int)((rayy_n_1.z))]
                    pixel_values[(int)((rayy.x))][(int)((rayy.y))][(int)((rayy.z))]=f_data[(int)((rayy.x))][(int)((rayy.y))][(int)(rayy.z)]


                    if (pixel_values[(int)(rayy.x)][(int)(rayy.y)][(int)(rayy.z)]==255 and temp==0):
                                # print(i,j)
                                t_list5[i][j][n2]=(["in",(int)((rayy.x)),(int)((rayy.y)),(int)((rayy.z))])
                                # print(t_list[i][j][n2])
                                n2=n2+1



                    elif(pixel_values[(int)(rayy.x)][(int)((rayy.y))][(int)((rayy.z))]==0 and temp==255):
                                # print(i,j)
                                t_list5[i][j][n2]=(["out",(int)((rayy.x)),(int)((rayy.y)),(int)((rayy.z))])
                                # print(t_list[i][j][n2])
                                n2=n2+1

                n=n+1


    f_data=[]

    return(t_list),f_data,grid1,grid2,t_list1,t_list2,t_list3,t_list4,t_list5







def main3_3(f_data,width,height,depth,cameraEye):

    cameraEye=glm.vec3(cameraEye)

    cameraUp=glm.vec3(0,1,0)
    volume_centre=glm.vec3(int(width/2),int(height/2),int(depth/2))
    grid1,grid2=draw_grid2(width,height,depth,cameraEye,volume_centre,cameraUp)



    stepSize = 1#0.01



    pixel_values=f_data
    t_list=np.empty((grid2.shape[0],grid2.shape[1],1000),dtype=object)
    for i in range(0,grid2.shape[0],5):
      print(i)

      for j in range(0,grid2.shape[1],5):
        # print(i,j)
        n2=0
        temp=0
        # pixelPosition1=Vec3(grid1[i,j,0],grid1[i,j,1],grid1[i,j,2])
        pixelPosition2=glm.vec3(grid2[i,j,0],grid2[i,j,1],grid2[i,j,2])
        ray = glm.normalize(pixelPosition2 - cameraEye)
        n=((magnitude((pixelPosition2 - cameraEye))-(max(width,height,depth)/2)))
        nn=(n+max(width,height,depth))/stepSize
        while(n<nn):
            rayy= glm.vec3(cameraEye + ray * stepSize * (float)(n))
            rayy_n_1= glm.vec3(cameraEye + ray * stepSize * (float)(n-1))
            while (int)(rayy.x)==(int)(rayy_n_1.x) and (int)(rayy.y)==(int)(rayy_n_1.y) and (int)(rayy.z)==(int)(rayy_n_1.z):
                n=n+1
                rayy= glm.vec3(cameraEye + ray * stepSize * (float)(n))


            if(n<nn):
                # print(i,j)
                if(rayy.x<width and rayy.x>=0 and rayy.y<height and rayy.y>=0 and rayy.z<depth and rayy.z>=0 and rayy_n_1.x<width and rayy_n_1.x>=0 and rayy_n_1.y<height and rayy_n_1.y>=0 and rayy_n_1.z<depth and rayy_n_1.z>=0):
                    # print(i,j)
                    temp=pixel_values[(int)((rayy_n_1.x))][(int)((rayy_n_1.y))][(int)((rayy_n_1.z))]
                    # pixel_values[(int)((rayy.x))][(int)((rayy.y))][(int)((rayy.z))]=f_data[(int)((rayy.x))][(int)((rayy.y))][(int)(rayy.z)]


                    if (pixel_values[(int)(rayy.x)][(int)(rayy.y)][(int)(rayy.z)]==255 and temp==0):
                                # print(i,j)
                                t_list[i][j][n2]=(["in",(int)((rayy.x)),(int)((rayy.y)),(int)((rayy.z))])
                                # print(t_list[i][j][n2])
                                n2=n2+1



                    elif(pixel_values[(int)(rayy.x)][(int)((rayy.y))][(int)((rayy.z))]==0 and temp==255):
                                # print(i,j)
                                t_list[i][j][n2]=(["out",(int)((rayy.x)),(int)((rayy.y)),(int)((rayy.z))])
                                # print(t_list[i][j][n2])
                                n2=n2+1

                n=n+1


    f_data=[]
    return(t_list),f_data,grid1,grid2
