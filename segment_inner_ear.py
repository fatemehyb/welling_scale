# https://www.researchgate.net/post/How_do_I_convert_a_mesh_in_a_voxelized_volume_3d_image2
# import SimpleITK as sitk
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# from scipy.spatial import ConvexHull
# import numpy as np
# import scipy.io as sio
# from stl import mesh
# # import shapely
# import random
# import mayavi
# from mayavi import mlab
# # mesh created with
# # verts, faces = skimage.measure.marching_cubes(volume, level, spacing=(1.0, 1.0, 1.0))
# import digastric_ridge_identification_function as dr
from mayavi import mlab
# import vtk
import os
# import openmesh as om
import argparse
import itk
import SimpleITK as sitk
import numpy as np
import read_image_m as RIM
import glm
import time
# import wx
# from openmesh import *
from skimage.measure import marching_cubes_lewiner
# from scipy.spatial import ConvexHull
X_2=[]
Y_2=[]
Z_2=[]

def bbox2_3D(img):

    r = np.any(img, axis=(1, 2))
    c = np.any(img, axis=(0, 2))
    z = np.any(img, axis=(0, 1))

    rmin, rmax = np.where(r)[0][[0, -1]]
    cmin, cmax = np.where(c)[0][[0, -1]]
    zmin, zmax = np.where(z)[0][[0, -1]]

    return rmin, rmax, cmin, cmax, zmin, zmax

def Cross(a, b):
    x = a.y * b.z - a.z * b.y
    y = a.z * b.x - a.x * b.z
    z = a.x * b.y - a.y * b.x

    # //    return (a.y*b.z-a.z*b.y,a.z*b.x-a.x*b.z,a.x*b.y-a.y*b.x);
    return glm.vec3(x, y, z)
def Dot(a, b):
    x = a.x * b.x
    y = a.y * b.y
    z = a.z * b.z

    # //    return (a.y*b.z-a.z*b.y,a.z*b.x-a.x*b.z,a.x*b.y-a.y*b.x);
    return glm.vec3(x, y, z)


index_changed=[]
index_changed2=[]
faces2=[]
verts2=[]
def main_normal(myvolume,spacing,verts2,faces2):
        # verts, faces = skimage.measure.marching_cubes(volume, level, spacing=(1,1,1))
        verts, faces, normals, values = marching_cubes_lewiner(myvolume, 0, spacing)

        mesh=mlab.triangular_mesh([vert[0] for vert in verts],
                             [vert[1] for vert in verts],
                             [vert[2] for vert in verts],
                             faces)
        mesh.mlab_source.dataset.cell_data.scalars = np.zeros(faces.size)
        mesh.actor.mapper.scalar_visibility=True
        mlab.gcf().scene.parallel_projection = True

        mesh.mlab_source.update()
        # mlab.show()
        mesh_external=mesh

        ########################these two lines give you information about celles try to color cells tomorrow#######################################


        # result=read_trimesh(mesh,myvolume)
        faces2.append(faces)
        verts2.append(verts)


        # A first plot in 3D
        fig = mlab.figure(1)
        # for f in faces:
        #     if faces[f,0]==vertex
        # face_index=verts.index(vertex)

        cursor3d = mlab.points3d(0., 0., 0., mode='axes',
                                        color=(0, 0, 0),
                                        scale_factor=0.5)
        mlab.title('Click on the volume to determine 3 points(consider right hand rule)')



        ################################################################################
        # Some logic to select 'mesh' and the data index when picking.
        def picker_callback2(picker_obj):

            picked = picker_obj.actors




            if mesh.actor.actor._vtk_obj in [o._vtk_obj for o in picked]:



                point_id=index_changed2.pop()



                index_to_change2=np.where(point_id==(faces2[0].transpose())[:])
                ##################################################################################mayavi puck surface point python no depth
                for i in range(0,index_to_change2[1].size):
                     mesh.mlab_source.dataset.cell_data.scalars[int(index_to_change2[1][i])]=0
                mesh.mlab_source.dataset.cell_data.scalars.name = 'Cell data'


                mesh2= mlab.pipeline.set_active_attribute(mesh,cell_scalars='Cell data')
                mlab.pipeline.surface(mesh2)

                ###################################################################################







        def picker_callback(picker_obj):
            # picker_obj.tolerance=1
            picked = picker_obj.actors
            # picker_obj.GetActore()



            if mesh.actor.actor._vtk_obj in [o._vtk_obj for o in picked]:


                x_2, y_2, z_2 = picker_obj.pick_position


                index_to_change2=np.where(picker_obj.point_id==(faces2[0].transpose())[:])
                ##################################################################################mayavi puck surface point python no depth
                for i in range(0,index_to_change2[1].size):
                     mesh.mlab_source.dataset.cell_data.scalars[int(index_to_change2[1][i])]=255
                mesh.mlab_source.dataset.cell_data.scalars.name = 'Cell data'

                mesh2= mlab.pipeline.set_active_attribute(mesh,cell_scalars='Cell data')
                mlab.pipeline.surface(mesh2)

                # mesh.mlab_source.update()
                # wx.Yield()
                ###################################################################################
                if picker_obj.pick_position[0]>0 and picker_obj.pick_position[1]>0 and picker_obj.pick_position[2]>0 :
                    index_changed2.append( picker_obj.point_id)





                    # x_2, y_2, z_2 = picker_obj.mapper_position
                    X_2.append(x_2/spacing[0])
                    Y_2.append(y_2/spacing[1])
                    Z_2.append(z_2/spacing[2])


                    print("Data indices: %f, %f, %f" % (x_2, y_2, z_2))
                    print("point ID: %f"% (picker_obj.point_id))
                    index_changed.append((np.asarray(picker_obj.pick_position))/0.12)
                    print("cell ID: %f"% (picker_obj.cell_id))
                # index_changed.append(int(picker_obj.cell_id))

        picker_obj=fig.on_mouse_pick(picker_callback,type='cell')
        fig.on_mouse_pick(picker_callback2,type='cell',button='Right')



        mlab.show()

        ############################################################################

        The_Normal2=glm.vec3(1,1,1)



        return ((The_Normal2),index_changed)

        # From:
        # http://scikit-image.org/docs/dev/api/skimage.measure.html?highlight=marching_cubes#skimage.measure.marching_cubes
        # http://scikit-image.org/docs/dev/auto_examples/plot_marching_cubes.html
# time.time()
startTime = time.time()

# bytes(path, "utf-8").decode("unicode_escape")

parser = argparse.ArgumentParser(
    description = """This program uses ray casting method to detect overhang problem""")
parser.add_argument("-args0", type = str, default = (('\\\\samba.cs.ucalgary.ca\\fatemeh.yazdanbakhsh\Documents\Data_Sets\Kowther\Specimen2501L\Specimen2501L\Segmentations\InnerEar.nrrd')), help = "Inner ear")
# parser.add_argument("-args1",type=str, default=('\\\\samba.cs.ucalgary.ca\\fatemeh.yazdanbakhsh\Documents\Data_Sets\Kowther\Specimen2501L\Specimen2501L\\2501L_reduced'),help="intact image address")
# parser.add_argument("-args2", type = str, default = (("\\\\samba.cs.ucalgary.ca\\fatemeh.yazdanbakhsh\Documents\Data_Sets\Kowther\Specimen2501L\Specimen2501L\Segmentations\FacialNerve.nrrd")), help = "facial nerve")
# # parser.add_argument("-args0", type = str, default = "U:\Documents\Data_Sets\Calgary\TBone-2015\TBoneCBCT-2015-10\L3016_modified_19_nov", help = "dicome image address")
# parser.add_argument("-args3", type = str, default = ( "\\\\samba.cs.ucalgary.ca\\fatemeh.yazdanbakhsh\Documents\Data_Sets\Kowther\Specimen2501L\Specimen2501L\Segmentations\SigmoidSinus.nrrd"),
#     help = "address of sigmoid sinus mask")

args = parser.parse_args()
low=1000
high=4000



######################################################################################################################

#####################################################################
def finish():
    print('\n', "bye", '\n')
    input('Press Enter to quit: ')
###############################################################################################################################



##########################################################################################################################


# read the original volume
ext=os.path.splitext(args.args0)[1]
m_string=args.args0
if (ext==".nii" or ext==".nrrd"):

    input_volume=sitk.ReadImage(m_string)
else:
    input_volume=RIM.dicom_series_reader(m_string)

# spacing=input_volume.GetSpacing()
spacing=[0.12,0.12,0.12]
origin=input_volume.GetOrigin()
try:
    myvolume=sitk.GetArrayFromImage(input_volume)
except:
    myvolume=itk.GetArrayFromImage(input_volume)

###############################################################################################################################
#############################Reading Intact Volume################################


# ext=os.path.splitext(str(( args.args1)))[1]
# m_string3=str(( args.args1))
# if (ext==".nii" or ext==".nrrd" or ext==".nhdr"):
#
#     intact_volume=sitk.ReadImage(m_string3)
#     intact_array=sitk.GetArrayFromImage(intact_volume)
# else:
#     intact_volume=RIM.dicom_series_reader(m_string3)
#     intact_array=itk.GetArrayFromImage(intact_volume)
# intact_volume=RIM.dicom_series_reader(str(unicode('\\\\samba.cs.ucalgary.ca\\fatemeh.yazdanbakhsh\Documents\Data_Sets\Calgary\TBone-2015\TBoneCBCT-2015-10\L2963L','utf-8')))

#######################################################################################################
#

#do binary threshoulding on the original image

# PixelType = itk.ctype('signed short')
# Dimension = 3
# try:
#     thresholdFilter= sitk.BinaryThresholdImageFilter()
#     intact_volume_thr=thresholdFilter.Execute(intact_volume,low,high,255,0)
# except:print(0)
# try:
#     ImageType_threshold = itk.Image[PixelType, Dimension]
#     thresholdFilter= itk.BinaryThresholdImageFilter[ImageType_threshold,ImageType_threshold].New()
#     # input_volume=thresholdFilter.Execute(input_volume,low,high,0,255)
#     thresholdFilter.SetInput((intact_volume))
#
#     thresholdFilter.SetLowerThreshold(low)
#     thresholdFilter.SetUpperThreshold(high)
#     thresholdFilter.SetOutsideValue(0)
#     thresholdFilter.SetInsideValue(255)
#     thresholdFilter.Update()
#     intact_volume_thr=thresholdFilter.GetOutput()
#
# except:print(0)
#
# try:
#     thr_intact_matrix=sitk.GetArrayFromImage(intact_volume_thr)
# except:
#     thr_intact_matrix=itk.GetArrayFromImage(intact_volume_thr)
###############################################################################################################################
from scipy.spatial import ConvexHull

w1 = myvolume.shape[2]
h1 = myvolume.shape[1]
d1 = myvolume.shape[0]

# rmin,rmax,cmin,cmax,zmin,zmax=dr.main(args.args2,args.args3,args.args0,args.args1)
rmin,rmax,cmin,cmax,zmin,zmax=bbox2_3D(myvolume)
The_Normal,index_changed=main_normal(myvolume[rmin:rmax,cmin:cmax,zmin:zmax],spacing,verts2,faces2)
# rmin=rmin-100
# rmax=rmax+100
# cmin=cmin-100
# cmax=cmax+100
# zmin=zmin-100
# zmax=zmax+100
hull=ConvexHull(index_changed)
points=hull.points
aaa=np.zeros((w1,h1,d1))
for i in range(0,points.shape[0]):
    aaa[int((points[i][0])),int((points[i][1])),int((points[i][2]))]=1

rmin2,rmax2,cmin2,cmax2,zmin2,zmax2=bbox2_3D(aaa)

myvolume2=myvolume
w1 = myvolume.shape[2]
h1 = myvolume.shape[1]
d1 = myvolume.shape[0]
del(myvolume)

def ismember(A, B):
    return [ np.sum(a == B) for a in A ]




# The_Normal,p1,p2,p3,index_changed=main_normal(myvolume,spacing,verts2,faces2)
# hull=ConvexHull(myvolume)

from stl import mesh
import stl
mlab.triangular_mesh(points[:,0],points[:,1],points[:,2], hull.simplices)
# mesh.save('mesh.stl')
mlab.show()
cube = mesh.Mesh(np.zeros(hull.simplices.shape[0], dtype=mesh.Mesh.dtype))
# cube = mesh.Mesh(np.zeros((myvolume2.shape), dtype=mesh.Mesh.dtype))
for i, f in enumerate(hull.simplices):
    for j in range(3):
        cube.vectors[i][j] = hull.points[f[j],:]

# Write the mesh to file "cube.stl"
cube.save('D:\matlab_useful_codes\Mesh_voxelisation\Mesh_voxelisation\cube2.stl')
############################running matlab scripts in python #################################
from oct2py import octave as oct
# octave = oct.oct2py('D:\Octave\Octave-5.1.0.0\mingw64\\bin\octave-cli.exe')
import os
oct.eval("cd D:\matlab_useful_codes\Mesh_voxelisation\Mesh_voxelisation")
cwd = os.getcwd()
oct.addpath(cwd)
oct.addpath('D:\matlab_useful_codes\Mesh_voxelisation\Mesh_voxelisation')
oct.feval('VOXELISE_example_function','cube2.stl',rmax2-rmin2,cmax2-cmin2,zmax2-zmin2)
# oct.feval('VOXELISE_example_function','cube2.stl',100,100,100)
oct.eval("cd D:\matlab_useful_codes\Mesh_voxelisation")
# oct.eval("save -v7 myworkspace.mat")
from scipy.io import loadmat
D = loadmat("D:\matlab_useful_codes\Mesh_voxelisation\Mesh_voxelisation\myworkspace.mat")
print(D.keys())
########reading the .mat matrix convert it to numpy array and save it as .nrrd image using SimpleITK
# z=sio.loadmat('test_voxel.mat')
zz=np.zeros((d1,h1,w1),dtype=int)
z2=D['OUTPUTgrid']
# os.remove('cube2.stl')
rmin3,rmax3,cmin3,cmax3,zmin3,zmax3=bbox2_3D(z2)
# zz[rmin2:,cmin2:,zmin2:]=z2
zz[rmin+rmin2+rmin3:rmin+rmin2+rmax3,cmin+cmin2+cmin3:cmin+cmin2+cmax3,zmin+zmin2+zmin3:zmin+zmin2+zmax3]=z2[rmin3:rmax3,cmin3:cmax3,zmin3:zmax3]
zz[np.where(zz==1.0)]=255
zz2=np.logical_and(zz,thr_intact_matrix)
zz_Image=sitk.GetImageFromArray(zz)
sitk.WriteImage(zz_Image,'output.nrrd')
fig = mlab.figure("part of intact")

zz2=np.asarray(zz2,dtype=int)
verts, faces, normals, values = marching_cubes_lewiner(zz2, 0, spacing)
mesh=mlab.triangular_mesh([vert[0] for vert in verts],
                             [vert[1] for vert in verts],
                             [vert[2] for vert in verts],
                            faces)
mlab.show()
