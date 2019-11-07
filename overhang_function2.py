#call this function like this
#https://stackoverflow.com/questions/31400769/bounding-box-of-numpy-array
# python call_function.py
for name in dir():
    if not name.startswith('_'):
        del globals()[name]

import numpy as np
import itk
import os
import clustering_3D_2 as cluster

import finall_point_ray_casting_3D as prc
import Evaluation_3D as eva


import SimpleITK as sitk
import time
import sys
import gc
# import math
import glm

#Store the start time of the program

def main(dis_volume,inta_volume,sig_volume,ossic_volume,fac_volume,Inn_volume,dis_volume_add,Normall,Spacing,Low,High):
    # time.time()
    startTime = time.time()

    # bytes(path, "utf-8").decode("unicode_escape")

    # parser = argparse.ArgumentParser(
    #     description = """This program uses ray casting method to detect overhang problem""")
    # parser.add_argument("-args0", type = str, default = (('D:\evan_campton\\fatemeh_overhang3')), help = "dissected image address")
    # parser.add_argument("-args1",type=str, default=('D:\evan_campton\\1601L_FLIPPED_CROPPED_EDITED\\1601L_FLIPPED_CROPPED_EDITED\IMAGES\\1601L_FLIPPED_CROPPED_EDITED'),help="intact image address")
    #
    # parser.add_argument("-args2", type = int, default = 1000,
    #     help = "low")
    # parser.add_argument("-args3", type = int, default = 4000,
    #     help = "high")
    #
    # # parser.add_argument("-args4",type=str, default=('D:\evan_campton\\1601L_FLIPPED_CROPPED_EDITED\\1601L_FLIPPED_CROPPED_EDITED\SEGMENTATIONS\edited\SigmoidSinus.nrrd'),help="sigmoid address")
    # parser.add_argument("-args4",type=str, default=('D:\evan_campton\\1601L_FLIPPED_CROPPED_EDITED\\1601L_FLIPPED_CROPPED_EDITED\SEGMENTATIONS\edited\\SigmoidSinus.nrrd'),help="sigmoid address")
    #
    # parser.add_argument("-args5",type=str, default=('D:\evan_campton\\1601L_FLIPPED_CROPPED_EDITED\\1601L_FLIPPED_CROPPED_EDITED\SEGMENTATIONS\edited\\Ossicles.nrrd'),help="Ossicle address")
    # parser.add_argument("-args6",type=str, default=('D:\evan_campton\\1601L_FLIPPED_CROPPED_EDITED\\1601L_FLIPPED_CROPPED_EDITED\SEGMENTATIONS\edited\\\\FacialNerve.nrrd'),help="FacialNerve address")
    # parser.add_argument("-args7",type=str, default=('D:\evan_campton\\1601L_FLIPPED_CROPPED_EDITED\\1601L_FLIPPED_CROPPED_EDITED\SEGMENTATIONS\edited\\\\InnerEar.nrrd'),help="inner ear address")
    # # Get your arguments
    # # Get your arguments
    # args = parser.parse_args()


    low=Low
    high=High


    ######################################################################################################################
    # def Cross(a,b):
    #       x=a.y*b.z-a.z*b.y
    #       y=a.z*b.x-a.x*b.z
    #       z=a.x*b.y-a.y*b.x
    #       return glm.vec3(x,y,z)
    ################################################################################################
    # def Dot(a, b):
    #     x = a.x * b.x
    #     y = a.y * b.y
    #     z = a.z * b.z
    #
    #     # //    return (a.y*b.z-a.z*b.y,a.z*b.x-a.x*b.z,a.x*b.y-a.y*b.x);
    #     return (x+y+z)

    #####################################################################
    def finish():
        print('\n', "bye", '\n')
        input('Press Enter to quit: ')
    ###############################################################################################################################
    #******flood filling algorithm for filling the holes in the image******
    # def flood_fill(test_array,h_max=255):
    #     input_array = np.copy(test_array)
    #     el = scipy.ndimage.generate_binary_structure(2,2).astype(np.int)
    #     inside_mask = scipy.ndimage.binary_erosion(~np.isnan(input_array), structure=el)
    #     output_array = np.copy(input_array)
    #     output_array[inside_mask]=h_max
    #     output_old_array = np.copy(input_array)
    #     output_old_array.fill(0)
    #     el = scipy.ndimage.generate_binary_structure(2,1).astype(np.int)
    #     while not np.array_equal(output_old_array, output_array):
    #         output_old_array = np.copy(output_array)
    #         output_array = np.maximum(input_array,scipy.ndimage.grey_erosion(output_array, size=(3,3), footprint=el))
    #     return output_array
    #################################################################################################################################
    #find the window around the image


    # def bbox2_ND(img):
    #     N = img.ndim
    #     out = []
    #     for ax in itertools.combinations(range(N), N - 1):
    #         nonzero = np.any(img, axis=ax)
    #         out.extend(np.where(nonzero)[0][[0, -1]])
    #     return tuple(out)
    def bbox2_3D(img):

        r = np.any(img, axis=(1, 2))
        c = np.any(img, axis=(0, 2))
        z = np.any(img, axis=(0, 1))

        rmin, rmax = np.where(r)[0][[0, -1]]
        cmin, cmax = np.where(c)[0][[0, -1]]
        zmin, zmax = np.where(z)[0][[0, -1]]

        return rmin, rmax, cmin, cmax, zmin, zmax
    def bbox2(img):

        rows = np.any(img, axis=1)
        cols = np.any(img, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]

        return rmin, rmax, cmin, cmax
    ############################################################################################################################
    def clamp_image(img):
        # input_matrix[np.where((input_matrix == [255.0]))] = [1.0]
        img=(img-min(img))/(max(img)-min(img))
        img=img*65535
        return img


    ##########################################################################################################################


    # read the original volume
    # ext=os.path.splitext(args.args0)[1]
    # m_string=args.args0
    # if (ext==".nii" or ext==".nrrd"):
    #
    #     input_volume=sitk.ReadImage(m_string)
    # else:
    #     input_volume=RIM.dicom_series_reader(m_string)
    #
    #
    input_volume=dis_volume
    # spacing=input_volume.GetSpacing()
    spacing=Spacing
    # origin=input_volume.GetOrigin()
    try:
        myvolume=sitk.GetArrayFromImage(input_volume)
    except:
        myvolume=itk.GetArrayFromImage(input_volume)
    w1 = myvolume.shape[2]
    h1 = myvolume.shape[1]
    d1 = myvolume.shape[0]
    # The_Normal,p1,p2,p3=call_normal_computation.main_normal(myvolume,spacing)
    # The_Normal=[round(The_Normal[0]),round(The_Normal[1]),round(The_Normal[2])]
    # print(The_Normal)
    # The_Normal=[0,0,-1]
    The_Normal=Normall
    # The_Normal=[-1,0,0]
    del(myvolume)
    #############################################################


    # ext=os.path.splitext(str(( args.args1)))[1]
    # m_string3=str(( args.args1))
    intact_volume=inta_volume
    try:

        # intact_volume=sitk.ReadImage(m_string3)
        intact_array=sitk.GetArrayFromImage(intact_volume)
    except:
        # intact_volume=RIM.dicom_series_reader(m_string3)
        intact_array=itk.GetArrayFromImage(intact_volume)
    # intact_volume=RIM.dicom_series_reader(str(unicode('\\\\samba.cs.ucalgary.ca\\fatemeh.yazdanbakhsh\Documents\Data_Sets\Calgary\TBone-2015\TBoneCBCT-2015-10\L2963L','utf-8')))


    intact_array_original=intact_array
    del(intact_array)
    #######################################################################################################
    #
    try:
      input_matrix2 = itk.GetArrayFromImage(input_volume)
    except: print(0)
    try:
        input_matrix2 = sitk.GetArrayFromImage(input_volume)
    except: print(0)
    input_matrix_original=input_matrix2

    input_volume2=input_matrix2
    blobs2=np.zeros(input_volume2.shape,dtype=bool)
    blobsTwo=np.zeros(input_volume2.shape,dtype=bool)

    #do binary threshoulding on the original image

    PixelType = itk.ctype('signed short')
    Dimension = 3
    try:
        thresholdFilter= sitk.BinaryThresholdImageFilter()
        input_volume_thr=thresholdFilter.Execute(input_volume,low,high,255,0)
    except:print(0)
    try:
        ImageType_threshold = itk.Image[PixelType, Dimension]
        thresholdFilter= itk.BinaryThresholdImageFilter[ImageType_threshold,ImageType_threshold].New()
        # input_volume=thresholdFilter.Execute(input_volume,low,high,0,255)
        thresholdFilter.SetInput(input_volume)

        thresholdFilter.SetLowerThreshold(low)
        thresholdFilter.SetUpperThreshold(high)
        thresholdFilter.SetOutsideValue(0)
        thresholdFilter.SetInsideValue(255)
        thresholdFilter.Update()
        input_volume_thr=thresholdFilter.GetOutput()

    except:print(0)

    #####################################################

    #do binary threshoulding on the intact image
    try:
        thresholdFilter= sitk.BinaryThresholdImageFilter()
        intact_volume2_thr=thresholdFilter.Execute(intact_volume,low,high,255,0)
    except:print(0)
    #
    try:
        PixelType = itk.ctype('signed short')
        Dimension = 3
        ImageType_threshold = itk.Image[PixelType, Dimension]
        thresholdFilter= itk.BinaryThresholdImageFilter[ImageType_threshold,ImageType_threshold].New()
        thresholdFilter.SetInput(intact_volume)

        thresholdFilter.SetLowerThreshold(low)
        thresholdFilter.SetUpperThreshold(high)
        thresholdFilter.SetOutsideValue(0)
        thresholdFilter.SetInsideValue(255)
        thresholdFilter.Update()
        intact_volume2_thr=thresholdFilter.GetOutput()
    #
    except:print(0)
    #intact_array=itk.GetArrayFromImage(intact_volume2)
    try:
        intact_array_thr=sitk.GetArrayFromImage(intact_volume2_thr)
    except:print(0)
    try:
        intact_array_thr=itk.GetArrayFromImage(intact_volume2_thr)
    except:print(0)



    #####################################################
    try:
        input_matrix = sitk.GetArrayFromImage(input_volume_thr)
    except:print(0)
    try:
        input_matrix = itk.GetArrayFromImage(input_volume_thr)
    except:print(0)


    #############################################################
    # read the sigmoid volume
    # ext=os.path.splitext(args.args4)[1]
    # m_string=args.args4
    sigmoid_volume=sig_volume
    # if (ext==".nii" or ext==".nrrd"):
    #
    #     sigmoid_volume=sitk.ReadImage(m_string)
    # else:
    #     sigmoid_volume=RIM.dicom_series_reader(m_string)

    # spacing=[0.12,0.12,0.12]
    # spacing=input_volume.GetSpacing()
    # origin=input_volume.GetOrigin()
    try:
        mysigmoid=sitk.GetArrayFromImage(sigmoid_volume)
    except:
        mysigmoid=itk.GetArrayFromImage(sigmoid_volume)

    # w1 = myvolume.shape[2]
    # h1 = myvolume.shape[1]
    # d1 = myvolume.shape[0]
    del(sigmoid_volume)

    #######################################################################################################
    # read the ossicle volume
    # ext=os.path.splitext(args.args5)[1]
    # m_string=args.args5
    Ossicles_volume=ossic_volume
    # if (ext==".nii" or ext==".nrrd"):
    #
    #     Ossicles_volume=sitk.ReadImage(m_string)
    # else:
    #     Ossicles_volume=RIM.dicom_series_reader(m_string)

    # spacing=[0.12,0.12,0.12]
    # spacing=input_volume.GetSpacing()
    # origin=input_volume.GetOrigin()
    try:
        myOssicles=sitk.GetArrayFromImage(Ossicles_volume)
    except:
        myOssicles=itk.GetArrayFromImage(Ossicles_volume)


    del(Ossicles_volume)
    rmino,rmaxo,cmino,cmaxo,zmino,zmaxo=bbox2_3D(myOssicles)
    myOssicles=myOssicles[rmino:rmaxo,cmino:cmaxo,zmino:zmaxo]
    w1 = myOssicles.shape[0]
    h1 = myOssicles.shape[1]
    d1 = myOssicles.shape[2]
    center_ossicles=(int(w1/2)+rmino,int(h1/2)+cmino,int(d1/2)+zmino)
    #######################################################################################################
    #######################################################################################################
    # read the ossicle volume
    Facial_volume=fac_volume
    # ext=os.path.splitext(args.args6)[1]
    # m_string=args.args6
    # if (ext==".nii" or ext==".nrrd"):
    #
    #     Facial_volume=sitk.ReadImage(m_string)
    # else:
    #     Facial_volume=RIM.dicom_series_reader(m_string)

    # spacing=[0.12,0.12,0.12]
    # spacing=input_volume.GetSpacing()
    # origin=input_volume.GetOrigin()
    try:
        myFacial=sitk.GetArrayFromImage(Facial_volume)
    except:
        myFacial=itk.GetArrayFromImage(Facial_volume)


    del(Facial_volume)

    #######################################################################################################
    #######################################################################################################
    # read the ossicle volume
    Inner_volume=Inn_volume
    # ext=os.path.splitext(args.args7)[1]
    # m_string=args.args7
    # if (ext==".nii" or ext==".nrrd"):
    #
    #     Inner_volume=sitk.ReadImage(m_string)
    # else:
    #     Inner_volume=RIM.dicom_series_reader(m_string)

    # spacing=[0.12,0.12,0.12]
    # spacing=input_volume.GetSpacing()
    # origin=input_volume.GetOrigin()
    try:
        myInner=sitk.GetArrayFromImage(Inner_volume)
    except:
        myInner=itk.GetArrayFromImage(Inner_volume)


    del(Inner_volume)

    #######################################################################################################
    roi2=myInner+myFacial
    rmin,rmax,cmin,cmax,zmin,zmax=bbox2_3D(roi2)
    roi3=np.zeros(roi2.shape)
    roi3[rmin:rmax,cmin:cmax,zmin:zmax]=roi2[rmin:rmax,cmin:cmax,zmin:zmax]
    del(roi2)
    del(myFacial)
    del(myInner)
    ########################################################################################################
    #subtract the original image and intact image
    # subtract_array=intact_array_original-input_matrix_original
    subtract_array=intact_array_thr-input_matrix
    sub_res=np.logical_and(intact_array_thr,subtract_array)
    sub_res=np.asarray(sub_res,dtype=float)
    np.asarray(sub_res,dtype=float)
    sub_res[np.where((sub_res == [True]))] = [255.0]


    del(intact_volume2_thr)
    gc.collect()
    del(input_matrix2)
    del(input_matrix_original)
    del(input_volume_thr)
    del(intact_volume)

    # del(intact_array)
    del(subtract_array)
    # sub_res2=np.zeros(input_volume2.shape,dtype=bool)
    #subtract the result from bone
    nrrd_volume_bone_changed=intact_array_thr-sub_res
    sub_res2=np.zeros(input_volume2.shape,dtype=bool)
    rmin_s,rmax_s,cmin_s,cmax_s,zmin_s,zmax_s=bbox2_3D(mysigmoid)
    # if(The_Normal[0]>=0 and The_Normal[1]>=0 and The_Normal[2]>=0):

    r1=rmin_s-1000*abs(The_Normal[0])-10
    r2=rmax_s+1000*abs(The_Normal[0])+10
    c1=cmin_s-1000*abs(The_Normal[1])-10
    c2=cmax_s+1000*abs(The_Normal[1])+10
    z1=zmin_s-1000*abs(The_Normal[2])-10
    z2=zmax_s+1000*abs(The_Normal[2])+10
    if r1<=0:
        r1=0
    if c1<=0:
        c1=0
    if z1<=0:
        z1=0
    if r2>=input_volume2.shape[0]:
        r2=input_volume2.shape[0]
    if c2>=input_volume2.shape[1]:
        c2=input_volume2.shape[1]
    if z2>=input_volume2.shape[2]:
        z2=input_volume2.shape[2]
    sub_res2[r1:r2,c1:c2,z1:z2]=sub_res[r1:r2,c1:c2,z1:z2]

    # rmin_s,rmax_s,cmin_s,cmax_s,zmin_s,zmax_s=bbox2_3D(mysigmoid)
    # if(The_Normal[0]>=0 and The_Normal[1]>=0 and The_Normal[2]>=0):
    # sub_res2[rmin_s-1000*abs(The_Normal[0]):rmax_s+1000*abs(The_Normal[0]),cmin_s-1000*abs(The_Normal[1]):cmax_s+1000*abs(The_Normal[1]),zmin_s-1000*abs(The_Normal[2]):zmax_s+1000*abs(The_Normal[2])]=sub_res[rmin_s-1000*abs(The_Normal[0]):rmax_s+1000*abs(The_Normal[0]),cmin_s-1000*abs(The_Normal[1]):cmax_s+1000*abs(The_Normal[1]),zmin_s-1000*abs(The_Normal[2]):zmax_s+1000*abs(The_Normal[2])]
    # if(The_Normal[0]<=0 and The_Normal[1]<=0 and The_Normal[2]<=0):
    #     sub_res2=sub_res[rmin_s+1000*The_Normal[0]:rmax_s,cmin_s+1000*The_Normal[1]:cmax_s,zmin_s+1000*The_Normal[2]:zmax_s]
    del(intact_array_thr)

    #####################################################
    #apply logical And to get mask of sigmoid sinus

    input_matrix[np.where((input_matrix == [255.0]))] = [1.0]

    #########################################################################
    #find
    region_of_interest_small=np.logical_and(sub_res2,sub_res)
    # region_of_interest=np.logical_and(nrrd_volume_bone_changed,input_matrix)
    # region_of_interest=input_volume2-intact_array_original
    rmin3, rmax3, cmin3, cmax3, zmin3, zmax3=bbox2_3D(region_of_interest_small)
    region_test_center=nrrd_volume_bone_changed[rmin3:rmax3, cmin3:cmax3, zmin3:zmax3]
    # dd_shape=region_test_center.shape[2]
    # hh_shape=region_test_center.shape[1]
    # ww_shape=region_test_center.shape[0]
    #####################################################################
    rmin3_t, rmax3_t, cmin3_t, cmax3_t, zmin3_t, zmax3_t=bbox2_3D(sub_res)
    sub_res_region=nrrd_volume_bone_changed[rmin3_t:rmax3_t, cmin3_t:cmax3_t, zmin3_t:zmax3_t]
    dd_shape=sub_res_region.shape[0]
    hh_shape=sub_res_region.shape[1]
    ww_shape=sub_res_region.shape[2]
    ####################################################################

    ############center of dissected area################################
    center_volume=((dd_shape/2)+rmin3_t,(hh_shape/2)+cmin3_t, (ww_shape/2)+zmin3_t)
    # view_test=((dd_shape/2)+rmin3,(hh_shape/2)+cmin3, (ww_shape/2)+zmin3)
    # CameraEye=(view_test[0],view_test[1]+200, view_test[2])

    ########################################################################
    #************************

    ####################################################################
    # del(region_of_interest_small)
    # del(nrrd_volume)

    del(intact_array_original)
    #######################################################################
    #applying algorithm only on the rectangular part of binary threshold image
    # input_matrix[np.where((input_matrix == [1.0]))] = [255.0]
    del(input_matrix)
    # region_of_interest2=np.zeros(nrrd_volume_bone_changed.shape)
    # region_of_interest2[rmin+Winw:rmax+Win_W,cmin-Win_H:cmax+Win_H,zmin:zmax]=nrrd_volume_bone_changed[rmin+Winw:rmax+Win_W,cmin-Win_H:cmax+Win_H,zmin:zmax]
    # region_of_interest_small2=np.zeros(nrrd_volume_bone_changed.shape)
    # region_of_interest_small2[rmin3:rmax3,cmin3:cmax3,zmin3:zmax3]=nrrd_volume_bone_changed[rmin3:rmax3,cmin3:cmax3,zmin3:zmax3]
    # original_segmented=np.logical_and(region_of_interest2,region_of_interest_small2)
    original_segmented=np.zeros(nrrd_volume_bone_changed.shape)
    original_segmented[rmin3:rmax3, cmin3:cmax3, zmin3:zmax3]=region_test_center
    # region_of_interest2=np.asarray(region_of_interest2,dtype=float)
    # del(region_of_interest_small2)
    d=original_segmented.shape[0]
    h=original_segmented.shape[1]
    w=original_segmented.shape[2]
    original_segmented=np.asarray(original_segmented,dtype=float)
    original_segmented[np.where((original_segmented == [1.0]))] = [255.0]

    original_segmented_filled=np.zeros((original_segmented.shape[0],original_segmented.shape[1],original_segmented.shape[2]))
    ##############################################################################################################
    roi2=np.logical_and(roi3,original_segmented)
    del(original_segmented)
    try:
        rmin,rmax,cmin,cmax,zmin,zmax=bbox2_3D(roi2)
    except:
        print("one region will be searched")
        rmin=0
        rmax=0
        cmin=0
        cmax=0
        zmin=0
        zmax=0
    del(roi2)
    region2=np.zeros(nrrd_volume_bone_changed.shape)
    region1=np.zeros(nrrd_volume_bone_changed.shape)
    region2[rmin:rmax,cmin:cmax,zmin:zmax]=nrrd_volume_bone_changed[rmin:rmax,cmin:cmax,zmin:zmax]
    region1[rmin3:rmax3, cmin3:cmax3, zmin3:zmax3]=nrrd_volume_bone_changed[rmin3:rmax3, cmin3:cmax3, zmin3:zmax3]
    # del(nrrd_volume_bone_changed)
    del(mysigmoid)
    del(myOssicles)
    del(input_volume)
    # del(input_volume2)
    del(region_test_center)
    original_segmented_filled=region1-region2
    # del(region2)

    rmin_f,rmax_f,cmin_f,cmax_f,zmin_f,zmax_f=bbox2_3D(original_segmented_filled)
    ##############################################################################################################
    original_segmented_filled=original_segmented_filled[rmin_f:rmax_f,cmin_f:cmax_f,zmin_f:zmax_f]

    d_shape=original_segmented_filled.shape[0]
    h_shape=original_segmented_filled.shape[1]
    w_shape=original_segmented_filled.shape[2]

    ############considering camera eye in the direction of the center of volume to the center of dissected area
    Center=center_volume
    # volume_center=(int(d1)/2,int(h1)/2,int(w1)/2)
    # direction_vector=glm.vec3(Center)-glm.vec3(volume_center)
    # CameraEye2=(glm.vec3(Center)+glm.vec3(direction_vector))
    # CameraEye2=(CameraEye2[0]-rmin3,CameraEye2[1]-cmin3, CameraEye2[2]-zmin3)

    ###############################considering cameraeye as the point in the direction of the normal from the center of dissected area################################3
    Camera1=(glm.vec3(Center)-500*(glm.vec3(The_Normal)))
    Camera2=(glm.vec3(Center)+500*(glm.vec3(The_Normal)))
    # Vector_test=glm.vec3(Center)-glm.vec3(p1)
    # theta1=math.acos(glm.dot(Camera1,glm.vec3(Vector_test))/(glm.length(Camera1)*glm.length(Vector_test)))
    # theta2=math.acos(glm.dot(Camera2,glm.vec3(Vector_test))/(glm.length(Camera2)*glm.length(Vector_test)))
    # print("theta1")
    # print(theta1)
    # print("theta2")
    # print(theta2)
    # if theta1>theta2:
    #     CameraEye3=Camera2
    # else:
    #     CameraEye3=Camera1
    CameraEye3=Camera1
    print(Center)
    print(CameraEye3)
    CameraEye3=(CameraEye3[0]-rmin_f,CameraEye3[1]-cmin_f, CameraEye3[2]-zmin_f)

    center_volume2=(int(dd_shape/2)+rmin3,int(hh_shape/2)+cmin3,int(ww_shape/2)+zmin3)
    center_volume2=(center_volume2[0]-rmin_f,center_volume2[1]-cmin_f,center_volume2[2]-zmin_f)
    startTime = time.time()

    # t_list1,rgb_image,grid1,grid2=prc.main3_editted(original_segmented_filled,d_shape,h_shape,w_shape,CameraEye3,center_volume2)
    t_list1,rgb_image,grid1,grid2=prc.main3_editted(original_segmented_filled,d_shape,h_shape,w_shape,CameraEye3,center_volume2)
####################################################################################################################################

    #find the the index of the voxels exposed to overhang problem for both volumes
    final_list_1=eva.evaluate2(t_list1)


    final_list_1=list(filter(lambda a: a!=[],final_list_1))

    # final_list=filter(lambda a: a==Vertex,final_list)
    final_list2=final_list_1
    final_list2=np.asarray(final_list2)
    k=0
    ff=[]
    for i in range(0,np.asarray(final_list_1.__len__())):
        for j in range(0,np.asarray(final_list_1[i]).__len__()):
            ff.append(final_list2[i][j])
            k=k+1
    final_list_1=ff




    input_volume3=input_volume2[rmin3:rmax3,cmin3:cmax3,zmin3:zmax3]
    blobs_1=np.zeros(input_volume3.shape,dtype=bool)
    for k in range(0,final_list_1.__len__()):


            blobs_1[(int)(final_list_1[k][0])][(int)(final_list_1[k][1])][(int)(final_list_1[k][2])]=True

    blobs2[(int)(rmin3):(int)(rmax3),(int)(cmin3):(int)(cmax3),int(zmin3):int(zmax3)]=blobs_1
    print(np.sum(blobs2))

    blobs2=np.logical_and(blobs2,sub_res)
    # im=input_matrix
    # im = input_volume2


    final_number=np.sum(blobs2)
    print(final_number)
    if final_number>200:
        print(final_number)
    #     print("overhang problem has occured")
    #     res1=0
    #     res2=0
    #
    # else:
    #     print("No overhang detected")
    #     res1=1
    ##############################################################
    #******write the final volume with the colored voxel******
    # x=args.args0.split("\\")
    # write_string=os.path.join("D:\Results"+ "\\" +x[-2]+"_"+x[-1]+"_"+"overhang"+ "." + "nrrd")
    # write_string=str(("D:\Results\label_3d_9_3_oct_2_test_bigger_no.nrrd"))
    os.path.split(dis_volume_add)[0]
    x1=dis_volume_add.split("/")
    x2=dis_volume_add.split("\\")
    if x1.__len__()>x2.__len__():
        x=x1
    else:
        x=x2

    # print(x[-2]+"\\"+x[-1])
    write_string=os.path.join(".\\"+ "\\" +x[-2]+"_"+x[-1]+"_"+"overhang_Label"+ "." + "nrrd")

    gc.collect()
    blobs2=np.asarray(blobs2,dtype=float)

    blobs2[np.where((blobs2 == [1.0]))] = [255.0]
    output_volume=sitk.GetImageFromArray(blobs2)

    output_volume=sitk.Cast(output_volume,sitk.sitkFloat64)
    output_volume.SetSpacing(spacing)
    print(spacing)

    sitk.WriteImage(output_volume,write_string)
    res=cluster.main(write_string)

    f= open("Karname.txt","a+")
    f.write("over hang in sigmoid sinus      %d\r\n" %(res))
    if res==0:
        f= open("Karname.txt","a+")
        f.write("compelete Saucerization      %d\r\n" %(res))
        sys.exit()
    elif res==1:
        f= open("Karname.txt","a+")
        f.write("over hang in sigmoid sinus      %d\r\n" %(res))
#####################################################################################################
        del(grid1)
    del(grid2)
    del(rgb_image)
    del(roi3)
    test2_region=region2
    # test2_region=sub_res-(region1-region2)
    del(original_segmented_filled)
    del(region1)
    # test2_region=sub_res-np.logical_and(sub_res,region_of_interest_small)
    del(region_of_interest_small)
    try:
        rmin2,rmax2,cmin2,cmax2,zmin2,zmax2=bbox2_3D(test2_region)
    except:
        rmin2=0
        rmax2=0
        cmin2=0
        cmax2=0
        zmax2=0
        zmin2=0
    test2_region=nrrd_volume_bone_changed[rmin2:rmax2,cmin2:cmax2,zmin2:zmax2]
    del(nrrd_volume_bone_changed)
    try:


        t_list2,rgb_image,grid1,grid2=prc.main3_editted(test2_region,test2_region.shape[0],test2_region.shape[1],test2_region.shape[2],center_volume2,center_ossicles)
        del(grid1)
        del(grid2)
        del(rgb_image)
    except:
        t_list2=0
####################################################################



    try:
        #find the the index of the voxels exposed to overhang problem for both volumes
        final_list_2=eva.evaluate2(t_list2)


        final_list_2=list(filter(lambda a: a!=[],final_list_2))

        # final_list=filter(lambda a: a==Vertex,final_list)
        final_list2=final_list_2
        final_list2=np.asarray(final_list2)
        k=0
        ff=[]
        for i in range(0,np.asarray(final_list_2.__len__())):
            for j in range(0,np.asarray(final_list_2[i]).__len__()):
                ff.append(final_list2[i][j])
                k=k+1
        final_list_2=ff
    except:
        final_list_2=0





    print ("Execution time: %8.2f seconds." % (time.time() - startTime))

    blobs_2=np.zeros((input_volume2[rmin2:rmax2,cmin2:cmax2,zmin2:zmax2]).shape,dtype=bool)





    try:
        for k in range(0,final_list_2.__len__()):


                blobs_2[(int)(final_list_2[k][0])][(int)(final_list_2[k][1])][(int)(final_list_2[k][2])]=True

    except:
        blobs_2=0

    # input_volume2[rmin3:rmax3,cmin3:cmax3,zmin3:zmax3]=input_volume3
    # input_volume2=input_volume3

    # if blobs_2!=0:
    blobsTwo[(int)(rmin2):(int)(rmax2),(int)(cmin2):(int)(cmax2),int(zmin2):int(zmax2)]=blobs_2
    blobs2=blobsTwo
    # bolbs2=np.logical_or(blobs2,blobsTwo)
    blobs2=np.logical_and(blobs2,sub_res)

    #
    ################################################################
    ##label saving
    ##label saving
    gc.collect()

    del(sub_res)

    # del(blobs)
    del(final_list2)
    del(final_list_1)
    # del(original_segmented)
    # del(original_segmented_filled)
    # del(grid1)
    # del(grid2)
    # del(rgb_image)
    del(t_list1)
    del(t_list2)
    # del(t_listf)
    # del(im)

    del(input_volume3)
    blobs2=np.asarray(blobs2,dtype=float)

    blobs2[np.where((blobs2 == [1.0]))] = [255.0]
    # os.path.split(dis_volume_add)[0]
    x1=dis_volume_add.split("/")
    x2=dis_volume_add.split("\\")
    if x1.__len__()>x2.__len__():
        x=x1
    else:
        x=x2

    # print(x[-2]+"\\"+x[-1])
    write_string=os.path.join(".\\"+ "\\" +x[-2]+"_"+x[-1]+"_"+"Saucerization_Label"+ "." + "nrrd")





    # del(region_of_interest2)
    #
    # del(region_of_interest_small2)


    gc.collect()

    output_volume=sitk.GetImageFromArray(blobs2)

    output_volume=sitk.Cast(output_volume,sitk.sitkFloat64)
    output_volume.SetSpacing(spacing)
    print(spacing)

    sitk.WriteImage(output_volume,write_string)

    ###############################################################
    #******write the final volume with the colored voxel******
    # write_string2=str("\\\\samba.cs.ucalgary.ca\\fatemeh.yazdanbakhsh\Documents\medical_imaging\\results\Totlavolume_3D_9_3_oct_19_1_test.nrrd")
    # output_volume=np.zeros((d,h,w))
    # output_volume=sitk.GetImageFromArray(input_volume2)
    #
    # output_volume=sitk.Cast(output_volume,sitk.sitkFloat64)
    # output_volume.SetSpacing(np.asarray(spacing))
    #
    # sitk.WriteImage(output_volume,write_string2)
    try:
        res2=cluster.main(write_string)
    except:
        res2=1
    f= open("Karname.txt","a+")
    f.write("Complete Saucerization      %d\r\n" %(res))

    return res1,res2


    ####################################################################
