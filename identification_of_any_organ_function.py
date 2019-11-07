import SimpleITK as sitk
# import read_image_m as RIM
# import os
#
#
# import argparse
# import pymesh
import scipy.ndimage.morphology as morph
# from skimage.measure import marching_cubes_lewiner
# from mayavi import mlab
# from scipy.spatial.distance import cdist
import numpy as np
import itk
def bbox2_3D(img):

    r = np.any(img, axis=(1, 2))
    c = np.any(img, axis=(0, 2))
    z = np.any(img, axis=(0, 1))

    rmin, rmax = np.where(r)[0][[0, -1]]
    cmin, cmax = np.where(c)[0][[0, -1]]
    zmin, zmax = np.where(z)[0][[0, -1]]

    return rmin, rmax, cmin, cmax, zmin, zmax
ress=0
def main(diss_volume,inta_volume,diga_volume):
    # parser = argparse.ArgumentParser(
    #     description = """This program uses ray casting method to detect overhang problem""")
    # parser.add_argument("-args0", type = str, default = (("D:\evan_campton\\1601L_FLIPPED_CROPPED_EDITED\\1601L_FLIPPED_CROPPED_EDITED\SEGMENTATIONS\edited\FacialNerve.nrrd")), help = "facial nerve")
    # # parser.add_argument("-args0", type = str, default = "U:\Documents\Data_Sets\Calgary\TBone-2015\TBoneCBCT-2015-10\L3016_modified_19_nov", help = "dicome image address")
    # parser.add_argument("-args1", type = str, default = ( "D:\evan_campton\\1601L_FLIPPED_CROPPED_EDITED\\1601L_FLIPPED_CROPPED_EDITED\SEGMENTATIONS\edited\SigmoidSinus.nrrd"),
    #     help = "address of sigmoid sinus mask")
    # parser.add_argument("-args2", type = str, default = (('D:\evan_campton\JL_1_POSTFINAL')), help = "dissected image address")
    # parser.add_argument("-args3",type=str, default=('D:\evan_campton\\1601L_FLIPPED_CROPPED_EDITED\\1601L_FLIPPED_CROPPED_EDITED\IMAGES\\1601L_FLIPPED_CROPPED_EDITED'),help="intact image address")
    # parser.add_argument("-args4", type = int, default = 1000,
    #     help = "low")
    # parser.add_argument("-args5", type = int, default = 4000,
    #     help = "high")
    # parser.add_argument("-args6",type=str, default=('D:\evan_campton\\1601L_FLIPPED_CROPPED_EDITED\\1601L_FLIPPED_CROPPED_EDITED\SEGMENTATIONS\edited\SigmoidSinus.nrrd'),help="intact image address")
    #
    # # Get your arguments
    # args = parser.parse_args()

    # low=Low
    # high=High

    ################################################Reading Dissected Volume##########################################################################

    ########################################################################################################################################
    def finish():
        print('\n', "bye", '\n')
        input('Press Enter to quit: ')
    ################################################Reading Dissected Volume##########################################################################

    # read the original volume
    # ext=os.path.splitext(args.args2)[1]
    # m_string=args.args2
    # if (ext==".nii" or ext==".nrrd"):
    #
    #     input_volume=sitk.ReadImage(m_string)
    # else:
    #     input_volume=RIM.dicom_series_reader(m_string)


    input_volume=diss_volume
    spacing=input_volume.GetSpacing()
    origin=input_volume.GetOrigin()
    try:
        dissected_matrix=sitk.GetArrayFromImage(input_volume)
    except:
        dissected_matrix=itk.GetArrayFromImage(input_volume)
    w1 = dissected_matrix.shape[2]
    h1 = dissected_matrix.shape[1]
    d1 = dissected_matrix.shape[0]


    #############################Reading Intact Volume################################
    intact_volume=inta_volume

    # ext=os.path.splitext(str(( args.args3)))[1]
    # m_string3=str(( args.args3))
    # if (ext==".nii" or ext==".nrrd" or ext==".nhdr"):
    try:
        # intact_volume=sitk.ReadImage(m_string3)
        intact_array=sitk.GetArrayFromImage(intact_volume)
    except:
        # intact_volume=RIM.dicom_series_reader(m_string3)
        intact_array=itk.GetArrayFromImage(intact_volume)
    # intact_volume=RIM.dicom_series_reader(str(unicode('\\\\samba.cs.ucalgary.ca\\fatemeh.yazdanbakhsh\Documents\Data_Sets\Calgary\TBone-2015\TBoneCBCT-2015-10\L2963L','utf-8')))

    #######################################################################################################
    #

    # #do binary threshoulding on the original image
    #
    # PixelType = itk.ctype('signed short')
    # Dimension = 3
    # try:
    #     thresholdFilter= sitk.BinaryThresholdImageFilter()
    #     input_volume_thr=thresholdFilter.Execute(input_volume,low,high,255,0)
    # except:print(0)
    # try:
    #     ImageType_threshold = itk.Image[PixelType, Dimension]
    #     thresholdFilter= itk.BinaryThresholdImageFilter[ImageType_threshold,ImageType_threshold].New()
    #     # input_volume=thresholdFilter.Execute(input_volume,low,high,0,255)
    #     thresholdFilter.SetInput((input_volume))
    #
    #     thresholdFilter.SetLowerThreshold(low)
    #     thresholdFilter.SetUpperThreshold(high)
    #     thresholdFilter.SetOutsideValue(0)
    #     thresholdFilter.SetInsideValue(255)
    #     thresholdFilter.Update()
    #     input_volume_thr=thresholdFilter.GetOutput()
    #
    # except:print(0)

    #####################################################

    # #do binary threshoulding on the intact image
    # try:
    #     thresholdFilter= sitk.BinaryThresholdImageFilter()
    #     intact_volume2_thr=thresholdFilter.Execute(intact_volume,low,high,255,0)
    # except:print(0)
    # #
    # try:
    #     PixelType = itk.ctype('signed short')
    #     Dimension = 3
    #     ImageType_threshold = itk.Image[PixelType, Dimension]
    #     thresholdFilter= itk.BinaryThresholdImageFilter[ImageType_threshold,ImageType_threshold].New()
    #     thresholdFilter.SetInput(intact_volume)
    #
    #     thresholdFilter.SetLowerThreshold(low)
    #     thresholdFilter.SetUpperThreshold(high)
    #     thresholdFilter.SetOutsideValue(0)
    #     thresholdFilter.SetInsideValue(255)
    #     thresholdFilter.Update()
    #     intact_volume2_thr=thresholdFilter.GetOutput()
    #
    # except:print(0)
    # #intact_array=itk.GetArrayFromImage(intact_volume2)
    # try:
    #     intact_array_thr=sitk.GetArrayFromImage(intact_volume2_thr)
    # except:print(0)
    # try:
    #     intact_array_thr=itk.GetArrayFromImage(intact_volume2_thr)
    # except:print(0)
    #
    #
    #
    # #####################################################
    # try:
    #     dissected_array_thr = sitk.GetArrayFromImage(input_volume_thr)
    # except:print(0)
    # try:
    #     dissected_array_thr = itk.GetArrayFromImage(input_volume_thr)
    # except:print(0)







    ######################################Read digastric Ridge volume#################################

    # # read digastric ridge segmentaion
    # ext=os.path.splitext(args.args6)[1]
    # m_string=args.args6
    # if (ext==".nii" or ext==".nrrd"):
    #
    #     digastric_volume=sitk.ReadImage(m_string)
    # else:
    #     digastric_volume=RIM.dicom_series_reader(m_string)


    digastric_volume=diga_volume
    try:
        digastric_matrix=sitk.GetArrayFromImage(digastric_volume)
    except:
        digastric_matrix=itk.GetArrayFromImage(digastric_volume)


    rmin_o,rmax_o,cmin_o,cmax_o,zmin_o,zmax_o=bbox2_3D(digastric_matrix)

    ###################expand digastric ridge ##################################
    #expanding digastric ridge using morphology dilation
    a=np.ones((3,3,3))

    expanded_digastrix=morph.binary_dilation(digastric_matrix,a,1)
    ##calculate the difference of the expanded version and original version of digastric ridge
    diff_digastric=expanded_digastrix-digastric_matrix

    ###calculate the voxels around digastric ridge in the dissected image and original image
    A_ridge=np.logical_and(diff_digastric,dissected_matrix)
    B_ridge=np.logical_and(diff_digastric,intact_array)
    #########calculate the voxels around the digastric ridge that have been changed
    diff_A_B=np.asarray(B_ridge,dtype=int)-np.asarray(A_ridge,dtype=int)
    # print(diff_A_B)
    try:
        rmin4,rmax4,cmin4,cmax4,zmin4,zmax4=bbox2_3D(diff_A_B)
    except:
        print("the organ is not identified")
        ress=0
        return ress
        # finish()

    proportion=len(diff_A_B[rmin4:rmax4,cmin4:cmax4,zmin4:zmax4])/len(digastric_matrix[rmin_o:rmax_o,cmin_o:cmax_o,zmin_o:zmax_o])
    print("portion")
    print(proportion)
    if(proportion>0.2):
        print("the organ is identified")
        ress=1
        # print("wait for calculating whether it is cutt away inadvertantly")
    else:
        print("the organ is not identified")
        ress=0
    # if(np.abs(np.sum(np.asarray(np.logical_and(digastric_matrix,dissected_matrix),dtype=int)-np.asarray(np.logical_and(digastric_matrix,intact_array),dtype=int)))>10):
    #     print("the organ is cut away")
    #     ress=0
    return ress






