import SimpleITK as sitk
import read_image_m as RIM
import os


import argparse
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

parser = argparse.ArgumentParser(
    description = """This program uses ray casting method to detect overhang problem""")
parser.add_argument("-args0", type = str, default = (("D:\evan_campton\\1601L_FLIPPED_CROPPED_EDITED\\1601L_FLIPPED_CROPPED_EDITED\SEGMENTATIONS\edited\FacialNerve.nrrd")), help = "facial nerve")
# parser.add_argument("-args0", type = str, default = "U:\Documents\Data_Sets\Calgary\TBone-2015\TBoneCBCT-2015-10\L3016_modified_19_nov", help = "dicome image address")
parser.add_argument("-args1", type = str, default = ( "D:\evan_campton\\1601L_FLIPPED_CROPPED_EDITED\\1601L_FLIPPED_CROPPED_EDITED\SEGMENTATIONS\edited\SigmoidSinus.nrrd"),
    help = "address of sigmoid sinus mask")
parser.add_argument("-args2", type = str, default = (('D:\evan_campton\JL_1_POSTFINAL')), help = "dissected image address")
parser.add_argument("-args3",type=str, default=('D:\evan_campton\\1601L_FLIPPED_CROPPED_EDITED\\1601L_FLIPPED_CROPPED_EDITED\IMAGES\\1601L_FLIPPED_CROPPED_EDITED'),help="intact image address")
parser.add_argument("-args4", type = int, default = 1000,
    help = "low")
parser.add_argument("-args5", type = int, default = 4000,
    help = "high")
parser.add_argument("-args6",type=str, default=('D:\evan_campton\\1601L_FLIPPED_CROPPED_EDITED\\1601L_FLIPPED_CROPPED_EDITED\SEGMENTATIONS\edited\InnerEar.nrrd'),help="organ to check skeletonization")

# Get your arguments
args = parser.parse_args()

low=args.args4
high=args.args5
########################################################################################################################################
def finish():
    print('\n', "bye", '\n')
    input('Press Enter to quit: ')
################################################Reading Dissected Volume##########################################################################


# read the original volume
ext=os.path.splitext(args.args2)[1]
m_string=args.args2
if (ext==".nii" or ext==".nrrd"):

    input_volume=sitk.ReadImage(m_string)
else:
    input_volume=RIM.dicom_series_reader(m_string)



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


ext=os.path.splitext(str(( args.args3)))[1]
m_string3=str(( args.args3))
if (ext==".nii" or ext==".nrrd" or ext==".nhdr"):

    intact_volume=sitk.ReadImage(m_string3)
    intact_array=sitk.GetArrayFromImage(intact_volume)
else:
    intact_volume=RIM.dicom_series_reader(m_string3)
    intact_array=itk.GetArrayFromImage(intact_volume)
# intact_volume=RIM.dicom_series_reader(str(unicode('\\\\samba.cs.ucalgary.ca\\fatemeh.yazdanbakhsh\Documents\Data_Sets\Calgary\TBone-2015\TBoneCBCT-2015-10\L2963L','utf-8')))

#######################################################################################################
#

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
    thresholdFilter.SetInput((input_volume))

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
    dissected_array_thr = sitk.GetArrayFromImage(input_volume_thr)
except:print(0)
try:
    dissected_array_thr = itk.GetArrayFromImage(input_volume_thr)
except:print(0)




#######################################################################################################

# read nrrd or segmented volume of sigmoid sinus to be used as mask
ext=os.path.splitext(args.args1)[1]
m_string2=args.args1
if (ext==".nii" or ext==".nrrd"):

    sigmoid_volume=sitk.ReadImage(m_string2)
else:
    sigmoid_volume=RIM.dicom_series_reader(m_string2)
# nrrd_volume=sitk.ReadImage(m_string2)

spacing=sigmoid_volume.GetSpacing()
sigmoid_volume=sitk.GetArrayFromImage(sigmoid_volume)
sigmoid_volume=np.ndarray.transpose(np.ndarray.transpose(sigmoid_volume))




############################################ read facialnerve#################################################
#####reading facial nerve to be used as mask
ext=os.path.splitext(args.args0)[1]
m_string=args.args0
if (ext==".nii" or ext==".nrrd"):

    facial_volume=sitk.ReadImage(m_string)
else:
    facial_volume=RIM.dicom_series_reader(m_string)

facial_volume=sitk.GetArrayFromImage(facial_volume)
facial_volume=np.ndarray.transpose(np.ndarray.transpose(facial_volume))



######################################Read digastric Ridge volume#################################

# read digastric ridge segmentaion
ext=os.path.splitext(args.args6)[1]
m_string=args.args6
if (ext==".nii" or ext==".nrrd"):

    digastric_volume=sitk.ReadImage(m_string)
else:
    digastric_volume=RIM.dicom_series_reader(m_string)



try:
    digastric_matrix=sitk.GetArrayFromImage(digastric_volume)
except:
    digastric_matrix=itk.GetArrayFromImage(digastric_volume)
rmin,rmax,cmin,cmax,zmin,zmax=bbox2_3D(digastric_matrix)
#############################################################################
diff_image=intact_array_thr-dissected_array_thr
rmind,rmaxd,cmind,cmaxd,zmind,zmaxd=bbox2_3D(diff_image)
#########################################################
#extend the organ by 1 milimeter
pixel_number=1/spacing[0]
a=np.ones((3,3,3))
expanded_digastrix=morph.binary_dilation(digastric_matrix,a,int(pixel_number))
#calculate bounding box of the extended organ
rmino,rmaxo,cmino,cmaxo,zmino,zmaxo=bbox2_3D(expanded_digastrix)
intersection_area=np.logical_and(diff_image,expanded_digastrix)
#calculate bounding box of the intersection area
try:
    rmini,rmaxi,cmini,cmaxi,zmini,zmaxi=bbox2_3D(intersection_area)
except:
    print("the organ is not skeletonized")
    rmini=0
    rmaxi=0
    cmin=0
    cmax=0
    zmin=0
    zmax=0
    finish()

proportion1=len(digastric_matrix[rmin:rmax,cmin:cmax,zmin:zmax])/len(diff_image[rmind:rmaxd,cmind:cmaxd,zmind:zmaxd])
proportion2=len(intersection_area[rmini:rmaxi,cmini:cmaxi,zmini:zmaxi])/len(digastric_matrix[rmin:rmax,cmin:cmax,zmin:zmax])
proportion3=len(intersection_area[rmini:rmaxi,cmini:cmaxi,zmini:zmaxi])/len(diff_image[rmind:rmaxd,cmind:cmaxd,zmind:zmaxd])
###############################################################################################################################
print("proportion")
print(proportion3)
if proportion3>0.2:
    print("the organ is skeletonize")
else:
    print("the organ is not skeletonize")
#############################################################################################################################





