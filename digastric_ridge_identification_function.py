import SimpleITK as sitk
import read_image_m as RIM
import os


import argparse
# import pymesh
import scipy.ndimage.morphology as morph
from skimage.measure import marching_cubes_lewiner
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
def computeQualityMeasures(lP,lT):
    quality=dict()
    labelPred=sitk.GetImageFromArray(lP)
    labelTrue=sitk.GetImageFromArray(lT)
    hausdorffcomputer=sitk.HausdorffDistanceImageFilter()
    hausdorffcomputer.Execute(labelTrue>0.5,labelPred>0.5)
    quality["avgHausdorff"]=hausdorffcomputer.GetAverageHausdorffDistance()
    quality["Hausdorff"]=hausdorffcomputer.GetHausdorffDistance()

    dicecomputer=sitk.LabelOverlapMeasuresImageFilter()
    dicecomputer.Execute(labelTrue>0.5,labelPred>0.5)
    quality["dice"]=dicecomputer.GetDiceCoefficient()

    return quality

def main(args0,args1,args2,args3):
    parser = argparse.ArgumentParser(
        description = """This program uses ray casting method to detect overhang problem""")
    parser.add_argument("-args0", type = str, default = (("\\\\samba.cs.ucalgary.ca\\fatemeh.yazdanbakhsh\Documents\Data_Sets\Kowther\Specimen2501L\Specimen2501L\Segmentations\FacialNerve.nrrd")), help = "facial nerve")
    # parser.add_argument("-args0", type = str, default = "U:\Documents\Data_Sets\Calgary\TBone-2015\TBoneCBCT-2015-10\L3016_modified_19_nov", help = "dicome image address")
    parser.add_argument("-args1", type = str, default = ( "\\\\samba.cs.ucalgary.ca\\fatemeh.yazdanbakhsh\Documents\Data_Sets\Kowther\Specimen2501L\Specimen2501L\Segmentations\SigmoidSinus.nrrd"),
        help = "address of sigmoid sinus mask")
    parser.add_argument("-args2", type = str, default = (('\\\\samba.cs.ucalgary.ca\\fatemeh.yazdanbakhsh\Documents\Data_Sets\Kowther\Specimen2501L\dissected_27_feb_2019')), help = "dissected image address")
    parser.add_argument("-args3",type=str, default=('\\\\samba.cs.ucalgary.ca\\fatemeh.yazdanbakhsh\Documents\Data_Sets\Kowther\Specimen2501L\Specimen2501L\\2501L_reduced'),help="intact image address")
    parser.add_argument("-args4", type = int, default = 1000,
        help = "low")
    parser.add_argument("-args5", type = int, default = 4000,
        help = "high")

    # Get your arguments
    args = parser.parse_args()
    args.args0=args0
    args.args1=args1
    args.args2=args2
    args.args3=args3
    low=args.args4
    high=args.args5

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




    # read facialnerve
    ext=os.path.splitext(args.args0)[1]
    m_string=args.args0
    if (ext==".nii" or ext==".nrrd"):

        facial_volume=sitk.ReadImage(m_string)
    else:
        facial_volume=RIM.dicom_series_reader(m_string)

    facial_volume=sitk.GetArrayFromImage(facial_volume)
    facial_volume=np.ndarray.transpose(np.ndarray.transpose(facial_volume))


    ##########################chebyshev distance###########################
    # allDist = squareform( pdist2( set1, set2 ) );
    # [minDist nni] = min( allDist, [], 2 );
    ###########python version#################

    #
    # X = sigmoid_volume
    # Y = facial_volume
    #
    # allDist=cdist(X, Y)
    # distancee=min(allDist)
    mesh=sigmoid_volume
    pts=facial_volume
    # squared_distances,face_indices,closest_point=pymesh.distance_to_mesh(mesh, pts, engine='auto')
    # print(squared_distances)
    # print(face_indices)
    # print(closest_point)
    #######################################################################

    # quality=computeQualityMeasures(facial_volume,sigmoid_volume)
    # print(quality)
    a=np.ones((3,3,3))
    for i in range(10,50):
        b=morph.binary_dilation(sigmoid_volume,a,i)
        c=morph.binary_dilation(facial_volume,a,i)
        d=np.logical_and(b,c)
        d[np.where((d == [True]))] = [1.0]
        if np.sum(d)>10000:
            print(np.sum(d))

            print(i)
            break
    print(np.where((d == [1.0])))

    sum_image=b+c
    #############visualizing the extended sigmoid sinus and facial nerve
    verts, faces, normals, values = marching_cubes_lewiner(sum_image, 0, spacing)
    # mesh=mlab.triangular_mesh([vert[0] for vert in verts],
    #                              [vert[1] for vert in verts],
    #                              [vert[2] for vert in verts],
    #                              faces)
    # fig = mlab.figure(1)
    # mlab.show()
    #################calculate the bounding box of the intersection point
    d[np.where((d == [True]))] = [1.0]
    rmin3, rmax3, cmin3, cmax3, zmin3, zmax3=bbox2_3D(d)
    original_sum=sigmoid_volume+facial_volume
    original_sum[np.where((original_sum == [True]))] = [1.0]
    ##################extracting region of interest
    segmented_area=dissected_array_thr[rmin3:rmax3,cmin3:cmax3,zmin3:zmax3]
    print(segmented_area.shape)
    verts, faces, normals, values = marching_cubes_lewiner(segmented_area, 0, spacing)
    # mesh=mlab.triangular_mesh([vert[0] for vert in verts],
    #                              [vert[1] for vert in verts],
    #                              [vert[2] for vert in verts],
    #                              faces)
    # mlab.title("part of dissected")
    # fig = mlab.figure("part of dissected")


    segmented_area=intact_array_thr[rmin3:rmax3,cmin3:cmax3,zmin3:zmax3]
    print(segmented_area.shape)
    verts, faces, normals, values = marching_cubes_lewiner(segmented_area, 0, spacing)
    # mesh=mlab.triangular_mesh([vert[0] for vert in verts],
    #                              [vert[1] for vert in verts],
    #                              [vert[2] for vert in verts],
    #                             faces)
    # mlab.title("part of intact")
    # fig = mlab.figure("part of intact")

    segmented_area=dissected_array_thr[rmin3:rmax3,cmin3:cmax3,zmin3:zmax3]
    print(segmented_area.shape)
    verts, faces, normals, values = marching_cubes_lewiner(dissected_array_thr, 0, spacing)
    # mesh=mlab.triangular_mesh([vert[0] for vert in verts],
    #                              [vert[1] for vert in verts],
    #                              [vert[2] for vert in verts],
    #                             faces)
    # mlab.title('complete image')
    # fig=mlab.figure('complete image')
    # mlab.show()


    ##############################results####################################
    subtracted_array=intact_array_thr-dissected_array_thr

    subtracted_region=subtracted_array[rmin3:rmax3,cmin3:cmax3,zmin3:zmax3]
    num1=np.sum(subtracted_region)
    print(num1)
    if(num1>0):
        print("digastric ridge is identified")
    else:
        print("digastric ridge is not identified")


    print("rmin")
    print(rmin3)
    print("rmax")
    print(rmax3)
    print("cmin")
    print(cmin3)
    print("cmax")
    print(cmax3)
    print("zmin")
    print(zmin3)
    print("zmax")
    print(zmax3)

    return rmin3,rmax3,cmin3,cmax3,zmin3,zmax3
