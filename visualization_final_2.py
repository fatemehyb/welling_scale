
#This program performs surface and volume rendering for a dataset.
import vtk
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk
import argparse
import numpy as np

#Gaussian filter
def gaussianFilter(image):

    filter = vtk.vtkImageGaussianSmooth()

    filter.SetInputConnection(image.GetOutputPort())
    filter.SetStandardDeviation(1)
    filter.SetRadiusFactors(1,1,1)
    filter.SetDimensionality(3)
    filter.Update()

    return filter.GetOutput()

# Thresholding
def threshold(image, lowerThreshold, upperThreshold):

    thresh = vtk.vtkImageThreshold()
    thresh.SetInputData(image)

    thresh.ThresholdBetween(lowerThreshold, upperThreshold)
    thresh.ReplaceInOn()
    thresh.SetInValue(1)
    thresh.ReplaceOutOn()
    thresh.SetOutValue(0)
    thresh.Update()

    return thresh.GetOutput()

# Creates mesh using marching cubes for an image
def createMesh(mask, threshold1,threshold2):

    mesh = vtk.vtkMarchingCubes()
    mesh.SetInputData(mask)
    mesh.SetValue(threshold1, threshold2)
    mesh.Update()

    return mesh.GetOutput()

# Creates mesh for a thresholded image
def createThresholdMesh(mask,threshold1,threshold2):

    mesh = vtk.vtkDiscreteMarchingCubes()
    mesh.SetInputData(mask)
    mesh.SetValue(threshold1,threshold2)
    # mesh.GenerateValues(threshold2-threshold1+1,threshold1,threshold2)
    mesh.Update()

    return mesh.GetOutput()

def createThresholdMesh2(mask,threshold1,threshold2):

    mesh = vtk.vtkDiscreteMarchingCubes()
    mesh.SetInputData(mask)
    # mesh.SetValue(threshold1,threshold2)
    mesh.GenerateValues(threshold2-threshold1+1,threshold1,threshold2)
    mesh.Update()

    return mesh.GetOutput()
# Thresholds an image and renders the surface
def surfaceRendering(image, lowerThreshold, upperThreshold):

    ren = vtk.vtkRenderer()
    renWin = vtk.vtkRenderWindow()
    renWin.AddRenderer(ren)
    renWin.PolygonSmoothingOn()
    renWin.SetSize(400,400)

    # create a renderwindowinteractor
    iren = vtk.vtkRenderWindowInteractor()
    iren.SetRenderWindow(renWin)

    # mask = threshold(image, lowerThreshold, upperThreshold)
    # mesh = createThresholdMesh(mask)
    mesh = createMesh(image, 255)


    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(mesh)
    mapper.ScalarVisibilityOff()

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetDiffuseColor(0.7,0.7,0.7)
    actor.GetProperty().SetSpecular(0.3)
    actor.GetProperty().SetAmbient(0.2)
    actor.GetProperty().SetDiffuse(0.7)


    ren.AddActor(actor)

    iren.Initialize()
    renWin.Render()
    iren.Start()

# Extracts two isosurfaces representing skin and bone
def dualSurfaceRendering(image1,image2):

    ren = vtk.vtkRenderer()
    renWin = vtk.vtkRenderWindow()
    renWin.AddRenderer(ren)
    renWin.PolygonSmoothingOn()
    renWin.SetSize(400,400)

    # create a renderwindowinteractor
    iren = vtk.vtkRenderWindowInteractor()
    iren.SetRenderWindow(renWin)

    # skinMesh = createMesh(image1,(int)(image1.GetScalarRange()[0]),(int)(image1.GetScalarRange()[1]-1))
    skinMesh = createMesh(image1,0,255)
    boneMesh = createThresholdMesh(image2,0, 255)
    ##########################################################
    # polydata=vtk.vtkPolyData()
    # polydata.SetPoints(boneMesh.GetPoints())
    splatter=vtk.vtkGaussianSplatter()
    splatter.SetInputData(boneMesh)
    splatter.SetRadius(0.02)
    surface=vtk.vtkContourFilter()
    surface.SetInputConnection(splatter.GetOutputPort())
    surface.SetValue(0,0.01)
    # # Convert the image to a polydata
    # imageDataGeometryFilter = vtk.vtkImageDataGeometryFilter()
    # imageDataGeometryFilter.SetInputData(image2)
    # imageDataGeometryFilter.Update()
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(surface.GetOutputPort())
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    # actor.GetProperty().SetPointSize(0.01)
    #############################################################
    # sigmoidMesh = createThresholdMesh(sigmoid_image,0,1)
    # boneMesh = createThresholdMesh(image1,(int)(image1.GetScalarRange()[1]-1), (int)(image1.GetScalarRange()[1]))
    skinMapper = vtk.vtkPolyDataMapper()
    skinMapper.SetInputData(skinMesh)
    skinMapper.ScalarVisibilityOff()

    skinActor = vtk.vtkActor()
    skinActor.SetMapper(skinMapper)
    skinActor.GetProperty().SetColor(1.0,1.0,1.0)
    skinActor.GetProperty().SetOpacity(1.0)
    skinActor.GetProperty().SetAmbient(0.1)
    skinActor.GetProperty().SetDiffuse(0.7)
    skinActor.GetProperty().SetSpecular(0.1)

    boneMapper = vtk.vtkPolyDataMapper()
    boneMapper.SetInputData(boneMesh)
    boneMapper.ScalarVisibilityOff()

    boneActor = vtk.vtkActor()
    boneActor.SetMapper(boneMapper)
    boneActor.GetProperty().SetColor(1.0,0.0,0.0)
    boneActor.GetProperty().SetAmbient(0.2)
    boneActor.GetProperty().SetDiffuse(0.7)
    boneActor.GetProperty().SetSpecular(0.2)
    boneActor.GetProperty().SetOpacity(1.0)
    # boneActor.GetProperty().SetPointSize(100)

    # sigmoidMapper = vtk.vtkPolyDataMapper()
    # sigmoidMapper.SetInputData(sigmoidMesh)
    # sigmoidMapper.ScalarVisibilityOff()
    #
    # sigmoidActor = vtk.vtkActor()
    # sigmoidActor.SetMapper(sigmoidMapper)
    # sigmoidActor.GetProperty().SetColor(0.0,1.0,0.0)
    # sigmoidActor.GetProperty().SetAmbient(0.2)
    # sigmoidActor.GetProperty().SetDiffuse(0.7)
    # sigmoidActor.GetProperty().SetSpecular(0.2)
    # sigmoidActor.GetProperty().SetOpacity(1.0)


    sphere_view = vtk.vtkSphereSource()
    # sphere_view.SetOrigin([-48.96,-48.96,-48.42])
    # sphere_view.SetSpacing(0.18,0.18,0.18)
    # 210.0, 271.5, 314.5
    sphere_view.SetCenter(int(814*0.12),int(271*0.12),int(710*0.12))
    # sphere_view.SetCenter(int(250*0.18),int(150*0.18),int(600*0.18))
    sphere_view.SetRadius(1.0)
    # sphere_view.SetColor(0.0,1.0,1.0)
    # mapper
    mapper = vtk.vtkPolyDataMapper()
    if vtk.VTK_MAJOR_VERSION <= 5:
       mapper.SetInput(sphere_view.GetOutput())
    else:
       mapper.SetInputConnection(sphere_view.GetOutputPort())

    # actor
    sphere_actor = vtk.vtkActor()
    sphere_actor.SetMapper(mapper)
    sphere_actor.GetProperty().SetColor(1.0,0.0,1.0)


    ren.AddActor(boneActor)
    ren.AddActor(skinActor)
    ren.AddActor(sphere_actor)
    # ren.AddActor(sigmoidActor)
    ren.AddActor(actor)

    iren.Initialize()
    renWin.Render()
    iren.Start()


# Volume rendering
def dual_volumeRendering(image,image2, lowerThreshold, upperThreshold,sigmoid_image):

    color = vtk.vtkColorTransferFunction()
    color.AddRGBPoint(0,0.0,0.0,0.0) # background
    color.AddRGBPoint(lowerThreshold,0.0,0.0,0.0)
    color.AddRGBPoint(upperThreshold,1.0,1.0,1.0)

    opacity = vtk.vtkPiecewiseFunction()
    opacity.AddPoint(0,0.0)
    opacity.AddPoint(lowerThreshold,0.0)
    opacity.AddPoint(upperThreshold,1)

    gradient = vtk.vtkPiecewiseFunction()
    gradient.AddPoint(0,0.0)
    # gradient.AddPoint(90, 0.5)
    gradient.AddPoint(255,1.0)

    volumeProperty = vtk.vtkVolumeProperty()
    volumeProperty.SetColor(color)
    volumeProperty.SetScalarOpacity(opacity)
    volumeProperty.SetGradientOpacity(gradient)
    volumeProperty.SetInterpolationTypeToLinear()
    volumeProperty.ShadeOn()
    volumeProperty.SetAmbient(0.1)
    volumeProperty.SetDiffuse(0.7)
    volumeProperty.SetSpecular(0.1)

    color2 = vtk.vtkColorTransferFunction()
    color2.AddRGBPoint(0,0.0,0.0,0.0) # background
    color2.AddRGBPoint(lowerThreshold,0.0,0.0,0.0)
    color2.AddRGBPoint(upperThreshold,1.0,0.0,0.0)

    opacity2 = vtk.vtkPiecewiseFunction()
    opacity2.AddPoint(0,0.0)
    opacity2.AddPoint(lowerThreshold,0.0)
    opacity2.AddPoint(upperThreshold,1.0)

    gradient2 = vtk.vtkPiecewiseFunction()
    gradient2.AddPoint(0,0.0)
    # gradient2.AddPoint(90, 0.5)
    # gradient2.AddPoint(100,1.0)
    gradient2.AddPoint(255,1.0)

    volumeProperty2 = vtk.vtkVolumeProperty()
    volumeProperty2.SetColor(color2)
    volumeProperty2.SetScalarOpacity(opacity2)
    volumeProperty2.SetGradientOpacity(gradient2)
    volumeProperty2.SetInterpolationTypeToLinear()
    volumeProperty2.ShadeOn()
    volumeProperty2.SetAmbient(0.1)
    volumeProperty2.SetDiffuse(1.0)
    volumeProperty2.SetSpecular(0.1)

    #######################################
    color3 = vtk.vtkColorTransferFunction()
    color3.AddRGBPoint(0,0.0,0.0,0.0) # background
    color3.AddRGBPoint(lowerThreshold,0.0,0.0,0.0)
    color3.AddRGBPoint(upperThreshold,0.0,1.0,0.0)

    opacity3 = vtk.vtkPiecewiseFunction()
    opacity3.AddPoint(0,0.0)
    opacity3.AddPoint(lowerThreshold,0.0)
    opacity3.AddPoint(upperThreshold,1)

    gradient3 = vtk.vtkPiecewiseFunction()
    gradient3.AddPoint(0,0.0)
    # gradient.AddPoint(90, 0.5)
    gradient3.AddPoint(255,1.0)

    volumeProperty3 = vtk.vtkVolumeProperty()
    volumeProperty3.SetColor(color3)
    volumeProperty3.SetScalarOpacity(opacity)
    volumeProperty3.SetGradientOpacity(gradient)
    volumeProperty3.SetInterpolationTypeToLinear()
    volumeProperty3.ShadeOn()
    volumeProperty3.SetAmbient(0.1)
    volumeProperty3.SetDiffuse(0.7)
    volumeProperty3.SetSpecular(0.1)
    ###########################################

    volumeMapper = vtk.vtkFixedPointVolumeRayCastMapper()
    volumeMapper.SetInputData(image)

    volumeMapper2 = vtk.vtkFixedPointVolumeRayCastMapper()
    volumeMapper2.SetInputData(image2)

    volumeMapper3 = vtk.vtkFixedPointVolumeRayCastMapper()
    volumeMapper3.SetInputData(sigmoid_image)

    volume = vtk.vtkVolume()
    volume.SetMapper(volumeMapper)
    volume.SetProperty(volumeProperty)
    volume.Update()

    volume2 = vtk.vtkVolume()
    volume2.SetMapper(volumeMapper2)
    volume2.SetProperty(volumeProperty2)
    volume2.Update()

    volume3 = vtk.vtkVolume()
    volume3.SetMapper(volumeMapper3)
    volume3.SetProperty(volumeProperty3)
    volume3.Update()

    sphere_view = vtk.vtkSphereSource()
    sphere_view.SetCenter(30,0,550)
    sphere_view.SetRadius(5.0)
    # sphere_view.SetColor(0.0,1.0,1.0)
    # mapper
    mapper = vtk.vtkPolyDataMapper()
    if vtk.VTK_MAJOR_VERSION <= 5:
       mapper.SetInput(sphere_view.GetOutput())
    else:
       mapper.SetInputConnection(sphere_view.GetOutputPort())

    # actor
    sphere_actor = vtk.vtkActor()
    sphere_actor.SetMapper(mapper)
    sphere_actor.GetProperty().SetColor(1.0,0.0,1.0)

    ren = vtk.vtkRenderer()
    ren.AddActor(sphere_actor)
    # ren.AddActor(vol2_actor)
    ren.AddViewProp(volume)
    ren.AddViewProp(volume2)
    ren.AddViewProp(volume3)
    ren.SetBackground(0, 0, 0)
    renWin = vtk.vtkRenderWindow()
    renWin.AddRenderer(ren)
    renWin.SetSize(400,400)

    # create a renderwindowinteractor
    iren = vtk.vtkRenderWindowInteractor()
    iren.SetRenderWindow(renWin)

    iren.Initialize()
    renWin.Render()
    iren.Start()


def volumeRendering(image,image2, lowerThreshold, upperThreshold):

    color = vtk.vtkColorTransferFunction()
    color.AddRGBPoint(0,0.0,0.0,0.0) # background
    color.AddRGBPoint(lowerThreshold,1.0,0.0,0.0)
    color.AddRGBPoint(upperThreshold,0.0,1.0,0.0)

    opacity = vtk.vtkPiecewiseFunction()
    opacity.AddPoint(0,0.0)
    opacity.AddPoint(lowerThreshold,0.15)
    opacity.AddPoint(upperThreshold,0.85)

    gradient = vtk.vtkPiecewiseFunction()
    gradient.AddPoint(0,0.0)
    gradient.AddPoint(90, 0.5)
    gradient.AddPoint(100,1.0)

    volumeProperty = vtk.vtkVolumeProperty()
    volumeProperty.SetColor(color)
    volumeProperty.SetScalarOpacity(opacity)
    volumeProperty.SetGradientOpacity(gradient)
    volumeProperty.SetInterpolationTypeToLinear()
    volumeProperty.ShadeOn()
    volumeProperty.SetAmbient(0.2)
    volumeProperty.SetDiffuse(0.7)
    volumeProperty.SetSpecular(0.3)

    volumeMapper = vtk.vtkFixedPointVolumeRayCastMapper()
    volumeMapper.SetInputData(image)

    volume = vtk.vtkVolume()
    volume.SetMapper(volumeMapper)
    volume.SetProperty(volumeProperty)
    volume.Update()

    ren = vtk.vtkRenderer()
    ren.AddViewProp(volume)
    ren.SetBackground(0, 0, 0)
    renWin = vtk.vtkRenderWindow()
    renWin.AddRenderer(ren)
    renWin.SetSize(400,400)

    # create a renderwindowinteractor
    iren = vtk.vtkRenderWindowInteractor()
    iren.SetRenderWindow(renWin)

    iren.Initialize()
    renWin.Render()
    iren.Start()
# Reads, smooths and renders the dataset
def rendering(
        dataPath,label,
        lowerThreshold, upperThreshold,surface,
        dualSurface,volume,dual_volume,sigmoid):

    # Read the data
    if dataPath.endswith('.nrrd'):
        reader = vtk.vtkNrrdReader()
        reader.SetFileName(dataPath)
        reader.Update()
    # if dataPath.endswith('.nii'):
    #     reader = vtk.vtkNIFTIImageReader()
    #     reader.SetFileName(dataPath)
    #     reader.Update()
    else:
        reader = vtk.vtkDICOMImageReader()
        reader.SetDirectoryName(dataPath)
        reader.Update()


    if label.endswith('.nrrd'):
        reader2 = vtk.vtkNrrdReader()
        reader2.SetFileName(label)
        reader2.Update()
    else:
        reader2 = vtk.vtkDICOMImageReader()
        reader2.SetDirectoryName(label)
        reader2.Update()

    # if sigmoid.endswith('.nii'):
    #     reader3 = vtk.vtkNIFTIImageReader()
    #     reader3.SetFileName(sigmoid)
    #     reader3.Update()
    # else:
    #     reader3 = vtk.vtkDICOMImageReader()
    #     reader3.SetDirectoryName(sigmoid)
    #     reader3.Update()
    # Smooth the image
    filteredImage = gaussianFilter(reader)

    # Surface rendering
    if surface:
        surfaceRendering(reader2.GetOutput(), lowerThreshold, upperThreshold)

    if dualSurface:
        dualSurfaceRendering(filteredImage,reader2.GetOutput())


    # Volume rendering
    if volume:
        volumeRendering(reader2.GetOutput(), lowerThreshold, upperThreshold)
    if dual_volume:
        dual_volumeRendering(reader.GetOutput(),reader2.GetOutput(), lowerThreshold, upperThreshold,reader3.GetOutput())


# Argument Parsing
# parser = argparse.ArgumentParser(
#         description = """Surface and Volume rendering""")

def visualize(path_string,label,sigmoid_sinus):
    # Data file path
    # parser.add_argument("dataPath")
    dataPath=path_string
    Label=label
    sigmoid=sigmoid_sinus

    lowerThreshold=0

    upperThreshold=255

    surface=0

    dualSurface=1

    volume=0
    dual_volume=0



    rendering(dataPath,Label, lowerThreshold, upperThreshold, surface,
    dualSurface, volume,dual_volume,sigmoid)

parser = argparse.ArgumentParser(
    description = """visualizing the results""")
parser.add_argument("-args0", type = str, default = ((".\Data_Sets_test_volume2.nrrd")),
    help = "address of label in .nii format")
parser.add_argument("-args1", type = str, default = (".\Data_Sets_test_volume22.nrrd"),
    help = "address of volume in .nii format")
parser.add_argument("-args2", type = int, default = 0,
    help = "lower threshold")
parser.add_argument("-args3", type = int, default = 255,
    help = "upper threshold")
parser.add_argument("-args4", type = int, default = 0,
    help = "surface rendering")
parser.add_argument("-args5", type = int, default = 1,
    help = "dual surface rendering")
parser.add_argument("-args6", type = int, default = 0,
    help = "volume rendering")
parser.add_argument("-args7", type = int, default = 0,
    help = "dual volume rendering")
# parser.add_argument("-args8", type = str, default = (unicode("\\\\samba.cs.ucalgary.ca\\fatemeh.yazdanbakhsh\Documents\medical_imaging\Results\sig.nii",'utf-8')),
#     help = "sigmoid sinus directory")
args = parser.parse_args()
label=args.args0
lowerThreshold=args.args2
upperThreshold=args.args3
surface=args.args4
dualSurface=args.args5
volume=args.args6
dual_voulme=args.args7
sigmoid_sinus=1
visualize(args.args1,label,sigmoid_sinus)
# visualize("U:\Documents\Data_Sets\Calgary\TBone-2015\TBoneCBCT-2015-10\L2963_L_modified_1_oct_2018",label)
