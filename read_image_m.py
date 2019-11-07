#!/usr/bin/env python
# import os

import itk

#####################################################################
def dicom_reader(m):
    #main function
    # inputImage = "U:\Documents\medical_imaging\D3Slice17.dcm"
    inputImage=m
    outputImage = "U:\Documents\medical_imaging\dcm-test.dcm"
    sigma = float(0.5)

    PixelType = itk.ctype('signed short')
    Dimension = 2

    ImageType = itk.Image[PixelType, Dimension]

    reader = itk.ImageFileReader[ImageType].New()
    reader.SetFileName(inputImage)
    reader.Update()
    image_input=reader.GetOutput()
    return image_input
###################################################################################################
def dicom_series_reader(m):
    print("Usage: " + "U:\Documents\Data_Sets\Calgary\TBone-2015\TBoneCBCT-2015-10\2963L" +
              " [DicomDirectory [outputFileName [seriesName]]]")
    print("If DicomDirectory is not specified, current directory is used\n")

    # current directory by default
    dirName = m

    PixelType = itk.ctype('signed short')
    Dimension = 3

    ImageType = itk.Image[PixelType, Dimension]

    namesGenerator = itk.GDCMSeriesFileNames.New()
    namesGenerator.SetUseSeriesDetails(True)
    namesGenerator.AddSeriesRestriction("0008|0021")
    namesGenerator.SetGlobalWarningDisplay(False)
    namesGenerator.SetDirectory(dirName)

    seriesUID = namesGenerator.GetSeriesUIDs()

    if len(seriesUID) < 1:
        print('No DICOMs in: ' + dirName)
       # sys.exit(1)

    print('The directory: ' + dirName)
    print('Contains the following DICOM Series: ')
    for uid in seriesUID:
        print(uid)

    seriesFound = False
    for uid in seriesUID:
        seriesIdentifier = uid
        # if len(sys.argv) > 3:
        #     seriesIdentifier = sys.argv[3]
        #     seriesFound = True
        print('Reading: ' + seriesIdentifier)
        fileNames = namesGenerator.GetFileNames(seriesIdentifier)

        reader = itk.ImageSeriesReader[ImageType].New()
        dicomIO = itk.GDCMImageIO.New()
        reader.SetImageIO(dicomIO)
        reader.SetFileNames(fileNames)
        reader.Update()
        volume=reader.GetOutput()
        return volume
