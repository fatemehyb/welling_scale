import SimpleITK as sitk
import read_image_m as RIM
import read_dicom_series_sitk as risk
import os
# import itk
def read(read_string):
    ext=os.path.splitext(read_string)[1]
    m_string=read_string
    if (ext==".nii" or ext==".nrrd" or ext==".nhdr"):

        Inner_volume=sitk.ReadImage(m_string)
    else:
        Inner_volume=risk.read_series_sitk(m_string)

    return Inner_volume
