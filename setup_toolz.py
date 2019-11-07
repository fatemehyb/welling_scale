# from distutils.core import setup
# import py2exe
#
# setup(console=['call_all_function.py'])
import os
from cx_Freeze import setup, Executable

from PyInstaller import log as logging
from PyInstaller import compat
import seaborn
from os import listdir
import numpy
import os
import scipy
import scipy.ndimage._ni_support
import cytoolz
import skimage
import mayavi
import itk
import mayavi.mlab
import SimpleITK
import matplotlib
import itkLazy
matplotlib.use("Qt5Agg")

#################################
from PyInstaller.utils.hooks import collect_data_files

hiddenimports = ['new']

# If ITK is pip installed, gets all the files.
itk_datas = collect_data_files('itk',  include_py_files=True)
# itk_datas= [('C:\\Users\\fatemeh.yazdanbakhsh\\AppData\\Local\\Continuum\\anaconda2_2\\envs\\py36\\lib\\site-packages\\SimpleITK\\SimpleITK.py', 'SimpleITK'), ('C:\\Users\\fatemeh.yazdanbakhsh\\AppData\\Local\\Continuum\\anaconda2_2\\envs\\py36\\lib\\site-packages\\SimpleITK\\_SimpleITK.cp36-win_amd64.pyd', 'SimpleITK'), ('C:\\Users\\fatemeh.yazdanbakhsh\\AppData\\Local\\Continuum\\anaconda2_2\\envs\\py36\\lib\\site-packages\\SimpleITK\\__init__.py', 'SimpleITK'), ('C:\\Users\\fatemeh.yazdanbakhsh\\AppData\\Local\\Continuum\\anaconda2_2\\envs\\py36\\lib\\site-packages\\SimpleITK\\__pycache__\\SimpleITK.cpython-36.pyc', 'SimpleITK\\__pycache__'), ('C:\\Users\\fatemeh.yazdanbakhsh\\AppData\\Local\\Continuum\\anaconda2_2\\envs\\py36\\lib\\site-packages\\SimpleITK\\__pycache__\\__init__.cpython-36.pyc', 'SimpleITK\\__pycache__')]
datas = [x for x in itk_datas if '__pycache__' not in x[0]]
################################

includefiles_list=[]
matplotlib_path = os.path.dirname(matplotlib.__file__)
includefiles_list.append(matplotlib_path)

sitk_path = os.path.dirname(SimpleITK.__file__)
includefiles_list.append(sitk_path)

itk_path='C:\\Users\\fatemeh.yazdanbakhsh\\AppData\\Local\\Continuum\\anaconda2_2\\envs\\py36\\lib\\site-packages\\itk'
# itk_path = os.path.dirname(itk.__file__)
includefiles_list.append(itk_path)

itkLazy_path = os.path.dirname(itkLazy.__file__)
includefiles_list.append(itkLazy_path)


cytoolz_path = os.path.dirname(cytoolz.__file__)
includefiles_list.append(cytoolz_path)

mayavi_path = os.path.dirname(mayavi.__file__)
includefiles_list.append(mayavi_path)

skimage_path = os.path.dirname(skimage.__file__)
includefiles_list.append(skimage_path)

seaborn_path = os.path.dirname(seaborn.__file__)
includefiles_list.append(seaborn_path)

# includefiles_list=[]
scipy_path = os.path.dirname(scipy.__file__)
includefiles_list.append(scipy_path)

libdir = compat.base_prefix + "/lib"
mkllib = filter(lambda x : x.startswith('libmkl_'), listdir(libdir))
if mkllib != []:
   logger = logging.getLogger(__name__)
   logger.info("MKL installed as part of numpy, importing that!")
   binaries = map(lambda l: (libdir + "/" + l, ''), mkllib)

PYTHON_INSTALL_DIR = os.path.dirname(os.path.dirname(os.__file__))
os.environ['TCL_LIBRARY'] = os.path.join(PYTHON_INSTALL_DIR, 'tcl', 'tcl8.6')
os.environ['TK_LIBRARY'] = os.path.join(PYTHON_INSTALL_DIR, 'tcl', 'tk8.6')

build_exe_options = {'packages': ['numpy','scipy','mayavi'],
                     'includes': ['matplotlib.backends.backend_qt5agg','numpy.core._methods','scipy.ndimage._ni_support','mayavi.mlab','matplotlib',includefiles_list],
                     'include_files': [(os.path.join(PYTHON_INSTALL_DIR, 'DLLs', 'tcl86t.dll'),
                                        os.path.join('lib', 'tcl86t.dll')),
                                       (os.path.join(PYTHON_INSTALL_DIR, 'DLLs', 'tk86t.dll'),
                                        os.path.join('lib', 'tk86t.dll'))
                                       # add here further files which need to be included as described in 1.
                                      ]}

#result, datas = datas, []
result, includefiles_list = includefiles_list, []
result2, datas = datas, []
result3=result+[i[0] for i in result2]
setup(name = "call_all_function" ,
options={'build_exe': {"includes": ["numpy.core._methods","scipy.ndimage._ni_support","scipy","cytoolz","skimage","mayavi","mayavi.mlab","seaborn","itk","SimpleITK","itkLazy","sklearn"],"include_files":[i for i in result3]}},
     version = "0.1" ,
      description = "" ,
      executables = [Executable("call_all_function.py")])


