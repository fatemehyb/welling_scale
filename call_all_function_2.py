# import skeletonization_function as skel
# import identification_of_any_organ_function as iden
import overhang_function_3 as over
import read_file_function as read
import tkinter as tk
import re
# from tkinter import filedialog
import argparse
Low=1000
High=4000

parser = argparse.ArgumentParser(
    description = """This program uses ray casting method to detect overhang problem""")
parser.add_argument("-args0", type = str, default = (('\\\\samba.cs.ucalgary.ca\\fatemeh.yazdanbakhsh\Documents\Data_Sets\Calgary\TBone-2015\MicroCT\dissected\L2944_complete_dissection')), help = "dissected image address")
parser.add_argument("-args1",type=str, default=('\\\\samba.cs.ucalgary.ca\\fatemeh.yazdanbakhsh\Documents\Data_Sets\Calgary\TBone-2015\MicroCT\\2944L_reduced'),help="intact image address")
parser.add_argument("-args2",type=str, default=('\\\\samba.cs.ucalgary.ca\\fatemeh.yazdanbakhsh\Documents\Data_Sets\Calgary\TBone-2015\Segmentations\\2944L SEGMENTATIONS\\final\\final\\SigmoidSinus.nrrd'),help="sigmoid address")
parser.add_argument("-args3",type=str, default=('\\\\samba.cs.ucalgary.ca\\fatemeh.yazdanbakhsh\Documents\Data_Sets\Calgary\TBone-2015\Segmentations\\2944L SEGMENTATIONS\\final\\final\\FacialNerve.nrrd'),help="FacialNerve address")

# parser.add_argument("-args2", type = int, default = 1000,
#     help = "low")
# parser.add_argument("-args3", type = int, default = 4000,
#     help = "high")

# parser.add_argument("-args4",type=str, default=('D:\evan_campton\\1601L_FLIPPED_CROPPED_EDITED\\1601L_FLIPPED_CROPPED_EDITED\SEGMENTATIONS\edited\SigmoidSinus.nrrd'),help="sigmoid address")


parser.add_argument("-args4",type=str, default=('\\\\samba.cs.ucalgary.ca\\fatemeh.yazdanbakhsh\Documents\Data_Sets\Calgary\TBone-2015\Segmentations\\2944L SEGMENTATIONS\\final\\final\\malleus_incus.nrrd'),help="Ossicle address")

parser.add_argument("-args5",type=str, default=('\\\\samba.cs.ucalgary.ca\\fatemeh.yazdanbakhsh\\Documents\Data_Sets\\Calgary\\TBone-2015\Segmentations\\2944L SEGMENTATIONS\\final\\final\\cochlea.nrrd'),help="inner ear address")
parser.add_argument("-args6",type=str, default=('-1,0,0'),help="Normal of surface")
parser.add_argument("-args7",type=str, default=('\\\\samba.cs.ucalgary.ca\\fatemeh.yazdanbakhsh\Documents\Data_Sets\Calgary\TBone-2015\Segmentations\\2944L SEGMENTATIONS\\final\\final\\2944L_Tegmen.nrrd'),help="Tegmen")
# Get your arguments
# Get your arguments
args = parser.parse_args()


root = tk.Tk()
root.withdraw()
#select directory of files
intact_volume_add = args.args1
dissected_volume_add = args.args0
sigmoid_volume_add = args.args2
facial_volume_add = args.args3
# posterior_volume_add = filedialog.askopenfilename(title="open posterior canal image",initialdir=str("D:\\fatemeh\\UWO_CASES"))
# Horizental_volume_add = filedialog.askopenfilename(title="open horizental canal image",initialdir=str("D:\\fatemeh\\UWO_CASES"))
# superior_volume_add = filedialog.askopenfilename(title="open superior canal image",initialdir=str("D:\\fatemeh\\UWO_CASES"))
# digastric_volume_add = filedialog.askopenfilename(title="open digastric ridge image",initialdir=str("D:\\fatemeh\\UWO_CASES"))
ossicle_volume_add = args.args4
Inner_volume_add = args.args5
Tegmen_volume_add=args.args7
#read files using itk or sitk
intact_volume=read.read(intact_volume_add)
dissected_volume=read.read(dissected_volume_add)
sigmoid_volume=read.read(sigmoid_volume_add)
facial_volume=read.read(facial_volume_add)
# posterior_volume=read.read(posterior_volume_add)
# Horizental_volume=read.read(Horizental_volume_add)
# superior_volume=read.read(superior_volume_add)
# digastric_volume=read.read(digastric_volume_add)
ossicle_volume=read.read(ossicle_volume_add)
Inner_volume=read.read(Inner_volume_add)
Tegmen_volume=read.read(Tegmen_volume_add)
Normall=args.args6
b=re.findall('-\d|\d',Normall)
Normall=[int(s) for s in b ]
# Normall=[0.584682,0.584682,-0.23108]
#Normal of Volume
# Normall=args.args6.split(',')
#
# Normall = [ int(s) for s in Normall ]
Spacing=facial_volume.GetSpacing()
# Horizental=skel.main(dissected_volume,intact_volume,Horizental_volume,Low,High)
# f= open("Karname.txt","a+")
# f.write("skeletonization in Horizental semi-circular canal      %d\r\n" %(Horizental))


# superior=skel.main(dissected_volume,intact_volume,superior_volume,Low,High)
# f= open("Karname.txt","a+")
# f.write("skeletonization in Superior semi-circular canal      %d\r\n" %(superior))


# posterior=skel.main(dissected_volume,intact_volume,posterior_volume,Low,High)
# f= open("Karname.txt","a+")
# f.write("skeletonization in posterior semi-circular canal      %d\r\n" %(posterior))


# digastric=iden.main(dissected_volume,intact_volume,digastric_volume)
# f= open("Karname.txt","a+")
# f.write("digastric ridge identification      %d\r\n" %(digastric))


# facial=iden.main(dissected_volume,intact_volume,facial_volume)
# f= open("Karname.txt","a+")
# f.write("facial nerve identification      %d\r\n" %(facial))

overhang,saucerization=over.main(dissected_volume,intact_volume,sigmoid_volume,ossicle_volume,facial_volume,Inner_volume,dissected_volume_add,Normall,Spacing,Low,High,Tegmen_volume)
f= open("Karname.txt","w+")
f.write("overhang detection in sigmoid sinus      %d\r\n" %(overhang))
f= open("Karname.txt","a+")
f.write("Saucerization      %d\r\n" %(saucerization))
