import skeletonization_function as skel
import identification_of_any_organ_function as iden
import overhang_function_3 as over
import read_file_function as read
import tkinter as tk
from tkinter import filedialog
Low=1000
High=4000


root = tk.Tk()
root.withdraw()
#select directory of files
intact_volume_add = filedialog.askdirectory(title="open intact image",initialdir=str("D:\\fatemeh\\UWO_CASES"))
dissected_volume_add = filedialog.askdirectory(title="open dissected image",initialdir=str("D:\\fatemeh\\UWO_CASES"))
sigmoid_volume_add = filedialog.askopenfilename(title="open sigmoid image",initialdir=str("D:\\fatemeh\\UWO_CASES"))
facial_volume_add = filedialog.askopenfilename(title="open facial image",initialdir=str("D:\\fatemeh\\UWO_CASES"))
posterior_volume_add = filedialog.askopenfilename(title="open posterior canal image",initialdir=str("D:\\fatemeh\\UWO_CASES"))
Horizental_volume_add = filedialog.askopenfilename(title="open horizental canal image",initialdir=str("D:\\fatemeh\\UWO_CASES"))
superior_volume_add = filedialog.askopenfilename(title="open superior canal image",initialdir=str("D:\\fatemeh\\UWO_CASES"))
digastric_volume_add = filedialog.askopenfilename(title="open digastric ridge image",initialdir=str("D:\\fatemeh\\UWO_CASES"))
ossicle_volume_add = filedialog.askopenfilename(title="open osssicle image",initialdir=str("D:\\fatemeh\\UWO_CASES"))
Inner_volume_add = filedialog.askopenfilename(title="open InnerEar image",initialdir=str("D:\\fatemeh\\UWO_CASES"))

#read files using itk or sitk
intact_volume=read.read(intact_volume_add)
dissected_volume=read.read(dissected_volume_add)
sigmoid_volume=read.read(sigmoid_volume_add)
facial_volume=read.read(facial_volume_add)
posterior_volume=read.read(posterior_volume_add)
Horizental_volume=read.read(Horizental_volume_add)
superior_volume=read.read(superior_volume_add)
digastric_volume=read.read(digastric_volume_add)
ossicle_volume=read.read(ossicle_volume_add)
Inner_volume=read.read(Inner_volume_add)

#Normal of Volume
Normall=[0,0,1]
Spacing=facial_volume.GetSpacing()
Horizental=skel.main(dissected_volume,intact_volume,Horizental_volume,Low,High)
f= open("Karname.txt","a+")
f.write("skeletonization in Horizental semi-circular canal      %d\r\n" %(Horizental))


superior=skel.main(dissected_volume,intact_volume,superior_volume,Low,High)
f= open("Karname.txt","a+")
f.write("skeletonization in Superior semi-circular canal      %d\r\n" %(superior))


posterior=skel.main(dissected_volume,intact_volume,posterior_volume,Low,High)
f= open("Karname.txt","a+")
f.write("skeletonization in posterior semi-circular canal      %d\r\n" %(posterior))


digastric=iden.main(dissected_volume,intact_volume,digastric_volume)
f= open("Karname.txt","a+")
f.write("digastric ridge identification      %d\r\n" %(digastric))


facial=iden.main(dissected_volume,intact_volume,facial_volume)
f= open("Karname.txt","a+")
f.write("facial nerve identification      %d\r\n" %(facial))

overhang,saucerization=over.main(dissected_volume,intact_volume,sigmoid_volume,ossicle_volume,facial_volume,Inner_volume,dissected_volume_add,Normall,Spacing,Low,High)
f= open("Karname.txt","a+")
f.write("overhang detection in sigmoid sinus      %d\r\n" %(overhang))
f= open("Karname.txt","a+")
f.write("Saucerization      %d\r\n" %(saucerization))
