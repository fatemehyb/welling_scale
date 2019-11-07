import numpy as np
# from collections import Counter
import re
# def Diff(li1, li2):
#     return (list(set(li1) - set(li2)))

def Diff1(li1, li2):
    li_dif = [i for i in li1 + li2 if i not in li1 or i not in li2]
    return li_dif

def Diff2(li1, li2):
    li_dif = [i for i in li2  if i not in li1]
    #liii=[i for k in li2 if k not in li1]
    return li_dif

def Diff3(li1,li2):
    li3=np.zeros(li2.__len__(),dtype=object)

    for i in range(0,li2.__len__()):
        li3[i]=[j for j in li2[i] if j not in li1[i]]
    return li3

def Diff4(li1,li2):
    li3=[]
    li3=[j for j in li2 if j not in li1]
    return li3
# def Diff3(li1,li2):
#     li3=np.zeros(li2.__len__(),dtype=object)
#
#     for i in range(0,li2.__len__()):
#         li3[i]=[j for j in li2[i] if j not in li1[i]]
#     return li3
def zeross(li1):
    return [0 for i in li1]

def differ_1list(li1):
    li3=np.zeros((li1.shape[0],li1.shape[1]),dtype=object)
    for i in range(0,li1.shape[0]):
        for j in range(0,li1.shape[1]):
        # li3[i]=[j for j in li2[i] if j not in li1[i]]
            if li1[i][j].__len__()>0:
                a=li1[i][j][0][0]
                b=li1[i][j][-1][0]
                if a=='in' and b=='out':
                   li3[i][j]=[li1[i][j][1:-1]]
                   print("li3[i][j]")
                   print(li3[i][j])
                elif a=='in' and b=='in':
                    li3[i][j]=[li1[i][j][1:]]
                elif a=='out' and b=='out':
                    li3[i][j]=[li1[i][j][:-1]]
                elif a=='out' and b=='in':
                    li3[i]=[li1[i][j][:]]
            else: li3[i][j]=[]


    return li3

def differ_1list_2(li1):
    [I,J,K]=np.where(li1!=0)
    li3=[]
    li4=[]
    count1=0
    count2=0
    tempi=I[0]
    tempj=J[0]
    i=0
    while i<(I.__len__()-2):

        tempi=I[i]
        tempj=J[i]
        while tempi==I[i] and tempj==J[i]:
            li3.append(li1[I[i]][J[i]][K[i]])
            tempi=I[i]
            tempj=J[i]
            if i<I.__len__()-2:
                i=i+1
            else:
                break
            # print(li3)

        li4.append(li3.copy())
        li3.clear()





    # for i,j,k in zip(I,J,K):
    #
    #     # tmpi=i
    #     # tmpj=j
    #     while tempi==i and tempj==j:
    #         li3.append(li1[i][j][k])
    #
    #     li4.append(li3)
    #     li3.clear()
##############################################
    # if li1[i][j].__len__()>0:
    #     a=str.split(li1[i][j][0]," ")[0]
    #     b=str.split(li1[i][j][-1]," ")[0]
    #     # a=li1[i][j][0][0]
    #     # b=li1[i][j][-1][0]
    #     if a=='in' and b=='out':
    #        li3[i][j]=[li1[i][j][1:-1]]
    #     elif a=='in' and b=='in':
    #         li3[i][j]=[li1[i][j][1:]]
    #     elif a=='out' and b=='out':
    #         li3[i][j]=[li1[i][j][:-1]]
    #     elif a=='out' and b=='in':
    #         li3[i]=[li1[i][j][:]]
    # else: li3[i][j]=[]

#############################################
    return li4


def differ_1list_modified(li1):
    li3=np.zeros(li1.__len__(),dtype=object)
    for i in range(0,li1.__len__()):
        # li3[i]=[j for j in li2[i] if j not in li1[i]]

       li3[i]=[li1[i][1:-1]]
    return li3

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    j=1
    listj=np.zeros((l.__len__()//3)+1,dtype=object)
    [l[i:i + n] for i in range(0, len(l), n)]
    for i in range(0, len(l), n):
        listj[j]= l[i:i + n]
        j=j+1
    return listj
def chunks_2(l, n):
    """Yield successive n-sized chunks from l."""
    j=1
    listj=np.zeros((l.__len__()//3),dtype=object)
    [l[i:i + n] for i in range(0, len(l), n)]
    for i in range(0, len(l), n):
        listj[j]= l[i:i + n]
        j=j+1
    return listj

def evaluate(tlist1,tlist_original):

    # tlist_difer1=(Diff1(tlist_original, tlist1))
    tlist_trim=np.zeros(tlist1.__len__(),dtype=object)
    tlist_trim_original=np.zeros(tlist1.__len__(),dtype=object)
    for i in range(0,tlist1.__len__()):
        tlist_trim[i]=list(filter(lambda a: a!=None,tlist1[i]))
        tlist_trim_original[i]=list(filter(lambda a: a!=None,tlist_original[i]))
        even_test=(tlist_trim[i]).__len__()

    tlist_difer=(Diff3(tlist_trim_original, tlist_trim))
    for i in range(0,tlist_difer.__len__()):
        even_test=tlist_difer[i].__len__()
        if even_test>1:
            print("visibility error detected",i,",",(tlist_difer[i]))





    return tlist_difer

# list_of_numbers=[]
# for k in range(0,tlist1.__len__()):
#     list_of_numbers.append([float(s) for s in re.findall(r'-?\d+\.?\d*', str(tlist1[k]))])

def evaluate2_2(tlist1):
    # tlist1=tlist1[0]
    tlist1=np.asarray(tlist1)
    # tlist_trim=np.zeros((tlist1.shape[0],tlist1.shape[1]),dtype=object)

    # tlist_trim=np.zeros(tlist1[0].shape[0],dtype=object)
    # for i in range(0,tlist1.shape[0]):
    #     for j in range(0,tlist1.shape[1]):
    #         # tlist_trim=tlist1[np.where(tlist1!=0.0)]
    #         tlist_trim[i,j]=(list(filter(lambda a: a!=[],list(filter(lambda a: a!=0.0,tlist1[i][j])))))
    #         # tlist_trim[i,j]=(list(filter(lambda a: a!=[],list(filter(lambda a: a!=None,tlist1[i][j])))))
    #         # tlist_trim[i][j]=list(filter(lambda a: a!=[],tlist_trim[i][j]))

        # even_test=(tlist_trim[i]).__len__()

    tlist_difer=(differ_1list_2(tlist1))
    # tlist_difer=list(filter(lambda a: a!=[],tlist_difer))
    # tlist_difer=tlist_trim

    # list_of_numbers=[float(s) for s in re.findall(r'-?\d+\.?\d*', str(tlist_difer))]
    list_of_numbers=[]
    final_list=[]
    for k in range(0,tlist_difer.__len__()):
        list_of_numbers.append([float(s) for s in re.findall(r'-?\d+\.?\d*', str(tlist_difer[k]))])
        final_list.append(chunks(list_of_numbers[k][:],3))

    for i in range(0,final_list.__len__()):
        final_list[i]=list(filter(lambda a: a!=0,final_list[i]))





    return final_list

def evaluate2_test(tlist1):
    # tlist1=tlist1[0]
    tlist1=np.asarray(tlist1)
    tlist_trim=np.zeros((tlist1.shape[0],tlist1.shape[1]),dtype=object)

    # tlist_trim=np.zeros(tlist1[0].shape[0],dtype=object)
    for i in range(0,tlist1.shape[0]):
        for j in range(0,tlist1.shape[1]):
            # tlist_trim=tlist1[np.where(tlist1!=0.0)]
            tlist_trim[i,j].append(list(filter(lambda a: a!=[],list(filter(lambda a: a!=0.0,tlist1[i][j])))))
            # tlist_trim[i,j]=(list(filter(lambda a: a!=[],list(filter(lambda a: a!=None,tlist1[i][j])))))
            # tlist_trim[i][j]=list(filter(lambda a: a!=[],tlist_trim[i][j]))

        # even_test=(tlist_trim[i]).__len__()

    tlist_difer=(differ_1list_2(tlist_trim))
    tlist_difer=list(filter(lambda a: a!=[],tlist_difer))
    # tlist_difer=tlist_trim

    # list_of_numbers=[float(s) for s in re.findall(r'-?\d+\.?\d*', str(tlist_difer))]
    list_of_numbers=[]
    final_list=[]
    for k in range(0,tlist_difer.__len__()):
        list_of_numbers.append([float(s) for s in re.findall(r'-?\d+\.?\d*', str(tlist_difer[k]))])
        final_list.append(chunks(list_of_numbers[k][:],3))

    for i in range(0,final_list.__len__()):
        final_list[i]=list(filter(lambda a: a!=0,final_list[i]))





    return final_list

def evaluate2(tlist1):
    # tlist1=tlist1[0]
    tlist1=np.asarray(tlist1)
    tlist_trim=np.zeros((tlist1.shape[0],tlist1.shape[1]),dtype=object)

    # tlist_trim=np.zeros(tlist1[0].shape[0],dtype=object)
    for i in range(0,tlist1.shape[0]):
        for j in range(0,tlist1.shape[1]):
            tlist_trim[i][j]=list(filter(lambda a: a!=None,tlist1[i][j]))
            tlist_trim[i][j]=list(filter(lambda a: a!=[],tlist_trim[i][j]))

        # even_test=(tlist_trim[i]).__len__()

    tlist_difer=(differ_1list(tlist_trim))
    tlist_difer=list(filter(lambda a: a!=[],tlist_difer))
    # tlist_difer=tlist_trim

    # list_of_numbers=[float(s) for s in re.findall(r'-?\d+\.?\d*', str(tlist_difer))]
    list_of_numbers=[]
    final_list=[]
    for k in range(0,tlist_difer.__len__()):
        list_of_numbers.append([float(s) for s in re.findall(r'-?\d+\.?\d*', str(tlist_difer[k]))])
        final_list.append(chunks(list_of_numbers[k][:],3))

    for i in range(0,final_list.__len__()):
        final_list[i]=list(filter(lambda a: a!=0,final_list[i]))






    return final_list
