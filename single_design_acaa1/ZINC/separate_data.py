#!/usr/bin/env python
import sys

USAGE="""

"""
import numpy as np

file_input="data_5.txt"
fp_input=open(file_input)
lines=fp_input.readlines()
fp_input.close()

logP_list=[]
SAS_list=[]
QED_list=[]
MW_list=[]
TPSA_list=[]

for i in range(len(lines)):
    line=lines[i]
    if line[0]=='#':
        continue
    arr=line.split()
    logP=float(arr[1])
    SAS=float(arr[2])
    QED=float(arr[3])
    MW=float(arr[4])
    TPSA=float(arr[5])
    logP_list+=[logP]
    SAS_list+=[SAS]
    QED_list+=[QED]
    MW_list+=[MW]
    TPSA_list+=[TPSA]

logP_array=np.array(logP_list)
SAS_array=np.array(SAS_list)
QED_array=np.array(QED_list)
MW_array=np.array(MW_list)
TPSA_array=np.array(TPSA_list)

print(logP_array.min(),logP_array.max())
print(SAS_array.min(),SAS_array.max())
print(QED_array.min(),QED_array.max())
print(MW_array.min(),MW_array.max())
print(TPSA_array.min(),TPSA_array.max())

Ndata=len(lines)-1
index=np.arange(0,Ndata)
np.random.shuffle(index)
Ntest=10000
Ntrain=Ndata-Ntest
train_index=index[0:Ntrain]
test_index=index[Ntrain:Ndata]
train_index.sort()
test_index.sort()

file_output="train_5.txt"
fp_out=open(file_output,"w")
line_out="#smi logP SAS QED MW TPSA\n"
fp_out.write(line_out)
for i in train_index:
    line=lines[i+1]
    fp_out.write(line)
fp_out.close()


file_output="test_5.txt"
fp_out=open(file_output,"w")
line_out="#smi logP SAS QED MW TPSA\n"
fp_out.write(line_out)
for i in test_index:
    line=lines[i+1]
    fp_out.write(line)
fp_out.close()


