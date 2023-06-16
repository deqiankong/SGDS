#!/usr/bin/env python
import sys

USAGE="""

"""
import numpy as np
from rdkit import Chem

from rdkit.Chem.Descriptors import ExactMolWt
from rdkit.Chem.Crippen import MolLogP
from rdkit.Chem.Crippen import MolMR

from rdkit.Chem.rdMolDescriptors import CalcNumHBD
from rdkit.Chem.rdMolDescriptors import CalcNumHBA
from rdkit.Chem.rdMolDescriptors import CalcTPSA

from rdkit.Chem.QED import qed
#sys.path.insert(0,'/home/shade/SA_Score')
import sascorer


file_input="250k_rndm_zinc_drugs_clean_3.csv"
fp_input=open(file_input)
lines=fp_input.readlines()
fp_input.close()

file_output="data_5.txt"
fp_out=open(file_output,"w")

line_out="#smi logP SAS QED MW TPSA\n"
fp_out.write(line_out)
logP_list=[]
SAS_list=[]
QED_list=[]
MW_list=[]
TPSA_list=[]

for i in range(len(lines)):
    line=lines[i]
    if line[0]!='"':
        continue
    if line[1]!=",":
        smi=line[1:].strip()
        continue
    m=Chem.MolFromSmiles(smi)
    smi2=Chem.MolToSmiles(m)

    property0=line[2:].split(",")
#    logP=float(property0[0])
#    SAS=float(property0[2])
#    QED=float(property0[1])

    logP=MolLogP(m)
    SAS=sascorer.calculateScore(m)
    QED=qed(m)

    MW=ExactMolWt(m)
    TPSA=CalcTPSA(m)
    line_out="%s %6.3f %6.3f %6.3f %6.3f %6.3f\n" %(smi2,logP,SAS,QED,MW,TPSA)
    fp_out.write(line_out)
    logP_list+=[logP]
    SAS_list+=[SAS]
    QED_list+=[QED]
    MW_list+=[MW]
    TPSA_list+=[TPSA]


fp_out.close()

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

