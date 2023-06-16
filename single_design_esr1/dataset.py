import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from ZINC.char import char_list, char_dict
import selfies as sf
import json
from rdkit.Chem import Draw
from rdkit import Chem
from rdkit.Chem.Crippen import MolLogP
from rdkit.Chem.QED import qed
from rdkit.Chem.Descriptors import ExactMolWt
from rdkit.Chem.rdMolDescriptors import CalcTPSA
import sascorer
import math


# max_len = 72
# num_of_embeddings = 109
class MolDataset(Dataset):
    def __init__(self, datadir, dname):
        Xdata_file = datadir + "/X" + dname + ".npy"
        self.Xdata = torch.tensor(np.load(Xdata_file), dtype=torch.long)  # number-coded molecule
        Ldata_file = datadir + "/L" + dname + ".npy"
        self.Ldata = torch.tensor(np.load(Ldata_file), dtype=torch.long)  # length of each molecule
        self.len = self.Xdata.shape[0]
        sasdata_file = datadir + "/sas_" + dname + ".npy"
        self.sas = torch.tensor(np.load(sasdata_file), dtype=torch.float32)
        qed_data_file = datadir + "/qed_" + dname + ".npy"
        self.qed = torch.tensor(np.load(qed_data_file), dtype=torch.float32)

        ba0data_file = datadir + "/ba0_" + dname + ".npy"
        self.ba0 = torch.tensor(np.load(ba0data_file), dtype=torch.float32)
        # ba1data_file = datadir + "/ba1_" + dname + ".npy"
        # self.ba1 = torch.tensor(np.load(ba1data_file), dtype=torch.float32)
        # self.preprocess()

    def preprocess(self):
        ind = torch.nonzero(self.ba0)
        self.ba0 = self.ba0[ind].squeeze()
        self.sas = self.sas[ind].squeeze()
        self.qed = self.qed[ind].squeeze()
        self.Xdata = self.Xdata[ind].squeeze()
        self.Ldata = self.Ldata[ind].squeeze()
        self.len = self.Xdata.shape[0]

    def __getitem__(self, index):
        # Add sos=108 for each sequence and this sos is not shown in char_list and char_dict as in selfies.
        mol = self.Xdata[index]
        sos = torch.tensor([108], dtype=torch.long)
        mol = torch.cat([sos, mol], dim=0).contiguous()
        mask = torch.zeros(mol.shape[0] - 1)
        mask[:self.Ldata[index] + 1] = 1.
        return (mol, mask, -1 * self.ba0[index])

    def __len__(self):
        return self.len

    def save_mol_png(self, label, filepath, size=(600, 600)):
        m_smi = self.label2sf2smi(label)
        m = Chem.MolFromSmiles(m_smi)
        Draw.MolToFile(m, filepath, size=size)

    def label2sf2smi(self, label):
        m_sf = sf.encoding_to_selfies(label, char_dict, enc_type='label')
        m_smi = sf.decoder(m_sf)
        m_smi = Chem.CanonSmiles(m_smi)
        return m_smi

    @staticmethod
    def delta_to_kd(x):
        return math.exp(x / (0.00198720425864083 * 298.15))

