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


if __name__ == '__main__':
    datadir = '../data'
    # Xdata_file = datadir + "/Xtrain.npy"
    # x_train = np.load(Xdata_file)
    # x_test = np.load('../data/Xtest.npy')
    # x_all = np.concatenate((x_train, x_test), axis=0)
    # print(x_all.shape)
    #
    # ba_train = np.load('../data/ba0_train.npy')
    # ba_test = np.load('../data/ba0_test.npy')
    # ba_all = np.concatenate((ba_train, ba_test), axis=0)
    # print(ba_all.shape)
    # ind = np.argsort(ba_all)
    # ind = ind[:10000]
    #
    # sas_train = np.load('../data/sas_train.npy')
    # sas_test = np.load('../data/sas_test.npy')
    # sas_all = np.concatenate((sas_train, sas_test), axis=0)
    #
    # l_train = np.load('../data/Ltrain.npy')
    # l_test = np.load('../data/Ltest.npy')
    # l_all = np.concatenate((l_train, l_test), axis=0)
    #
    # qed_train = np.load('../data/qed_train.npy')
    # qed_test = np.load('../data/qed_test.npy')
    # qed_all = np.concatenate((sas_train, sas_test), axis=0)
    #
    # sas_design = sas_all[ind]
    # ba_design = ba_all[ind]
    # x_design = x_all[ind]
    # qed_design = qed_all[ind]
    # l_design = l_all[ind]
    #
    # np.save('../data/Xdesign.npy', x_design)
    # np.save('../data/ba0_design.npy', ba_design)
    # np.save('../data/qed_design.npy', qed_design)
    # np.save('../data/sas_design.npy', sas_design)
    # np.save('../data/Ldesign.npy', l_design)

    ds = MolDataset(datadir, 'train')
    ds_loader = DataLoader(dataset=ds, batch_size=100,
                           shuffle=True, drop_last=True, num_workers=2)

    s = ds.ba1
    print(len(s))
    print(s[:10])
    # print(ds.Xdata[:10])
    # print(ds.len)
    # non_zero = torch.nonzero(s)
    # print(non_zero)
    # s = s[non_zero]
    # print(len(s))
    # print(s[1000:2000])
    # print(torch.max(s), torch.min(s))
    # print(ds.delta_to_kd(torch.max(s)), ds.delta_to_kd(torch.min(s)))
    # max_len = 0
    # for i in range(len(ds)):
    #     print(len(ds))
    #     m, len, ba, sas, qed = ds[i]
    #     m = m.numpy()
    #     mol = ds.label2sf2smi(m[1:])
    #     print(mol)
    #     print(ba, sas, qed)
    #     break

    # print(char_dict)
    # print(ds.sf2smi(m), s)
