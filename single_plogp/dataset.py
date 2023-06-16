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


# max_len = 72
# num_of_embeddings = 109
class MolDataset(Dataset):
    def __init__(self, datadir, dname):
        Xdata_file = datadir + "/X" + dname + ".npy"
        self.Xdata = torch.tensor(np.load(Xdata_file), dtype=torch.long)  # number-coded molecule
        Ldata_file = datadir + "/L" + dname + ".npy"
        self.Ldata = torch.tensor(np.load(Ldata_file), dtype=torch.long)  # length of each molecule
        self.len = self.Xdata.shape[0]
        LogPdata_file = datadir + "/LogP" + dname + ".npy"
        self.LogP = torch.tensor(np.load(LogPdata_file), dtype=torch.float32)
        qed_data_file = datadir + "/qed_" + dname + ".npy"
        self.qed = torch.tensor(np.load(qed_data_file), dtype=torch.float32)
        PlogPdata_file = datadir + "/PlogP" + dname + ".npy"
        self.PlogP = torch.tensor(np.load(PlogPdata_file), dtype=torch.float32)

    def __getitem__(self, index):
        # Add sos=108 for each sequence and this sos is not shown in char_list and char_dict as in selfies.
        mol = self.Xdata[index]
        sos = torch.tensor([108], dtype=torch.long)
        mol = torch.cat([sos, mol], dim=0).contiguous()
        mask = torch.zeros(mol.shape[0] - 1)
        mask[:self.Ldata[index] + 1] = 1.
        return (mol, mask, self.PlogP[index])
        # return (mol, mask, self.LogP[index])
        # print(self.Ldata[index])
        # print(mask)
        # print(mol[:self.Ldata[index] + 2])
        # return (mol, self.Ldata[index], self.LogP[index])

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


if __name__ == '__main__':
    datadir = '../data'
    ds = MolDataset(datadir, 'train')
    ds_loader = DataLoader(dataset=ds, batch_size=100,
                           shuffle=True, drop_last=True, num_workers=2)

    max_len = 0
    for i in range(len(ds)):
        m, len, s = ds[i]
        m = m.numpy()
        mol = ds.label2sf2smi(m[1:])
        print(mol)
        ds.save_mol_png(label=m[1:], filepath='../a.png')
        break

    # print(char_dict)
    # print(ds.sf2smi(m), s)
