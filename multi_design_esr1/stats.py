from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
from rdkit.Chem.QED import qed
import sascorer


def property_plot(logps, sass, qeds, data_properties):
    from matplotlib import pyplot as plt
    import seaborn as sns
    import pandas as pd
    data_logps, data_sass, data_qeds = data_properties
    assert len(data_logps) == len(data_sass) == len(data_qeds)
    assert len(logps) == len(sass) == len(qeds)
    assert len(data_logps) == len(logps)
    labels = ['Test'] * len(data_logps) + ['Train'] * len(logps)
    logp_df = pd.DataFrame(data={'model': labels, 'logP': data_logps + logps})
    logp_df["logP"] = pd.to_numeric(logp_df["logP"])
    sas_df = pd.DataFrame(data={'model': labels, 'SAS': data_sass + sass})
    sas_df["SAS"] = pd.to_numeric(sas_df["SAS"])
    qed_df = pd.DataFrame(data={'model': labels, 'QED': data_qeds + qeds})
    qed_df["QED"] = pd.to_numeric(qed_df["QED"])

    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=[6.4 * 3, 4.8])
    # fig.suptitle("Epoch: " + str(epoch), fontsize=12)
    logp_plot = sns.kdeplot(data=logp_df, x='logP', hue='model', ax=ax[0])
    sas_plot = sns.kdeplot(data=sas_df, x='SAS', hue='model', ax=ax[1])
    qed_plot = sns.kdeplot(data=qed_df, x='QED', hue='model', ax=ax[2])

    plt.show()
    plt.close(fig)


def get_data_properties(path, seq_length=110):
    with open(path) as f:
        data_lines = f.readlines()

    logps = []
    sass = []
    qeds = []
    mols = []
    title = ""
    for line in data_lines:
        if line[0] == "#":
            title = line[1:-1]
            title_list = title.split()
            print(title_list)
            continue
        arr = line.split()
        # print(arr)
        if len(arr) < 2:
            continue
        smiles = arr[0]
        if len(smiles) > seq_length:
            continue
        assert len(arr) == 6
        mols.append(arr[0])
        logps.append(arr[1])
        sass.append(arr[2])
        qeds.append(arr[3])
        # smiles0 = smiles.ljust(seq_length, '>')
        # smiles_list += [smiles]
        # Narr = len(arr)
        # # cdd = []
        # for i in range(1, Narr):
        #     if title_list[i] == "logP":
        #         cdd += [float(arr[i])/10.0]
        #     elif title_list[i] == "SAS":
        #         cdd += [float(arr[i])/10.0]
        #     elif title_list[i] == "QED":
        #         cdd += [float(arr[i])/1.0]
        #     elif title_list[i] == "MW":
        #         cdd += [float(arr[i])/500.0]
        #     elif title_list[i] == "TPSA":
        #         cdd += [float(arr[i])/150.0]
    return mols, (logps, sass, qeds)


def _convert_keys_to_int(dict):
    new_dict = {}
    for k, v in dict.items():
        try:
            new_key = int(k)
        except ValueError:
            new_key = k
        new_dict[new_key] = v
    return new_dict


if __name__ == '__main__':
    from args import get_args
    import numpy as np
    from rdkit import Chem
    from rdkit.Chem.Crippen import MolLogP
    import json
    import selfies as sf
    from tqdm import tqdm
    import random

    args = get_args()
    mols, data_properties = get_data_properties(args.test_file)
    # print(mols, len(mols))

    # smiles
    # logps, sass, qeds = [], [], []
    # for i, smi in enumerate(mols):
    #     m = Chem.MolFromSmiles(smi)
    #     p = MolLogP(m)
    #     logps.append(p)
    #     try:
    #         SAS = sascorer.calculateScore(m)
    #     except ZeroDivisionError:
    #         SAS = 2.8
    #     sass.append(SAS)
    #     QED = qed(m)
    #     qeds.append(QED)
    #
    # property_plot(logps, sass, qeds, data_properties)

    # selfies
    logps, sass, qeds = [], [], []
    dir_test = '../data/Xtrain.npy'
    sf_test = np.load(dir_test)
    json_file = json.load(open('../data/info.json'))
    vocab_itos = json_file['vocab_itos'][0]
    # change the keywords of vocab_itos from string to int
    vocab_itos = _convert_keys_to_int(vocab_itos)

    randomlist = random.sample(range(0, len(sf_test)), 10000)
    random_data = sf_test[randomlist]
    for i, data in enumerate(tqdm(random_data)):
        m_sf = sf.encoding_to_selfies(sf_test[i], vocab_itos, enc_type='label')
        m_smi = sf.decoder(m_sf)
        m = Chem.MolFromSmiles(m_smi)
        p = MolLogP(m)
        logps.append(p)
        try:
            SAS = sascorer.calculateScore(m)
        except ZeroDivisionError:
            SAS = 2.8
        sass.append(SAS)
        QED = qed(m)
        qeds.append(QED)
    property_plot(logps, sass, qeds, data_properties)
