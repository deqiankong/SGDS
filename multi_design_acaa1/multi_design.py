import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.nn.functional as F
from dataset import MolDataset
from args import get_args
from utils import get_output_dir, copy_source, set_gpu
from model import RNNEBM
from evaluation import evaluation
import logger
from ZINC.char import char_dict
from rdkit import Chem
from rdkit.Chem.Crippen import MolLogP
from rdkit.Chem.QED import qed
import sascorer
import selfies as sf
from evaluation import is_valid, get_data_properties
from tqdm import tqdm
import os
from rdkit.Chem import Draw
import subprocess
import time


def sample_p_0(x, args):
    if args.ref_dist == 'gaussian':
        return args.init_factor * torch.randn(*[x.size(0), args.latent_dim], device=x.device)
    else:
        return torch.Tensor(*[x.size(0), args.latent_dim]).uniform_(-1, 1).to(x.device)


def label2sf2smi(out_num):
    m_sf = sf.encoding_to_selfies(out_num, char_dict, enc_type='label')
    m_smi = sf.decoder(m_sf)
    m_smi = Chem.CanonSmiles(m_smi)
    return m_smi


def cal_logp(smi):
    m = Chem.MolFromSmiles(smi)
    logP = MolLogP(m)
    return logP





def property_plot(args, logps, y_hat, data_properties):
    from matplotlib import pyplot as plt
    import seaborn as sns
    import pandas as pd
    data_logps, data_sass, data_qeds = data_properties
    l = len(logps)

    data_logps, data_sass, data_qeds = data_logps[:l], data_sass[:l], data_qeds[:l]

    assert len(logps) == len(data_logps)
    print(len(logps), len(data_logps), len(y_hat))
    labels = ['data'] * len(data_logps) + ['Latent EBM'] * len(logps) + ['LSEBM Pred'] * len(y_hat)
    logp_df = pd.DataFrame(data={'model': labels, 'logP': data_logps + logps + y_hat})
    print(y_hat[0], y_hat[1000])
    logp_df["logP"] = pd.to_numeric(logp_df["logP"])

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=[6.4 * 1, 4.8])
    fig.suptitle("logP", fontsize=12)
    logp_plot = sns.kdeplot(data=logp_df, x='logP', hue='model', ax=ax)

    plt.show()
    plt.close(fig)
    # plt.savefig(os.path.join(args.output_dir, 'epoch_{:03d}.png'.format(epoch)))


def single_property_plot_qed(args, epoch, qeds, y_hat, data_properties):
    from matplotlib import pyplot as plt
    import seaborn as sns
    import pandas as pd
    data_logps, data_sass, data_qeds = data_properties
    l = len(qeds)

    data_logps, data_sass, data_qeds = data_logps[:l], data_sass[:l], data_qeds[:l]

    assert len(qeds) == len(data_qeds)
    print(len(qeds), len(data_qeds), len(y_hat))
    labels = ['data'] * len(data_qeds) + ['Latent EBM'] * len(qeds) + ['LSEBM Pred'] * len(y_hat)
    qed_df = pd.DataFrame(data={'model': labels, 'qed': data_qeds + qeds + y_hat})

    qed_df["qed"] = pd.to_numeric(qed_df["qed"])

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=[6.4 * 1, 4.8])
    fig.suptitle("QED", fontsize=12)
    qed_plot = sns.kdeplot(data=qed_df, x='qed', hue='model', ax=ax)

    plt.show()
    plt.close(fig)
    # plt.savefig(os.path.join(args.output_dir, 'single_qed_{:03d}.png'.format(epoch)))


def single_property_plot_plogp(args, epoch, plogps, y_hat, data_plogp):
    from matplotlib import pyplot as plt
    import seaborn as sns
    import pandas as pd

    print(len(plogps), len(data_plogp), len(y_hat))
    labels = ['data'] * len(data_plogp) + ['Latent EBM'] * len(plogps) + ['LSEBM Pred'] * len(y_hat)
    logp_df = pd.DataFrame(data={'model': labels, 'PlogP': data_plogp + plogps + y_hat})
    logp_df["PlogP"] = pd.to_numeric(logp_df["PlogP"])

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=[6.4 * 1, 4.8])
    fig.suptitle("PlogP", fontsize=12)
    logp_plot = sns.kdeplot(data=logp_df, x='PlogP', hue='model', ax=ax)

    plt.show()
    plt.close(fig)
    # plt.savefig(os.path.join(args.output_dir, 'single_plogp_{:03d}.png'.format(epoch)))


def smiles_to_affinity(smiles, autodock, protein_file, num_devices=torch.cuda.device_count()):
    if not os.path.exists('ligands'):
        os.mkdir('ligands')
    if not os.path.exists('outs'):
        os.mkdir('outs')
    subprocess.run('rm core.*', shell=True, stderr=subprocess.DEVNULL)
    subprocess.run('rm outs/*.xml', shell=True, stderr=subprocess.DEVNULL)
    subprocess.run('rm outs/*.dlg', shell=True, stderr=subprocess.DEVNULL)
    subprocess.run('rm -rf ligands/*', shell=True, stderr=subprocess.DEVNULL)
    for device in range(num_devices):
        os.mkdir(f'ligands/{device}')
    device = 0
    for i, hot in enumerate(tqdm(smiles, desc='preparing ligands')):
        subprocess.Popen(
            f'obabel -:"{smiles[i]}" -O ligands/{device}/ligand{i}.pdbqt -p 7.4 --partialcharge gasteiger --gen3d',
            shell=True, stderr=subprocess.DEVNULL)
        device += 1
        if device == num_devices:
            device = 0
    while True:
        total = 0
        for device in range(num_devices):
            total += len(os.listdir(f'ligands/{device}'))
        if total == len(smiles):
            break
    time.sleep(1)
    print('running autodock..')
    if len(smiles) == 1:
        subprocess.run(f'{autodock} -M {protein_file} -s 0 -L ligands/0/ligand0.pdbqt -N outs/ligand0', shell=True,
                       stdout=subprocess.DEVNULL)
    else:
        ps = []
        for device in range(num_devices):
            ps.append(subprocess.Popen(
                f'{autodock} -M {protein_file} -s 0 -B ligands/{device}/ligand*.pdbqt -N ../../outs/ -D {device + 1}',
                shell=True, stdout=subprocess.DEVNULL))
        stop = False
        while not stop:
            for p in ps:
                stop = True
                if p.poll() is None:
                    time.sleep(1)
                    stop = False
    affins = [0 for _ in range(len(smiles))]
    for file in tqdm(os.listdir('outs'), desc='extracting binding values'):
        if file.endswith('.dlg') and '0.000   0.000   0.000  0.00  0.00' not in open(f'outs/{file}').read():
            affins[int(file.split('ligand')[1].split('.')[0])] = float(
                subprocess.check_output(f"grep 'RANKING' outs/{file} | tr -s ' ' | cut -f 5 -d ' ' | head -n 1",
                                        shell=True).decode('utf-8').strip())
    return [min(affin, 0) for affin in affins]


class DesignDataset(Dataset):
    def __init__(self, x, ba, sas, qed):
        self.Xdata = x
        self.ba = ba
        self.sas = sas
        self.qed = qed
        self.len = len(ba)

    def __getitem__(self, index):
        # Add sos=108 for each sequence and this sos is not shown in char_list and char_dict as in selfies.
        mol = self.Xdata[index]
        sos = torch.tensor([108], dtype=torch.long)
        mol = torch.cat([sos, mol], dim=0).contiguous()
        # return (mol, self.ba[index], self.sas[index], self.qed[index])
        return (mol, torch.tensor(self.ba[index]), torch.tensor(self.sas[index]), torch.tensor(self.qed[index]))

    def __len__(self):
        return self.len


def multi_prop_design(model_path, args):
    args = args
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    test_data = MolDataset(args.data_dir, "design")

    logger.info('loading model from ' + model_path)
    vocab_size = 109
    model = RNNEBM(args, vocab_size=vocab_size,
                   dec_word_dim=args.dec_word_dim,
                   dec_h_dim=args.dec_h_dim,
                   dec_num_layers=args.dec_num_layers,
                   dec_dropout=args.dec_dropout,
                   latent_dim=args.latent_dim,
                   max_sequence_length=args.max_len)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model'].state_dict())
    model.cuda()
    print(model)

    args.beta = 1
    num_design = 30

    sos = torch.tensor([108], dtype=torch.long)
    criterion = nn.NLLLoss(reduction='sum')

    prior_params = [p[1] for p in model.named_parameters() if 'prior' in p[0] and p[1].requires_grad is True]
    likelihood_params = [p[1] for p in model.named_parameters() if 'prior' not in p[0] and p[1].requires_grad is True]

    optimizer_prior = torch.optim.Adam(prior_params, lr=args.prior_lr, weight_decay=args.ebm_reg)
    optimizer = torch.optim.Adam(likelihood_params, lr=args.lr)

    properties = []
    Xdata, ba_data, sas_data, qed_data = 0, 0, 0, 0
    for epoch in range(num_design):
        if epoch == 0:
            design_data = DesignDataset(x=test_data.Xdata, ba=test_data.ba1, sas=test_data.sas, qed=test_data.qed)
        else:
            design_data = DesignDataset(x=Xdata, ba=ba_data, sas=sas_data, qed=qed_data)
        design_loader = DataLoader(dataset=design_data, batch_size=1000, shuffle=True, drop_last=False)

        # Train Model based on new dataset
        model.train()
        train_data = design_data
        train_loader = DataLoader(dataset=train_data, batch_size=1000, shuffle=True, drop_last=False)
        for i, data in enumerate(tqdm(train_loader)):
            x, ba, sas, qed = data
            x, ba, sas, qed = x.cuda(), ba.cuda(), sas.cuda(), qed.cuda()
            target = x.detach().clone()
            target = target[:, 1:]
            batch_size = x.size(0)

            # generator update
            optimizer.zero_grad()

            z_0_posterior = sample_p_0(x, args)
            z_samples, z_grads = model.infer_z(z=z_0_posterior, x=x, x_len=None, y=[ba, sas, qed], beta=args.beta,
                                               step_size=args.z_step_size, training=True,
                                               dropout=args.dec_dropout, debug=args.debug)
            preds = model.decoder(x, z_samples, dropout=args.dec_dropout)

            preds = preds.view(-1, preds.size(2))
            target = target.reshape(-1)

            abp_loss = criterion(preds, target) / batch_size

            mlp_error = 0
            # if args.single_design:
            #     y_hat = model.mlp(z_samples)
            #     mlp_error = args.prop_coefficient * F.mse_loss(y_hat, y, reduction='mean')
            #     abp_loss += mlp_error

            if args.multi_design:
                ba_hat = model.mlp_ba(z_samples)
                qed_hat = model.mlp_qed(z_samples)
                sas_hat = model.mlp_sas(z_samples)

                ba_loss = args.ba * F.mse_loss(ba_hat, ba, reduction='mean')
                sas_loss = args.sas * F.mse_loss(sas_hat, sas, reduction='mean')
                qed_loss = args.qed * F.mse_loss(qed_hat, qed, reduction='mean')
                mlp_error = ba_loss + sas_loss + qed_loss
                abp_loss += mlp_error

            # generator update
            optimizer.zero_grad()
            optimizer_prior.zero_grad()

            abp_loss.backward()
            optimizer.step()

            # ebm update
            num_ebm_updates = 2
            for j in range(num_ebm_updates):
                optimizer_prior.zero_grad()

                z_0_prior = sample_p_0(x, args)
                z_prior, z_prior_grads_norm = model.infer_prior_z(z_0_prior, args)
                positive_potential = model.ebm_prior(z_samples, args).mean()
                negative_potential = model.ebm_prior(z_prior, args).mean()
                cd = positive_potential - negative_potential
                negative_cd = -cd
                negative_cd.backward()
                optimizer_prior.step()

        # controlled generation
        model.eval()
        Plogps, SASs, QEDs = [], [], []
        x_old, x_new = [], []
        sas_old, sas_new = [], []
        qed_old, qed_new = [], []
        ba_old, ba_new = [], []

        generated_samples = []
        for i, data in enumerate(tqdm(design_loader)):
            x, ba, sas, qed = data
            x, ba, sas, qed = x.cuda(), ba.cuda(), sas.cuda(), qed.cuda()
            target = x.detach().clone()
            target = target[:, 1:]
            batch_size = x.size(0)

            z_0 = sample_p_0(x, args)
            z_y, _ = model.infer_z_given_y(z_0, [ba + 0.5, sas - 0.05, qed + 0.05], n_iter=30, step_size=0.5)

            samples, _ = model.inference(x.device, sos_idx=108, z=z_y, training=False)
            # ba_hat = model.mlp_ba(z_y)
            # sas_hat = model.mlp_sas(z_y)
            # qed_hat = model.mlp_qed(z_y)
            for idx, s in enumerate(samples):
                generated_samples.append(label2sf2smi(s.cpu().numpy()))
                x_old.append(x[idx, 1:].cpu().numpy())
                # mol = torch.cat([sos, s[:-1].cpu()], dim=0).contiguous()
                x_new.append(s.cpu().numpy())
                ba_old.append(ba[idx].cpu().numpy())
                qed_old.append(qed[idx].cpu().numpy())
                sas_old.append(sas[idx].cpu().numpy())

        unique_set = set(generated_samples)  # Todo: two different smiles might be the same molecule
        validity = 1
        uniqueness = len(unique_set) / len(generated_samples)

        # novelty
        ZINC_file = "../ZINC/test_5.txt"
        ZINC_data = [x.strip().split()[0] for x in open(ZINC_file) if not x.startswith("#smi")]
        ZINC_set = set(ZINC_data)
        novel_list = list(unique_set - ZINC_set)
        novelty = len(novel_list) / len(generated_samples)
        print('Here', validity, uniqueness, novelty)
        logger.record_tabular('Prior Validity', validity)
        logger.record_tabular('Prior Uniqueness', uniqueness)
        logger.record_tabular('Prior Novelty', novelty)
        # logger.record_tabular('Posterior Validity', posterior_prop_valid)

        max_sa, min_qed = 5.5, 0.4
        filtered_samples = []

        # if epoch > 0:
        #     for i in range(len(x_old)):
        #         x_new.append(x_old[i])
        # generated_samples.append(label2sf2smi(x_old[i]))

        IDs = []
        for id, s in enumerate(tqdm(generated_samples, desc='filtering')):
            _is_valid = is_valid(s)
            if _is_valid != 0:
                logP, SAS, QED, plogp, cycle_count = _is_valid[1]
                if cycle_count == 0 and (SAS < max_sa) and (QED > min_qed):
                    SASs.append(SAS)
                    QEDs.append(QED)
                    filtered_samples.append(s)
                    IDs.append(id)
        num_print = 30
        x_new = np.array(x_new)
        x_new = x_new[IDs]
        qed_new = np.array(QEDs)
        # ba_new = torch.zeros(len(qed_new))
        ba_new = smiles_to_affinity(filtered_samples, args.autodock_executable, args.protein_file, num_devices=1)
        ba_new = np.array(ba_new) * (-1)
        ind = np.argsort(ba_new)
        ind = ind[::-1][:num_print]
        kd = np.exp(ba_new[ind] * (-1) / (0.00198720425864083 * 298.15)).flatten()
        sas_new = np.array(SASs) / 10.

        name = str(epoch) + '_ba.npy'
        np.save(name, ba_new)
        name = str(epoch) + '_qed.npy'
        np.save(name, qed_new)
        name = str(epoch) + '_sa.npy'
        np.save(name, sas_new)

        mols = []
        props = []
        print('ba     Kd     SAs     QED     smi')
        for i in range(len(ind)):
            id = ind[i]
            print(ba_new[id], kd[i], sas_new[id] * 10, qed_new[id], filtered_samples[id])
            m = Chem.MolFromSmiles(filtered_samples[id])
            mols.append(m)
            # s = '{:.5f}\n'.format(kd[i] * 10 ** 9)
            s = '{:.3f} '.format(kd[i] * 10 ** 9) + '{:.2f} '.format(sas_new[id] * 10) + '{:.3f}\n'.format(qed_new[id])
            props.append(s)
        fig = Draw.MolsToGridImage(mols, molsPerRow=5, legends=props)
        fig.save('Kd_epoch' + str(epoch) + '.png')



        # Prepare New dataset
        x_old, x_new = torch.tensor(np.array(x_old)), torch.tensor(np.array(x_new))
        print(x_old.shape, x_new.shape)
        # if x_old.shape[1] != x_new.shape[1]:
        #     x_tmp = torch.zeros(x_old.shape[0], x_new.shape[1])
        #     x_tmp[:, :] = 58.
        #     x_tmp[:, :x_old.shape[1]] = x_old
        #     x_old = x_tmp
        ba_new = torch.tensor(np.array(ba_new), dtype=torch.float).squeeze()
        sas_new = torch.tensor(np.array(sas_new), dtype=torch.float).squeeze()
        qed_new = torch.tensor(np.array(qed_new), dtype=torch.float).squeeze()

        ba_old = torch.tensor(np.array(ba_old), dtype=torch.float).squeeze()
        sas_old = torch.tensor(np.array(sas_old), dtype=torch.float).squeeze()
        qed_old = torch.tensor(np.array(qed_old), dtype=torch.float).squeeze()

        assert len(ba_old) == len(sas_old) == len(qed_old)

        Xdata = torch.cat([x_old, x_new], dim=0)
        ba_data = torch.cat([ba_old, ba_new])
        sas_data = torch.cat([sas_old, sas_new])
        qed_data = torch.cat([qed_old, qed_new])

        max_len = 10000
        if len(ba_data) > max_len:
            ba_new = torch.tensor(ba_new)
            sorted, indices = torch.sort(ba_new, descending=True)
            Xdata = Xdata[indices[:max_len], :]
            ba_data = ba_data[indices[:max_len]]
            sas_data = sas_data[indices[:max_len]]
            qed_data = qed_data[indices[:max_len]]


if __name__ == '__main__':
    # smi2selfies()
    args = get_args()
    # print(args.mask)
    exp_id = 'ebm_multi_design_ba1'
    output_dir = get_output_dir(exp_id, fs_prefix='../exp_')
    # copy_source(__file__, output_dir)
    set_gpu(args.gpu)

    args.output_dir = output_dir
    args.max_len = 72
    model_path = '24.pt'
    # single_prop_design(model_path, args)
    # check_model(model_path, args)
    multi_prop_design(model_path, args)
