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
    def __init__(self, x, y):
        self.Xdata = x
        self.y = y
        self.len = len(y)

    def __getitem__(self, index):
        # Add sos=108 for each sequence and this sos is not shown in char_list and char_dict as in selfies.
        mol = self.Xdata[index]
        sos = torch.tensor([108], dtype=torch.long)
        mol = torch.cat([sos, mol], dim=0).contiguous()
        return (mol, self.y[index])

    def __len__(self):
        return self.len


def single_prop_design(model_path, args):
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
    num_design = 50

    bs = 1000

    sos = torch.tensor([108], dtype=torch.long)
    criterion = nn.NLLLoss(reduction='sum')

    prior_params = [p[1] for p in model.named_parameters() if 'prior' in p[0] and p[1].requires_grad is True]
    likelihood_params = [p[1] for p in model.named_parameters() if 'prior' not in p[0] and p[1].requires_grad is True]

    optimizer_prior = torch.optim.Adam(prior_params, lr=args.prior_lr, weight_decay=args.ebm_reg)
    optimizer = torch.optim.Adam(likelihood_params, lr=args.lr)

    Xdata, ba_data, sas_data, qed_data = 0, 0, 0, 0
    x_old, x_new = [], []
    y_old, y_new = [], []
    for epoch in range(num_design):
        if epoch == 0:
            design_data = DesignDataset(x=test_data.Xdata, y=test_data.ba1)
        else:
            design_data = DesignDataset(x=Xdata, y=Ydata)
        design_loader = DataLoader(dataset=design_data, batch_size=bs, shuffle=True, drop_last=False)

        # Train Model based on new dataset
        model.train()
        train_data = design_data
        train_loader = DataLoader(dataset=train_data, batch_size=bs, shuffle=True, drop_last=False)

        num_iters = 1
        for iter in range(num_iters):
            for i, data in enumerate(tqdm(train_loader, desc='Training')):
                x, y = data
                # print('Wrong here', x.shape, y.shape)
                # if epoch >= 1:
                #     input()
                x, y = x.cuda(), y.cuda()
                target = x.detach().clone()
                target = target[:, 1:]
                batch_size = x.size(0)

                # generator update
                optimizer.zero_grad()

                z_0_posterior = sample_p_0(x, args)
                z_samples, z_grads = model.infer_z(z=z_0_posterior, x=x, x_len=None, y=y, beta=args.beta,
                                                   step_size=args.z_step_size, training=True,
                                                   dropout=args.dec_dropout, debug=args.debug)
                preds = model.decoder(x, z_samples, dropout=args.dec_dropout)

                preds = preds.view(-1, preds.size(2))
                target = target.reshape(-1)

                abp_loss = criterion(preds, target) / batch_size

                mlp_error = 0
                if args.single_design:
                    y_hat = model.mlp(z_samples)
                    mlp_error = args.prop_coefficient * F.mse_loss(y_hat, y, reduction='mean')
                    abp_loss += mlp_error

                # if args.multi_design:
                #     ba_hat = model.mlp_ba(z_samples)
                #     qed_hat = model.mlp_qed(z_samples)
                #     sas_hat = model.mlp_sas(z_samples)
                #
                #     ba_loss = args.ba * F.mse_loss(ba_hat, ba, reduction='mean')
                #     sas_loss = args.sas * F.mse_loss(sas_hat, sas, reduction='mean')
                #     qed_loss = args.qed * F.mse_loss(qed_hat, qed, reduction='mean')
                #     mlp_error = ba_loss + sas_loss + qed_loss
                #     abp_loss += mlp_error

                # generator update
                optimizer.zero_grad()
                optimizer_prior.zero_grad()

                abp_loss.backward()
                optimizer.step()

                # ebm update
                num_ebm_updates = 1
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
        x_new = []
        generated_samples = []
        for i, data in enumerate(tqdm(design_loader, desc='SGDS')):
            if i >= 2:
                break
            x, y = data
            x, y = x.cuda(), y.cuda()
            target = x.detach().clone()

            z_0 = sample_p_0(x, args)
            if epoch > 0:
                z_0 = latent_z_old[i * bs:(i + 1) * bs, :]
            z_y, _ = model.infer_z_given_y(z_0, y + 1, n_iter=2, step_size=0.5)

            if i == 0:
                latent_z_new = z_y
            else:
                latent_z_new = torch.cat((latent_z_new, z_y), 0)

            samples, _ = model.inference(x.device, sos_idx=108, z=z_y, training=False)
            # ba_hat = model.mlp_ba(z_y)
            # sas_hat = model.mlp_sas(z_y)
            # qed_hat = model.mlp_qed(z_y)
            for idx, s in enumerate(samples):
                generated_samples.append(label2sf2smi(s.cpu().numpy()))
                # x_old.append(x[idx, 1:].cpu().numpy())
                # mol = torch.cat([sos, s[:-1].cpu()], dim=0).contiguous()
                x_new.append(s.cpu().numpy())
                # y_old.append(y[idx].cpu().numpy())


        if epoch == 0:
            x_old = design_data.Xdata
            y_old = design_data.y

        num_print = 30
        x_new = np.array(x_new)
        y_new = smiles_to_affinity(generated_samples, args.autodock_executable, args.protein_file, num_devices=1)
        y_new = np.array(y_new) * (-1)

        # here save
        name = str(epoch) + '.npy'
        np.save(name, y_new)

        ind = np.argsort(y_new)
        ind = ind[::-1][:num_print]
        kd = np.exp(y_new[ind] * (-1) / (0.00198720425864083 * 298.15)).flatten()

        mols = []
        props = []
        print('ba     Kd     smi')
        for i in range(len(ind)):
            id = ind[i]
            print(y_new[id], kd[i], generated_samples[id])
            m = Chem.MolFromSmiles(generated_samples[id])
            mols.append(m)
            s = '{:.5f}\n'.format(kd[i] * 10 ** 9)
            props.append(s)
        fig = Draw.MolsToGridImage(mols, molsPerRow=5, legends=props)
        fig.save('Kd_epoch' + str(epoch) + '.png')

        # Prepare New dataset
        x_old, x_new = torch.tensor(np.array(x_old)), torch.tensor(np.array(x_new))
        y_new = torch.tensor(np.array(y_new), dtype=torch.float).squeeze()
        y_old = torch.tensor(np.array(y_old), dtype=torch.float).squeeze()

        x_data = torch.cat([x_old, x_new], dim=0)
        y_data = torch.cat([y_old, y_new])

        # print(x_data.shape, y_data.shape)

        if epoch > 0:
            total_samples = 2000
            sorted, indices = torch.sort(y_data, descending=True)
            Xdata = x_data[indices[:total_samples], :]
            Ydata = y_data[indices[:total_samples]]
            latent_z_all = torch.cat([latent_z_old, latent_z_new], dim=0)
            latent_z_old = latent_z_all[indices[:total_samples], :]
            # print(Xdata.shape, Ydata.shape, latent_z_old.shape, latent_z_new.shape, latent_z_all.shape)
            # input()
        else:
            Xdata = x_new
            Ydata = y_new
            latent_z_old = latent_z_new

        # print(Xdata.shape, Ydata.shape, latent_z_old.shape, latent_z_new.shape)
        # max_len = 10000
        # if len(Ydata) > max_len:
        #     Ydata = torch.tensor(Ydata)
        #     sorted, indices = torch.sort(Ydata, descending=True)
        #     Xdata = Xdata[indices[:max_len], :]
        #     Ydata = Ydata[indices[:max_len]]

        x_old = Xdata
        y_old = Ydata



if __name__ == '__main__':
    # smi2selfies()
    args = get_args()
    # print(args.mask)
    exp_id = 'ebm_single_design_ba1'
    output_dir = get_output_dir(exp_id, fs_prefix='../exp_')
    # copy_source(__file__, output_dir)
    set_gpu(args.gpu)

    args.output_dir = output_dir
    args.max_len = 72
    model_path = '16.pt'
    # single_prop_design(model_path, args)
    # check_model(model_path, args)
    single_prop_design(model_path, args)
