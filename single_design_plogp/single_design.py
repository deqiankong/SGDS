import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.nn.functional as F
from dataset import MolDataset
from args import get_args
from utils import get_output_dir, copy_source, set_gpu
from model_plogp import RNNEBM
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

    test_data = MolDataset(args.data_dir, "test")
    test_loader = DataLoader(dataset=test_data, batch_size=1000,
                             shuffle=True, drop_last=False, num_workers=2)

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

    sos = torch.tensor([108], dtype=torch.long)
    criterion = nn.NLLLoss(reduction='sum')

    prior_params = [p[1] for p in model.named_parameters() if 'prior' in p[0] and p[1].requires_grad is True]
    likelihood_params = [p[1] for p in model.named_parameters() if 'prior' not in p[0] and p[1].requires_grad is True]

    optimizer_prior = torch.optim.Adam(prior_params, lr=args.prior_lr, weight_decay=args.ebm_reg)
    optimizer = torch.optim.Adam(likelihood_params, lr=args.lr)

    properties = []
    Xdata, Ydata = 0, 0
    for epoch in range(num_design):
        if epoch == 0:
            design_data = DesignDataset(x=test_data.Xdata, y=test_data.PlogP)
        else:
            design_data = DesignDataset(x=Xdata, y=Ydata)
        design_loader = DataLoader(dataset=design_data, batch_size=1000, shuffle=True, drop_last=False)
        # controlled generation
        model.eval()
        Plogps = []
        x_old, x_new = [], []
        y_old, y_new = [], []

        generated_samples = []
        for i, data in enumerate(tqdm(design_loader, desc='Generating')):
            x, y = data
            x, y = x.cuda(), y.cuda()
            target = x.detach().clone()
            target = target[:, 1:]
            batch_size = x.size(0)

            z_0 = sample_p_0(x, args)
            if epoch > 0:
                z_0 = latent_z_old[i * 1000:(i + 1) * 1000, :]
            z_y, _ = model.infer_z_given_y(z_0, y + 1, n_iter=2, step_size=0.5)

            # z_y shape: 1000, 100
            if i == 0:
                latent_z_new = z_y
            else:
                latent_z_new = torch.cat((latent_z_new, z_y), 0)

            samples, _ = model.inference(x.device, sos_idx=108, z=z_y, training=False)
            y_hat = model.mlp(z_y)
            for idx, s in enumerate(samples):
                generated_samples.append(label2sf2smi(s.cpu().numpy()))
                x_old.append(x[idx, 1:].cpu().numpy())
                # mol = torch.cat([sos, s[:-1].cpu()], dim=0).contiguous()
                x_new.append(s.cpu().numpy())
                y_old.append(y[idx].cpu().numpy())

        for _, s in enumerate(tqdm(generated_samples, desc='Calling Oracle')):
            _is_valid = is_valid(s)
            if _is_valid != 0:
                logP, SAS, QED, plogp = _is_valid[1]
                Plogps.append(plogp)
        y_new = Plogps
        Plogps = np.array(Plogps)
        ind = np.argsort(Plogps)
        # sorted = -np.sort(-QEDs)
        # print(sorted[:10])
        ind = ind[::-1][:20]
        print(Plogps[ind])

        name = str(epoch) + '.npy'
        np.save(name, Plogps)
        mols = []
        props = []
        for i in range(len(ind)):
            id = ind[i]
            m = Chem.MolFromSmiles(generated_samples[id])
            mols.append(m)
            s = '{:.5f}\n'.format(Plogps[id])
            print(s, generated_samples[id])
            props.append(s)
        fig = Draw.MolsToGridImage(mols, molsPerRow=5, legends=props)
        fig.save('Epoch_plogp' + str(epoch) + '.png')

        # Prepare New dataset
        x_old, x_new = torch.tensor(np.array(x_old)), torch.tensor(np.array(x_new))
        # print(x_old.shape, x_new.shape)
        if x_old.shape[1] != x_new.shape[1]:
            x_tmp = torch.zeros_like(x_new)
            x_tmp[:, :] = 58.
            x_tmp[:, :x_old.shape[1]] = x_old
            x_old = x_tmp
        y_new, y_old = torch.tensor(np.array(y_new), dtype=torch.float).squeeze(), torch.tensor(np.array(y_old),
                                                                                                dtype=torch.float).squeeze()

        if epoch > 0:
            y_data = torch.cat([y_old, y_new])
            x_data = torch.cat([x_old, x_new], dim=0)
            sorted, indices = torch.sort(y_data, descending=True)
            Xdata = x_data[indices[:10000], :]
            Ydata = y_data[indices[:10000]]
            latent_z_all = torch.cat([latent_z_old, latent_z_new], dim=0)
            latent_z_old = latent_z_all[indices[:10000], :]
        else:
            Xdata = x_new
            Ydata = y_new
            latent_z_old = latent_z_new
            # print(latent_z_old.shape)

        # Train Model based on new dataset
        model.train()
        train_data = DesignDataset(x=Xdata, y=Ydata)
        train_loader = DataLoader(dataset=train_data, batch_size=1000, shuffle=True, drop_last=False)
        for i, data in enumerate(tqdm(train_loader, desc='SGDS')):
            x, y = data
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
                # generator update
            optimizer.zero_grad()
            optimizer_prior.zero_grad()

            abp_loss.backward()
            optimizer.step()

            # ebm update
            optimizer_prior.zero_grad()

            z_0_prior = sample_p_0(x, args)
            z_prior, z_prior_grads_norm = model.infer_prior_z(z_0_prior, args)
            positive_potential = model.ebm_prior(z_samples, args).mean()
            negative_potential = model.ebm_prior(z_prior, args).mean()
            cd = positive_potential - negative_potential
            negative_cd = -cd
            negative_cd.backward()
            optimizer_prior.step()


if __name__ == '__main__':
    # smi2selfies()
    args = get_args()
    # print(args.mask)
    exp_id = 'ebm_design'
    output_dir = get_output_dir(exp_id, fs_prefix='../alienware_')
    # copy_source(__file__, output_dir)
    set_gpu(args.gpu)

    args.output_dir = output_dir
    args.max_len = 72
    model_path = '23.pt'
    single_prop_design(model_path, args)
    # check_model(model_path, args)
