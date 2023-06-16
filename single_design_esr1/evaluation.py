import time
import torch.nn as nn
import selfies as sf
from ZINC.char import char_dict
import torch
from rdkit import Chem
from rdkit.Chem.Crippen import MolLogP
from rdkit.Chem.QED import qed
import sascorer
import os
from tqdm import tqdm
import numpy as np
import subprocess
import math


def sample_p_0(x, args):
    if args.ref_dist == 'gaussian':
        return args.init_factor * torch.randn(*[x.size(0), args.latent_dim], device=x.device)
    else:
        return torch.Tensor(*[x.size(0), args.latent_dim]).uniform_(-1, 1).to(x.device)


##--------------------------------------------------------------------------------------------------------------------##
def evaluation(args, epoch, test_loader, model, logger, writer, mode='Val'):
    model.prior_network.eval()
    criterion = nn.NLLLoss(reduction='sum')
    total_nll_abp = 0.
    pps = 0.
    nps = 0.
    cds = 0.
    n_cd = 0.
    test_data_size = 0
    generated_samples = []
    bas, qeds, sass = [], [], []
    for i, data in enumerate(tqdm(test_loader)):
        # if i > 2:
        #     break
        x, x_len, ba, sas, qed = data
        x, ba, sas, qed = x.cuda(), ba.cuda(), sas.cuda(), qed.cuda()
        x_len = x_len.cuda()
        target = x.detach().clone()
        target = target[:, 1:]
        batch_size = x.size(0)

        z_0 = sample_p_0(x, args)
        z_samples, z_grads = model.infer_z(z=z_0, x=x, x_len=x_len, y=[ba, sas, qed], beta=args.beta,
                                           step_size=args.z_step_size, training=False,
                                           dropout=args.dec_dropout, debug=args.debug)
        # z_samples, _ = model.infer_z(z=z_0, x=x, beta=args.beta,
        #                              step_size=args.z_step_size, training=False,
        #                              dropout=args.dec_dropout)
        preds = model.decoder(x, z_samples, training=False)
        abp_loss = criterion(preds.view(-1, preds.size(2)), target.reshape(-1)) / batch_size

        z_0_prior = sample_p_0(x, args)
        z_prior, _ = model.infer_prior_z(z_0_prior, args)
        positive_potential = model.ebm_prior(z_samples, args).mean()
        negative_potential = model.ebm_prior(z_prior, args).mean()
        cd = positive_potential - negative_potential
        pps += positive_potential
        nps += negative_potential
        cds += cd
        n_cd += 1.

        total_nll_abp += abp_loss.item()
        test_data_size += batch_size

        samples, _ = model.inference(x.device, sos_idx=108, z=z_prior, training=False)
        qed_hat = model.mlp_qed(z_prior)
        ba_hat = model.mlp_ba(z_prior)
        sas_hat = model.mlp_sas(z_prior)
        for idx, s in enumerate(samples):
            generated_samples.append(label2sf2smi(s.cpu().numpy()))
            bas.append(ba_hat[idx].detach().cpu().numpy())
            qeds.append(qed_hat[idx].detach().cpu().numpy())
            s_score = sas_hat[idx] * 10
            sass.append(s_score.detach().cpu().numpy())

    bas = np.array(bas)
    bas = list(bas)
    qeds = np.array(qeds)
    qeds = list(qeds)
    sass = np.array(sass)
    sass = list(sass)

    logPs, SASs, QEDs, PlogPs = [], [], [], []
    num_valid = 0
    for s in generated_samples:
        _is_valid = is_valid(s)
        if _is_valid != 0:
            num_valid += _is_valid[0]
            logP, SAS, QED, PlogP = _is_valid[1]
            logPs.append(logP)
            SASs.append(SAS)
            QEDs.append(QED)
            PlogPs.append(PlogP)
    validity = num_valid / len(generated_samples)
    data_properties = get_data_properties(args.test_file)

    if args.tb:
        plot = property_plot(args, logPs, SASs, QEDs, epoch, data_properties)
    else:
        property_plot(args, logPs, SASs, QEDs, epoch, data_properties)

    from dataset import MolDataset
    test_data = MolDataset(args.data_dir, "test")
    data_ba0 = test_data.ba0
    data_ba0 = data_ba0.numpy() * (-1.)
    data_ba0 = data_ba0.tolist()

    single_property_plot_ba_simple(args, epoch, bas, data_ba0)
    single_property_plot_qed(args, epoch, QEDs, qeds, data_properties)
    single_property_plot_sas(args, epoch, SASs, sass, data_properties)

    # uniqueness
    unique_set = set(generated_samples)  
    validity = num_valid / len(generated_samples)
    uniqueness = len(unique_set) / len(generated_samples)

    # novelty
    ZINC_file = "../ZINC/test_5.txt"
    ZINC_data = [x.strip().split()[0] for x in open(ZINC_file) if not x.startswith("#smi")]
    ZINC_set = set(ZINC_data)
    novel_list = list(unique_set - ZINC_set)
    novelty = len(novel_list) / len(generated_samples)

    rec_abp = total_nll_abp / test_data_size
    pe = pps / n_cd
    ne = nps / n_cd
    mcd = cds / n_cd

    logger.record_tabular('ABP REC', rec_abp)
    logger.record_tabular('PP', pe)
    logger.record_tabular('NP', ne)
    logger.record_tabular('CD', mcd)
    logger.record_tabular('Prior Validity', validity)
    logger.record_tabular('Prior Uniqueness', uniqueness)
    logger.record_tabular('Prior Novelty', novelty)
    # logger.record_tabular('Posterior Validity', posterior_prop_valid)

    logger.dump_tabular()

    if writer is not None:
        writer.add_scalar(mode + '/PP', pe, epoch)
        writer.add_scalar(mode + '/NP', ne, epoch)
        writer.add_scalar(mode + '/CD', mcd, epoch)

    model.train()
    return writer, validity


def label2sf2smi(out_num):
    m_sf = sf.encoding_to_selfies(out_num, char_dict, enc_type='label')
    m_smi = sf.decoder(m_sf)
    m_smi = Chem.CanonSmiles(m_smi)
    return m_smi


def is_valid(smi):
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return 0
    if Chem.SanitizeMol(m, catchErrors=True):
        return 0
    smi2 = Chem.MolToSmiles(m)
    return 1, calc_prop(m)


def calc_prop(m):
    logP = MolLogP(m)
    try:
        SAS = sascorer.calculateScore(m)
    except ZeroDivisionError:
        SAS = 2.8
    QED = qed(m)
    PlogP = logP - SAS
    cycle_count = 0
    for ring in m.GetRingInfo().AtomRings():
        if len(ring) > 6:
            PlogP -= 1
        if not (4 < len(ring) < 7):
            cycle_count += 1
    return (logP, SAS, QED, PlogP, cycle_count)


def strip_special(smi0):
    index = smi0.find('>')
    smi = smi0[0:index].strip('<')
    return smi


def property_plot(args, logps, sass, qeds, epoch, data_properties):
    from matplotlib import pyplot as plt
    import seaborn as sns
    import pandas as pd
    data_logps, data_sass, data_qeds = data_properties
    assert len(data_logps) == len(data_sass) == len(data_qeds)
    assert len(logps) == len(sass) == len(qeds)
    l = len(logps)
    data_logps, data_sass, data_qeds = data_logps[:l], data_sass[:l], data_qeds[:l]

    assert len(logps) == len(data_logps)
    labels = ['data'] * len(data_logps) + ['Latent EBM'] * len(logps)
    logp_df = pd.DataFrame(data={'model': labels, 'logP': data_logps + logps})
    logp_df["logP"] = pd.to_numeric(logp_df["logP"])
    sas_df = pd.DataFrame(data={'model': labels, 'SAS': data_sass + sass})
    sas_df["SAS"] = pd.to_numeric(sas_df["SAS"])
    qed_df = pd.DataFrame(data={'model': labels, 'QED': data_qeds + qeds})
    qed_df["QED"] = pd.to_numeric(qed_df["QED"])

    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=[6.4 * 3, 4.8])
    fig.suptitle("Epoch: " + str(epoch), fontsize=12)
    logp_plot = sns.kdeplot(data=logp_df, x='logP', hue='model', ax=ax[0])
    sas_plot = sns.kdeplot(data=sas_df, x='SAS', hue='model', ax=ax[1])
    qed_plot = sns.kdeplot(data=qed_df, x='QED', hue='model', ax=ax[2])

    if args.tb:
        return fig
    else:
        # plt.show()
        # plt.close(fig)
        plt.savefig(os.path.join(args.output_dir, 'epoch_{:03d}.png'.format(epoch)))


def single_property_plot_logp(args, epoch, logps, y_hat, data_properties):
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
    logp_df["logP"] = pd.to_numeric(logp_df["logP"])

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=[6.4 * 1, 4.8])
    fig.suptitle("logP", fontsize=12)
    logp_plot = sns.kdeplot(data=logp_df, x='logP', hue='model', ax=ax)

    # plt.show()
    # plt.close(fig)
    plt.savefig(os.path.join(args.output_dir, 'single_epoch_{:03d}.png'.format(epoch)))


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

    # plt.show()
    # plt.close(fig)
    plt.savefig(os.path.join(args.output_dir, 'single_plogp_{:03d}.png'.format(epoch)))


def single_property_plot_ba(args, epoch, bas, y_hat, data_ba):
    from matplotlib import pyplot as plt
    import seaborn as sns
    import pandas as pd

    print(len(bas), len(data_ba), len(y_hat))
    l = len(bas)
    data_ba = data_ba[:l]
    labels = ['data'] * len(data_ba) + ['Latent EBM'] * len(bas) + ['LSEBM Pred'] * len(y_hat)
    logp_df = pd.DataFrame(data={'model': labels, 'ba': data_ba + bas + y_hat})
    logp_df["ba"] = pd.to_numeric(logp_df["ba"])

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=[6.4 * 1, 4.8])
    fig.suptitle("ba", fontsize=12)
    logp_plot = sns.kdeplot(data=logp_df, x='ba', hue='model', ax=ax)

    # plt.show()
    # plt.close(fig)
    plt.savefig(os.path.join(args.output_dir, 'single_ba_{:03d}.png'.format(epoch)))


def single_property_plot_ba_simple(args, epoch, y_hat, data_ba):
    from matplotlib import pyplot as plt
    import seaborn as sns
    import pandas as pd

    print(len(data_ba), len(y_hat))
    labels = ['data'] * len(data_ba) + ['LSEBM Pred'] * len(y_hat)
    logp_df = pd.DataFrame(data={'model': labels, 'ba': data_ba + y_hat})
    logp_df["ba"] = pd.to_numeric(logp_df["ba"])

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=[6.4 * 1, 4.8])
    fig.suptitle("ba", fontsize=12)
    logp_plot = sns.kdeplot(data=logp_df, x='ba', hue='model', ax=ax)

    # plt.show()
    # plt.close(fig)
    plt.savefig(os.path.join(args.output_dir, 'single_ba_{:03d}.png'.format(epoch)))


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
    # print(y_hat[0], y_hat[1000])
    qed_df["qed"] = pd.to_numeric(qed_df["qed"])

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=[6.4 * 1, 4.8])
    fig.suptitle("QED", fontsize=12)
    qed_plot = sns.kdeplot(data=qed_df, x='qed', hue='model', ax=ax)

    # plt.show()
    # plt.close(fig)
    plt.savefig(os.path.join(args.output_dir, 'single_qed_{:03d}.png'.format(epoch)))

def single_property_plot_sas(args, epoch, sass, y_hat, data_properties):
    from matplotlib import pyplot as plt
    import seaborn as sns
    import pandas as pd
    data_logps, data_sass, data_qeds = data_properties
    l = len(sass)

    data_logps, data_sass, data_qeds = data_logps[:l], data_sass[:l], data_qeds[:l]

    assert len(sass) == len(data_sass)
    print(len(sass), len(data_sass), len(y_hat))
    labels = ['data'] * len(data_sass) + ['Latent EBM'] * len(sass) + ['LSEBM Pred'] * len(y_hat)
    qed_df = pd.DataFrame(data={'model': labels, 'sas': data_sass + sass + y_hat})
    # print(y_hat[0], y_hat[1000])
    qed_df["sas"] = pd.to_numeric(qed_df["sas"])

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=[6.4 * 1, 4.8])
    fig.suptitle("SAS", fontsize=12)
    sas_plot = sns.kdeplot(data=qed_df, x='sas', hue='model', ax=ax)
    plt.savefig(os.path.join(args.output_dir, 'single_sas_{:03d}.png'.format(epoch)))

def get_data_properties(path, seq_length=110):
    with open(path) as f:
        data_lines = f.readlines()

    logps = []
    sass = []
    qeds = []
    title = ""
    for line in data_lines:
        if line[0] == "#":
            title = line[1:-1]
            title_list = title.split()
            print(title_list)
            continue
        arr = line.split()
        if len(arr) < 2:
            continue
        smiles = arr[0]
        if len(smiles) > seq_length:
            continue
        assert len(arr) == 6
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
    return (logps, sass, qeds)

