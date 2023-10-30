import os
import logger
import numpy as np
import time

import torch
from dataset import MolDataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
from model import RNNEBM, NLLLoss
import torch.nn as nn
from evaluation import evaluation
from utils import get_output_dir, copy_all_files, set_gpu
from args import get_args
from ZINC.char import char_list

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


def sample_p_0(x, args):
    if args.ref_dist == 'gaussian':
        return args.init_factor * torch.randn(*[x.size(0), args.latent_dim], device=x.device)
    else:
        return torch.Tensor(*[x.size(0), args.latent_dim]).uniform_(-1, 1).to(x.device)


def main(args, output_dir):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    train_data = MolDataset(args.data_dir, "train")
    test_data = MolDataset(args.data_dir, "test")
    train_loader = DataLoader(dataset=train_data, batch_size=args.batch_size,
                              shuffle=True, drop_last=True, num_workers=2)
    test_loader = DataLoader(dataset=test_data, batch_size=args.eval_batch_size,
                             shuffle=True, drop_last=False, num_workers=2)

    N_train = len(train_data)
    N_test = len(test_data)
    vocab_size = len(char_list)  

    if args.warmup == 0:
        args.beta = 1.
    else:
        args.beta = 0.001

    checkpoint_dir = output_dir
    args.output_dir = output_dir
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    suffix = "%s.pt" % (args.model)
    checkpoint_path = os.path.join(checkpoint_dir, suffix)

    if args.tb:
        writer = SummaryWriter(log_dir=output_dir)
    else:
        writer = None

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # device = 'cpu'
    if args.train_from == '':
        model = RNNEBM(args, vocab_size=vocab_size,
                       dec_word_dim=args.dec_word_dim,
                       dec_h_dim=args.dec_h_dim,
                       dec_num_layers=args.dec_num_layers,
                       dec_dropout=args.dec_dropout,
                       latent_dim=args.latent_dim,
                       max_sequence_length=args.max_len)
        for param in model.parameters():
            param.data.uniform_(-0.1, 0.1)
    else:
        logger.info('loading model from ' + args.train_from)
        checkpoint = torch.load(args.train_from)
        model = checkpoint['model']

    logger.info("model architecture")
    print(model)

    prior_params = [p[1] for p in model.named_parameters() if 'prior' in p[0] and p[1].requires_grad is True]
    likelihood_params = [p[1] for p in model.named_parameters() if 'prior' not in p[0] and p[1].requires_grad is True]

    optimizer_prior = torch.optim.Adam(prior_params, lr=args.prior_lr, weight_decay=args.ebm_reg)
    optimizer = torch.optim.Adam(likelihood_params, lr=args.lr)

    criterion = nn.NLLLoss(reduction='sum')  
    model.to(device)
    criterion.to(device)
    model.train()

    val_stats = []
    best_prior_prop_valid = 0.

    if True:
        epoch = 0
        logger.info('--------------------------------')
        logger.info('Checking validation perf...')
        logger.record_tabular('Epoch', epoch)
        logger.record_tabular('Mode', 'Val')
        logger.record_tabular('LR', args.lr)
        writer, prior_prop_valid = evaluation(args, epoch, test_loader, model, logger, writer, epoch)

    for epoch in range(args.num_epochs):
        start_time = time.time()
        logger.info('Starting epoch %d' % epoch)

        for i, data in enumerate(tqdm(train_loader)):

            x, x_len, y = data
            x, y = x.to(device), y.to(device)
            x_len = x_len.to(device)
            target = x.detach().clone()
            target = target[:, 1:]
            batch_size = x.size(0)

            # generator update
            optimizer.zero_grad()
            optimizer_prior.zero_grad()

            z_0_posterior = sample_p_0(x, args)
            z_samples, z_grads = model.infer_z(z=z_0_posterior, x=x, x_len=x_len, y=y, beta=args.beta,
                                               step_size=args.z_step_size, training=True,
                                               dropout=args.dec_dropout, debug=args.debug)
            preds = model.decoder(x, z_samples, dropout=args.dec_dropout)

            preds = preds.view(-1, preds.size(2))
            target = target.reshape(-1)
            x_len = x_len.reshape(-1)

            if args.mask:
                abp_loss = NLLLoss(preds, target, x_len, reduction='sum')
                abp_loss = abp_loss / batch_size
            else:
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
            if args.max_grad_norm > 0:
                llhd_grad_norm = torch.nn.utils.clip_grad_norm_(likelihood_params, args.max_grad_norm)
            else:
                llhd_grad_norm = torch.norm(
                    torch.stack([torch.norm(p.grad.clone().detach()) for p in likelihood_params]))
            optimizer.step()

            # ebm update
            optimizer.zero_grad()
            optimizer_prior.zero_grad()

            z_0_prior = sample_p_0(x, args)
            z_prior, z_prior_grads_norm = model.infer_prior_z(z_0_prior, args)
            positive_potential = model.ebm_prior(z_samples, args).mean()
            negative_potential = model.ebm_prior(z_prior, args).mean()
            cd = positive_potential - negative_potential
            negative_cd = -cd

            optimizer.zero_grad()
            optimizer_prior.zero_grad()
            negative_cd.backward()
            if args.max_grad_norm_prior > 0:
                prior_grad_norm = torch.nn.utils.clip_grad_norm_(prior_params, args.max_grad_norm_prior)
            else:
                prior_grad_norm = torch.norm(torch.stack([torch.norm(p.grad.clone().detach()) for p in prior_params]))
            optimizer_prior.step()

            if i % args.print_every == 0:
                with torch.no_grad():
                    positive_potential_2 = model.ebm_prior(z_samples, args).mean()
                    negative_potential_2 = model.ebm_prior(z_prior, args).mean()

                    prior_param_norm = torch.norm(torch.stack([torch.norm(p.clone().detach()) for p in prior_params]))
                    llhd_param_norm = torch.norm(
                        torch.stack([torch.norm(p.clone().detach()) for p in likelihood_params]))
                    param_norm_str = '[ebm:{:8.2f} lh:{:8.2f}]'.format(prior_param_norm, llhd_param_norm)

                    grad_norm_str = '[ebm:{:8.2f} lh:{:8.2f}]'.format(prior_grad_norm, llhd_grad_norm)

                    posterior_z_disp_str = torch.norm(z_0_posterior - z_samples, dim=1).mean()
                    prior_z_disp_str = torch.norm(z_0_prior - z_prior, dim=1).mean()
                    z_disp_str = '[pr:{:8.2f} po:{:8.2f}]'.format(prior_z_disp_str, posterior_z_disp_str)

                    prior_posterior_z_norm_str = '[noise:{:8.2f} pr:{:8.2f} po:{:8.2f}]'.format(
                        torch.norm(z_0_prior, dim=1).mean(),
                        torch.norm(z_prior, dim=1).mean(),
                        torch.norm(z_samples, dim=1).mean())

                    prior_z_grad_norm_str = ' '.join(['{:8.2f}'.format(g) for g in z_prior_grads_norm])
                    posterior_z_f_grad_norm_str = ' '.join(['{:8.2f}'.format(g) for g in z_grads[0]])
                    posterior_z_nll_grad_norm_str = ' '.join(['{:8.2f}'.format(g) for g in z_grads[1]])

                    prior_moments = '[{:8.2f}, {:8.2f}, {:8.2f}]'.format(z_prior.mean(), z_prior.std(),
                                                                         z_prior.abs().max())
                    posterior_moments = '[{:8.2f}, {:8.2f}, {:8.2f}]'.format(z_samples.mean(), z_samples.std(),
                                                                             z_samples.abs().max())

                logger.info('Epoch={:4d}, Batch={:4d}/{:4d}, LR={:8.6f}, Rec={:10.4f}, MSE={:10.4f}'
                            ' CD={:10.4f}, PP={:10.4f} / {:10.4f} / {:10.4f}, NP={:10.4f} / {:10.4f} / {:10.4f}, |params|={}, |grad|={}, |z|={},'
                            ' z_disp={}, |prior_z_grad|={}, |posterior_z_f_grad|={}, |posterior_z_nll_grad|={}, prior_moments={}, posterior_moments={}, '
                            ' Beta={:10.4f}, best_prior_valid={:8.2f}'.format(
                    epoch, i + 1, len(train_loader), args.lr, abp_loss - mlp_error, mlp_error, cd, positive_potential,
                    positive_potential_2, positive_potential_2 - positive_potential,
                    negative_potential, negative_potential_2, negative_potential_2 - negative_potential,
                    param_norm_str, grad_norm_str, prior_posterior_z_norm_str, z_disp_str,
                    prior_z_grad_norm_str, posterior_z_f_grad_norm_str, posterior_z_nll_grad_norm_str,
                    prior_moments, posterior_moments, args.beta, best_prior_prop_valid))

        epoch_train_time = time.time() - start_time
        logger.info('Time Elapsed: %.1fs' % epoch_train_time)

        logger.info('--------------------------------')
        logger.info('Checking validation perf...')
        logger.record_tabular('Epoch', epoch)
        logger.record_tabular('Mode', 'Val')
        logger.record_tabular('LR', args.lr)
        logger.record_tabular('Epoch Train Time', epoch_train_time)
        writer, prior_prop_valid = evaluation(args, epoch, test_loader, model, logger, writer, epoch)
        val_stats.append(prior_prop_valid)

        if prior_prop_valid > best_prior_prop_valid:
            best_prior_prop_valid = prior_prop_valid
            model.cpu()
            checkpoint = {
                'args': args.__dict__,
                'model': model,
                'val_stats': val_stats
            }
            logger.info('Save checkpoint to %s' % checkpoint_path)
            torch.save(checkpoint, checkpoint_path)
            model.cuda()
        else:
            if epoch >= args.min_epochs:
                args.decay = 1

        name = "%s.pt" % (str(epoch))
        ckpt_path = os.path.join(checkpoint_dir, name)
        model.cpu()
        checkpoint = {
            'args': args.__dict__,
            'model': model,
        }
        logger.info('Save checkpoint to %s' % ckpt_path)
        torch.save(checkpoint, ckpt_path)
        model.cuda()

        # if args.decay == 1:
        #   args.lr = args.lr*0.5
        #   for param_group in optimizer.param_groups:
        #     param_group['lr'] = args.lr
        #   if args.lr < 0.03:
        #     break


if __name__ == '__main__':
    args = get_args()
    exp_id = 'ebm_plogp'
    output_dir = get_output_dir(exp_id, fs_prefix='../exp_')
    copy_all_files(__file__, output_dir)
    set_gpu(args.gpu)

    with logger.session(dir=output_dir, format_strs=['stdout', 'csv', 'log']):
        main(args, output_dir)
