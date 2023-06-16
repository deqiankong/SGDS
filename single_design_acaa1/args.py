import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", default=False, type=bool)
    parser.add_argument("--tb", default=False, type=bool)

    # Input data
    parser.add_argument('--data_dir', default='../data/ba1')
    parser.add_argument('--test_file', default='../ZINC/test_5.txt')
    parser.add_argument('--autodock_executable', type=str, default='~/AutoDock-GPU/bin/autodock_gpu_128wi')
    parser.add_argument('--protein_file', type=str, default='../2iik/2iik.maps.fld')
    parser.add_argument('--train_from', default='')
    parser.add_argument('--max_len', default=110, type=int)
    parser.add_argument('--batch_size', default=1024 * 1, type=int)
    parser.add_argument('--eval_batch_size', default=500 * 1, type=int)

    # General options
    parser.add_argument('--z_n_iters', type=int, default=20)
    parser.add_argument('--z_step_size', type=float, default=0.5)
    parser.add_argument('--z_with_noise', type=int, default=1)
    parser.add_argument('--num_z_samples', type=int, default=10)
    parser.add_argument('--model', type=str, default='mol_ebm')
    parser.add_argument('--mask', type=bool, default=False)
    parser.add_argument('--single_design', default=True, type=bool)
    parser.add_argument('--multi_design', default=False, type=bool)

    # EBM
    parser.add_argument('--prior_hidden_dim', type=int, default=200)
    parser.add_argument('--z_prior_with_noise', type=int, default=1)
    parser.add_argument('--prior_step_size', type=float, default=0.5)
    parser.add_argument('--z_n_iters_prior', type=int, default=20)
    parser.add_argument('--max_grad_norm_prior', default=1, type=float)
    parser.add_argument('--ebm_reg', default=0.0, type=float)
    parser.add_argument('--ref_dist', default='gaussian', type=str, choices=['gaussian', 'uniform'])
    parser.add_argument('--ref_sigma', type=float, default=0.5)
    parser.add_argument('--init_factor', type=float, default=1.)
    parser.add_argument('--noise_factor', type=float, default=0.5)

    # Decoder and MLP options
    parser.add_argument('--mlp_hidden_dim', default=100, type=int)
    parser.add_argument('--latent_dim', default=100, type=int)
    parser.add_argument('--dec_word_dim', default=512, type=int)
    parser.add_argument('--dec_h_dim', default=1024, type=int)
    parser.add_argument('--dec_num_layers', default=1, type=int)
    parser.add_argument('--dec_dropout', default=0.2, type=float)
    parser.add_argument('--train_n2n', default=1, type=int)
    parser.add_argument('--train_kl', default=1, type=int)

    # prop coefficients
    parser.add_argument('--prop_coefficient', default=10., type=float)
    parser.add_argument('--ba', default=10., type=float)
    parser.add_argument('--sas', default=10., type=float)
    parser.add_argument('--qed', default=10., type=float)

    # Optimization options
    parser.add_argument('--log_dir', default='../log/')
    parser.add_argument('--checkpoint_dir', default='models')
    # parser.add_argument('--slurm', default=0, type=int)
    parser.add_argument('--warmup', default=0, type=int)
    parser.add_argument('--num_epochs', default=30, type=int)
    parser.add_argument('--min_epochs', default=15, type=int)
    parser.add_argument('--start_epoch', default=0, type=int)
    parser.add_argument('--eps', default=1e-5, type=float)
    parser.add_argument('--decay', default=0, type=int)
    parser.add_argument('--momentum', default=0.5, type=float)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--prior_lr', default=0.0001, type=float)
    parser.add_argument('--max_grad_norm', default=5, type=float)
    parser.add_argument('--gpu', default=1, type=int)
    parser.add_argument('--seed', default=3435, type=int)
    parser.add_argument('--print_every', type=int, default=100)
    parser.add_argument('--sample_every', type=int, default=1000)
    parser.add_argument('--kl_every', type=int, default=100)
    parser.add_argument('--compute_kl', type=int, default=1)
    parser.add_argument('--test', type=int, default=0)
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    print(args)
