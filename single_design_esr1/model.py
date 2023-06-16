import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class RNNEBM(nn.Module):
    def __init__(self, args, vocab_size=10000,
                 dec_word_dim=512,
                 dec_h_dim=1024,
                 dec_num_layers=1,
                 dec_dropout=0.5,
                 latent_dim=32,
                 max_sequence_length=40):
        super(RNNEBM, self).__init__()
        self.args = args
        self.dec_h_dim = dec_h_dim
        self.dec_num_layers = dec_num_layers
        self.embedding_size = dec_word_dim
        self.latent_dim = latent_dim
        self.max_sequence_length = max_sequence_length
        self.vocab_size = vocab_size

        self.dec_word_vecs = nn.Embedding(vocab_size, dec_word_dim)
        dec_input_size = dec_word_dim
        dec_input_size += latent_dim
        self.dec_rnn = nn.LSTM(dec_input_size, dec_h_dim, num_layers=dec_num_layers,
                               batch_first=True)
        # self.input_dropout = nn.Dropout(dec_dropout)
        # self.out_dropout = nn.Dropout(dec_dropout)
        self.dec_linear = nn.Sequential(*[nn.Linear(dec_h_dim, vocab_size),
                                          nn.LogSoftmax(dim=-1)])
        self.dec = nn.ModuleList([self.dec_word_vecs, self.dec_rnn, self.dec_linear])
        if latent_dim > 0:
            self.latent_hidden_linear = nn.Linear(latent_dim, dec_h_dim)
            self.dec.append(self.latent_hidden_linear)

        # ebm prior
        self.prior_dim = self.latent_dim
        self.prior_hidden_dim = args.prior_hidden_dim

        self.prior_network = nn.Sequential(
            nn.Linear(self.prior_dim, self.prior_hidden_dim),
            GELU(),
            nn.Linear(self.prior_hidden_dim, self.prior_hidden_dim),
            GELU(),
            nn.Linear(self.prior_hidden_dim, 1)
        )

        # Property design
        if args.single_design:
            self.mlp = MLP(args)

        if args.multi_design:
            self.mlp_qed = MLP_qed(args)
            self.mlp_ba = MLP_ba(args)
            self.mlp_sas = MLP_sas(args)


    def ebm_prior(self, z, args):
        return self.prior_network(z)

    def decoder(self, x, q_z, init_h=True, training=True, dropout=0.2):
        self.word_vecs = F.dropout(self.dec_word_vecs(x[:, :-1]), training=training, p=dropout)
        if init_h:
            self.h0 = Variable(
                torch.zeros(self.dec_num_layers, self.word_vecs.size(0), self.dec_h_dim).type_as(self.word_vecs.data),
                requires_grad=False)
            self.c0 = Variable(
                torch.zeros(self.dec_num_layers, self.word_vecs.size(0), self.dec_h_dim).type_as(self.word_vecs.data),
                requires_grad=False)
        else:
            self.h0.data.zero_()
            self.c0.data.zero_()

        if q_z is not None:
            q_z_expand = q_z.unsqueeze(1).expand(self.word_vecs.size(0),
                                                 self.word_vecs.size(1), q_z.size(1))
            dec_input = torch.cat([self.word_vecs, q_z_expand], 2)
        else:
            dec_input = self.word_vecs
        if q_z is not None:
            self.h0[-1] = self.latent_hidden_linear(q_z)
        memory, _ = self.dec_rnn(dec_input, (self.h0, self.c0))
        dec_linear_input = memory.contiguous()
        dec_linear_input = F.dropout(dec_linear_input, training=training, p=dropout)
        preds = self.dec_linear(dec_linear_input.view(
            self.word_vecs.size(0) * self.word_vecs.size(1), -1)).view(
            self.word_vecs.size(0), self.word_vecs.size(1), -1)
        return preds

    def infer_prior_z(self, z, args, n_steps=0, debug=False):
        z_prior_grads_norm = []

        if debug:
            print('-----------------')
            print('-----------------')
            print('-----------------')

        if n_steps < args.z_n_iters_prior:
            _n_steps = args.z_n_iters_prior
        else:
            _n_steps = n_steps
        for i in range(_n_steps):
            z = z.detach().clone().requires_grad_(True)
            assert z.grad is None
            f = self.ebm_prior(z, args).sum()
            negative_f = -f
            negative_f.backward()
            if debug:
                print(negative_f.cpu().item())

            z_grad = z.grad.detach().clone()
            if args.ref_dist == 'gaussian':
                z = z - 0.5 * args.prior_step_size * args.prior_step_size * (
                        z.grad + z / (args.ref_sigma * args.ref_sigma))
            else:
                z = z - 0.5 * args.prior_step_size * args.prior_step_size * z.grad
            if args.z_prior_with_noise:
                z += args.noise_factor * args.prior_step_size * torch.randn_like(z)
            z_prior_grads_norm.append(torch.norm(z_grad, dim=1).mean().cpu().numpy())

        z = z.detach().clone()

        return z, z_prior_grads_norm

    def infer_z(self, z, x, x_len=None, y=None, beta=1., step_size=0.8, training=True, dropout=0.2, debug=False):
        args = self.args
        target = x.detach().clone()
        target = target[:, 1:]
        z_f_grads_norm = []
        z_nll_grads_norm = []

        if debug:
            print('-----------------')
            print('-----------------')
            print('-----------------')

        for i in range(args.z_n_iters):
            z = z.detach().clone()
            z.requires_grad = True
            assert z.grad is None
            logp = self.decoder(x, z, training=training, dropout=dropout)  
            nll = 0
            logp = logp.view(-1, logp.size(2))
            target = target.reshape(-1)
            if x_len is not None and args.mask:
                x_len = x_len.reshape(-1)
                nll = NLLLoss(logp, target, x_len, reduction='sum')
            else:
                nll = F.nll_loss(logp, target, reduction='sum')
            f = self.ebm_prior(z, self.args).sum()
            z_grad_f = torch.autograd.grad(-f, z)[0]
            z_grad_nll = torch.autograd.grad(nll, z)[0]
            _z_grad_f = z_grad_f.detach().clone()
            _z_grad_nll = z_grad_nll.detach().clone()
            if args.ref_dist == 'gaussian':
                z = z - 0.5 * step_size * step_size * (
                        z_grad_nll + beta * z_grad_f + beta * z / (args.ref_sigma * args.ref_sigma))
            else:
                z = z - 0.5 * step_size * step_size * (z_grad_nll + beta * z_grad_f)

            if args.z_with_noise:
                z += args.noise_factor * step_size * torch.randn_like(z)

            if args.single_design and y is not None:
                y_z = self.mlp(z)
                mse = F.mse_loss(y_z, y, reduction='sum')
                z_grad_mse = torch.autograd.grad(mse, z)[0]
                z = z - 0.5 * step_size * step_size * z_grad_mse * args.prop_coefficient

            if args.multi_design and y is not None:
                ba, sas, qed = y

                ba_z = self.mlp_ba(z)
                mse_ba = F.mse_loss(ba_z, ba, reduction='sum')
                z_grad_mse_ba = torch.autograd.grad(mse_ba, z)[0]
                z = z - 0.5 * step_size * step_size * z_grad_mse_ba * args.ba

                sas_z = self.mlp_sas(z)
                mse_sas = F.mse_loss(sas_z, sas, reduction='sum')
                z_grad_mse_sas = torch.autograd.grad(mse_sas, z)[0]
                z = z - 0.5 * step_size * step_size * z_grad_mse_sas * args.sas

                qed_z = self.mlp_qed(z)
                mse_qed = F.mse_loss(qed_z, qed, reduction='sum')
                z_grad_mse_qed = torch.autograd.grad(mse_qed, z)[0]
                z = z - 0.5 * step_size * step_size * z_grad_mse_qed * args.qed

            z_f_grads_norm.append(torch.norm(_z_grad_f, dim=1).mean().cpu().numpy())
            z_nll_grads_norm.append(torch.norm(_z_grad_nll, dim=1).mean().cpu().numpy())

            if debug:
                print(nll.cpu().item())
                print('+++++++++++++++++++++++++++++')
                print((-f).cpu().item())

        z = z.detach().clone()

        return z, (z_f_grads_norm, z_nll_grads_norm)

    def inference(self, device, sos_idx=108, max_len=None, z=None, init_h=True, training=True):
        batch_size = z.size(0)
        if init_h:
            self.h0 = torch.zeros((self.dec_num_layers, batch_size, self.dec_h_dim), dtype=torch.float, device=device,
                                  requires_grad=False)
            self.c0 = torch.zeros((self.dec_num_layers, batch_size, self.dec_h_dim), dtype=torch.float, device=device,
                                  requires_grad=False)
        else:
            self.h0.data.zero_()
            self.c0.data.zero_()

        self.h0[-1] = self.latent_hidden_linear(z)
        if max_len is None:
            max_len = self.max_sequence_length
        generations = torch.zeros(batch_size, max_len, dtype=torch.long, device=device)
        preds_sequence = torch.zeros(batch_size, max_len, self.vocab_size, dtype=torch.float, device=device)
        input_sequence = torch.tensor([sos_idx] * batch_size, dtype=torch.long, device=device)

        hidden = (self.h0, self.c0)
        for i in range(max_len):
            input_embedding = F.dropout(self.dec_word_vecs(input_sequence).view(batch_size, 1, self.embedding_size),
                                        training=training)
            dec_input = torch.cat([input_embedding, z.view(batch_size, 1, self.latent_dim)],
                                  dim=2) 
            output, hidden = self.dec_rnn(dec_input, hidden)
            dec_linear_input = output.contiguous()
            dec_linear_input = F.dropout(dec_linear_input, training=training) 
            preds = self.dec_linear(dec_linear_input.view(batch_size, self.dec_h_dim))
            preds[:, -1] = -1e5
            probs = F.softmax(preds, dim=1)
            samples = torch.multinomial(probs, 1)
            generations[:, i] = samples.view(-1).data
            preds_sequence[:, i, :] = preds
            input_sequence = samples.view(-1)

        return generations, preds_sequence

    def infer_z_given_y(self, z, y, n_iter=20, step_size=0.8):
        args = self.args

        z_f_grads_norm = []
        z_mse_grads_norm = []

        for i in range(n_iter):
            z = z.detach().clone()
            z.requires_grad = True
            assert z.grad is None

            f = self.ebm_prior(z, self.args).sum()
            z_grad_f = torch.autograd.grad(-f, z)[0]
            _z_grad_f = z_grad_f.detach().clone()

            if args.ref_dist == 'gaussian':
                z = z - 0.5 * step_size * step_size * (
                        z_grad_f + z / (args.ref_sigma * args.ref_sigma))
            else:
                z = z - 0.5 * step_size * step_size * z_grad_f

            if args.z_with_noise:
                z += args.noise_factor * step_size * torch.randn_like(z)

            _z_grad_mse = 0
            if args.single_design and y is not None:
                y_z = self.mlp(z)
                mse = F.mse_loss(y_z, y, reduction='sum')
                z_grad_mse = torch.autograd.grad(mse, z)[0]
                _z_grad_mse = z_grad_mse.detach().clone()
                z = z - 0.5 * step_size * step_size * z_grad_mse * args.prop_coefficient

            if args.multi_design and y is not None:
                ba, sas, qed = y

                ba_z = self.mlp_ba(z)
                mse_ba = F.mse_loss(ba_z, ba, reduction='sum')
                z_grad_mse_ba = torch.autograd.grad(mse_ba, z)[0]
                z = z - 0.5 * step_size * step_size * z_grad_mse_ba * args.ba

                sas_z = self.mlp_sas(z)
                mse_sas = F.mse_loss(sas_z, sas, reduction='sum')
                z_grad_mse_sas = torch.autograd.grad(mse_sas, z)[0]
                z = z - 0.5 * step_size * step_size * z_grad_mse_sas * args.sas

                qed_z = self.mlp_qed(z)
                mse_qed = F.mse_loss(qed_z, qed, reduction='sum')
                z_grad_mse_qed = torch.autograd.grad(mse_qed, z)[0]
                z = z - 0.5 * step_size * step_size * z_grad_mse_qed * args.qed

            z_f_grads_norm.append(torch.norm(_z_grad_f, dim=1).mean().cpu().numpy())
            # z_mse_grads_norm.append(torch.norm(_z_grad_mse, dim=1).mean().cpu().numpy())

        z = z.detach().clone()
        return z, (z_f_grads_norm)

class MLP(nn.Module):
    def __init__(self, args):
        super(MLP, self).__init__()
        self.args = args
        self.prior_dim = args.latent_dim
        self.mlp_hidden_dim = args.mlp_hidden_dim
        self.mlp = nn.Sequential(
            nn.Linear(self.prior_dim, self.mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.mlp_hidden_dim, self.mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.mlp_hidden_dim, 1),
        )

    def forward(self, z):
        score = torch.sigmoid(self.mlp(z))
        # score = self.mlp(z)
        return score.squeeze()

class MLP_qed(nn.Module):
    def __init__(self, args):
        super(MLP_qed, self).__init__()
        self.args = args
        self.prior_dim = args.latent_dim
        self.mlp_hidden_dim = args.mlp_hidden_dim
        self.mlp = nn.Sequential(
            nn.Linear(self.prior_dim, self.mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.mlp_hidden_dim, self.mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.mlp_hidden_dim, 1),
        )

    def forward(self, z):
        # [0, 1] maximize
        score = torch.sigmoid(self.mlp(z))
        return score.squeeze()

class MLP_sas(nn.Module):
    def __init__(self, args):
        super(MLP_sas, self).__init__()
        self.args = args
        self.prior_dim = args.latent_dim
        self.mlp_hidden_dim = args.mlp_hidden_dim
        self.mlp = nn.Sequential(
            nn.Linear(self.prior_dim, self.mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.mlp_hidden_dim, self.mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.mlp_hidden_dim, 1),
        )

    def forward(self, z):
        # [-1, 0] maximize
        score = torch.sigmoid(self.mlp(z))
        return score.squeeze()

class MLP_ba(nn.Module):
    def __init__(self, args):
        super(MLP_ba, self).__init__()
        self.args = args
        self.prior_dim = args.latent_dim
        self.mlp_hidden_dim = args.mlp_hidden_dim
        self.mlp = nn.Sequential(
            nn.Linear(self.prior_dim, self.mlp_hidden_dim),
            Swish(),
            nn.Linear(self.mlp_hidden_dim, self.mlp_hidden_dim),
            Swish(),
            nn.Linear(self.mlp_hidden_dim, 1),
            # nn.ReLU()
        )

    def forward(self, z):
        # [0, infty] maximize
        score = self.mlp(z)
        return score.squeeze()

class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return F.gelu(x)


class Mish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * (torch.tanh(F.softplus(x)))


class Swish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * F.sigmoid(x)


def NLLLoss(logs, targets, mask=None, reduction='sum'):
    '''
    NLL loss with mask
    :param logs: log prob (output from logsoftmax) N by C
    :param targets: N
    :param mask: N
    :param reduction: sum or mean
    :return:
    '''
    out = torch.diag(logs[:, targets])
    if mask is not None:
        out = out * mask
    if reduction == 'sum':
        return -torch.sum(out)
    elif reduction == 'mean':
        return -torch.mean(out)
    else:
        return NotImplemented
