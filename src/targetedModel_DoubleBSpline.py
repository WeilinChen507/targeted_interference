import argparse
import torch
import pickle as pkl
import torch.nn as nn
import utils as utils
import numpy as np
from modules import GCN, NN, Predictor, Discriminator, Density_Estimator, Discriminator_simplified

class Truncated_power():
    def __init__(self, degree, knots):
        """
        This class construct the truncated power basis; the data is assumed in [0,1]
        :param degree: int, the degree of truncated basis
        :param knots: list, the knots of the spline basis; two end points (0,1) should not be included
        """
        self.degree = degree
        self.knots = knots
        self.num_of_basis = self.degree + 1 + len(self.knots)
        self.relu = nn.ReLU(inplace=True)

        if self.degree == 0:
            print('Degree should not set to be 0!')
            raise ValueError

        if not isinstance(self.degree, int):
            print('Degree should be int')
            raise ValueError

    def forward(self, x):
        """
        :param x: torch.tensor, batch_size * 1
        :return: the value of each basis given x; batch_size * self.num_of_basis
        """
        x = x.squeeze()
        out = torch.zeros(x.shape[0], self.num_of_basis).cuda()
        for _ in range(self.num_of_basis):
            if _ <= self.degree:
                if _ == 0:
                    out[:, _] = 1.
                else:
                    out[:, _] = x**_
            else:
                if self.degree == 1:
                    out[:, _] = (self.relu(x - self.knots[_ - self.degree]))
                else:
                    out[:, _] = (self.relu(x - self.knots[_ - self.degree - 1])) ** self.degree

        return out # bs, num_of_basis


class TR(nn.Module):
    def __init__(self, degree, knots):
        super(TR, self).__init__()
        self.spb = Truncated_power(degree, knots)
        self.d = self.spb.num_of_basis # num of basis
        self.weight = torch.nn.Parameter(torch.rand(self.d, device='cuda'), requires_grad=True)

    def forward(self, t):
        out = self.spb.forward(t)
        out = torch.matmul(out, self.weight)
        return out

    def _initialize_weights(self):
        # self.weight.data.normal_(0, 0.1)
        self.weight.data.zero_()


class TargetedModel_DoubleBSpline(nn.Module):

    def __init__(self, Xshape, hidden, dropout, num_grid=None, init_weight=True, tr_knots=0.25, cfg_density=None):
        super(TargetedModel_DoubleBSpline, self).__init__()
        if num_grid is None:
            num_grid = 20

        self.encoder = GCN(nfeat=Xshape, nclass=hidden, dropout=dropout)
        self.X_XN = Predictor(input_size=hidden + Xshape, hidden_size1=hidden, hidden_size2=hidden, output_size=int(hidden/2))
        self.Q1 = Predictor(input_size=int(hidden/2) + 1, hidden_size1=int(hidden*2), hidden_size2=hidden, output_size=1)
        self.Q0 = Predictor(input_size=int(hidden/2) + 1, hidden_size1=int(hidden*2), hidden_size2=hidden, output_size=1)
        self.g_T = Discriminator_simplified(input_size=int(hidden/2), hidden_size1=hidden, output_size=1)
        self.g_Z = Density_Estimator(input_size=int(hidden/2), num_grid=num_grid)
        tr_knots = list(np.arange(tr_knots, 1, tr_knots))
        tr_degree = 2
        self.tr_reg_t1 = TR(tr_degree, tr_knots)
        self.tr_reg_t0 = TR(tr_degree, tr_knots)

        if init_weight:
            self.encoder._initialize_weights()
            self.X_XN._initialize_weights()
            self.Q1._initialize_weights()
            self.Q0._initialize_weights()
            self.g_Z._initialize_weights()
            self.g_T._initialize_weights()
            self.tr_reg_t1._initialize_weights()
            self.tr_reg_t0._initialize_weights()

    def parameter_base(self):
        return list(self.encoder.parameters()) +\
            list(self.X_XN.parameters()) +\
            list(self.Q1.parameters())+list(self.Q0.parameters())+\
            list(self.g_T.parameters())+\
            list(self.g_Z.parameters())

    def parameter_trageted(self):
        return list(self.tr_reg_t0.parameters()) + list(self.tr_reg_t1.parameters())

    def tr_reg(self, T, neighborAverageT):
        tr_reg_t1 = self.tr_reg_t1(neighborAverageT)
        tr_reg_t0 = self.tr_reg_t0(neighborAverageT)
        regur = torch.where(T==1, tr_reg_t1, tr_reg_t0)
        return regur



    def forward(self, A, X, T, Z=None):
        embeddings = self.encoder(X, A)  # X_i,X_N
        embeddings = self.X_XN(torch.cat((embeddings,X), dim=1))

        g_T_hat = self.g_T(embeddings)  # X_i,X_N -> T_i
        if Z is None:
            neighbors = torch.sum(A, 1)
            neighborAverageT = torch.div(torch.matmul(A, T.reshape(-1)), neighbors)  # treated_neighbors / all_neighbors
        else:
            neighborAverageT = Z


        g_Z_hat = self.g_Z(embeddings, neighborAverageT)  # X_i,X_N -> Z
        g_Z_hat = g_Z_hat.unsqueeze(1)


        embed_avgT = torch.cat((embeddings, neighborAverageT.reshape(-1, 1)), 1)

        Q_hat = T.reshape(-1, 1) * self.Q1(embed_avgT) + (1-T.reshape(-1, 1)) * self.Q0(embed_avgT)

        epsilon = self.tr_reg(T, neighborAverageT)  # epsilon(T,Z)


        return g_T_hat, g_Z_hat, Q_hat, epsilon, embeddings, neighborAverageT

    def infer_potential_outcome(self, A, X, T, Z=None):
        embeddings = self.encoder(X, A)  # X_i,X_N
        embeddings = self.X_XN(torch.cat((embeddings,X), dim=1))

        g_T_hat = self.g_T(embeddings)  # X_i,X_N -> T_i
        g_T_hat = g_T_hat.squeeze(1)

        if Z is None:
            neighbors = torch.sum(A, 1)
            neighborAverageT = torch.div(torch.matmul(A, T.reshape(-1)), neighbors)  # treated_neighbors / all_neighbors
        else:
            neighborAverageT = Z


        g_Z_hat = self.g_Z(embeddings, neighborAverageT)  # X_i,X_N -> Z



        embed_avgT = torch.cat((embeddings, neighborAverageT.reshape(-1, 1)), 1)

        Q_hat = T.reshape(-1, 1) * self.Q1(embed_avgT) + (1-T.reshape(-1, 1)) * self.Q0(embed_avgT)

        epsilon = self.tr_reg(T, neighborAverageT)  # epsilon(T,Z)
        # epsilon = epsilon.squeeze(1)


        return Q_hat.reshape(-1) + (epsilon.reshape(-1) * 1/(g_Z_hat.reshape(-1)*g_T_hat.reshape(-1) + 1e-6))

