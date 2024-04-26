import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import GraphConvolution


ini_normal_variance=0.1

class GCN(nn.Module):
    def __init__(self, nfeat, nclass, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nclass).cuda()
        self.dropout = dropout

    def forward(self, x, adj):
        num = adj.shape[0]
        diag = torch.diag(torch.cuda.FloatTensor([1 for _ in range(num)]))
        x = F.relu(self.gc1(x, adj+diag))
        x = F.dropout(x, self.dropout)
        return x

    def _initialize_weights(self):
        self.gc1.weight.data.normal_(0, ini_normal_variance)

class NN(nn.Module):
    def __init__(self,in_dim,out_dim):
        super(NN,self).__init__()

        self.fc = nn.Linear(in_dim, out_dim, device='cuda')
        self.act = nn.LeakyReLU(0.2, inplace=True).cuda()
        self.dropout = nn.Dropout(0.1).cuda()
    def forward(self,x):
        x = F.relu(self.fc(x))
        x = self.dropout(x)
        return x

    def _initialize_weights(self):
        self.fc.weight.data.normal_(0, ini_normal_variance)


class Predictor(nn.Module):

    def __init__(self,input_size,hidden_size1,hidden_size2,output_size):
        super(Predictor, self).__init__()

        self.predict1 = nn.Linear(input_size,hidden_size1, device='cuda')
        self.predict2 = nn.Linear(hidden_size1,hidden_size2, device='cuda')
        self.predict3 = nn.Linear(hidden_size2,output_size, device='cuda')
        self.act = nn.LeakyReLU(0.2, inplace=True).cuda()
        self.dropout = nn.Dropout(0.1).cuda()

    def forward(self,x):
        x = self.predict1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.predict2(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.predict3(x)
        return  x

    def _initialize_weights(self):
        self.predict1.weight.data.normal_(0, ini_normal_variance)
        self.predict2.weight.data.normal_(0, ini_normal_variance)
        self.predict3.weight.data.normal_(0, ini_normal_variance)

class Discriminator(nn.Module):
    def __init__(self,input_size,hidden_size1,hidden_size2,output_size):
        super(Discriminator,self).__init__()

        self.disc1 = nn.Linear(input_size,hidden_size1, device='cuda')
        self.disc2 = nn.Linear(hidden_size1,hidden_size2, device='cuda')
        self.disc3 = nn.Linear(hidden_size2,output_size, device='cuda')
        self.act = nn.LeakyReLU(0.2, inplace=True).cuda()
        self.dropout = nn.Dropout(0.1).cuda()


    def forward(self,x):
        x = self.disc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.disc2(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.disc3(x)
        x = torch.sigmoid(x)
        return x

    def _initialize_weights(self):
        self.disc1.weight.data.normal_(0, ini_normal_variance)
        self.disc2.weight.data.normal_(0, ini_normal_variance)
        self.disc3.weight.data.normal_(0, ini_normal_variance)

class Discriminator_simplified(nn.Module):
    def __init__(self,input_size,hidden_size1,output_size):
        super(Discriminator_simplified,self).__init__()

        self.disc1 = nn.Linear(input_size,hidden_size1, device='cuda')
        self.disc3 = nn.Linear(hidden_size1,output_size, device='cuda')
        self.act = nn.LeakyReLU(0.2, inplace=True).cuda()

    def forward(self,x):
        x = self.disc1(x)
        x = self.act(x)
        x = self.disc3(x)
        x = torch.sigmoid(x)
        return x

    def _initialize_weights(self):
        self.disc1.weight.data.normal_(0, ini_normal_variance)
        self.disc3.weight.data.normal_(0, ini_normal_variance)

def comp_grid(y, num_grid):

    # L gives the lower index
    # U gives the upper index
    # inter gives the distance to the lower int

    U = torch.ceil(y * num_grid)
    inter = 1 - (U - y * num_grid)
    L = U - 1
    L += (L < 0).int()

    return L.int().tolist(), U.int().tolist(), inter


class Density_Block(nn.Module):
    def __init__(self, num_grid, ind, isbias=1):
        super(Density_Block, self).__init__()
        """
        Assume the variable is bounded by [0,1]
        the output grid: 0, 1/B, 2/B, ..., B/B; output dim = B + 1; num_grid = B
        """
        self.ind = ind
        self.num_grid = num_grid
        self.outd = num_grid + 1

        self.isbias = isbias

        self.weight = nn.Parameter(torch.rand(self.ind, self.outd, device='cuda'), requires_grad=True)
        if self.isbias:
            self.bias = nn.Parameter(torch.rand(self.outd, device='cuda'), requires_grad=True)
        else:
            self.bias = None

        self.softmax = nn.Softmax(dim=1)

    def forward(self, z, x):
        out = torch.matmul(x, self.weight)
        if self.isbias:
            out += self.bias
        out = self.softmax(out)

        x1 = list(torch.arange(0, x.shape[0]))
        L, U, inter = comp_grid(z, self.num_grid)

        L_out = out[x1, L]
        U_out = out[x1, U]

        out = L_out + (U_out - L_out) * inter

        return out

    def _initialize_weights(self):
        self.weight.data.normal_(0, ini_normal_variance)
        if self.isbias:
            self.bias.data.normal_(0, ini_normal_variance)
        # self.disc3.weight.data.zero_()
        return


class Density_Estimator(nn.Module):

    def __init__(self, input_size, num_grid):
        super().__init__()
        # input_size is the size of embeddings of X_i,X_N
        self.num_grid=num_grid
        self.density_estimator_head = Density_Block(self.num_grid, input_size, isbias=1)

    def forward(self, x, z):
        g_Z = self.density_estimator_head(z, x)
        return g_Z

    def _initialize_weights(self):
        self.density_estimator_head._initialize_weights()


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
        self.relu = nn.ReLU(inplace=True).cuda()

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
        out = torch.zeros(x.shape[0], self.num_of_basis, device='cuda')
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