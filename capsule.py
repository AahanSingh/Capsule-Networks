import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.autograd import Variable

use_cuda = torch.cuda.is_available()

def Squash(x):
    l2norm = x.norm(dim=-1,keepdim=True)
    unit_v = x/l2norm
    squashed_v = l2norm.pow(2)/(1+l2norm.pow(2))
    x = unit_v*squashed_v
    return x

class Capsule_conv(nn.Module):
    def __init__(self,in_channels,out_channels,cap_dim):
        super(Capsule_conv,self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.cap_dim = cap_dim
        self.kernel_size = 9
        self.stride = 2
        self.conv = nn.Conv2d(in_channels=self.in_channels,out_channels=(self.out_channels*self.cap_dim),
                              kernel_size=self.kernel_size,stride=self.stride)

    def forward(self,x):
        '''

        :param x: shape = 256 x 20 x 20. Output of convolution operation
        :return: output of primary capsules
        '''
        x = self.conv(x)
        x = x.view(x.shape[0],-1,self.cap_dim)
        x = Squash(x)
        return x

class Capsule_fc(nn.Module):
    def __init__(self,in_cap_dim,num_in_caps,out_cap_dim,num_out_caps,r = 3):
        super(Capsule_fc, self).__init__()
        self.num_in_caps = num_in_caps
        self.num_out_caps = num_out_caps
        self.in_cap_dim = in_cap_dim
        self.out_cap_dim = out_cap_dim
        self.W = nn.Parameter(torch.randn(self.num_in_caps,self.num_out_caps,self.out_cap_dim,self.in_cap_dim))
        self.routing_iterations = r

    def forward(self,x):
        '''

        :param x: shape = num_in_caps x in_cap_dim || eg. 1152 x 8
        :return: output after routing for r iterations
        '''
        x = torch.matmul(self.W,x.unsqueeze(-1).unsqueeze(-3)).squeeze()
        # shape of x is now B X NUM_IN_CAPS X NUM_OUT_CAPS  X 16
        # x is now U j|i or the PREDICTION VECTORS
        coupling_coef = Variable(torch.zeros([*x.shape]))
        if use_cuda:
            coupling_coef = coupling_coef.cuda()
        b = coupling_coef
        for r in range(1,self.routing_iterations+1):                                                    # STEP 3
            coupling_coef = F.softmax(b,dim=1)                                                         # STEP 4
            s = coupling_coef * x                                                                       # STEP 5
            s = s.sum(dim=1,keepdim=True)                                                              # STEP 5
            v = Squash(s)                                                                          # STEP 6
            if r!=self.routing_iterations:
                b = b + (v * x).sum(dim=-1, keepdim=True)                                               # STEP 7
        return v.squeeze()


def MarginLoss(output,one_hot):
    # output = B X 10 X 16, TARGET = B X CLASS NUMBER
    downweighting = 0.5
    m_plus = 0.9
    m_minus = 0.1
    l2norm = output.norm(dim=-1)
    l2norm = F.softmax(l2norm,dim=-1)
    term1 = F.relu(m_plus-l2norm)**2
    #term1 = term1.squeeze()
    term2 = F.relu(l2norm-m_minus)**2
    #term2 = term2.squeeze()
    loss_vec = one_hot * term1 + downweighting*((1-one_hot)*term2)
    total_loss = loss_vec.sum(dim=-1)
    return total_loss.mean()

def ReconLoss(original,recon):
    # original = B X 1 X 28 X 28, recon = B X 784
    original = original.view(-1,28*28)
    loss_vec = (original.data-recon.data)**2
    loss_vec = loss_vec.sum(-1)
    return loss_vec.mean()

def CapsuleLoss(out,label,original,recon):
    loss_m = MarginLoss(out, label)
    loss_r = ReconLoss(original, recon)
    loss = loss_m+ 0.0005*loss_r
    return loss

class Capsule_Net(nn.Module):
    def __init__(self):
        super(Capsule_Net, self).__init__()
        self.conv1 = nn.Conv2d(1,256,9)
        self.primary_caps = Capsule_conv(256,32,8)
        self.digcaps = Capsule_fc(8,32*6*6,16,10)
        self.decoder = nn.Sequential(
            nn.Linear(10*16,512),
            nn.ReLU(),
            nn.Linear(512,1024),
            nn.ReLU(),
            nn.Linear(1024,784),
            nn.Sigmoid()
        )
        self.mask = Variable(torch.eye(10))
        if use_cuda:
            self.mask = self.mask.cuda()

    def forward(self,x,label=None):
        x = self.conv1(x)
        x = self.primary_caps(x)
        x = self.digcaps(x)
        if label is None:
            logits = x.norm(dim=-1)
            _, label = torch.max(logits.data, dim=1)
            label = Variable(label)
            if use_cuda:
                label=label.cuda()

        one_hot = self.mask.index_select(dim=0,index = label)
        recon = one_hot.unsqueeze(-1) * x # B x 10 x 16
        recon = recon.view(-1,x.shape[1]*x.shape[2])
        recon = self.decoder(recon) # B x 784
        return (x,recon,one_hot)
