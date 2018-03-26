import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.autograd import Variable

use_cuda = torch.cuda.is_available()

class Squash(nn.Module):
    def __init__(self):
        super(Squash,self).__init__()
    def forward(self,x):
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
        self.squash = Squash()

    def forward(self,x):
        '''

        :param x: shape = 256 x 20 x 20. Output of convolution operation
        :return: output of primary capsules
        '''
        x = self.conv(x)
        # reshaping the tensor
        #o = x.shape[-1]
        #x = x.view(-1,self.out_channels,self.cap_dim,o,o)
        # move axis of cap_dim to -1
        #x = x.permute(0,1,3,2,4)
        #x = x.permute(0,1,2,4,3)
        #x = x.contiguous()
        x = x.view(x.shape[0],-1,self.cap_dim)
        x = self.squash(x)
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
        self.squash = Squash()

    def forward(self,x):
        '''

        :param x: shape = num_in_caps x in_cap_dim || eg. 1152 x 8
        :return: output after routing for r iterations
        '''
        x = torch.matmul(self.W,x.unsqueeze(-1).unsqueeze(-3)).squeeze()
        # shape of x is now B X (NUM_OUT_CAPS X OUT_CAP_DIM). Reshape to B X NUM_OUT_CAPS X OUT_CAP_DIM
        # x is now U j|i or the PREDICTION VECTORS
        coupling_coef = Variable(torch.zeros((self.num_in_caps, self.num_out_caps)))
        if use_cuda:
            coupling_coef = coupling_coef.cuda()
        b = coupling_coef
        s = None
        for r in range(self.routing_iterations+1):                                                      # STEP 3
            coupling_coef = F.softmax(b,dim=-1)                                                         # STEP 4
            s = coupling_coef.unsqueeze(dim=-1) * x                                                     # STEP 5
            s = s.sum(dim=-3)                                                                           # STEP 5
            s = self.squash(s)                                                                          # STEP 6
            if r<self.routing_iterations:
                b = b + torch.matmul(x.unsqueeze(-2),s.unsqueeze(-1).unsqueeze(1)).squeeze().sum(dim=0) # STEP 7
        return s

class MarginLoss(nn.Module):
    def __init__(self,m_plus=0.9,m_minus=0.1,downweighting=0.5):
        super(MarginLoss, self).__init__()
        self.m_plus = m_plus
        self.m_minus = m_minus
        self.downweighting = downweighting

    def forward(self,output,target):
        # output = B X 10 X 16, TARGET = B X CLASS NUMBER
        loss_vec = Variable(torch.zeros(output.shape[0], output.shape[1]))
        one_hot = Variable(torch.zeros_like(loss_vec.data))
        if use_cuda:
            loss_vec = loss_vec.cuda()
            one_hot = one_hot.cuda()

        for i,lab in enumerate(target.data): ## POSSIBLE BOTTLENECK
            one_hot[i,lab] = 1

        l2norm = output.norm(dim=-1, keepdim=True)
        term1 = F.relu(self.m_plus-l2norm)**2
        term1 = term1.squeeze()
        term2 = self.downweighting * F.relu(l2norm-self.m_minus)**2
        term2 = term2.squeeze()
        loss_vec = torch.mul(term1,one_hot) + torch.mul(term2,1-one_hot)

        total_loss = loss_vec.mean()
        return total_loss

class ReconLoss(nn.Module):
    def __init__(self):
        super(ReconLoss, self).__init__()

    def forward(self,original,recon):
        # original = B X 1 X 28 X 28, recon = B X 784
        original = original.view(-1,28*28)
        loss_vec = (original.data-recon.data).norm(p=2,dim=-1)**2
        loss_vec = Variable(loss_vec)
        return loss_vec.mean()

class CapsuleLoss(nn.Module):
    def __init__(self):
        super(CapsuleLoss, self).__init__()
        self.marginloss = MarginLoss()
        self.reconloss = ReconLoss()

    def forward(self,out,label,original,recon):
        loss_m = self.marginloss(out, label)
        loss_r = self.reconloss(original, recon)
        loss = loss_m.data+ 0.0005*loss_r.data
        loss = Variable(loss)
        loss.requires_grad = True
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

    def forward(self,x,label=None):
        original = x
        x = self.conv1(x)
        x = self.primary_caps(x)
        x = self.digcaps(x)
        one_hot = Variable(torch.zeros(x.shape[0], x.shape[1]))  # B x 10
        if use_cuda:
            one_hot = one_hot.cuda()

        if label is None:
            logits = x.norm(dim=-1)
            _, label = torch.max(logits.data, dim=1)
            label = Variable(label)
            if use_cuda:
                label=label.cuda()

        for i, lab in enumerate(label.data):
            one_hot[i, lab] = 1

        recon = one_hot.unsqueeze(-1) * x # B x 10 x 16
        recon = recon.view(-1,x.shape[1]*x.shape[2])
        recon = self.decoder(recon) # B x 784
        return (x,recon)
