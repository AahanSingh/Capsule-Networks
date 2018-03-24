import torch.nn as nn
import torch,sys
import torch.nn.functional as F
import numpy as np
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
        #print('Norms=',l2norm)
        #print('Unit_v=',unit_v)
        #print('squashed=',squashed_v)
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
        x = self.conv(x)
        # reshaping the tensor
        o = x.shape[-1]
        x = x.view(-1,self.out_channels,self.cap_dim,o,o)
        # move axis of cap_dim to -1
        x = x.permute(0,1,3,2,4)
        x = x.permute(0,1,2,4,3)
        x = x.contiguous()
        #print(x.is_contiguous())
        x = x.view(-1,self.out_channels*o*o,self.cap_dim)
        #print(x)
        x = self.squash(x)
        return x

class Capsule_fc(nn.Module):
    def __init__(self,in_cap_dim,num_in_caps,out_cap_dim,num_out_caps,r = 1):
        super(Capsule_fc, self).__init__()
        self.num_in_caps = num_in_caps
        #print('created ninc')
        self.num_out_caps = num_out_caps
        self.in_cap_dim = in_cap_dim
        self.out_cap_dim = out_cap_dim
        #print('created ocdim')
        #self.W = nn.Linear((num_in_caps*in_cap_dim),(num_in_caps*num_out_caps*out_cap_dim),bias=False)
        self.W = nn.ModuleList([nn.Linear(self.in_cap_dim,self.out_cap_dim,bias=False) for i in range(self.num_in_caps*self.num_out_caps)])
        #print('created W')
        self.routing_iterations = r
        #self.coupling_coef = Variable(torch.zeros((self.num_in_caps, self.num_out_caps)))
        #if use_cuda:
        #    self.coupling_coef = Variable(torch.zeros((self.num_in_caps, self.num_out_caps)).cuda())
        self.squash = Squash()

    def forward(self,x):
        # reshape x: B x NUM_IN_CAPS X IN_CAP_DIM -> B X (NUM_IN_CAPS X IN_CAP_DIM)
        #x = x.view(-1,self.num_in_caps*self.in_cap_dim)
        #x = self.W(x)
        #print('here dummy   ',x.data.__class__)
        u_ji = []
        for i in range(self.num_in_caps):
            temp = []
            for j in range(self.num_out_caps):
                #print(x[:,i,:].data.__class__)
                res = self.W[i*self.num_out_caps+j](x[:,i,:])
                temp.append(res.data.cpu().numpy())
            u_ji.append(temp)
        u_ji = np.array(u_ji)
        if use_cuda:
            u_ji = Variable(torch.from_numpy(u_ji).cuda())
        else:
            u_ji = Variable(torch.from_numpy(u_ji))
        u_ji = u_ji.permute(0, 2, 1, 3)
        u_ji = u_ji.permute(1, 0, 2, 3)
        #print('UJI=',u_ji.shape)
        x=u_ji
        # shape of x is now B X (NUM_OUT_CAPS X OUT_CAP_DIM). Reshape to B X NUM_OUT_CAPS X OUT_CAP_DIM
        #x = x.view(-1,self.num_in_caps,self.num_out_caps,self.out_cap_dim)
        # x is now U j|i or the PREDICTION VECTORS
        if use_cuda:
            coupling_coef = Variable(torch.zeros((self.num_in_caps, self.num_out_caps)).cuda())
        else:
            coupling_coef = Variable(torch.zeros((self.num_in_caps, self.num_out_caps)))
        b = coupling_coef
        s = None
        for r in range(self.routing_iterations):            # STEP 3
            sys.stdout.write('\nr={}\r'.format(r))
            sys.stdout.flush()
            coupling_coef = F.softmax(b,dim=-1)        # STEP 4
            s = coupling_coef.unsqueeze(dim=-1) * x    # STEP 5
            s = s.sum(dim=-3)                               # STEP 5
            s = self.squash(s)                              # STEP 6
            '''
            for bat in range(s.shape[0]):                   # STEP 7
                for i in range(self.num_in_caps):
                    for j in range(self.num_out_caps):
                        b[i,j] = b[i,j] + (x[bat,i,j].dot(s[bat,j]))
            '''
            b = b + torch.matmul(x.unsqueeze(-2),s.unsqueeze(-1).unsqueeze(1)).squeeze().sum(dim=0)
        return s

class MarginLoss(nn.Module):
    def __init__(self,m_plus=0.9,m_minus=0.1,downweighting=0.5):
        super(MarginLoss, self).__init__()
        self.m_plus = m_plus
        self.m_minus = m_minus
        self.downweighting = downweighting

    def forward(self,output,target):
        # output = B X 10 X 16, TARGET = B X CLASS NUMBER
        if use_cuda:
            loss_vec = Variable(torch.zeros(output.shape[0],output.shape[1]).cuda()) # B x 10
            one_hot = Variable(torch.zeros_like(loss_vec.data).cuda()) # B x 10
        else:
            loss_vec = Variable(torch.zeros(output.shape[0], output.shape[1]))
            one_hot = Variable(torch.zeros_like(loss_vec.data))
        for i,lab in enumerate(target.data2):
            one_hot[i,lab] = 1
        one_hot_inv = (one_hot-1) * -1

        l2norm = output.norm(dim=-1, keepdim=True)  # Bx10
        term1 = F.relu(self.m_plus-l2norm)**2  # Bx10
        term2 = self.downweighting * F.relu(l2norm-self.m_minus)**2  # Bx10
        loss_vec = torch.mul(term1,one_hot) + torch.mul(term2,one_hot_inv)
        '''
        for bat in range(target.shape[0]):
            for i in range(output.shape[1]):
                if i == target.data[bat]:
                    loss_vec[bat,i] = term1[bat,i]
                else:
                    loss_vec[bat,i] = term2[bat,i]
        '''
        # loss_vec contains capsule wise loss
        total_loss = loss_vec.sum()
        total_loss.requires_grad=True
        return total_loss


class ReconLoss(nn.Module):
    def __init__(self):
        super(ReconLoss, self).__init__()

    def forward(self,original,recon):
        # original = B X 1 X 28 X 28, recon = B X 784
        original = original.view(-1,28*28)
        loss_vec = (original.data-recon.data).norm(p=2,dim=-1)
        loss_vec = Variable(loss_vec)
        loss = loss_vec.sum()
        loss.required_grad=True
        return loss

class Capsule_Net(nn.Module):
    def __init__(self):
        super(Capsule_Net, self).__init__()
        self.conv1 = nn.Conv2d(1,256,9)
        #print('created conv1')
        self.primary_caps = Capsule_conv(256,32,8)
        #print('created pc')
        self.digcaps = Capsule_fc(8,32*6*6,16,10)
        #print('created dc')
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
        #print('AFTER conv1 ',x.shape)
        x = self.primary_caps(x)
        #print('AFTER Primary CAPS=',x.shape)
        x = self.digcaps(x)
        #print('AFTER DIGCAPS=',x.shape)
        if use_cuda:
            one_hot = Variable(torch.zeros(x.shape[0], x.shape[1]).cuda())  # B x 10
        else:
            one_hot = Variable(torch.zeros(x.shape[0], x.shape[1]))  # B x 10

        if label is None:
            logits = x.norm(dim=-1)
            _, label = torch.max(logits.data, dim=1)

        for i, lab in enumerate(label.data):
            one_hot[i, lab] = 1

        recon = one_hot.unsqueeze(-1) * x # B x 10 x 16
        recon = recon.view(-1,x.shape[1]*x.shape[2])
        recon = self.decoder(recon) # B x 784
        return (x,recon)
