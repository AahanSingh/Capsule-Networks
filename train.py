from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.optim as optim
import sys, argparse
from capsule import Capsule_Net, MarginLoss, ReconLoss
import torchvision.datasets as dset
import torchvision.transforms as transforms

def save_model(model,path):
    torch.save(model.state_dict(),path)

def load_model(model,path):
    model.load_state_dict(torch.load(path), strict=False)
    print('######## MODEL LOADED ########')
    return model

def print_blobs(self, input, output):
    # input is a tuple of packed inputs
    # output is a Variable. output.data is the Tensor we are interested
    print('\nInside ' + self.__class__.__name__ + ' forward')
    print('')
    print('input: ', type(input))
    print('input[0]: ', type(input[0]))
    print('output: ', type(output))
    print('')
    print('input size:', input[0].size())
    print('output size:', output.data.size())

def add_fwd_hook(net):
    net.conv1.register_backward_hook(print_blobs)
    net.primary_caps.register_backward_hook(print_blobs)
    net.digcaps.register_backward_hook(print_blobs)

########################################################################################
###################### LOADER CODE BELOW TAKEN FROM TUTORIAL############################
########################################################################################
use_cuda = torch.cuda.is_available()

root = './data'
download = True  # download MNIST dataset or not

trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])
train_set = dset.MNIST(root=root, train=True, transform=trans, download=download)
test_set = dset.MNIST(root=root, train=False, transform=trans)

batch_size = 128

train_loader = torch.utils.data.DataLoader(
                 dataset=train_set,
                 batch_size=batch_size,
                 shuffle=True)
test_loader = torch.utils.data.DataLoader(
                dataset=test_set,
                batch_size=batch_size,
                shuffle=False)

print('==>>> total training batch number: {}'.format(len(train_loader)))
print('==>>> total testing batch number: {}'.format(len(test_loader)))

model = Capsule_Net()
print(model)
if use_cuda:
    model = model.cuda()

optimizer = optim.Adam(model.parameters())
loss_margin = MarginLoss()
loss_recon = ReconLoss()
for epoch in range(10):
    # training
    ave_loss = 0
    for batch_idx, (x, target) in enumerate(train_loader):
        optimizer.zero_grad()
        if use_cuda:
            x, target = x.cuda(), target.cuda()
        x, target = Variable(x), Variable(target)
        out,recon = model(x, target)
        loss_m = loss_margin(out, target)
        loss_r = loss_recon(x,recon)
        #print(loss_m)
        #print(loss_r)
        loss = loss_m.data+ 0.0005*loss_r.data
        loss = Variable(loss)
        loss.requires_grad=True
        loss.backward()
        optimizer.step()
        sys.stdout.write('==>>> epoch: {}, batch index: {}, train loss: {}\r'.format(epoch, batch_idx + 1, loss.data[0]))
        sys.stdout.flush()
    # testing
    correct_cnt, ave_loss = 0, 0
    total_cnt = 0
    for batch_idx, (x, target) in enumerate(test_loader):
        if use_cuda:
            x, targe = x.cuda(), target.cuda()
        x, target = Variable(x, volatile=True), Variable(target, volatile=True)
        out = model(x,train=False)
        loss_m = loss_margin(out)
        logits = out.norm(dim=-1)
        _, pred_label = torch.max(logits.data, dim=1) # cool trick
        total_cnt += x.data.size()[0]
        correct_cnt += (pred_label == target.data).sum()
        # smooth average
        ave_loss = ave_loss * 0.9 + loss.data[0] * 0.1

        if (batch_idx + 1) % 100 == 0 or (batch_idx + 1) == len(test_loader):
            print
            '==>>> epoch: {}, batch index: {}, test loss: {:.6f}, acc: {}'.format(
                epoch, batch_idx + 1, ave_loss, correct_cnt * 1.0 / total_cnt)

torch.save(model.state_dict(), model.name())
