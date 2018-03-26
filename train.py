from torch.autograd import Variable
import torch
import torch.optim as optim
import sys, time
from capsule import Capsule_Net, CapsuleLoss
import torchvision.datasets as dset
import torchvision.transforms as transforms

def save_model(model,path):
    torch.save(model.state_dict(),path)

def load_model(model,path):
    model.load_state_dict(torch.load(path), strict=False)
    print('######## MODEL LOADED ########')
    return model

########################################################################################
###################### LOADER CODE BELOW TAKEN FROM TUTORIAL############################
########################################################################################
use_cuda = torch.cuda.is_available()

root = './data'
download = True  # download MNIST dataset or not

trans = transforms.Compose([transforms.ToTensor()])#, transforms.Normalize((0.5,), (1.0,))])
train_set = dset.MNIST(root=root, train=True, transform=trans, download=download)
test_set = dset.MNIST(root=root, train=False, transform=trans)

batch_size = 5

train_loader = torch.utils.data.DataLoader(
                 dataset=train_set,
                 batch_size=batch_size,
                 shuffle=True)
test_loader = torch.utils.data.DataLoader(
                dataset=test_set,
                batch_size=batch_size,
                shuffle=False)

print('==>>> total training batch number: {}'.format(len(train_loader)))
print('==>>> total testing batch number: {}\n'.format(len(test_loader)))

model = Capsule_Net()
print(model)
print ("# parameters: ", sum(param.numel() for param in model.parameters()))
if use_cuda:
    model = model.cuda()

optimizer = optim.Adam(model.parameters())
loss_fn = CapsuleLoss()
best_acc = 0.0
for epoch in range(10):
    # training
    avg_loss = 0.0
    train_acc=0.0
    start_time = time.time()
    for batch_no, (x, target) in enumerate(train_loader):

        if use_cuda:
            x, target = x.cuda(), target.cuda()
        x, target = Variable(x), Variable(target)

        # CLEAR GRADIENT TO PREVENT ACCUMULATION
        optimizer.zero_grad()
        # COMPUTE OUTPUT
        out,recon = model(x, target)
        # COMPUTE LOSS
        loss = loss_fn(out,target,x,recon)
        # FIND GRADIENTS
        loss.backward()
        # UPDATE WEIGHTS
        optimizer.step()

        # OBTAIN ACCURACY ON BATCH
        logits = out.norm(dim=-1)
        _, pred_label = torch.max(logits.data, dim=1)
        if use_cuda:
            pred_label = pred_label.cuda()
        train_acc = (pred_label == target.data).double().mean()
        #if batch_no%batch_size==0:
        sys.stdout.write('Epoch = {0}\t Batch n.o.={1}\t Loss={2:.4f}\t Train_acc={3:.4f}\n'.format(epoch,batch_no,loss.data[0],train_acc))
        sys.stdout.flush()
        avg_loss+=loss
    total_time = time.time()-start_time
    sys.stdout.write('\nAvg Loss={0:.4f}\t time taken = {1:0.2f}'.format(avg_loss.data[0]/len(train_loader),total_time))
    # testing
    correct_cnt=0
    total_cnt = 0
    for batch_idx, (x, target) in enumerate(test_loader):
        if use_cuda:
            x, targe = x.cuda(), target.cuda()
        x, target = Variable(x, volatile=True), Variable(target, volatile=True)
        out,recon = model(x)
        logits = out.norm(dim=-1)
        _, pred_label = torch.max(logits.data, dim=1) # cool trick
        if use_cuda:
            pred_label=pred_label.cuda()
        total_cnt += x.data.size()[0]
        correct_cnt += (pred_label == target.data.cuda()).sum()
    test_acc = correct_cnt/total_cnt
    print('\nTest Accuracy={}'.format(correct_cnt * 1.0 / total_cnt))
    if test_acc>best_acc:
        best_acc=test_acc
        save_model(model,'model_acc_{}.pkl'.format(best_acc))
