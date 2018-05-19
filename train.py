import torch
import torch.optim as optim
import sys, time
from capsule import Capsule_Net, CapsuleLoss
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.nn.functional as F




def save_model(model,path):
    torch.save(model.state_dict(),path)

def load_model(model,path):
    model.load_state_dict(torch.load(path), strict=False)
    print('######## MODEL LOADED ########')
    return model

def printgradnorm(self, grad_input, grad_output):
    print('\nInside ' + self.__class__.__name__ + ' backward')
    print('Inside class:' + self.__class__.__name__)
    print('')
    print('grad_input: ', type(grad_input))
    print('grad_input[0]: ', type(grad_input[0]))
    print('grad_input[1]: ', type(grad_input[1]))
    print('grad_input[2]: ', type(grad_input[2]))
    #print('grad_input[0].data: ', type(grad_input[0].data))
    print('grad_input[1]: ', type(grad_input[1]))
    print('grad_input[1].data: ', type(grad_input[1].data))

    print('grad_output: ', type(grad_output))
    print('grad_output[0]: ', type(grad_output[0]))
    print('grad_output[0].data: ', type(grad_output[0].data))
    print('')
    print('grad_input size:', grad_input[1].size())
    print('grad_input size:', grad_input[2].size())
    print('grad_output size:', grad_output[0].size())
    print('grad_input norm:', grad_input[0].data.norm())

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
########################################################################################
###################### LOADER CODE BELOW TAKEN FROM TUTORIAL############################
########################################################################################
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Using device:\t',device)

root = './data'
download = True  # download MNIST dataset or not

trans = transforms.Compose([transforms.ToTensor()])#, transforms.Normalize((0.5,), (1.0,))])
train_set = dset.MNIST(root=root, train=True, transform=trans, download=download)
test_set = dset.MNIST(root=root, train=False, transform=trans)

batch_size = 256

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

model = Capsule_Net().to(device)
print(model)
#model.conv1.register_backward_hook(printgradnorm)
#model.digcaps.register_forward_hook(print_blobs)
print ("# parameters: ", sum(param.numel() for param in model.parameters()))

optimizer = optim.Adam(model.parameters())
best_acc = 0.0
for epoch in range(10):
    # training
    avg_loss = 0.0
    train_acc=0.0
    start_time = time.time()
    for batch_no, (x, target) in enumerate(train_loader):

        x, target = x.to(device), target.to(device)

        # CLEAR GRADIENT TO PREVENT ACCUMULATION
        optimizer.zero_grad()
        # COMPUTE OUTPUT
        out,recon,mask = model(x, target)
        # COMPUTE LOSS
        loss = CapsuleLoss(out,mask,x,recon)
        # FIND GRADIENTS
        loss.backward()
        # UPDATE WEIGHTS
        optimizer.step()
        # OBTAIN ACCURACY ON BATCH
        logits = F.softmax(out.norm(dim=-1),dim=-1)
        _, pred_label = torch.max(logits.data, dim=1)
        pred_label = pred_label.to(device)
        train_acc = (pred_label == target.data).double().sum()
        #if batch_no%batch_size==0:
        sys.stdout.write('Epoch = {0}\t Batch n.o.={1}\t Loss={2:.4f}\t Train_acc={3:.4f}\r'.format(epoch,batch_no,loss.item(),train_acc))
        sys.stdout.flush()
        avg_loss+=loss.item()
    total_time = time.time()-start_time
    sys.stdout.write('\nAvg Loss={0:.4f}\t time taken = {1:0.2f}'.format(avg_loss/len(train_loader),total_time))
    # testing
    correct_cnt=0
    total_cnt = 0
    for batch_idx, (x, target) in enumerate(test_loader):
        x, target = x.to(device), target.to(device)
        x.no_grad()
        target.no_grad()
        out,recon,_ = model(x)
        logits = out.norm(dim=-1)
        _, pred_label = torch.max(logits.data, dim=1) # cool trick
        pred_label=pred_label.to(device)
        total_cnt += x.data.size()[0]
        correct_cnt += (pred_label == target.data.cuda()).sum()
    test_acc = correct_cnt/total_cnt
    print('\nTest Accuracy={}'.format(correct_cnt * 1.0 / total_cnt))
    if test_acc>best_acc:
        best_acc=test_acc
        save_model(model,'model_acc_{}.pkl'.format(best_acc))

