import copy
import math
import logging
import argparse
import torch
import torch.optim as optim
import time
from capsule import Capsule_Net, CapsuleLoss
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.nn.functional as F
import mlflow
import mlflow.pytorch

parser = argparse.ArgumentParser(description="Capsule Nets Mnist MLFlow Example")
parser.add_argument("--batch-size", type=int, default=8)
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--optimizer", type=str, default="adam")
parser.add_argument("--lr", type=float, default=1e-2)
parser.add_argument("--gpu", action="store_true", default=False)
parser.add_argument("--log-interval", type=int, default=1)

logging.basicConfig(
    format="%(asctime)s %(levelname)s:%(message)s",
    datefmt="%m/%d/%Y %I:%M:%S %p",
    level=logging.INFO,
)

optimizers = {
    "adam": optim.Adam,
    "sgd": optim.SGD,
    "adadelta": optim.Adadelta,
    "rmsprop": optim.RMSprop,
}


def train_step(model, train_loader, device, optimizer, epoch, batch_size):
    # model.train()
    # training
    avg_loss = 0.0
    start_time = time.time()
    for batch_no, (x, target) in enumerate(train_loader):

        x, target = x.to(device), target.to(device)

        # CLEAR GRADIENT TO PREVENT ACCUMULATION
        optimizer.zero_grad()
        # COMPUTE OUTPUT
        out, recon, mask = model(x, target)
        # COMPUTE LOSS
        loss = CapsuleLoss(out, mask, x, recon)
        # FIND GRADIENTS
        loss.backward()
        # UPDATE WEIGHTS
        optimizer.step()
        # OBTAIN ACCURACY ON BATCH
        logits = F.softmax(out.norm(dim=-1), dim=-1)
        _, pred_label = torch.max(logits.data, dim=1)
        pred_label = pred_label.to(device)
        train_acc = (pred_label == target.data).double().sum()
        logging.info(
            "Epoch = {0}\t Batch n.o.={1}\t Loss={2:.4f}\t Batch_acc={3:.4f}\r".format(
                epoch, batch_no, loss.item(), train_acc / batch_size
            )
        )
        mlflow.log_metric(
            "Batch Accuracy",
            train_acc.item() / batch_size,
            step=math.ceil(epoch * len(train_loader) / batch_size) + batch_no,
        )
        mlflow.log_metric(
            "Loss",
            loss.item(),
            step=math.ceil(epoch * len(train_loader) / batch_size) + batch_no,
        )
        avg_loss += loss.item()
    total_time = time.time() - start_time
    avg_loss /= len(train_loader)
    logging.info("\nAvg Loss={0:.4f}\t time taken = {1:0.2f}".format(avg_loss, total_time))
    mlflow.log_metric("Average Loss", avg_loss, step=epoch)
    mlflow.log_metric("Time Taken", total_time, step=epoch)


def test_step(model, test_loader, device, epoch):
    # model.eval()
    # testing
    correct_cnt = 0
    total_cnt = 0
    for batch_idx, (x, target) in enumerate(test_loader):
        x, target = x.to(device), target.to(device)

        out, recon, _ = model(x)
        logits = out.norm(dim=-1)
        _, pred_label = torch.max(logits.data, dim=1)
        pred_label = pred_label.to(device)
        total_cnt += x.data.size()[0]
        correct_cnt += (pred_label == target).sum()
    test_acc = correct_cnt.item() * 1.0 / total_cnt
    logging.info("\nEpoch:{}\tTest Accuracy={}".format(epoch, test_acc))
    mlflow.log_metric("Test Accuracy", test_acc, step=epoch)
    return test_acc


def main():
    args = parser.parse_args()
    ########################################################################################
    ###################### LOADER CODE BELOW TAKEN FROM TUTORIAL############################
    ########################################################################################

    if args.optimizer not in optimizers:
        raise Exception(
            "Invalid optimizer given. Available optimizers: {}".format(list(optimizers.keys()))
        )

    if args.gpu:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")
    logging.info("Using device: {}".format(device))

    root = "./data"
    download = True  # download MNIST dataset or not

    trans = transforms.Compose([transforms.ToTensor()])  # , transforms.Normalize((0.5,), (1.0,))])
    train_set = dset.MNIST(root=root, train=True, transform=trans, download=download)
    test_set = dset.MNIST(root=root, train=False, transform=trans)

    train_loader = torch.utils.data.DataLoader(
        dataset=train_set, batch_size=args.batch_size, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_set, batch_size=args.batch_size, shuffle=False
    )

    logging.info("==>>> total training batch number: {}".format(len(train_loader)))
    logging.info("==>>> total testing batch number: {}\n".format(len(test_loader)))

    model = Capsule_Net().to(device)
    logging.info(model)
    # model.conv1.register_backward_hook(printgradnorm)
    # model.digcaps.register_forward_hook(print_blobs)
    logging.info("# parameters: {}".format(sum(param.numel() for param in model.parameters())))

    optimizer = optimizers[args.optimizer](model.parameters(), lr=args.lr)
    best_acc = 0.0
    training_parameters = {
        "Batch Size": args.batch_size,
        "Epochs": args.epochs,
        "Optimizer": args.optimizer,
        "Learning Rate": args.lr,
        "GPU Used": args.gpu,
        "Log Interval": args.log_interval,
    }
    best_epoch = 0
    with mlflow.start_run(experiment_id=experiment_id):
        mlflow.log_params(training_parameters)
        for epoch in range(args.epochs):
            train_step(
                model=model,
                train_loader=train_loader,
                device=device,
                optimizer=optimizer,
                epoch=epoch,
                batch_size=args.batch_size,
            )
            test_acc = test_step(model=model, test_loader=test_loader, device=device, epoch=epoch)
            if test_acc > best_acc:
                best_acc = test_acc
                best_epoch = epoch
                mlflow.pytorch.log_model(model, artifact_path="{}-{}".format(best_epoch, best_acc))


if __name__ == "__main__":
    mlflow.set_tracking_uri("http://localhost:5000")
    experiment_id = mlflow.get_experiment_by_name("Capsule Nets - MNIST")
    if experiment_id is None:
        experiment_id = mlflow.create_experiment("Capsule Nets - MNIST")
    else:
        experiment_id = experiment_id.experiment_id
    main()
