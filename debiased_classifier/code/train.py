import argparse
import torch
import os
from torch.autograd import Variable
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
import numpy as np
import warnings


warnings.filterwarnings("ignore")
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()

from dataset import create_dataset
from utilities import utils
from model import Network

# check for GPU
use_cuda = torch.backends.mps.is_available()
device = torch.device("mps" if use_cuda else "cpu")
if use_cuda:
    torch.backends.cudnn.benchmark = True


def train_model(config_file):
    """
    config_file: json file with informations regarding data paths and training parameters

    """

    config = utils.parse_configuration(config_file)
    model_name = config["selected_attribute"]

    [train_loader, val_loader, class_names] = create_dataset(config, mode="train")

    model = Network(len(class_names))
    model.to(device)

    print("Computing weights for balanced loss...")

    class_weights = utils.compute_weights(train_loader, len(class_names))
    class_weights = torch.FloatTensor(class_weights).to(device)
    print("Estimated class weights:")
    print(class_weights)

    # Loss criterion is defined depending on number of classes
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    criterion.to(device)

    # Optimizer with l2 regularization and lr decay is used
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=6, gamma=0.1)

    try:
        os.makedirs(config["save_path"])
    except FileExistsError:
        pass

    num_epochs = config["max_epochs"]
    print("Started Training....")

    train_loss = []
    val_loss = []
    best_loss = 100
    for epoch in range(1, num_epochs + 1):
        model.train()
        train_running_loss = []
        for batch in tqdm(train_loader, desc="Train epoch {}.".format(epoch)):
            data = Variable(batch[0].to(torch.float).to(device))
            labels_true = Variable(
                batch[1].to(torch.long).to(device, non_blocking=True)
            )
            optimizer.zero_grad()
            labels_pred = model(data)
            loss = criterion(labels_pred, labels_true)
            loss.backward()
            optimizer.step()
            train_running_loss.append(loss.item())

        train_loss_epoch = np.average(train_running_loss)
        train_loss.append(train_loss_epoch)
        writer.add_scalar("training loss", train_loss_epoch, epoch)

        # Each epoch of training is followed by validation and the best model is saved
        model.eval()
        test_running_loss = []
        for batch in tqdm(val_loader, desc="Validation epoch {}.".format(epoch)):
            labels_pred = model(batch[0].to(torch.float).to(device))
            labels_true = batch[1].to(torch.long).to(device)
            loss = criterion(labels_pred, labels_true)
            test_running_loss.append(loss.item())

        val_loss_epoch = np.average(test_running_loss)
        val_loss.append(val_loss_epoch)
        writer.add_scalar("validation loss", val_loss_epoch, epoch)

        print(f"{epoch =}, {train_loss_epoch =}, {val_loss_epoch =}")
        # save the best model
        is_best = val_loss_epoch < best_loss
        best_loss = min(val_loss_epoch, best_loss)
        if is_best:
            torch.save(
                model.state_dict(), config["save_path"] + "best_" + f"{model_name}.pth"
            )
        scheduler.step()

    print("Evaluating final model on validation set...")
    utils.evaluator(model, val_loader, class_names, device, writer)

    # serialize the model to disk
    print("Saving Final model...")

    torch.save(
        model.state_dict(), config["save_path"] + "last_epoch_" + f"{model_name}.pth"
    )
    writer.flush()
    writer.close()


if __name__ == "__main__":
    """
    Takes the config file with all parameters set
    Trains and validates the model
    """
    parser = argparse.ArgumentParser(
        description="Perform model training and validation"
    )
    parser.add_argument("configfile", help="path to the configfile")
    args = parser.parse_args()
    train_model(args.configfile)
