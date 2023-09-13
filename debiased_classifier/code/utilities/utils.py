import matplotlib.pyplot as plt
import json
from sklearn.utils import class_weight
import numpy as np
import torch
import torchvision
from sklearn.metrics import confusion_matrix, f1_score


def compute_weights(data_loader, num_classes):
    """Takes the dataloader
    and returns the class weights
    """
    y_train = torch.zeros(0, dtype=torch.long, device="cpu")
    for batch in data_loader:
        y_train = torch.cat([y_train, batch[1].view(-1).cpu()])
    weights = class_weight.compute_class_weight(
        "balanced", classes=np.unique(y_train), y=np.array(y_train)
    )
    print(weights / weights.sum())

    # Optionally normalize the class weights to sum to 1
    weights /= weights.sum()
    return weights


def imshow(input, title):
    input = input.numpy().transpose((1, 2, 0))
    fig = plt.figure()
    plt.imshow(input)
    plt.title(title)
    plt.show()
    return fig


def parse_configuration(config_file):
    """Loads config file if a string was passed
    and returns the input if a dictionary was passed.
    """
    if isinstance(config_file, str):
        with open(config_file) as json_file:
            return json.load(json_file)
    else:
        return config_file


def evaluator(model, data_loader, class_names, device, writer=None):
    """
    Calculates accuracy,confusion matrix and f1 score
    and plots the confusion matrix and  write to the summary writer
    """
    model.eval()
    correct = 0
    data_len = 0
    predlist = torch.zeros(0, dtype=torch.long, device="cpu")
    truelist = torch.zeros(0, dtype=torch.long, device="cpu")
    with torch.no_grad():
        for i, (data, label) in enumerate(data_loader):
            labels_pred = model(data.to(device))
            _, pred_idx = torch.max(labels_pred, 1)
            correct += (pred_idx == label.to(device)).sum().item()
            data_len += len(label)
            predlist = torch.cat([predlist, pred_idx.view(-1).detach().cpu()])
            truelist = torch.cat([truelist, label.view(-1).detach().cpu()])
            if i == 0:
                images = torchvision.utils.make_grid(data[:4], padding=2)

                label_tuples = [
                    (
                        list(class_names.keys())[int(label[i])],
                        list(class_names.keys())[pred_idx[i]],
                    )
                    for i in range(len(np.array(label)[:4]))
                ]
                plot = imshow(
                    images.cpu(),
                    title=label_tuples,
                )
               

    print("Accuracy: %d %%" % (100 * correct / data_len))
    cm = confusion_matrix(truelist.numpy(), predlist.numpy())
    f1score = f1_score(truelist.numpy(), predlist.numpy(), average="macro")
    fig = plot_confusion_matrix(cm, class_names, f1score)
    if writer is not None:
        writer.add_figure("Confusion_Matrix", fig)
        writer.add_figure("Network_predictions", plot)


def plot_confusion_matrix(
    cm, target_names, f1score, title="Confusion matrix", cmap=plt.cm.Blues
):
    import itertools

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy
    fig = plt.figure(figsize=(4, 4))
    plt.imshow(cm, interpolation="nearest", cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)
    thresh = cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(
            j,
            i,
            "{:,}".format(cm[i, j]),
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black",
        )

    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel(
        "Predicted label\naccuracy={:0.4f}; misclass={:0.4f}; f1score={:0.4f}".format(
            accuracy, misclass, f1score
        )
    )
    plt.show()
    return fig
