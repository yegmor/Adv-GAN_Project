import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
from multiprocessing import cpu_count

"""
Determine if any GPUs are available
"""
def define_device(use_cuda=True):
    is_cuda = torch.cuda.is_available()
    # print("CUDA Available: ", is_cuda)

    device = torch.device("cuda" if (use_cuda and is_cuda) else "cpu")
    print('Current device:', device)

    return device


def load_mnist(is_train, batch_size, shuffle):
    mnist_dataset = datasets.MNIST('./dataset', train=is_train, transform=transforms.ToTensor(), download=True)
    # dataloader = DataLoader(mnist_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=cpu_count())
    dataloader = DataLoader(mnist_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=1)
    return dataloader, len(mnist_dataset)


def plot_performance(counter, data, plt_names, fig_name, y_name, colors=None):
    if colors:
        for i, d in enumerate(data):
            plt.plot(counter, d, color=colors[i])
    else: 
        for d in data:
            plt.plot(counter, d)
    # plt.scatter(test_counter, test_losses, color='red')
    plt.legend(plt_names, loc='upper right')
    plt.grid()
    plt.title(fig_name)
    plt.xlabel('number of epochs passed')
    plt.ylabel(y_name)
    plt.savefig(f'./results/{fig_name}.png')
    plt.clf()
    plt.cla()
    plt.close()


# Add a clipping trick
def create_adv_example(data, perturbation, box_min, box_max):
    perturbation = torch.clamp(perturbation, -0.3, 0.3)
    adv_images = perturbation + data
    adv_images = torch.clamp(adv_images, box_min, box_max)
    return adv_images


def calculate_statistics(actual_labels, pred_labels):
    cf_matrix = confusion_matrix(actual_labels, pred_labels)
    per_class_accuracy = 100*cf_matrix.diagonal()/cf_matrix.sum(1)
    micro_f1 = f1_score(actual_labels, pred_labels, average='micro')
    weighted_f1 = f1_score(actual_labels, pred_labels, average='weighted')
    accuracy = 100*accuracy_score(actual_labels, pred_labels)
    return {"cf_matrix": cf_matrix, "per_class_accuracy": per_class_accuracy,
            "micro_f1": micro_f1, "weighted_f1": weighted_f1, "accuracy": accuracy}


def plot_confusion_matrix(cf_matrix, plt_name, cmap):
    classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')
    df_cm = pd.DataFrame(cf_matrix, index=[i for i in classes], columns=[
                         i for i in classes])
    plt.figure(figsize=(12, 7))
    plt.tight_layout()
    sn.heatmap(df_cm, annot=True, fmt='g', cmap=cmap)
    plt.title(plt_name)
    plt.xlabel("Predicted label")
    plt.ylabel("True Label (ground truth)")
    plt.savefig(f'./results/{plt_name}.png')
    plt.clf()
    plt.cla()
    plt.close()


def get_matrixed_imgs(adv_imgs, pred_labels, actual_labels, COLS=10, ROWS=10):
    matrix = {}
    for i in range(len(adv_imgs)):
        img = adv_imgs[i][0]
        actual = actual_labels[i]
        pred = pred_labels[i]

        matrix_idx = COLS*actual + pred + 1
        if matrix_idx not in matrix:
            matrix[matrix_idx] = []
        matrix[matrix_idx].append((img, actual, pred))

    print("len(images matrix)", len(matrix))
    return matrix


def plot_mnist(matrix_imgs, plt_name, COLS=10, ROWS=10):
    figure = plt.figure(figsize=(30, 25))
    # figure = plt.figure()
    # for i in range(1, COLS * ROWS + 1):
    for i, imgs in matrix_imgs.items():
        rand_idx = torch.randint(len(imgs), size=(1,)).item()
        img = imgs[rand_idx][0]
        actual = imgs[rand_idx][1]
        pred = imgs[rand_idx][2]
        
        figure.add_subplot(ROWS, COLS, i)
        
        plt.title('Actual: {}, Predicted: {}'.format(
            actual, pred), fontsize=15)
        plt.axis("off")
        plt.imshow(img, cmap="gray")
        plt.xticks([])
        plt.yticks([])
    plt.tight_layout()
    plt.show()
    plt.savefig(f'./results/{plt_name}.png')
    plt.clf()
    plt.cla()
    plt.close()
