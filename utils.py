import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt


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
    dataloader = DataLoader(
        mnist_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=1)
    return dataloader, len(mnist_dataset)


def plot_performance(counter, losses, plt_name, y_name):
    plt.plot(counter, losses, color='blue')
    # plt.scatter(test_counter, test_losses, color='red')
    plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
    plt.grid()
    plt.xlabel('number of epochs passed')
    plt.ylabel(y_name)
    # plt.show()
    plt.savefig(f'./results/{plt_name}.png')
    plt.clf()
    plt.cla()
    plt.close()
