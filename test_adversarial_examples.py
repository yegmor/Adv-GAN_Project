# Modified from https://github.com/mathcbc/advGAN_pytorch/blob/master/test_adversarial_examples.py

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import GAN_models as models
from target_models import MNIST_target_net
from utils import define_device


if __name__ == "__main__":
    image_nc = 1
    batch_size = 128
    gen_input_nc = image_nc
    targeted_model_path = './models/MNIST_target_model.pth'
    pretrained_generator_path = './models/GAN_generator_epoch_60.pth'

    device = define_device()

    # load the pretrained model
    target_model = MNIST_target_net().to(device)
    target_model.load_state_dict(torch.load(targeted_model_path))
    target_model.eval()

    # load the generator of adversarial examples
    pretrained_G = models.Generator(gen_input_nc, image_nc).to(device)
    pretrained_G.load_state_dict(torch.load(pretrained_generator_path))
    pretrained_G.eval()

    # test adversarial examples in MNIST training dataset
    mnist_dataset = datasets.MNIST('./dataset', train=True, transform=transforms.ToTensor(), download=True)
    train_dataloader = DataLoader(mnist_dataset, batch_size=batch_size, shuffle=False, num_workers=1)
    num_correct = 0
    for i, data in enumerate(train_dataloader, 0):
        test_img, test_label = data
        test_img, test_label = test_img.to(device), test_label.to(device)
        perturbation = pretrained_G(test_img)
        perturbation = torch.clamp(perturbation, -0.3, 0.3)
        adv_img = perturbation + test_img
        adv_img = torch.clamp(adv_img, 0, 1)
        pred_lab = torch.argmax(target_model(adv_img),1)
        num_correct += torch.sum(pred_lab==test_label,0)

    print('MNIST training dataset:')
    print('num_correct: ', num_correct.item())
    print('accuracy of adv imgs in training set: %f\n'%(num_correct.item()/len(mnist_dataset)))

    # test adversarial examples in MNIST testing dataset
    mnist_dataset_test = datasets.MNIST('./dataset', train=False, transform=transforms.ToTensor(), download=True)
    test_dataloader = DataLoader(mnist_dataset_test, batch_size=batch_size, shuffle=False, num_workers=1)
    num_correct = 0
    for i, data in enumerate(test_dataloader, 0):
        test_img, test_label = data
        test_img, test_label = test_img.to(device), test_label.to(device)
        perturbation = pretrained_G(test_img)
        perturbation = torch.clamp(perturbation, -0.3, 0.3)
        adv_img = perturbation + test_img
        adv_img = torch.clamp(adv_img, 0, 1)
        pred_lab = torch.argmax(target_model(adv_img),1)
        num_correct += torch.sum(pred_lab==test_label,0)

    print('num_correct: ', num_correct.item())
    print('accuracy of adv imgs in testing set: %f\n'%(num_correct.item()/len(mnist_dataset_test)))
