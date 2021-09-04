# Modified from https://github.com/mathcbc/advGAN_pytorch/blob/master/test_adversarial_examples.py

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import GAN_models
from target_models import MNIST_target_net
import utils 


BOX_MIN = 0
BOX_MAX = 1


def evaluate_target_model(target_model, dataloader):
    num_correct = 0

    for data in dataloader:
        img, label = data
        img, label = img.to(device), label.to(device)

        perturbation = generator(img)
        adv_img = utils.create_adv_example(img, perturbation, BOX_MIN, BOX_MAX)

        pred_label = torch.argmax(target_model(adv_img), 1)
        num_correct += torch.sum(pred_label == label, 0)

    return num_correct.item()


if __name__ == "__main__":
    image_nc = 1
    batch_size = 128
    gen_input_nc = image_nc
    targeted_model_path = './models/MNIST_target_model.pth'
    generator_path = './models/GAN_generator_epoch_60.pth'

    device = utils.define_device()

    # Load the pretrained target model
    target_model = MNIST_target_net().to(device)
    target_model.load_state_dict(torch.load(targeted_model_path))
    target_model.eval()

    # Load the generator of advGAN
    generator = GAN_models.Generator(gen_input_nc, image_nc).to(device)
    generator.load_state_dict(torch.load(generator_path))
    generator.eval()

    # Test adversarial examples in MNIST training dataset
    train_dataloader, train_data_count = utils.load_mnist(is_train=True, batch_size=batch_size, shuffle=False)
    
    train_num_correct = evaluate_target_model(target_model, train_dataloader)
    print('train_num_correct:', train_num_correct, '\ttotal train data:', train_data_count)
    print('accuracy of adv imgs in training set: %f\n' %(train_num_correct/train_data_count))


    # Test adversarial examples in MNIST testing dataset
    test_dataloader, test_data_count = utils.load_mnist(is_train=False, batch_size=batch_size, shuffle=False)
    
    test_num_correct = evaluate_target_model(target_model, test_dataloader)
    print('test_num_correct:', test_num_correct, '\ttotal test data:', test_data_count)
    print('accuracy of adv imgs in testing set: %f\n'%(test_num_correct/test_data_count))
