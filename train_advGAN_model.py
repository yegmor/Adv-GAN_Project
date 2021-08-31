# Modified from https://github.com/mathcbc/advGAN_pytorch/blob/master/main.py

import torch
from advGAN import AdvGAN_Attack
from target_models import MNIST_target_net
import utils


if __name__ == "__main__":
    training_parameters = {
        "EPOCHS": 60,
        "BATCH_SIZE": 128,
        "LEARNING_RATE": 0.001
    }
    targeted_model_file_name = './models/MNIST_target_model.pth'
    model_num_labels = 10
    image_nc = 1
    BOX_MIN = 0
    BOX_MAX = 1
    model_path = './models/'

    # Define what device we are using
    device = utils.define_device()

    # Load the pretrained targeted model
    targeted_model = MNIST_target_net().to(device)
    targeted_model.load_state_dict(torch.load(targeted_model_file_name))
    targeted_model.eval()

    # Load MNIST train dataset
    train_dataloader, train_data_count = utils.load_mnist(
        is_train=True, batch_size=training_parameters["BATCH_SIZE"], shuffle=True)
    
    # Train the AdvGAN model
    advGAN = AdvGAN_Attack(device, targeted_model, model_num_labels, image_nc,
                           BOX_MIN, BOX_MAX, training_parameters["LEARNING_RATE"], model_path=model_path)
    history = advGAN.train(train_dataloader, training_parameters["EPOCHS"])

    # Plot losses vs epoch
    utils.plot_performance(history["counter"], history["disc_losses"],
                           plt_name="discriminator_GAN_model_performance", y_name="discriminator's mse loss")
    utils.plot_performance(history["counter"], history["gen_losses"],
                           plt_name="generator_GAN_model_performance", y_name="generator's mse loss")
    utils.plot_performance(history["counter"], history["perturb_losses"],
                           plt_name="perturbation_GAN_model_performance", y_name="perturbation's loss")
    utils.plot_performance(history["counter"], history["adv_losses"],
                           plt_name="adversarial_GAN_model_performance", y_name="adversarial's loss")

