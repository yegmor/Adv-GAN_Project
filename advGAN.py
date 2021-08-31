# Modified from https://github.com/mathcbc/advGAN_pytorch/blob/master/advGAN.py

import torch
import numpy as np
import GAN_models
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os


SAVING_INTERVAL = 20
LOG_INTERVAL = 1

# Custom weights initialization called on GAN's generator and discriminator
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


"""
Network Architectures: discriminator and generator
"""
class AdvGAN_Attack:
    def __init__(self, device, model, model_num_labels, image_nc, box_min, box_max, learning_rate, model_path):
        self.device = device
        self.model = model
        self.model_num_labels = model_num_labels
        self.input_nc = image_nc
        self.output_nc = image_nc
        self.box_min = box_min
        self.box_max = box_max
        self.learning_rate = learning_rate
        self.model_path = model_path

        self.gen_input_nc = image_nc
        self.net_gen = GAN_models.Generator(self.gen_input_nc, image_nc).to(device)
        self.net_disc = GAN_models.Discriminator(image_nc).to(device)

        # Initialize all weights
        self.net_gen.apply(weights_init)
        self.net_disc.apply(weights_init)

        # Initialize optimizers
        self.opt_gen = optim.Adam(self.net_gen.parameters(), lr=self.learning_rate)
        self.opt_disc = optim.Adam(self.net_disc.parameters(), lr=self.learning_rate)


    # Add a clipping trick
    def create_adv_example(self, data, perturbation):
        adv_images = torch.clamp(perturbation, -0.3, 0.3) + data
        adv_images = torch.clamp(adv_images, self.box_min, self.box_max)
        return adv_images


    def train_batch(self, imgs, labels):
        # Optimizing and training the discriminator
        # Real inputs = actual images of the MNIST dataset
        # Fake inputs = from the generator
        # Real inputs should be classified as 1 and fake as 0
        for i in range(1):
            self.opt_disc.zero_grad()

            pred_real = self.net_disc(imgs)
            label_real = torch.ones_like(pred_real, device=self.device)

            loss_disc_real = F.mse_loss(pred_real, label_real)
            loss_disc_real.backward()

            perturbation = self.net_gen(imgs)
            adv_image = self.create_adv_example(imgs, perturbation)
            pred_fake = self.net_disc(adv_image.detach())
            label_fake = torch.zeros_like(pred_fake, device=self.device)

            loss_disc_fake = F.mse_loss(pred_fake, label_fake)
            loss_disc_fake.backward()
            
            loss_disc_GAN = loss_disc_fake + loss_disc_real
            self.opt_disc.step()


        # Optimizing and training the generator
        # For generator, goal is to make the discriminator believe everything is 1
        for i in range(1):
            self.opt_gen.zero_grad()

            pred_fake = self.net_disc(adv_image)
            target_fake = torch.ones_like(pred_fake, device=self.device)
            
            loss_gen_fake = F.mse_loss(pred_fake, target_fake)
            loss_gen_fake.backward(retain_graph=True)

            # Calculate perturbation norm
            # C = 0.1
            # loss_perturb = torch.max(loss_perturb - C, torch.zeros(1, device=self.device))
            loss_perturb = torch.mean(torch.norm(perturbation.view(perturbation.shape[0], -1), 2, dim=1))

            # Calculate adv loss
            logits_model = self.model(adv_image)
            probs_model = F.softmax(logits_model, dim=1)
            onehot_labels = torch.eye(self.model_num_labels, device=self.device)[labels]
            # C&W loss function
            real = torch.sum(onehot_labels * probs_model, dim=1)
            other, _ = torch.max((1 - onehot_labels) * probs_model - onehot_labels * 10000, dim=1)
            zeros = torch.zeros_like(other)
            loss_adv = torch.max(real - other, zeros)
            loss_adv = torch.sum(loss_adv)

            # maximize cross_entropy loss
            # loss_adv = - F.mse_loss(logits_model, onehot_labels)
            # loss_adv = - F.cross_entropy(logits_model, labels)

            adv_lambda = 10
            pert_lambda = 1
            loss_G = adv_lambda * loss_adv + pert_lambda * loss_perturb
            loss_G.backward()
            self.opt_gen.step()

        return loss_disc_GAN.item(), loss_gen_fake.item(), loss_perturb.item(), loss_adv.item()


    """
    Network training procedure
    Every step both the loss for disciminator and generator is updated
    Discriminator aims to classify reals and fakes
    Generator aims to generate images as realistic as possible
    """
    def train(self, train_dataloader, epochs):
        history = {"counter": [], "disc_losses": [],
                   "gen_losses": [], "perturb_losses": [], "adv_losses": []}

        for epoch in range(1, epochs+1):
            if epoch == 50:
                self.opt_gen = optim.Adam(
                    self.net_gen.parameters(), lr=self.learning_rate/10)
                self.opt_disc = optim.Adam(
                    self.net_disc.parameters(), lr=self.learning_rate/10)
            if epoch == 80:
                self.opt_gen = optim.Adam(
                    self.net_gen.parameters(), lr=self.learning_rate/100)
                self.opt_disc = optim.Adam(
                    self.net_disc.parameters(), lr=self.learning_rate/100)

            loss_disc_sum = 0
            loss_gen_fake_sum = 0
            loss_perturb_sum = 0
            loss_adv_sum = 0
            for data in train_dataloader:
                images, labels = data
                images, labels = images.to(self.device), labels.to(self.device)

                loss_disc_batch, loss_gen_fake_batch, loss_perturb_batch, loss_adv_batch = \
                    self.train_batch(images, labels)

                loss_disc_sum += loss_disc_batch
                loss_gen_fake_sum += loss_gen_fake_batch
                loss_perturb_sum += loss_perturb_batch
                loss_adv_sum += loss_adv_batch
            

            num_batch = len(train_dataloader)
            history["counter"].append(epoch)
            history["disc_losses"].append(loss_disc_sum/num_batch)
            history["gen_losses"].append(loss_gen_fake_sum/num_batch)
            history["perturb_losses"].append(loss_perturb_sum/num_batch)
            history["adv_losses"].append(loss_adv_sum/num_batch)
            
            if epoch % LOG_INTERVAL == 0:
                print("epoch %d:\nloss_D: %.3f, loss_G_fake: %.3f,\
                \nloss_perturb: %.3f, loss_adv: %.3f, \n" %
                    (epoch, loss_disc_sum/num_batch, loss_gen_fake_sum/num_batch,
                    loss_perturb_sum/num_batch, loss_adv_sum/num_batch))

            # Save the generator
            if epoch % SAVING_INTERVAL == 0:
                netGenerator_file_name = self.model_path + 'GAN_generator_epoch_' + str(epoch) + '.pth'
                torch.save(self.net_gen.state_dict(), netGenerator_file_name)
                # torch.save(G, 'Generator_epoch_{}.pth'.format(epoch))

        return history
