# Modified from https://github.com/mathcbc/advGAN_pytorch/blob/master/train_target_model.py

import torch
import torch.optim as optim
import torch.nn.functional as F
from target_models import  MNIST_target_net
import utils


LOG_INTERVAL = 1


def train_target_model(train_dataloader, training_parameters):
    target_model = MNIST_target_net().to(device)
    print(target_model)

    target_model.train()
    opt_model = optim.Adam(target_model.parameters(),
                           lr=training_parameters["LEARNING_RATE"])

    history = {"train_losses": [], "train_counter": []}
    for epoch in range(training_parameters["EPOCHS"]):
        loss_epoch = 0

        if epoch == 20:
            opt_model = optim.Adam(target_model.parameters(), 
                                   lr=training_parameters["LEARNING_RATE"]/10)

        for data in train_dataloader:
            train_imgs, train_labels = data
            train_imgs, train_labels = train_imgs.to(device), train_labels.to(device)

            logits_model = target_model(train_imgs)

            loss_model = F.cross_entropy(logits_model, train_labels)
            loss_epoch += loss_model
            
            # Clear gradients for this training step
            opt_model.zero_grad()
            # Backpropagation, compute gradients
            loss_model.backward()
            # Apply gradients
            opt_model.step()

        history["train_counter"].append(epoch+1)
        history["train_losses"].append(loss_epoch.item())

        if epoch % LOG_INTERVAL == 0:
            print('loss in epoch %d: %f' % (epoch, loss_epoch.item()))

    return target_model, history


def evaluate_target_model(target_model, test_dataloader, test_data_count):
    # Evaluate test dataset on target model
    num_correct = 0
    for data in test_dataloader:
        test_img, test_label = data
        test_img, test_label = test_img.to(device), test_label.to(device)

        pred_label = torch.argmax(target_model(test_img), 1)
        num_correct += torch.sum(pred_label == test_label, 0)

    print('num_correct: ', num_correct.item())
    print('accuracy in testing set: %f\n'%(num_correct.item()/test_data_count))
    return num_correct


if __name__ == "__main__":
    training_parameters = {
        "EPOCHS": 40,
        "BATCH_SIZE": 256,
        "LEARNING_RATE": 0.001
    }
    targeted_model_file_name = './models/MNIST_target_model.pth'
    
    # Define what device we are using
    device = utils.define_device()

    # Load MNIST train dataset
    train_dataloader, train_data_count = utils.load_mnist(
        is_train=True, batch_size=training_parameters["BATCH_SIZE"], shuffle=False)
    
    # Train the target model
    target_model, history = train_target_model(train_dataloader, training_parameters)

    # Plot losses vs epoch
    utils.plot_performance(history["train_counter"], history["train_losses"],
                           plt_name="targeted_model_performance", y_name="target model's cross entropy loss")

    # Save model
    torch.save(target_model.state_dict(), targeted_model_file_name)
    # torch.save(optimizer.state_dict(), '/results/optimizer.pth')

    # Load MNIST test dataset
    test_dataloader, test_data_count = utils.load_mnist(
        is_train=False, batch_size=training_parameters["BATCH_SIZE"], shuffle=True)
        
    # Evaluate test dataset on target model
    target_model.eval()
    num_correct = evaluate_target_model(target_model, test_dataloader, test_data_count)
