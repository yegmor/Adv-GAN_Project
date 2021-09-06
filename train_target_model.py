# Modified from https://github.com/mathcbc/advGAN_pytorch/blob/master/train_target_model.py
import torch
import torch.optim as optim
import torch.nn.functional as F
from target_models import  MNIST_target_net
import utils


def train_target_model(train_dataloader, train_data_count, test_dataloader, test_data_count, training_parameters):
    history = {"train_counter": [], "train_losses": [],
               "train_accuracy": [], "test_losses": [], "test_accuracy": []}
    
    target_model = MNIST_target_net().to(DEVICE)
    print(target_model)
    target_model.train()

    opt_model = optim.Adam(target_model.parameters(),
                           lr=training_parameters["LEARNING_RATE"])

    for epoch in range(training_parameters["EPOCHS"]):
        target_model.train()
        loss_epoch = 0

        if epoch == 20:
            opt_model = optim.Adam(target_model.parameters(), 
                                   lr=training_parameters["LEARNING_RATE"]/10)

        for data in train_dataloader:
            train_imgs, train_labels = data
            train_imgs, train_labels = train_imgs.to(DEVICE), train_labels.to(DEVICE)

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
        train_num_correct, _ = evaluate_target_model(target_model, train_dataloader)
        history["train_accuracy"].append(train_num_correct/train_data_count)
        
        test_num_correct, test_loss = evaluate_target_model(target_model, test_dataloader)
        history["test_losses"].append(test_loss)
        history["test_accuracy"].append(test_num_correct/test_data_count)

        if epoch % LOG_INTERVAL == 0:
            print('loss in epoch %d: %f' % (epoch+1, loss_epoch.item()))

    return target_model, history


def evaluate_target_model(target_model, dataloader):
    # Evaluate test dataset on target model
    target_model.eval()
    val_loss = 0
    num_correct = 0
    with torch.no_grad():
        for data in dataloader:
            test_img, test_label = data
            test_img, test_label = test_img.to(DEVICE), test_label.to(DEVICE)
            
            logits_model = target_model(test_img)

            val_loss += F.cross_entropy(logits_model, test_label)

            pred_label = torch.argmax(target_model(test_img), 1)
            num_correct += torch.sum(pred_label == test_label, 0)

    return num_correct.item(), val_loss.item()


if __name__ == "__main__":
    training_parameters = {
        "EPOCHS": 40,
        "BATCH_SIZE": 256,
        "LEARNING_RATE": 0.001
    }
    LOG_INTERVAL = 1
    targeted_model_file_name = './models/MNIST_target_model.pth'
    
    # Define what device we are using
    DEVICE = utils.define_device()

    # Load MNIST train dataset
    train_dataloader, train_data_count = utils.load_mnist(
        is_train=True, batch_size=training_parameters["BATCH_SIZE"], shuffle=False)
    # Load MNIST test dataset
    test_dataloader, test_data_count = utils.load_mnist(
        is_train=False, batch_size=training_parameters["BATCH_SIZE"], shuffle=True)

    # Train the target model
    target_model, history = train_target_model(
        train_dataloader, train_data_count, test_dataloader, test_data_count, training_parameters)

    # Save model
    torch.save(target_model.state_dict(), targeted_model_file_name)
    # torch.save(optimizer.state_dict(), '/results/optimizer.pth')

    # Plot loss vs epoch
    utils.plot_performance(history["train_counter"],
                           data=[history["train_losses"], history["test_losses"]],
                           plt_names=['Train Loss', 'Test Loss'],
                           fig_name="before_targeted_model_loss",
                           y_name="target model's cross entropy loss",
                           colors=['c', 'red'])
    # Plot accuracy vs epoch
    utils.plot_performance(history["train_counter"],
                           data=[history["train_accuracy"], history["test_accuracy"]],
                           plt_names=['Train Accuracy', 'Test Accuracy'],
                           fig_name="before_targeted_model_accuracy",
                           y_name="target model's accuracy",
                           colors=['orange', 'dodgerblue'])

    # Evaluate test dataset on target model
    test_num_correct, test_loss = evaluate_target_model(target_model, test_dataloader)
    print('test_num_correct:', test_num_correct, '\ttotal test data:', test_data_count)
    print('loss of adv imgs in testing set: %f' % (test_loss))
    print('accuracy of adv imgs in testing set: %f' %(test_num_correct/test_data_count))
    print('attack success rate on testing set: %f' %(100 - test_num_correct/test_data_count))
