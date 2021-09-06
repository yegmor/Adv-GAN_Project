# Modified from https://github.com/mathcbc/advGAN_pytorch/blob/master/test_adversarial_examples.py

import torch
import GAN_models
from target_models import MNIST_target_net
import utils 


def evaluate_target_model(dataloader):
    actual_labels = []
    pred_labels = []
    adv_imgs = []

    with torch.no_grad():
        for data in dataloader:
            img, label = data
            img, label = img.to(DEVICE), label.to(DEVICE)

            perturbation = GENERATOR(img)
            adv_img = utils.create_adv_example(img, perturbation, BOX_MIN, BOX_MAX)
            adv_imgs.extend(adv_img.detach().cpu().numpy())
            
            prediciton = torch.argmax(TARGET(adv_img), 1)
            pred_labels.extend(prediciton)
            actual_labels.extend(label.view_as(prediciton))

    return [i.item() for i in pred_labels], [i.item() for i in actual_labels], adv_imgs#torch.stack(adv_imgs)


if __name__ == "__main__":
    BOX_MIN, BOX_MAX = 0, 1
    COLS, ROWS = 10, 10
    image_nc = 1
    batch_size = 128
    gen_input_nc = image_nc
    targeted_model_path = './models/MNIST_target_model.pth'
    generator_path = './models/GAN_generator_epoch_60.pth'

    DEVICE = utils.define_device()

    # Load the pretrained target model
    TARGET = MNIST_target_net().to(DEVICE)
    TARGET.load_state_dict(torch.load(targeted_model_path))
    TARGET.eval()

    # Load the generator of advGAN
    GENERATOR = GAN_models.Generator(gen_input_nc, image_nc).to(DEVICE)
    GENERATOR.load_state_dict(torch.load(generator_path))
    GENERATOR.eval()

    # Test adversarial examples in MNIST training dataset
    train_dataloader, train_data_count = utils.load_mnist(is_train=True, batch_size=batch_size, shuffle=False)
    
    train_pred_labels, train_actual_labels, train_adv_imgs = evaluate_target_model(train_dataloader)


    train_stats = utils.calculate_statistics(train_actual_labels, train_pred_labels)
    
    utils.plot_confusion_matrix(
        train_stats["cf_matrix"], plt_name="trainset_confusion_matrix", cmap="viridis")

    print('Training set per-class accuracy:\n', list(zip(range(10), train_stats["per_class_accuracy"])))
    print('Training set F1 score (micro): %f' % train_stats["micro_f1"])
    print('Training set F1 score (weighted): %f' % train_stats["weighted_f1"])
    print('Training set Accuracy score: %f' % train_stats["accuracy"])
    print('Training set attack success rate: %f' %(100 - train_stats["accuracy"]))

    train_matrix_imgs = utils.get_matrixed_imgs(train_adv_imgs, train_pred_labels, train_actual_labels)
    utils.plot_mnist(train_matrix_imgs, plt_name="trainset_imgs_matrix")

    # # Test adversarial examples in MNIST testing dataset
    test_dataloader, test_data_count = utils.load_mnist(is_train=False, batch_size=batch_size, shuffle=False)

    test_pred_labels, test_actual_labels, test_adv_imgs = evaluate_target_model(test_dataloader)
    

    test_stats = utils.calculate_statistics(test_actual_labels, test_pred_labels)
    
    utils.plot_confusion_matrix(
        test_stats["cf_matrix"], plt_name="testset_confusion_matrix", cmap="viridis")

    print('\nTesting set per-class accuracy:\n', list(zip(range(10), test_stats["per_class_accuracy"])))
    print('Testing set F1 score (micro): %f' % test_stats["micro_f1"])
    print('Testing set F1 score (weighted): %f' % test_stats["weighted_f1"])
    print('Testing set Accuracy score: %f' % test_stats["accuracy"])
    print('Testing set attack success rate: %f' %(100 - test_stats["accuracy"]))

    test_matrix_imgs = utils.get_matrixed_imgs(test_adv_imgs, test_pred_labels, test_actual_labels)
    utils.plot_mnist(test_matrix_imgs, plt_name="testset_imgs_matrix")
