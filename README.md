# Generating Adversarial Examples using Adv-GAN

This project is adapted from  https://github.com/mathcbc/advGAN_pytorch, and it is a Pytorch implementation of the paper ["Generating Adversarial Examples with Adversarial Networks" (Adv-GAN)](https://arxiv.org/abs/1801.02610v5).


## Training the target model

```shell
python3 train_target_model.py
```

## Training the AdvGAN

```shell
python3 train_advGAN_model.py
```

## Testing adversarial examples using generator network of the AdvGAN

```shell
python3 test_adversarial_examples.py
```

## Results

**Attack success rate** on the MNIST test set: **99%**

**NOTE:** This implementation is a little different from the paper, because a clipping trick has been added.
