# advGAN_pytorch: https://github.com/mathcbc/advGAN_pytorch
a Pytorch implementation of the paper "Generating Adversarial Examples with Adversarial Networks" (advGAN).

## training the target model

```shell
python3 train_target_model.py
```

## training the advGAN

```shell
python3 train_advGAN_model.py
```

## testing adversarial examples

```shell
python3 test_adversarial_examples.py
```

## results

**attack success rate** in the MNIST test set: **99%**

**NOTE:** This implementation is a little different from the paper, because a clipping trick has been added.
