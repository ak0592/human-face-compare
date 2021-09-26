# PyTorch-ADDA
A PyTorch implementation for [Adversarial Discriminative Domain Adaptation](https://arxiv.org/abs/1702.05464).

Implementation reference : [pytorch-adda](https://github.com/corenel/pytorch-adda)
## Environment
- Python 3.6
- PyTorch 0.2.0
- Ubuntu 18.04

or
- google colaboratory
## Installation Steps
1.clone
```
$ git clone https://github.com/ak0592/human-face-compare.git
```
2.environment setup
```
$ cd human-face-compare/adda
$ cp docker/.env.sh docker/env.sh
$ sh docker/build.sh
$ sh docker/run.sh
$ sh docker/exec.sh
```
3.start jupyter
```
$ sh jupyter_run.sh
```
## Usage
1.Put **one** image you want to test in testdata/test

2.Use command:

```shell
$ python3 test.py
```
3. Delete the test image automatically．
## Network

In this experiment, I use three types of network. They are very simple.

- LeNet encoder

  ```
  LeNetEncoder (
    (encoder): Sequential (
      (0): Conv2d(1, 20, kernel_size=(4, 4), stride=(2, 2))
      (1): MaxPool2d (size=(2, 2), stride=(2, 2), dilation=(1, 1))
      (2): ReLU ()
      (3): Conv2d(20, 50, kernel_size=(4, 4), stride=(2, 2))
      (4): Dropout2d (p=0.5)
      (5): MaxPool2d (size=(2, 2), stride=(2, 2), dilation=(1, 1))
      (6): ReLU ()
    )
    (fc1): Linear (1250 -> 500)
  )
  ```

- LeNet classifier

  ```
  LeNetClassifier (
    (fc2): Linear (500 -> 4)
  )
  ```

- Discriminator

  ```
  Discriminator (
    (layer): Sequential (
      (0): Linear (500 -> 500)
      (1): ReLU ()
      (2): Linear (500 -> 500)
      (3): ReLU ()
      (4): Linear (500 -> 2)
      (5): LogSoftmax ()
    )
  )
  ```
