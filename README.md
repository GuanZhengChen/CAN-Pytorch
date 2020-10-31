## CAN: Co-embedding Attributed Networks
This repository contains the Python&Pytorch implementation for CAN. Further details about CAN can be found in 
The paper:
> Zaiqiao Meng, Shangsong Liang, Hongyan Bao, Xiangliang Zhang. Co-embedding Attributed Networks. (WSDM2019)

The orignal tensorflow implementation can be found in [CAN](https://github.com/mengzaiqiao/CAN)

A semi-supervised version of the CAN model  implemented by Pytorch can be found in [SCAN-Pytorch](https://github.com/GuanZhengChen/SCAN-Pytorch)

The orignal tensorflow implementation for SCAN can be found in [SCAN](https://github.com/mengzaiqiao/SCAN)
## Introduction

I try to keep the structure like tensorflow implementation,but there is also some changes:

>For computing the loss directly,i move part of the optimizer.py into train.py.

>I don't find the funtion like tf.nn.weighted_cross_entropy_with_logits() in pytorch,so I implemented by myself.It need computer torch.log(torch.sigmoid(logits)) and torch.log(1 - torch.sigmoid(logits)), if some values in logits too large or to small, act it by sigmod may get 1 or 0 and get -lnf after log. Therefore, I clamp the logits value from -10 to 10(I believe tensorflow do the same thing in the function)

```python
def weighted_cross_entropy_with_logits(logits, targets, pos_weight):
    logits=logits.clamp(-10,10)
    return targets * -torch.log(torch.sigmoid(logits)) *pos_weight + (1 - targets) * -torch.log(1 - torch.sigmoid(logits))
```


## Requirements

=================
* Pytorch (1.0 or later)
* python 3.6/3.7
* scikit-learn
* scipy

## Run the demo
=================

```bash
python train.py
```
