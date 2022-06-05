## CAN: Co-embedding Attributed Networks
This repository contains the Python&Pytorch implementation for CAN. Further details about CAN can be found in 
The paper:
> Zaiqiao Meng, Shangsong Liang, Hongyan Bao, Xiangliang Zhang. Co-embedding Attributed Networks. (WSDM2019)

The orignal tensorflow implementation can be found in [CAN](https://github.com/mengzaiqiao/CAN).

A semi-supervised version of the CAN model implemented by Pytorch can be found in [SCAN-Pytorch](https://github.com/GuanZhengChen/SCAN-Pytorch).

The orignal tensorflow implementation for SCAN can be found in [SCAN](https://github.com/mengzaiqiao/SCAN).

## Differences with tensorflow implementation

>For computing the loss directly, I move part of the optimizer.py into train.py.

>There is no funtion like tf.nn.weighted_cross_entropy_with_logits() in pytorch, so I implement it by myself. To avoid the overflowing issue dropped by sigmod (when computing the torch.log(torch.sigmoid(logits)) and torch.log(1 - torch.sigmoid(logits))), I clamp the logits value from -10 to 10.

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

## Result

The  Link prediction performance AUC&AP score :

| Dataset | AUC | AP |
| :--- | :------: | :------: |
| BLOGCATALOG | 0.820 | 0.822 |
| CORA | 0.989 | 0.988 |
| CITESEER | 0.993 | 0.989 |
| DBLP | 0.927 | 0.921 |
| FLICKR | 0.894 | 0.910 |
| FACEBOOK | 0.989 | 0.987 |

The  Attribute inference performance AUC&AP score :

| Dataset | AUC | AP |
| :--- | :------: | :------: |
| BLOGCATALOG | 0.876 | 0.876 |
| CORA | 0.932 | 0.916 |
| CITESEER | 0.949 | 0.934 |
| DBLP | 0.899 | 0.902 |
| FLICKR | 0.856 | 0.844 |
| FACEBOOK | 0.976 | 0.973 |

The  node classification performance Micro_F1&Macro_F1 score :

| Dataset | Micro_F1 | Macro_F1 |
| :--- | :------: | :------: |
| BLOGCATALOG | 0.760 | 0.755 |
| CORA | 0.869 | 0.857 |
| CITESEER | 0.743 | 0.689 |
| FLICKR |  0.634 | 0.629 |

