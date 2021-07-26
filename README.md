# 文件说明
- Traindata是数据集
- utils.py里面是存放各种工具函数的地方
- network.py是网络架构 
- train.py是训练文件
- evaluation.py是评价函数所在的文件

TP 就是正确的点，FP就是标错的点，FN就是标漏的点  
sensitivity (TPR) = TP / TP + FN  
precision (PPV) = TP / TP + FP  
TPR即为敏感度(sensitivity) 所有的点里面标对了多少个  
PPV即为精确率(Precision) 标的点里面对了多少个  


baseline augmentation
- rotation of ±60◦
- pixel shifting in a range of ±10%
- mask pixel shifting in a range of ±1%
- shearing in a range of ±0.1
- brightness in a range of ±0.2
- contrast in a range from 0.3 to 0.5
- random saturation in a range from 0.5 to 2.0 and hue in a range of ±0.1
- horizontally and vertically with a probability of 50%

baseline also use 5-fold

my augmentation
- horizontally and vertically with a probability of 50%
- rotation of ±40◦
- ColorJitter(p = 0.5)
- A.RandomBrightnessContrast(p=0.5),


## Simu
|  model         |  TPR(sensitivity)  |  PPV(Precision)    | F1 score |
|  ----          | ----  | ----    |  ---- |
| Simu-baseline  | 61.60% |  81.50%  |  69.78% |
| mag_lr_ResnextUnet |  78.07%     |   75.11%      | 76.56%  |
| dice_ResnextUnet |  82.27%     |   77.19%      | 79.65%  |

## Intra
|  model         |  TPR(sensitivity)  |  PPV(Precision)    | F1 score |
|  ----          | ----  | ----    | ---- |
| Intra-baseline | 24.45% |  66.68%  |   35.78%   |
| mag_lr_ResnextUnet  |    50.11%   |     63.43%    | 55.98%   |
| dice_ResnextUnet  |    55.19%   |     66.45%    | 60.30%   |


## 5-fold Simu
| Metric           | model    | f1    | f2    | f3    | f4    | f5    | μ±σ        |
| ---------------- | -------- | ----- | ----- | ----- | ----- | ----- | ---------- |
| PPV(Precision)   | baseline | -     | -     | -     | -     | -     | 81.50±5.77 |
| PPV(Precision)   | 5-fold   | 84.37 | 54.79 | 76.84 | 74.18 | 77.55 | 73.55±9.96 |
| TPR(sensitivity) | baseline | -     | -     | -     | -     | -     | 61.60±6.11 |
| TPR(sensitivity) | 5-fold   | 79.63 | 72.25 | 68.64 | 80.19 | 77.55 | 75.65±4.49 |
| F1 score         | baseline | -     | -     | -     | -     | -     | 69.78      |
| F1 score         | 5-fold   | 81.94 | 62.33 | 72.51 | 77.07 | 77.55 | 74.28±6.68 |

## 5-fold Intra
| Metric           | model    | f1    | f2    | f3    | f4    | μ±σ        |
| ---------------- | -------- | ----- | ----- | ----- | ----- | ---------- |
| PPV(Precision)   | baseline | -     | -     | -     | -     | 66.68±4.67 |
| PPV(Precision)   | 5-fold   | 62.24 | 67.35 | 54.92 | 66.54 | 62.76±4.93 |
| TPR(sensitivity) | baseline | -     | -     | -     | -     | 24.45±5.06 |
| TPR(sensitivity) | 5-fold   | 51.81 | 54.44 | 44.22 | 50.45 | 50.23±3.75 |
| F1 score         | baseline | -     | -     | -     | -     | 35.78%     |
| F1 score         | 5-fold   | 56.56 | 60.22 | 48.99 | 57.38 | 55.79±4.15 |

