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
- 


## Simu
|  model         |  TPR(sensitivity)  |  PPV(Precision)    |
|  ----          | ----  | ----    |
| Simu-baseline  | 61.60% |  81.50%  |
| Simu-ResnextUnet   |  62.72%     |   72.98%      |


## Intra
|  model         |  TPR(sensitivity)  |  PPV(Precision)    |
|  ----          | ----  | ----    |
| Intra-baseline | 24.45% |  66.68%  |
| Intra-ourUnet  |       |         |

