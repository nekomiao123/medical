# 文件说明
- Traindata是数据集
- lib里面是存放各种工具文件的地方
- network是网络架构 

TPR即为敏感度(sensitivity),衡量了分类器对正例的识别能力
PPV即为精确率(Precision),表示被分为正例的示例中实际为正例的比例

## Simu
|  model         |  TPR(sensitivity)  |  PPV(Precision)    |
|  ----          | ----  | ----    |
| Simu-baseline  | 61.60% |  81.50%  |
| Simu-ourUnet   |       |         |


## Intra
|  model         |  TPR(sensitivity)  |  PPV(Precision)    |
|  ----          | ----  | ----    |
| Intra-baseline | 24.45% |  66.68%  |
| Intra-ourUnet  |       |         |

