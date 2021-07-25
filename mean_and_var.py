
import numpy as np 
simu_f1 = [81.94, 62.33, 72.51, 77.07, 77.55]
simu_precision = [84.37,54.79,76.84,74.18,77.55]
simu_sensitivity = [79.63,72.25,68.64,80.19,77.55]

intra_f1 = [56.56, 60.22, 48.99, 57.38]
intra_Precision = [62.24,67.35,54.92,66.54]
intra_sensitivity = [51.81,54.44,44.22,50.45]

# arr = [68.46, 70.76, 72.88, 70.96]
arr = intra_sensitivity
#求均值
arr_mean = np.mean(arr)
#求总体方差
arr_var = np.var(arr)
#求总体标准差
arr_std = np.std(arr)
print("平均值为：%f" % arr_mean)
print("方差为：%f" % arr_var)
print("标准差为:%f" % arr_std)

