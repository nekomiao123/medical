import glob
import os
import json

def label2mask(file_name):
    file_in = json.load(open(file_name))
    points = file_in["points"]
    x = []
    if len(points):
        for point in points:
            if int(point["x"]) == 1:
                print(file_name)
            x.append(int(point["x"]))
            # print(point["x"])
        return min(x),max(x)
    else:
        return 512,0
    # point_outs = []
    # height = file_in["imageHeight"]
    # width = file_in["imageWidth"]
    # img = np.zeros((height,width,3),dtype = np.uint8)
    # for point in points:
    #     point_out = []
    #     point_out.append(point["x"])
    #     point_out.append(point["y"])
    #     point_outs.append(point_out)
    # circles = []
    # # 画圆
    # for a in point_outs:
    #     masks = generate_mask(height,width,6,a[1],a[0])
    #     xs,ys = np.where(masks == True)
    #     for i in range(len(xs)):
    #         circle = []
    #         circle.append(xs[i])
    #         circle.append(ys[i])
    #         circles.append(circle)
    # for b in circles:
    #     img[b[1],b[0]] = [1,1,1]
    # return img

paths = glob.glob(os.path.join("./Traindata/","*/*/point_labels/*.json"))
# print(paths)
x_min=[]
x_max=[]
for p in paths:
    # print(p)
    a,b = label2mask(p)
    x_min.append(a)
    x_max.append(b)
print(min(x_min),max(x_max))
