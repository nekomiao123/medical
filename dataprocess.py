import json
import glob
import os
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import torch
import matplotlib.pyplot as plt

def generate_mask(img_height,img_width,radius,center_x,center_y):
    y,x=np.ogrid[0:img_height,0:img_width]
    # circle mask
    mask = (x-center_x)**2+(y-center_y)**2<=radius**2
    return mask

def label2mask(file_name):
    file_in = json.load(open(file_name))
    points = file_in["points"]
    point_outs = []
    height = file_in["imageHeight"]
    width = file_in["imageWidth"]
    img = np.zeros((height,width,3),dtype = np.uint8)
    for point in points:
        point_out = []
        point_out.append(point["x"])
        point_out.append(point["y"])
        point_outs.append(point_out)
    circles = []
    # 画圆
    for a in point_outs:
        masks = generate_mask(height,width,6,a[1],a[0])
        xs,ys = np.where(masks == True)
        for i in range(len(xs)):
            circle = []
            circle.append(xs[i])
            circle.append(ys[i])
            circles.append(circle)
    for b in circles:
        img[b[1],b[0]] = [255,255,255]
    return img

class Medical_Data(Dataset):
    def __init__(self, data_path, data_mode, set_mode):
        '''
        data_path: data path
        data_mode: simulator or intra data
        set_mode:  train or test
        '''
        self.data_path = data_path
        if data_mode == "simulator":
            self.imgs_path = glob.glob(os.path.join(data_path,"aicm[1-9]/*/images/*.png"))
            self.imgs_path += glob.glob(os.path.join(data_path,"aicm10/*/images/*.png"))
        elif data_mode == "intra":
            self.imgs_path = glob.glob(os.path.join(data_path,"aicm1[1-4]/*/images/*.png"))

        print('Finished reading the {}_{} set of medical dataset ({} samples found)'
            .format(data_mode, set_mode, len(self.imgs_path)))

    def augment(self, image, code):
        print("data augment")

    def __getitem__(self, index):
        image_path = self.imgs_path[index]
        label_path = image_path.replace("images","point_labels").replace(".png",".json")
        img = Image.open(image_path)
        image = torch.from_numpy(np.array(img)) 
        label = torch.from_numpy(label2mask(label_path))
        return image,label
    
    def __len__(self):
        return len(self.imgs_path)

if __name__ == "__main__":
    simulator_dataset = Medical_Data("./Traindata/","simulator","test")
    intra_dataset = Medical_Data("./Traindata/","intra","test")
    simulator_loader = torch.utils.data.DataLoader(dataset=simulator_dataset,
                                               batch_size=1, 
                                               shuffle=True)
    dataiter = iter(simulator_loader)
    images, labels = dataiter.next()

    img_show = Image.fromarray(images.numpy()[0,:,:,:],"RGB")
    img_show.show()
    img_show.save("./pic/images.png")
    img_show = Image.fromarray(labels.numpy()[0,:,:,:],"RGB")
    img_show.show()
    img_show.save("./pic/labels.png")