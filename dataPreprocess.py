import numpy as np 
import pandas as pd 
import torch 
from torchvision import transforms, datasets
from skimage import io, transform
from shutil import copyfile
import os

trainPath = r"data\train\\"
testPath = r"data\test\\"
valPath = r"data\val\\"

root = r"Chars74K Dataset\\"
trainImgPathList = r"Chars74K Dataset\Annotations\English_Img\good_train.txt"
testImgPathList = r"Chars74K Dataset\Annotations\English_Img\good_test.txt"
valImgPathList = r"Chars74K Dataset\Annotations\English_Img\good_val.txt"

def copyImages(srcList,dst):
    with open(srcList,'r') as pathList:
        for idx, path in enumerate(pathList):
            path = path.strip()
            classNum = path[38:40]
            #print(path[34:], classNum)
            if '0' == classNum[0]:
                classNum = classNum[1]
            imgDir = dst+classNum
            if not os.path.exists(imgDir):
                os.makedirs(imgDir)
            imgDir = imgDir+'\\'+path[34:]
            #print(imgDir)
            copyfile(root+path.strip(),imgDir)
    print("%s Images copied"%str(idx+1))


trainCount = copyImages(trainImgPathList,trainPath)
testCount = copyImages(testImgPathList,testPath)
valCount = copyImages(valImgPathList,valPath)

# class Chars74K_Dataset:
#     def __init__(self,dataSet)

class Chars74K_Dataset:
    def __init__(self, dataset='train', resize=(64,64), colorMap = 'grayscale'):
        self.dataset = dataset
        self.datasetPath = r'data\\'+self.dataset
        self.resize = resize
        self.colorMap = colorMap
        self.imgCount = sum([len(files) for subdir,dirs,files in os.walk(self.datasetPath)])
        self.classCount = len(os.listdir(self.datasetPath))
        csls = []
        for subdir,dirs,files in os.walk(self.datasetPath):
            folderName = subdir[-2:]
            #csls.append(folderName)
            if "" != folderName and folderName.isdigit():
                if '0' != folderName[0]:
                    csls.append(folderName)
                else:
                    csls.append(folderName[-1])
        self.classes = csls
    
    def datagenerator(self):
        data_transform = transforms.Compose([
            transforms.Resize(self.resize),
            transforms.Grayscale(num_output_channels=1),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
            ])

    # def __len__(self):
    #     return self.imgCount       

    # def __getitem__(self,batch=10):
    #     if torch.is_tensor(idx):
    #         idx = idx.tolist()   
        


# num = len(Chars74K_Dataset())
# print(num)