import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision
import io
import time
import cv2
import logging as log
from typing import Optional
from PIL import Image
from fastai.vision import *
###deprecated functions
'''
def predictMask(frame, model,device='cpu'):
    traced_model = model
    vimage = frame
    x = vimage[:, :, (2, 1, 0)]

    x = torch.from_numpy(x).permute(2, 0, 1).to(device)
    to_norm_tensor = transforms.Compose([
            #transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    img_tensor = to_norm_tensor(x.float().div_(255))
    img_tensor.unsqueeze_(0)
    
    #TIMER
    toc1 = time.perf_counter()
    outputs = traced_model(img_tensor)
    
    #TIMER
    toc2 = time.perf_counter()
    print(f"3rd checkpoingtime is {toc2 - toc1:0.4f} seconds")
    score, predicted = outputs.max(1)

    return score, predicted

def loadtracedModel(model_path):
    device = 'cpu'
    with open(model_path,'rb') as f:
        buffer = io.BytesIO(f.read())
    traced_model = torch.jit.load(buffer, map_location=device)
    return traced_model

class Flatten(nn.Module):
    "Flatten `x` to a single dimension, often used at the end of a model. `full` for rank-1 tensor"
    def __init__(self, full:bool=False):
        super().__init__()
        self.full = full

    def forward(self, x):
        return x.view(-1) if self.full else x.view(x.size(0), -1)

class AdaptiveConcatPool2d(nn.Module):
    "Layer that concats `AdaptiveAvgPool2d` and `AdaptiveMaxPool2d`." # from pytorch
    def __init__(self, sz:Optional[int]=None): 
        "Output will be 2*sz or 2 if sz is None"
        super().__init__()
        self.output_size = sz or 1
        self.ap = nn.AdaptiveAvgPool2d(self.output_size)
        self.mp = nn.AdaptiveMaxPool2d(self.output_size)
    def forward(self, x): return torch.cat([self.mp(x), self.ap(x)], 1)
    
def myhead(nf, nc):
    return \
    nn.Sequential(        # the dropout is needed otherwise you cannot load the weights
            AdaptiveConcatPool2d(),
            Flatten(),
            nn.BatchNorm1d(nf),
            nn.Dropout(p=0.25),
            nn.Linear(nf, 512),
            nn.ReLU(True),
            nn.BatchNorm1d(512),
            nn.Dropout(p=0.5),
            nn.Linear(512, nc),
        )

'''

class maskPredictor:
    
    
    def __init__(self):
        #learn = cnn_learner('stage_2mn.pth')
        self.model = None

    def maskPredict(self, frame):
        device = 'cpu'
        vimage= frame
        x = vimage[:, :, (2, 1, 0)]
        x = torch.from_numpy(x).permute(2, 0, 1).to(device)
        to_norm_tensor = transforms.Compose([
            #transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        img_tensor = to_norm_tensor(x.float().div_(255))
        img_tensor.unsqueeze_(0)

        #TIMER
        toc1 = time.perf_counter()
        outputs = self.model(img_tensor)
        toc2 = time.perf_counter()
        softmaxer = torch.nn.Softmax(dim=1)
        out = softmaxer(outputs)
        print(out)
        
        #outputs = traced_model(img_tensor)
        #print (outputs)
        #TIMER
        
        print(f"3rd checkpoingtime is {toc2 - toc1:0.4f} seconds")
        score, predicted = out.max(1)
        return score, predicted

    def loadtrace(self, model_path):
        device = 'cpu'
        with open(model_path,'rb') as f:
            buffer = io.BytesIO(f.read())
        self.model = torch.jit.load(buffer, map_location=device)

class maskPredictor_fastai:
    def __init__(self):
        defaults.device = torch.device('cpu')
        
        classes = ['noMask','withMask']
        path = 'data'
        data2 = ImageDataBunch.single_from_classes(path, classes, ds_tfms=get_transforms(), size=224).normalize(imagenet_stats)
        
        learn = cnn_learner(data2, models.resnet34, metrics=error_rate)
        learn.load('res34_stage2_1')
        self.model = learn
        


    def maskPredict(self, face):
        
        img = Image(pil2tensor(face, dtype=np.float32).div_(255))
        #TIMER
        toc1 = time.perf_counter()
        pred_class,pred_idx,outputs = self.model.predict(img)
        toc2 = time.perf_counter()
        score = outputs.max()
        #outputs = traced_model(img_tensor)
        #print (outputs)
        #TIMER
        print(f"3rd checkpoingtime is {toc2 - toc1:0.4f} seconds")
        #print(str(pred_class))
        #print(str(pred_idx) + 'is the idx')
        return score, pred_idx

    def loadtrace(self, model_path):
        device = 'cpu'
        with open(model_path,'rb') as f:
            buffer = io.BytesIO(f.read())
        self.model = torch.jit.load(buffer, map_location=device)



    