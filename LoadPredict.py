import torch
import torch.nn as nn
import torchvision.transforms as transforms
import io
import time
import cv2

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
'''

class maskPredictor:
    
    
    def __init__(self):
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
        #toc1 = time.perf_counter()
        
        outputs = self.model(img_tensor)
        #outputs = traced_model(img_tensor)
        
        #TIMER
        #toc2 = time.perf_counter()
        #print(f"3rd checkpoingtime is {toc2 - toc1:0.4f} seconds")
        score, predicted = outputs.max(1)
        return score, predicted

    def loadtrace(self, model_path):
        device = 'cpu'
        with open(model_path,'rb') as f:
            buffer = io.BytesIO(f.read())
        self.model = torch.jit.load(buffer, map_location=device)
        