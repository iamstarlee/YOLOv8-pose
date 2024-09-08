import torch
import torch.nn as nn
import cv2
import numpy
import argparse
from utils import util

class ModifiedModel(nn.Module):
    def __init__(self, original_model):
        super(ModifiedModel, self).__init__()
        self.modified_layers = nn.ModuleDict()
        
        for name, module in original_model.named_children():
            if not name.startswith('head'):
                self.modified_layers[name] = module
                
    def forward(self, x):
        for name, layer in self.modified_layers.items():
            x = layer(x)
        return x

def truncate_model_without_structure():
    model = torch.load('weights/best.pt')

    modified_model = ModifiedModel(model['model'])

    for name, module in modified_model.named_parameters():
        print(name)
    torch.save(modified_model, 'weights/best-truncate.pt')

def truncate_model_with_structure():
    model = torch.load('weights/best.pt')
    # keep the structure!

def infer_truncate_model(args):
    model = torch.load('weights/best-truncate.pt').modified_layers.float()
    # print(model.modified_layers.net)
    image = cv2.imread('data/img0001.png')
    # print(f"model is {model}")
    
    stride = 32
    model.half()
    model.eval()
    shape = image.shape[:2]  # current shape [height, width]
    r = min(1.0, args.input_size / shape[0], args.input_size / shape[1])
    pad = int(round(shape[1] * r)), int(round(shape[0] * r))
    w = args.input_size - pad[0]
    h = args.input_size - pad[1]
    w = numpy.mod(w, stride)
    h = numpy.mod(h, stride)
    w /= 2
    h /= 2
    if shape[::-1] != pad:  # resize
        image = cv2.resize(image,
                            dsize=pad,
                            interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(h - 0.1)), int(round(h + 0.1))
    left, right = int(round(w - 0.1)), int(round(w + 0.1))
    image = cv2.copyMakeBorder(image,
                                top, bottom,
                                left, right,
                                cv2.BORDER_CONSTANT)  # add border
    # Convert HWC to CHW, BGR to RGB
    image = image.transpose((2, 0, 1))[::-1]
    image = numpy.ascontiguousarray(image)
    image = torch.from_numpy(image)
    image = image.unsqueeze(dim=0)
    # image = image.cuda()
    image = image.half()
    image = image / 255
    # Inference
    x = image
    for name, layer in model.items(): # for model without structure
        x = layer(x)
    for num in x:
        print(f"num is {num.shape}\n")
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-size', default=640, type=int)
    parser.add_argument('--batch-size', default=32, type=int)
    parser.add_argument('--local_rank', default=0, type=int)
    parser.add_argument('--epochs', default=1000, type=int)
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--demo', action='store_true')

    args = parser.parse_args()
    # model_dict = torch.load('weights/best.pt')
    # print(model_dict['model'].head)
    infer_truncate_model(args)
