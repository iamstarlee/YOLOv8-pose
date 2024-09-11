import torch
import torch.nn as nn
import cv2
import numpy
import argparse
from utils import util
from torchvision import datasets, transforms
from spikingjelly.activation_based import ann2snn

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
    model_dict = model['model']
    
    for key, value in model_dict.named_parameters():
        if key.startswith('head'):
            del dict[key]
    torch.save(model, 'weights/best-truncate-with-structure.pt')

def infer_truncate_model(args):
    # model = torch.load('weights/best-truncate.pt').modified_layers.float()
    model = torch.load('weights/best-truncate-with-structure.pt')
    # print(model.modified_layers.net)
    image = cv2.imread('data/img0001.png')
    
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
    # x = model(x)
    for num in x:
        print(f"num is {num.shape}\n")
    

def ann_2_snn():
    # Prepare for datasets
    dataset_train = datasets.CIFAR10(root="datasets", train=True, download=False, transform=transforms.ToTensor())
    train_data_loader = torch.utils.data.DataLoader(dataset_train, batch_size = 64)

    # Initialize the model    
    model = torch.load('weights/backbone_resnet50_fintune.pt', map_location=torch.device('cpu')).backbone
    
    # Convert ANN to SNN
    model_converter = ann2snn.Converter(mode='max', dataloader=train_data_loader)
    snn_model = model_converter(model)
    torch.save(snn_model, 'weights/backbone_resnet50_fintune-snn.pt')
    print(snn_model)

def infer_snn(args):
    snn_model = torch.load('weights/backbone_resnet50_fintune-snn.pt')

    image = cv2.imread('data/img0001.png')
    
    stride = 32
    snn_model.half()
    snn_model.eval()
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

    y1 = snn_model(image)

    ann_model1 = torch.load('weights/backbone_resnet50_fintune.pt', map_location=torch.device('cpu')).neck.half()
    ann_model2 = torch.load('weights/backbone_resnet50_fintune.pt', map_location=torch.device('cpu')).bbox_head.half()

    y2 = ann_model1(y1)

    y3 = ann_model2(y2)[0]
    print(f"y3[0] is {y3[0].shape}")
    print(f"y3[1] is {y3[1].shape}")
    print(f"y3[2] is {y3[2].shape}")
    

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
    # infer_truncate_model(args)
    # truncate_model_with_structure()
    infer_snn(args)
