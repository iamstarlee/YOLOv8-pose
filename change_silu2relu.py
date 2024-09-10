import torch
import os
import numpy
import cv2
from utils import util

class ModifiedModel(torch.nn.Module):
    def __init__(self, original_model):
        super(ModifiedModel, self).__init__()
        self.modified_layers = torch.nn.ModuleDict()
        
        for name, module in original_model.named_children():
            if not name.startswith('head'):
                self.modified_layers[name] = module
                
    def forward(self, x):
        for name, layer in self.modified_layers.items():
            x = layer(x)
        return x

class ModifiedHead(torch.nn.Module):
    def __init__(self, original_model):
        super(ModifiedHead, self).__init__()
        self.modified_head = torch.nn.ModuleDict()
        
        for name, module in original_model.named_children():
            if name.startswith('head'):
                self.modified_head[name] = module
                
    def forward(self, x):
        for name, layer in self.modified_head.items():
            x = layer(x)
        return x

def img2input(image):
    shape = image.shape[:2]  # current shape [height, width]
    shape = image.shape[:2]  # current shape [height, width]
    r = min(1.0, 640 / shape[0], 640 / shape[1])
    pad = int(round(shape[1] * r)), int(round(shape[0] * r))
    w = 640 - pad[0]
    h = 640 - pad[1]
    w = numpy.mod(w, 32)
    h = numpy.mod(h, 32)
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

    return image

def post_process(image, outputs):
    frame = cv2.imread('data/img0001.png')
    shape = frame.shape[:2]

    palette = numpy.array([[255, 128, 0], [255, 153, 51], [255, 178, 102], [230, 230, 0], [255, 153, 255],
                           [153, 204, 255], [255, 102, 255], [255, 51, 255], [102, 178, 255], [51, 153, 255],
                           [255, 153, 153], [255, 102, 102], [255, 51, 51], [153, 255, 153], [102, 255, 102],
                           [51, 255, 51], [0, 255, 0], [0, 0, 255], [255, 0, 0], [255, 255, 255]],
                          dtype=numpy.uint8)
    skeleton = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13], [6, 7], [6, 8], [7, 9],
                [8, 10], [9, 11], [2, 3], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]
    kpt_color = palette[[16, 16, 16, 16, 16, 0, 0, 0, 0, 0, 0, 9, 9, 9, 9, 9, 9]]
    limb_color = palette[[9, 9, 9, 9, 7, 7, 7, 0, 0, 0, 0, 0, 16, 16, 16, 16, 16, 16, 16]]
    # NMS
    outputs = util.non_max_suppression(outputs, 0.25, 0.7, 1)
    
    for output in outputs:
        output = output.clone()
        if len(output):
            box_output = output[:, :6]
            kps_output = output[:, 6:].view(len(output), 17 ,3)
        else:
            box_output = output[:, :6]
            kps_output = output[:, 6:]

        r = min(image.shape[2] / shape[0], image.shape[3] / shape[1])

        box_output[:, [0, 2]] -= (image.shape[3] - shape[1] * r) / 2  # x padding
        box_output[:, [1, 3]] -= (image.shape[2] - shape[0] * r) / 2  # y padding
        box_output[:, :4] /= r

        box_output[:, 0].clamp_(0, shape[1])  # x
        box_output[:, 1].clamp_(0, shape[0])  # y
        box_output[:, 2].clamp_(0, shape[1])  # x
        box_output[:, 3].clamp_(0, shape[0])  # y

        kps_output[..., 0] -= (image.shape[3] - shape[1] * r) / 2  # x padding
        kps_output[..., 1] -= (image.shape[2] - shape[0] * r) / 2  # y padding
        kps_output[..., 0] /= r
        kps_output[..., 1] /= r
        kps_output[..., 0].clamp_(0, shape[1])  # x
        kps_output[..., 1].clamp_(0, shape[0])  # y

        for box in box_output:
            box = box.cpu().numpy()
            x1, y1, x2, y2, score, index = box
            cv2.rectangle(frame,
                            (int(x1), int(y1)),
                            (int(x2), int(y2)),
                            (0, 255, 0), 2)
        for kpt in reversed(kps_output):
            for i, k in enumerate(kpt):
                color_k = [int(x) for x in kpt_color[i]]
                x_coord, y_coord = k[0], k[1]
                if x_coord % shape[1] != 0 and y_coord % shape[0] != 0:
                    if len(k) == 3:
                        conf = k[2]
                        if conf < 0.5:
                            continue
                    cv2.circle(frame,
                                (int(x_coord), int(y_coord)),
                                5, color_k, -1, lineType=cv2.LINE_AA)
            for i, sk in enumerate(skeleton):
                pos1 = (int(kpt[(sk[0] - 1), 0]), int(kpt[(sk[0] - 1), 1]))
                pos2 = (int(kpt[(sk[1] - 1), 0]), int(kpt[(sk[1] - 1), 1]))
                if kpt.shape[-1] == 3:
                    conf1 = kpt[(sk[0] - 1), 2]
                    conf2 = kpt[(sk[1] - 1), 2]
                    if conf1 < 0.5 or conf2 < 0.5:
                        continue
                if pos1[0] % shape[1] == 0 or pos1[1] % shape[0] == 0 or pos1[0] < 0 or pos1[1] < 0:
                    continue
                if pos2[0] % shape[1] == 0 or pos2[1] % shape[0] == 0 or pos2[0] < 0 or pos2[1] < 0:
                    continue
                cv2.line(frame,
                            pos1, pos2,
                            [int(x) for x in limb_color[i]],
                            thickness=2, lineType=cv2.LINE_AA)

    # cv2.imshow('Frame', frame)
    save_path = os.path.join('results/', f'img0001_silu_ann_out.png')  # 按数字命名文件
    return cv2.imwrite(save_path, frame)  # 保存图片
    

def infer():
    img = cv2.imread('data/img0001.png')
    input = img2input(img)

    # Inference
    backbone = torch.load('weights/backbone.pt')
    output1 = backbone(input)
    print()

    neck = torch.load('weights/neck.pt')
    output11 = neck(output1)

    head = torch.load('weights/best-truncate-head.pt')
    output2 = head(output11)

    # Post processing
    answer = post_process(input, output2)

    if(answer):
        print("Saved successfully!")
    else:
        print("Fail!")


def change_them():
    backbon = torch.load('weights/best-truncate-silu.pt').modified_layers.net
    neck = torch.load('weights/best-truncate-silu.pt').modified_layers.fpn
    # print(backbone_and_neck)

    # torch.save(backbone, 'weights/backbone.pt')
    # torch.save(neck, 'weights/neck.pt')
    print(torch.load('weights/neck.pt'))

if __name__ == '__main__':
    infer()