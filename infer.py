import torch
import cv2
import numpy
import argparse
from utils import util
import os
import glob

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
palette = numpy.array([[255, 128, 0], [255, 153, 51], [255, 178, 102], [230, 230, 0], [255, 153, 255],
                           [153, 204, 255], [255, 102, 255], [255, 51, 255], [102, 178, 255], [51, 153, 255],
                           [255, 153, 153], [255, 102, 102], [255, 51, 51], [153, 255, 153], [102, 255, 102],
                           [51, 255, 51], [0, 255, 0], [0, 0, 255], [255, 0, 0], [255, 255, 255]],
                          dtype=numpy.uint8)
skeleton = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13], [6, 7], [6, 8], [7, 9],
            [8, 10], [9, 11], [2, 3], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]
kpt_color = palette[[16, 16, 16, 16, 16, 0, 0, 0, 0, 0, 0, 9, 9, 9, 9, 9, 9]]
limb_color = palette[[9, 9, 9, 9, 7, 7, 7, 0, 0, 0, 0, 0, 16, 16, 16, 16, 16, 16, 16]]

def draw_rec():
    frame = cv2.imread("data/img0001.png")
    # cv2.rectangle(frame,
    #             (int(x1), int(y1)),
    #             (int(x2), int(y2)),
    #             (0, 255, 0), 2)

def infer_one_image(args):
    model = torch.load('./weights/best.pt', map_location='cuda')['model'].float()
    stride = int(max(model.stride.cpu().numpy()))
    model.half()
    model.eval()

    frame = cv2.imread("data/img0001.png")
    image = frame.copy()
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
    image = image.cuda()
    image = image.half()
    image = image / 255
    # Inference
    outputs = model(image)
    # NMS
    outputs = util.non_max_suppression(outputs, 0.25, 0.7, model.head.nc)
    for output in outputs:
        output = output.clone()
        if len(output):
            box_output = output[:, :6]
            kps_output = output[:, 6:].view(len(output), *model.head.kpt_shape)
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
    flag = cv2.imwrite('data/img0001_out.png', frame)
    if flag:
        print(f"saved.")
    else:
        print("fail.")

def infer_many_images(args):
    model = torch.load('./weights/best.pt', map_location='cuda')['model'].float()
    stride = int(max(model.stride.cpu().numpy()))
    model.half()
    model.eval()
    folder_path = 'data/raw_data/pose_right_rec'
    image_files = glob.glob(os.path.join(folder_path, '*.png'))

    for image in image_files:
        # 获取文件名
        file_w = os.path.basename(image)
        file_wo = os.path.splitext(file_w)[0]

        image = cv2.imread(image)
        frame = image.copy()
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
        image = image.cuda()
        image = image.half()
        image = image / 255
        # Inference
        outputs = model(image)
        # NMS
        outputs = util.non_max_suppression(outputs, 0.25, 0.7, model.head.nc)
        for output in outputs:
            output = output.clone()
            if len(output):
                box_output = output[:, :6]
                kps_output = output[:, 6:].view(len(output), *model.head.kpt_shape)
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
                with open('data/test_xy/right_xy.txt', 'a+') as f:
                    f.write(f"{file_w} {(x2-x1)/2.0:.2f} {(y2-y1)/2.0:.2f}\n")

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
        
        flag = cv2.imwrite(f'data/test_right/{file_wo}.png', frame)
        if flag:
            print(f"{file_w} saved.")
        else:
            print(f"{file_w} fail.")


def shrink_channels(input):
    print(f"input has {input.shape}")
    in_reshape = input.view(input.shape[0], input.shape[1] // 8, 8, input.shape[2], input.shape[3])
    in_avg = in_reshape.mean(dim=2)
    print(f"input_avg is {in_avg.shape}")
    return in_avg

def test_backbone():
    frame = cv2.imread("data/img0001.png")
    image = frame.copy()
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
    image = image.cuda()
    # image = image.half()
    image = image / 255
    
    model = torch.load('weights/backbone_resnet50_fintune.pt').backbone.to(device)
    # print(model)
    output = model(image)
    print(output[0].shape)
    print(output[1].shape)
    print(output[2].shape)
    
    # shrink channels
    # conv11 = torch.nn.Conv2d(in_channels=output[0].shape[1], out_channels=int(output[0].shape[1] / 8), kernel_size=1, stride=1, padding=0).to(device)
    # conv12 = torch.nn.Conv2d(in_channels=output[1].shape[1], out_channels=int(output[1].shape[1] / 8), kernel_size=1, stride=1, padding=0).to(device)
    # conv13 = torch.nn.Conv2d(in_channels=output[2].shape[1], out_channels=int(output[2].shape[1] / 8), kernel_size=1, stride=1, padding=0).to(device)
    # outx1 = conv11(output[0])
    # outx2 = conv12(output[1])
    # outx3 = conv13(output[2])
    s_output = shrink_channels(output[0])

def draw_xy():
    frame = cv2.imread('data/img0001.png')
    frame = cv2.rectangle(frame,
                (133, 126),
                (280, 512),
                (0, 255, 0), 2)
    cv2.imwrite('data/img0001_xy.png', frame)

def prefix_zero():
    # 指定图片所在文件夹路径
    folder_path = 'data/raw_data/pose_left_rec'  # 替换为你的文件夹路径

    # 获取文件夹中的所有文件名
    file_names = sorted([f for f in os.listdir(folder_path) if f.endswith('.png')])

    # 遍历文件并重命名
    for file_name in file_names:
        # 分离文件名和扩展名
        file_number = file_name.split('.')[0]  # 获取文件名的数字部分
        file_extension = file_name.split('.')[1]  # 获取扩展名部分

        # 如果文件名是数字，则补全为四位
        if file_number.isdigit():
            new_file_name = f"{int(file_number):04d}.{file_extension}"  # 将数字部分补齐为四位数
            old_file_path = os.path.join(folder_path, file_name)
            new_file_path = os.path.join(folder_path, new_file_name)

            # 重命名文件
            os.rename(old_file_path, new_file_path)

            print(f'Renamed: {file_name} -> {new_file_name}')



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

    args.local_rank = int(os.getenv('LOCAL_RANK', 0))
    args.world_size = int(os.getenv('WORLD_SIZE', 1))
    
    # infer_image(args)
    # prefix_zero()
    infer_many_images(args)