import matplotlib.pyplot as plt
import torch
import torchvision

def draw():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = torch.load('weights/backbone_resnet50_fintune.pt')
    model.to(device)

    model_weights = []   # append模型的权重
    conv_layers = []   # append模型的卷积层本身

    # get all the model children as list
    model_children = list(model.children())

    # counter to keep count of the conv layers
    counter = 0  # 统计模型里共有多少个卷积层

    model_children = list(model.children())
    print(model_children)

    # append all the conv layers and their respective wights to the list
    for i in range(len(model_children)):  # 遍历最表层(Sequential就是最表层)
        if type(model_children[i]) == nn.Conv2d:   # 最表层只有一个卷积层
            counter+=1
            model_weights.append(model_children[i].weight)
            conv_layers.append(model_children[i])

        elif type(model_children[i]) == nn.Sequential:
            for j in range(len(model_children[i])):
                for child in model_children[i][j].children():
                    if type(child) == nn.Conv2d:
                        counter+=1
                        model_weights.append(child.weight)
                        conv_layers.append(child)
    print(f"Total convolution layers: {counter}")

    outputs = []
    names = []

    for layer in conv_layers[0:]:    # conv_layers即是存储了所有卷积层的列表
        image = layer(image)  # 每个卷积层对image做计算，得到以矩阵形式存储的图片，需要通过matplotlib画出
        outputs.append(image)
        names.append(str(layer))
    print(len(outputs))

    for feature_map in outputs:
        print(feature_map.shape)

    processed = []

    for feature_map in outputs:
        feature_map = feature_map.squeeze(0)  # torch.Size([1, 64, 112, 112]) —> torch.Size([64, 112, 112])  去掉第0维 即batch_size维
        gray_scale = torch.sum(feature_map,0)
        gray_scale = gray_scale / feature_map.shape[0]  # torch.Size([64, 112, 112]) —> torch.Size([112, 112])   从彩色图片变为黑白图片  压缩64个颜色通道维度，否则feature map太多张
        processed.append(gray_scale.data.cpu().numpy())  # .data是读取Variable中的tensor  .cpu是把数据转移到cpu上  .numpy是把tensor转为numpy

    for fm in processed:
        print(fm.shape)

    fig = plt.figure(figsize=(30, 50))

    for i in range(len(processed)):   # len(processed) = 17
        a = fig.add_subplot(5, 4, i+1)
        img_plot = plt.imshow(processed[i])
        a.axis("off")
        a.set_title(names[i].split('(')[0], fontsize=30)   # names[i].split('(')[0] 结果为Conv2d

    plt.savefig('resnet18_feature_maps.jpg', bbox_inches='tight')  # 若不加bbox_inches='tight'，保存的图片可能不完整


if __name__ == '__main__':
    draw()