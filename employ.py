import PIL
import torch
from torchvision.transforms import transforms
import numpy as np
from model import RecognizeNet

transformation = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])



# 定义device，是否使用GPU，依据计算机配置自动会选择
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def predict_image(image_path, model_path, image_index):
    with torch.no_grad():
        print("Prediction in progress")
        image = PIL.Image.open(image_path)


        # 预处理图像,输出时已经归一化并且为Tensor型
        image = transformation(image)

        # 额外添加一个批次维度，因为PyTorch将所有的图像当做批次
        image = image.unsqueeze_(0)

        # 将图片数据迁移至GPU
        input = image.to(device)

        # 载入模型
        # model_evl = torch.load('{}model_{}.model'.format(model_path, 10))

        model_evl = RecognizeNet()

        model_evl.load_state_dict(torch.load('{}7_24model_{}.model'.format(model_path, 100)))  # 导入模型数据

        model_evl = model_evl.to(device)

        # 将模型设置为验证模式
        model_evl.eval()

        # 预测图像的类
        output = model_evl(input)

        index = output.data.cpu().argmax()

        print('预测的结果为:{}'.format(image_index[index.data]))


if __name__=='__main__':
    image_index = ['宫崎', '山下', '助手']
    model_path = 'D:/Pytorch/face_recognize/'
    image_path = 'D:/data/temp/val/1/134.jpg'
    predict_image(image_path, model_path, image_index)