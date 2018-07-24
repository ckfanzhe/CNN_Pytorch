import os


import torch
import torchvision
import torch.nn as nn
from model import RecognizeNet
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.transforms import transforms



# 数据准备
batch_size = 20
data_dir = 'D:/data/temp/'
model_path = 'D:/Pytorch/face_recognize/7_24'
if not os.path.exists(model_path):
    os.makedirs(model_path)


data_transforms = {
    'train': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(64),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}


image_datasets = {x: torchvision.datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}

dataloders = {x: DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4) for x in ['train', 'val']}


# 定义device，是否使用GPU，依据计算机配置自动会选择
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(device)

model = RecognizeNet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.00001)


def save_models(model_dir,epoch):
    torch.save(model.state_dict(), "{}model_{}.model".format(model_dir, epoch))
    print("Chekcpoint saved")



def train(epoch):
    model.train()
    batch_acc = 0
    for i in range(epoch+1):

        for data in dataloders['train']:
            inputs, labels = data
            input_s = inputs.to(device)
            labels = labels.to(device)
            # print(inputs.size())
            # print(labels.size())

            optimizer.zero_grad()
            #训练模型，输出结果
            out_put = model(input_s)
            # print(out_put)
            # print(labels)
            loss = criterion(out_put, labels)
            # print('loss:\n {}'.format(loss))
            _, prediction = torch.max(out_put.data, 1)
            # print('Predic:\n {} \nlabels:\n {}'.format(prediction, labels.data))

            batch_acc =batch_acc + torch.sum(prediction == labels.data)

            print('echo:{},batch_acc:{}'.format(i,batch_acc))
            # print('Epoch accuarcy: {} %'.format(batch_acc / 100))
            # print(batch_acc)
            #反向传播调整参数pytorch直接可以用loss
            loss.backward()
            #Adam刷新进步
            optimizer.step()
        if (i!=0)and(i % 10 == 0):
            print('Train Epoch: {} , loss:{}'.format(i*20, loss))
            print('Epoch accuarcy: {:.2f} %'.format(batch_acc/2))
            batch_acc = 0
            # print('Total accuarcy: {}'.format(total_acc))
            save_models(model_path, i)



if __name__ == '__main__':
    train(100)
