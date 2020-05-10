# 模型训练
import torch
import torch.nn as nn
from torch.autograd import Variable#pytorch自动求导机制的实现
import my_dataset#数据集的转载
from CNNmodel import CNN
import numpy as np
import captcha_setting#数据集的来源
import one_hot_encoding as one#对数据集进行one_hot操作
from shutil import copyfile#保存最好模型使用到

num_epochs = 312 #训练迭代次数
batch_size = 128 #训练批次大小
learning_rate = 0.001  #学习率

def predict():
    cnn = CNN()
    cnn.eval()
    cnn.load_state_dict(torch.load('model.pkl')) #加载模型
    print("....加载模型.......")
    test_dataloader = my_dataset.get_test_data_loader()#加载数据
    correct = 0
    total = 0
    for i, (images, labels) in enumerate(test_dataloader):
        image = Variable(images,requires_grad=False)#requires_grad是参不参与误差反向传播, 要不要计算梯度
        predict_label = cnn(image)
        c0 = captcha_setting.ALL_CHAR_SET[np.argmax(
            predict_label[0, 0:captcha_setting.ALL_CHAR_SET_LEN].data.numpy())]
        c1 = captcha_setting.ALL_CHAR_SET[np.argmax(
            predict_label[0, captcha_setting.ALL_CHAR_SET_LEN:2 * captcha_setting.ALL_CHAR_SET_LEN].data.numpy())]
        c2 = captcha_setting.ALL_CHAR_SET[np.argmax(
            predict_label[0, 2 * captcha_setting.ALL_CHAR_SET_LEN:3 * captcha_setting.ALL_CHAR_SET_LEN].data.numpy())]
        c3 = captcha_setting.ALL_CHAR_SET[np.argmax(
            predict_label[0, 3 * captcha_setting.ALL_CHAR_SET_LEN:4 * captcha_setting.ALL_CHAR_SET_LEN].data.numpy())]
        predict_label = '%s%s%s%s' % (c0, c1, c2, c3)
        true_label = one.decode(labels.numpy()[0])
        total += labels.size(0)
        if(predict_label == true_label):
            correct += 1
    newmax = 100*(correct/total)
    print("当前准确率：%f"% newmax)
    return newmax

def main():
    if torch.cuda.is_available(): #判断是否支持GPU加速，使用GPU来加载计算图形
        cnn = CNN().cuda()
    print('.......加载神经网络.........')
    criterion = nn.MultiLabelSoftMarginLoss()#使用多标签分类损失函数
    optimizer = torch.optim.Adam(#优化函数使用 Adam 自适应优化算法
        cnn.parameters(),
        lr=learning_rate)

    train_dataloader = my_dataset.get_train_data_loader()#加载训练数据

    passmax = 0 #初始化模型验证准确率
    for epoch in range(num_epochs): # 进行循环迭代训练
        # enumerate() 函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，
        # 同时列出数据和数据下标，一般用在 for 循环当中
        for i, (images, labels) in enumerate(train_dataloader):
            if torch.cuda.is_available():
                images = Variable(images,requires_grad=True).cuda()
                labels = Variable(labels.float()).cuda()
            optimizer.zero_grad()   # 将梯度归零
            predict_labels = cnn(images)    #预测图片中的数字
            loss = criterion(predict_labels, labels)    #从模型中得到预测值并确定损失
            loss.backward()     # 反向传播
            optimizer.step()    # 通过梯度做一步参数更新
            if (i+1) % 10 == 0:
                print("epoch:", epoch, "step:", i, "loss:", loss.item())
                strloss = []
                strloss.append(loss.item())
            if (i+1) % 100 == 0:
                #保存神经网络模型参数
                torch.save(cnn.state_dict(), "./model.pkl")
                print(" 保存模型  \n")
                # 加载保存模型进行验证
                newmax = predict()
                print("上次最好准确率为：%f"% passmax)
                if newmax > passmax:
                    passmax = newmax
                    # 将更好模型保存到good目录下
                    copyfile('./model.pkl', './good/model.pkl')
                    print("更好模型已经保存")
                else:
                    print("模型准确率没有上一次好，不做保存")#如果模型不好就不做保存，然后接着训练新的模型
                    break
    torch.save(cnn.state_dict(), "./model.pkl")   #current is model.pkl
    print("保存最后模型，结束训练。")
    print(strloss)

if __name__ == '__main__':
    main()