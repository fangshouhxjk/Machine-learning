#对模型进行测试
import numpy as np
import torch
from torch.autograd import Variable
import captcha_setting
import my_dataset
from CNNmodel import CNN
import one_hot_encoding as one

def main():
    cnn = CNN()
    cnn.eval()
    cnn.load_state_dict(torch.load('./good/model.pkl'))
    print("=====加载网络模型=====")
    test_dataloader = my_dataset.get_test_data_loader()
    correct = 0
    total = 0
    for i, (images, labels) in enumerate(test_dataloader):
        image = images
        vimage = Variable(image)
        predict_label = cnn(vimage)
        c0 = captcha_setting.ALL_CHAR_SET[np.argmax(predict_label[0, 0:captcha_setting.ALL_CHAR_SET_LEN].data.numpy())]
        c1 = captcha_setting.ALL_CHAR_SET[np.argmax(predict_label[0, captcha_setting.ALL_CHAR_SET_LEN:2 * captcha_setting.ALL_CHAR_SET_LEN].data.numpy())]
        c2 = captcha_setting.ALL_CHAR_SET[np.argmax(predict_label[0, 2 * captcha_setting.ALL_CHAR_SET_LEN:3 * captcha_setting.ALL_CHAR_SET_LEN].data.numpy())]
        c3 = captcha_setting.ALL_CHAR_SET[np.argmax(predict_label[0, 3 * captcha_setting.ALL_CHAR_SET_LEN:4 * captcha_setting.ALL_CHAR_SET_LEN].data.numpy())]
        predict_label = '%s%s%s%s' % (c0, c1, c2, c3)
        true_label = one.decode(labels.numpy()[0])
        total += labels.size(0)
        print('===加载测试图片===：',total)
        if(predict_label == true_label):
            correct += 1
    x = (correct/total)*100
    print('===测试准确率===: %f %%'% x)

if __name__ == '__main__':
    main()


