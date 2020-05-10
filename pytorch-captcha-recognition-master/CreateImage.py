# 验证码生成

from captcha.image import ImageCaptcha
from PIL import Image
import random
import time
import captcha_setting
import os

def random_captcha():
    captcha_text = []#定义一个列表用于存储随机获取到的数据
    for i in range(captcha_setting.MAX_CAPTCHA): #字符个数
        c = random.choice(captcha_setting.ALL_CHAR_SET)#随机获取一个数字
        captcha_text.append(c)#将获取的数字追加到列表中
    return ''.join(captcha_text)

# 生成字符对应的验证码
def creating_image():
    image = ImageCaptcha()
    captcha_text = random_captcha()
    captcha_image = Image.open(image.generate(captcha_text))
    return captcha_text, captcha_image

if __name__ == '__main__':
    count = 30000 #设置生成验证码的数量
    path = captcha_setting.TRAIN_DATASET_PATH # 通过改变此处目录，以生成 训练、测试和预测用的验证码集
    if not os.path.exists(path):
        os.makedirs(path)
    for i in range(count):
        now = str(int(time.time()))
        text, image = creating_image()
        filename = text+'_'+now+'.png'
        image.save(path  + os.path.sep +  filename)
        print('saved %d : %s' % (i+1,filename))

