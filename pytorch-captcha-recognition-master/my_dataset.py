#加载图片将图片和标签然后进行转换
import os
from torch.utils.data import DataLoader,Dataset
import torchvision.transforms as transforms#图像预处理包
from PIL import Image
import one_hot_encoding as one
import captcha_setting

class mydataset(Dataset):
    def __init__(self, folder, transform=None):
        self.train_image_file_paths = [os.path.join(folder, image_file)
                                       for image_file in os.listdir(folder)]
        self.transform = transform
    def __len__(self):
        return len(self.train_image_file_paths)
    def __getitem__(self, idx):
        image_root = self.train_image_file_paths[idx]
        image_name = image_root.split(os.path.sep)[-1]
        image = Image.open(image_root)
        if self.transform is not None:
            image = self.transform(image)
        # 为了减少重复率，在生成图片的时候，图片文件的命名格式 "4个数字_时间戳.PNG",同时对该值做 encode处理
        #  使用Python split()方法通过指定分隔符对标签字符串进行切片,仅获取标签前面四个数值进行one.encode()处理。
        label = one.encode(image_name.split('_')[0])
        return image, label

#转灰度图：transforms.Grayscale
# 转为tensor，并归一化至[0-1]：transforms.ToTenso
transform = transforms.Compose(
    [transforms.Grayscale(),
     transforms.ToTensor(),])

# 装载数据集时通过batch_size的值来确认每次加载的数据量大小，
# 通过shuffle的值来确认在装载的过程中打乱图片的顺序。
def get_train_data_loader():
    dataset = mydataset(captcha_setting.TRAIN_DATASET_PATH,
                        transform=transform)
    return DataLoader(dataset, batch_size=32, shuffle=True)
def get_test_data_loader():
    dataset = mydataset(captcha_setting.TEST_DATASET_PATH,
                        transform=transform)
    return DataLoader(dataset, batch_size=1, shuffle=True)
