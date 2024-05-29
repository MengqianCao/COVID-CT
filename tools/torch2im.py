import numpy as np
import torch


def tensor2im(input_image, imtype=np.uint8):
    """"
    Parameters:
        input_image (tensor) --  输入的tensor，维度为CHW，注意这里没有batch size的维度
        imtype (type)        --  转换后的numpy的数据类型
    """
    input_image = torch.squeeze(input_image, dim=0)
    mean = [0.485, 0.456, 0.406]  # dataLoader中设置的mean参数
    std = [0.229, 0.224, 0.225]  # dataLoader中设置的std参数

    # isinstance: 判断变量类型
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):
            image_tensor = input_image.data
        else:
            return input_image

        image_numpy = image_tensor.cpu().float().numpy()

        if image_numpy.shape[0] == 1:  # 灰度图转为RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))

        for i in range(len(mean)):  # 反标准化，乘以方差，加上均值
            image_numpy[i] = image_numpy[i] * std[i] + mean[i]

        image_numpy = image_numpy * 255  # 反ToTensor(),从[0,1]转为[0,255]
        
        # 从(channels, height, width)变为(height, width, channels)
        image_numpy = np.transpose(image_numpy, (1, 2, 0))

    else:  # 如果传入的是numpy数组,则不做处理
        image_numpy = input_image

    return image_numpy.astype(imtype)
