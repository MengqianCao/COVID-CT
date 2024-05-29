import torch
import warnings
import torchxrayvision as xrv
from torchvision import transforms
from torch.utils.data import DataLoader
from tools.dataload import CovidCTDataset
import matplotlib.pyplot as plt
from tools.torch2im import tensor2im

device = 'cuda'

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                 0.229, 0.224, 0.225])  # 依通道标准化

test_transformer = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize
])

if __name__ == '__main__':

    batchsize = 1

    testset = CovidCTDataset(root_dir='data',
                             txt_Severe='data/testCT_Severe.txt',
                             txt_Mild='data/testCT_Mild.txt',
                             transform=test_transformer)

    test_loader = DataLoader(
        testset, batch_size=batchsize, drop_last=False, shuffle=True)

    model = xrv.models.DenseNet(num_classes=2, in_channels=3).cuda()
    modelname = 'DenseNet_medical'
    model.load_state_dict(torch.load('model\DenseNet_medical_epoch1000.pt'))
    torch.cuda.empty_cache()

    warnings.filterwarnings('ignore')

    model.eval()

    with torch.no_grad():

        for batch_index, batch_samples in enumerate(test_loader):
            data, target = batch_samples['img'].to(
                device), batch_samples['label'].to(device)

            show_img = tensor2im(data)

            output = model(data)
            res = ''

            if output.argmax(dim=1) == 0:
                res += 'Predict: Severe, '

            else:
                res += 'Predict: Mild, '
            
            if batch_samples['label'].item() == 0:
                res += 'Label: Severe'

            else:
                res += 'Label: Mild'

            plt.figure("Predict")
            plt.imshow(show_img)
            plt.axis("off")
            plt.title(res)
            plt.show()
