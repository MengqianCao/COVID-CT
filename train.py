import os
import torch
import warnings
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torchxrayvision as xrv
from torchvision import transforms
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from tools.conduct import val
from tools.conduct import train
from tools.dataload import CovidCTDataset


#  预处理，标准化与图像增强
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                 0.229, 0.224, 0.225])  # 依通道标准化

train_transformer = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomResizedCrop((224), scale=(0.5, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize
])

val_transformer = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize
])

if __name__ == '__main__':

    batchsize = 32
    total_epoch = 3000
    votenum = 10

    # 实例化Dataset
    trainset = CovidCTDataset(root_dir='data',
                              txt_Severe='data/trainCT_Severe.txt',
                              txt_Mild='data/trainCT_Mild.txt',
                              transform=train_transformer)
    valset = CovidCTDataset(root_dir='data',
                            txt_Severe='data/valCT_Severe.txt',
                            txt_Mild='data/valCT_Mild.txt',
                            transform=val_transformer)
    print(trainset.__len__())
    print(valset.__len__())

    # 构建DataLoader
    train_loader = DataLoader(
        trainset, batch_size=batchsize, drop_last=False, shuffle=True)
    val_loader = DataLoader(valset, batch_size=batchsize,
                            drop_last=False, shuffle=False)

    model = xrv.models.DenseNet(
        num_classes=2, in_channels=3).cuda()  # DenseNet 模型，二分类
    modelname = 'DenseNet_medical'
    torch.cuda.empty_cache()

    criteria = nn.CrossEntropyLoss()  # 二分类交叉熵损失

    optimizer = optim.Adam(model.parameters(), lr=0.0001)  # Adam优化器
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=10)  # 动态调整学习率策略，初始学习率0.0001


    warnings.filterwarnings('ignore')

    TP = 0
    TN = 0
    FN = 0
    FP = 0

    r_list = []
    p_list = []
    acc_list = []
    AUC_list = []

    vote_pred = np.zeros(valset.__len__())
    vote_score = np.zeros(valset.__len__())

    # 迭代3000*14次

    for epoch in range(1, total_epoch + 1):

        train_loss = train(optimizer, epoch, model, train_loader, modelname, criteria, batchsize)
        
        # 用验证集验证
        targetlist, scorelist, predlist, val_loss = val(model, val_loader, criteria)
        print('target', targetlist)
        print('score', scorelist)
        print('predict', predlist)
        
        vote_pred = vote_pred + predlist
        vote_score = vote_score + scorelist
        
        if epoch % votenum == 0:  # 每10个epoch，计算一次准确率和召回率等

            # major vote
            vote_pred[vote_pred <= (votenum / 2)] = 0
            vote_pred[vote_pred > (votenum / 2)] = 1
            vote_score = vote_score / votenum

            print('vote_pred', vote_pred)
            print('targetlist', targetlist)

            TP = ((vote_pred == 1) & (targetlist == 1)).sum()
            TN = ((vote_pred == 0) & (targetlist == 0)).sum()
            FN = ((vote_pred == 0) & (targetlist == 1)).sum()
            FP = ((vote_pred == 1) & (targetlist == 0)).sum()

            print('TP=', TP, 'TN=', TN, 'FN=', FN, 'FP=', FP)
            print('TP+FP', TP + FP)
            p = TP / (TP + FP)
            print('precision', p)
            p = TP / (TP + FP)
            r = TP / (TP + FN)
            print('recall', r)
            F1 = 2 * r * p / (r + p)
            acc = (TP + TN) / (TP + TN + FP + FN)
            print('F1', F1)
            print('acc', acc)
            AUC = roc_auc_score(targetlist, vote_score)
            print('AUCp', roc_auc_score(targetlist, vote_pred))
            print('AUC', AUC)


            print(
                '\n The epoch is {}, average recall: {:.4f}, average precision: {:.4f},average F1: {:.4f}, '
                'average accuracy: {:.4f}, average AUC: {:.4f}'.format(
                    epoch, r, p, F1, acc, AUC))

            # 更新模型
            if os.path.exists('model') == 0:
                os.makedirs('model')
            torch.save(model.state_dict(), 'model/temp.pt')

            vote_pred = np.zeros(valset.__len__())
            vote_score = np.zeros(valset.__len__())
            f = open('performance/{}.txt'.format(modelname), 'a+')
            f.write(
                '\n The epoch is {}, average recall: {:.4f}, average precision: {:.4f},average F1: {:.4f}, '
                'average accuracy: {:.4f}, average AUC: {:.4f}'.format(
                    epoch, r, p, F1, acc, AUC))
            f.close()

        if epoch % (votenum*10) == 0:  # 每100个epoch，保存一次模型
            torch.save(model.state_dict(),
                       'model/{}_epoch{}.pt'.format(modelname, epoch))

    os.remove('model/temp.pt')
