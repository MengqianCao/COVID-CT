import os
import torch
import numpy as np
import torch.nn.functional as F

device = 'cuda'

def train(optimizer, epoch, model, train_loader, modelname, criteria, batch_size):
    
    model.train()
    train_loss = 0
    train_correct = 0

    for batch_index, batch_samples in enumerate(train_loader):

        data, target = batch_samples['img'].to(
            device), batch_samples['label'].to(device)

        optimizer.zero_grad()

        output = model(data)
        loss = criteria(output, target.long())
        train_loss += criteria(output, target.long())  # 后面求平均误差用的

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pred = output.argmax(dim=1, keepdim=True)
        # 累加预测与标签吻合的次数，用于后面算准确率
        train_correct += pred.eq(target.long().view_as(pred)).sum().item()

        if batch_index % 4 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tTrain Loss: {:.6f}'.format(
                epoch, batch_index, len(train_loader),
                100.0 * batch_index / len(train_loader), loss.item() / batch_size))

    print('\nTrain set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        train_loss /
        len(train_loader.dataset), train_correct, len(train_loader.dataset),
        100.0 * train_correct / len(train_loader.dataset)))

    if os.path.exists('performance') == 0:
        os.makedirs('performance')
    f = open('performance/{}.txt'.format(modelname), 'a+')
    f.write('\nTrain set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        train_loss /
            len(train_loader.dataset), train_correct, len(train_loader.dataset),
            100.0 * train_correct / len(train_loader.dataset)))
    f.write('\n')
    f.close()

    return train_loss / len(train_loader.dataset)  # 返回一个epoch的平均误差，用于可视化损失


def val(model, val_loader, criteria):

    model.eval()
    val_loss = 0

    # Don't update model
    with torch.no_grad():

        predlist = []
        scorelist = []
        targetlist = []
        # Predict
        for batch_index, batch_samples in enumerate(val_loader):
            data, target = batch_samples['img'].to(
                device), batch_samples['label'].to(device)

            output = model(data)

            val_loss += criteria(output, target.long())
            score = F.softmax(output, dim=1)
            pred = output.argmax(dim=1, keepdim=True)

            targetcpu = target.long().cpu().numpy()
            predlist = np.append(predlist, pred.cpu().numpy())
            scorelist = np.append(scorelist, score.cpu().numpy()[:, 1])
            targetlist = np.append(targetlist, targetcpu)

    return targetlist, scorelist, predlist, val_loss / len(val_loader.dataset)


def test(model, test_loader):
    model.eval()

    # Don't update model
    with torch.no_grad():

        predlist = []
        scorelist = []
        targetlist = []
     
        for batch_index, batch_samples in enumerate(test_loader):
            data, target = batch_samples['img'].to(
                device), batch_samples['label'].to(device)

            output = model(data)

            score = F.softmax(output, dim=1)
            pred = output.argmax(dim=1, keepdim=True)

            targetcpu = target.long().cpu().numpy()
            predlist = np.append(predlist, pred.cpu().numpy())
            scorelist = np.append(scorelist, score.cpu().numpy()[:, 1])
            targetlist = np.append(targetlist, targetcpu)

    return targetlist, scorelist, predlist
