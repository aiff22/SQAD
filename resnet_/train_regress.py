from CameraDataset import CameraCrop2Dataset, FullImageDataset
import numpy as np
from torch.utils.data import DataLoader
import torch
from torchvision import models, transforms
import torch.nn as nn
import argparse
import os


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
os.makedirs('result', exist_ok=True)

min_error = 10e5

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--factor', type=str, default='resolution', help="quality factor to train for, i.e. dr, check from GT json file")
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--train_epochs', type=int, default=200, help="number of epochs to train for")
    parser.add_argument('--save_eval_every', type=int, default=1, help="how often to evaluate and save model if satisfied")
    args = parser.parse_args()
    return args

def train(model, optimizer, criterion, train_loader, epoch):
    model.train()
    train_loss = 0
    for batch_idx, (x, target) in enumerate(train_loader):
        x, target = x.to(device), target.to(device)

        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out.reshape(-1), target)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()

        if (batch_idx + 1) % 80 == 0:
            print('==>>> train loss: {:.6f}'.format(loss.item()))
                
    print ('Train Epoch: {}\t Loss: {:.6f}'.format(epoch, train_loss/len(train_loader)))

def evaluation(model, criterion, test_loader, epoch):
    global min_error

    model.eval()
    test_loss = 0
    gt, pred = [], []
    for i, (x, target) in enumerate(test_loader):
        x, target = x.to(device), target.to(device)

        gt.append(target.cpu().numpy())
        with torch.no_grad():
            out = model(x)
            test_loss += criterion(out.reshape(-1), target).item()
            pred.append(out.reshape(-1).detach().cpu().numpy())

    test_loss /= len(test_loader)
        
    print("---- Evaluate model and save")
    print('Validation set: average loss = {:.4f}\n'.format(test_loss))
    
    pearson_coef = np.corrcoef(np.concatenate(pred,axis=0), np.concatenate(gt,axis=0))
    print('==>>> Pearson Coef: {:.4f}\n'.format(pearson_coef[0,1]))
    
    if test_loss < min_error:
        min_error = test_loss
        model_name = args.factor + '_epo'+ str(epoch) + '_err' + f'{min_error:.4f}' + '_coef' + f'{pearson_coef[0,1]:.4f}' + '.pkl'
        torch.save(model.state_dict(), os.path.join('result', model_name))



def main(args):

    trans = transforms.Compose([transforms.ToPILImage(),
                                transforms.RandomHorizontalFlip(),
                                transforms.RandomVerticalFlip(),
                                transforms.ToTensor(),
                                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])  # mean and std from ImageNet

    train_dataset = CameraCrop2Dataset(label_file='./data/processed/train.txt', data_dir='./', quality_factor=args.factor, transform=trans)
    eval_dataset = CameraCrop2Dataset(label_file='./data/processed/val.txt', data_dir='./', quality_factor=args.factor, transform=trans)

    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
    eval_loader = DataLoader(dataset=eval_dataset, batch_size=args.batch_size, shuffle=True)

    # Modify the last fc-layer to output single value for regression 
    sqad_RegNet = models.resnet50(pretrained=True)
    infc_num = sqad_RegNet.fc.in_features 
    sqad_RegNet.fc = nn.Linear(infc_num, 1)

    sqad_RegNet = sqad_RegNet.to(device)

    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam([{'params': sqad_RegNet.parameters()}], lr=0.001)


    for epoch in range(1, args.train_epochs + 1):
        train(sqad_RegNet, optimizer, criterion, train_loader, epoch)
        if epoch % args.save_eval_every == 0:
            evaluation(sqad_RegNet, criterion, eval_loader, epoch)


if __name__ == '__main__':
	args = parse_args()
	main(args)