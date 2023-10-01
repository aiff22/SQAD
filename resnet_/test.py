from CameraDataset import CameraCropDataset, FullImageDataset
import numpy as np
from torch.utils.data import DataLoader
import torch
from torchvision import models, transforms
import torch.nn as nn
import argparse
import os
from scipy.stats import spearmanr

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--factor', type=str, default='resolution', help="quality factor to train for, i.e. dr, check from GT json file")
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--crop_num', type=int, default=16, help="crop number for multi-crop evaluation")
    parser.add_argument('--checkpoint', type=str, help="file path of the checkpoint for model testing")
    parser.add_argument('--test_label_file', type=str, default='../data/processed/sample_test.txt', help="file path of the label")
    args = parser.parse_args()
    return args


def evaluation(args, model, criterion, test_loader):
    model.eval()
    
    test_loss = 0
    gt, pred = [], []
    for i, (x, target) in enumerate(test_loader):
        x, target = x.to(device), target.to(device)
        gt.append(target.cpu().numpy())
        
        with torch.no_grad():
            out = model(x)
            pred.append(out.reshape(-1).cpu().numpy())
            test_loss += criterion(out.reshape(-1), target).item()

    test_loss /= len(test_loader)
        
    print("---- Evaluate model ----")
    print('Test set: average loss = {:.4f}\n'.format(test_loss))
    
    c1 = np.concatenate(gt,axis=0)
    c2 = np.concatenate(pred,axis=0)
    comp_res = np.concatenate((c1.reshape(-1,1), c2.reshape(-1,1)), axis=1)
    
    print('---- (Single Crop) ----\n==>>> Pearson Coef:')
    print(np.corrcoef(c1, c2))
    print('\n==>>> Spearman Coef:')
    print(spearmanr(c1, c2))
    
    print('\n---- (Multiple Crops [{:d}]) ----'.format(args.crop_num))
    eve_patch = comp_res.reshape((-1, args.crop_num, 2))
    c1_ens, c2_ens = [], []
    ensem_loss = 0
    for i in range(eve_patch.shape[0]):
        gt_avg = np.median(eve_patch[i,:,0])    #eve_patch[i,:,0].mean()
        pred_avg = np.median(eve_patch[i,:,1])  #eve_patch[i,:,1].mean()

        c1_ens.append(gt_avg)
        c2_ens.append(pred_avg)
        ensem_loss += abs(gt_avg - pred_avg)
    
    print('==>>> Pearson Coef:')
    print(np.corrcoef(c1_ens, c2_ens))
    print('\n==>>> Spearman Coef:')
    print(spearmanr(c1_ens, c2_ens))


def main(args):

    trans = transforms.Compose([transforms.ToPILImage(),
                                transforms.ToTensor(),
                                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])  # mean and std from ImageNet

    
    eval_dataset = CameraCropDataset(label_file=args.test_label_file, data_dir='../', quality_factor=args.factor, transform=trans)

    eval_loader = DataLoader(dataset=eval_dataset, batch_size=args.batch_size)

    # Modify the last fc-layer to output single value for regression 
    sqad_RegNet = models.resnet50()
    infc_num = sqad_RegNet.fc.in_features 
    sqad_RegNet.fc = nn.Linear(infc_num, 1)
    
    # load checkpoint
    sqad_RegNet.load_state_dict(torch.load(args.checkpoint))
    sqad_RegNet = sqad_RegNet.to(device)

    criterion = nn.L1Loss()
    evaluation(args, sqad_RegNet, criterion, eval_loader)


if __name__ == '__main__':
	args = parse_args()
	main(args)
