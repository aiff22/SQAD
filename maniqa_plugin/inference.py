import os
import torch
import numpy as np
import random

from torchvision import transforms
from torch.utils.data import DataLoader
from config import Config
from utils.inference_process import ToTensor, Normalize, five_point_crop, sort_file
from data.camera29 import CameraCropDataset
from scipy.stats import spearmanr, pearsonr
# from tqdm import tqdm
from tqdm.notebook import tqdm


os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def setup_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def eval_epoch(config, net, test_loader):
    with torch.no_grad():
        net.eval()
        gt_list = []
        pred_list = []
        with open(config.valid_path + '/output.txt', 'w') as f:
            for data in tqdm(test_loader):
                pred = 0
                for i in range(config.num_avg_val):
                    x_d = data['image'].cuda()
                    # x_d = five_point_crop(i, d_img=x_d, config=config)
                    labels = data['target']
                    labels = torch.squeeze(labels.type(torch.FloatTensor)).cuda()
                    pred += net(x_d)

                pred /= config.num_avg_val
                gt = labels.data.cpu().numpy()
                pred = pred.cpu().numpy()
                gt_list.extend(gt)
                pred_list.extend(pred)
            for i in range(len(gt_list)):
                f.write(str(gt_list[i]) + ',' + str(pred_list[i]) + '\n')
            print(len(gt_list))

            # compute correlation coefficient
            rho_s, _ = spearmanr(np.array(pred_list), np.array(gt_list))
            rho_p, _ = pearsonr(np.array(pred_list), np.array(gt_list))
            print(('test result (single crop): SRCC:{:.4} / PLCC:{:.4}'.format(rho_s, rho_p)))
        f.close()


if __name__ == '__main__':
    cpu_num = 1
    os.environ['OMP_NUM_THREADS'] = str(cpu_num)
    os.environ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
    os.environ['MKL_NUM_THREADS'] = str(cpu_num)
    os.environ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
    os.environ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
    torch.set_num_threads(cpu_num)

    setup_seed(20)

    # config file
    config = Config({
        # dataset path
        "gt_file_name": "/data/home/zilin/meta_image/camera_ready/camera_ground_truth.json",
        "test_img_path": "/data/home/zilin/meta_image/camera_ready/",                                     # path for the folder where "data" folder is
        "test_txt_file_name": "/data/home/zilin/meta_image/camera_ready/data/processed/sample_test.txt",   # path for label file
        "factor": "resolution",
        
        # optimization
        "batch_size": 8,
        "num_avg_val": 1,
        "crop_size": 224,

        # device
        "num_workers": 8,

        # load & save checkpoint
        "valid": "./output/valid",
        "valid_path": "./output/valid/inference_valid",
        "model_path": "./output/models/model_maniqa/sqad_resolution"
    })

    if not os.path.exists(config.valid):
        os.mkdir(config.valid)

    if not os.path.exists(config.valid_path):
        os.mkdir(config.valid_path)
    
    # data load
    test_dataset = CameraCropDataset(
        gt_file=config.gt_file_name,
        img_path=config.test_img_path,
        txt_file_name=config.test_txt_file_name,
        quality_factor=config.factor,
        transform=transforms.Compose([transforms.ToPILImage(),
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        shuffle=False
    )
    net = torch.load(config.model_path)
    net = net.cuda()

    losses, scores = [], []
    eval_epoch(config, net, test_loader)
    sort_file(config.valid_path + '/output.txt')
    