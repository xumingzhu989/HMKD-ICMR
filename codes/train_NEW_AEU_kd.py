import argparse
import time
import datetime
import os
import shutil
import sys
import numpy as np

cur_path = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(cur_path)[0]
sys.path.append(root_path)

import torch
import torch.nn as nn
import torch.utils.data as data
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn.functional as F

from losses import *
from models.model_zoo import get_segmentation_model
from models.cross_attention_map_model import get_cross_attention_map_model
from models.cross_attention_map_5_model import get_cross_attention_map_5_model
from models.Grandy_edge_map_model import get_Grandy_edge_map_model
from models.GLA_model import get_GLA_model
from models.HFA_model import get_heterogeneous_feature_align_model
from models.FAM_model import get_FAM_model
from models.AICSD_model import get_AICSD_model
from models.DKD_model import get_DKD_model
from models.DTKD_model import get_DTKD_model

from models.uncertainty_model import get_uncertainty_model
from models.logit_model import get_logit_model
from utils.sagan import Discriminator
from utils.distributed import *
from utils.logger import setup_logger
from utils.score import SegmentationMetric
from utils.flops import cal_multi_adds, cal_param_size

from dataset.cityscapes import CSTrainValSet
from dataset.ade20k import ADETrainSet, ADEDataValSet
from dataset.camvid import CamvidTrainSet, CamvidValSet
from dataset.voc import VOCDataTrainSet, VOCDataValSet
from dataset.coco_stuff_164k import CocoStuff164kTrainSet, CocoStuff164kValSet


def parse_args():
    parser = argparse.ArgumentParser(description='Semantic Segmentation Training With Pytorch')
    # model and dataset
    parser.add_argument('--teacher-model', type=str, default='segformer',
                        help='model name')
    parser.add_argument('--student-model', type=str, default='deeplabv3',
                        help='model name')
    parser.add_argument('--student-backbone', type=str, default='resnet101',
                        help='backbone name')
    parser.add_argument('--teacher-backbone', type=str, default='MiT_B4',
                        help='backbone name')
    parser.add_argument('--dataset', type=str, default='camvid',
                        help='dataset name')
    parser.add_argument('--data', type=str, default='./data/Camvid/', #./data/cityscapes/
                        help='dataset directory')
    # parser.add_argument('--crop-size', type=int, default=[512, 1024], nargs='+',
    #                     help='crop image size: [height, width]')
    parser.add_argument('--crop-size', type=int, default=[360, 480], nargs='+',
                        help='crop image size: [height, width]')
    parser.add_argument('--workers', '-j', type=int, default=8,
                        metavar='N', help='dataloader threads')
    parser.add_argument('--ignore-label', type=int, default=-1, metavar='N',
                        help='ignore label')

    # training hyper params
    parser.add_argument('--aux', action='store_true', default=False,
                        help='Auxiliary loss')
    parser.add_argument('--batch-size', type=int, default=4, metavar='N',
                        help='input batch size for training (default: 8)')
    #parser.add_argument('--batch-size', type=int, default=8, metavar='N',
    #                    help='input batch size for training (default: 8)')
    parser.add_argument('--start_epoch', type=int, default=0,
                        metavar='N', help='start epochs (default:0)')
    parser.add_argument('--max-iterations', type=int, default=80000, metavar='N',
                        help='number of epochs to train (default: 50)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 1e-4)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=1e-4, metavar='M',
                        help='w-decay (default: 5e-4)')

    parser.add_argument("--kd-temperature", type=float, default=1.0, help="logits KD temperature")
    parser.add_argument("--lambda-kd", type=float, default=0., help="lambda_kd")
    parser.add_argument("--lambda-ATT", type=float, default=[1.,1.,1.,1.,1.], help="lambda_ATT")
    parser.add_argument("--lambda-Uncertainty", type=float, default=1., help="lambda_Uncertainty")
    parser.add_argument("--lambda-Logit", type=float, default=1., help="lambda_Logit")
    parser.add_argument("--lambda-edge", type=float, default=[1.,1.,1.,1.], help="lambda_edge")
    parser.add_argument("--lambda-gla", type=float, default=[1.,1.], help="lambda_gla")
    parser.add_argument("--lambda-adv", type=float, default=0., help="lambda adversarial loss")
    parser.add_argument("--lambda-d", type=float, default=0., help="lambda discriminator loss")
    parser.add_argument("--lambda-skd", type=float, default=0., help="lambda skd")
    parser.add_argument("--lambda-cwd-fea", type=float, default=0., help="lambda cwd feature")
    parser.add_argument("--lambda-cwd-logit", type=float, default=0., help="lambda cwd logit")
    parser.add_argument("--lambda-ifv", type=float, default=0., help="lambda ifvd")
    parser.add_argument("--lambda-fitnet", type=float, default=0., help="lambda fitnet")
    parser.add_argument("--lambda-at", type=float, default=0., help="lambda attention transfer")
    parser.add_argument("--lambda-psd", type=float, default=0., help="lambda pixel similarity KD")
    parser.add_argument("--lambda-csd", type=float, default=0., help="lambda category similarity KD")

    # cuda setting
    parser.add_argument('--gpu-id', type=str, default='0')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--local_rank', type=int, default=0)
    # checkpoint and log
    parser.add_argument('--resume', type=str, default=None,
                        help='put the path to resuming file if needed')
    parser.add_argument('--save-dir', default='./saves/models',
                        help='Directory for saving checkpoint models')
    parser.add_argument('--save-epoch', type=int, default=10,
                        help='save model every checkpoint-epoch')
    parser.add_argument('--log-dir', default='./runs/logs/',
                        help='Directory for saving checkpoint models')
    parser.add_argument('--log-iter', type=int, default=10,
                        help='print log every log-iter')
    parser.add_argument('--save-per-iters', type=int, default=800,
                        help='per iters to save')
    parser.add_argument('--val-per-iters', type=int, default=800,
                        help='per iters to val')
    # parser.add_argument('--teacher-pretrained-base', type=str, default='None',help='pretrained backbone')
    parser.add_argument('--teacher-pretrained', type=str,
                        default='mit_b4.pth',
                        help='pretrained seg model')
    parser.add_argument('--student-pretrained-base', type=str, default='resnet101-imagenet.pth',
                        help='pretrained backbone')
    # CIRKD1/CIRKD1127/saves/models/G12_deeplabv3_resnet18_citys_OM(None).pth
    parser.add_argument('--student-pretrained', type=str, default='None',
                        help='pretrained seg model')

    # evaluation only
    parser.add_argument('--val-epoch', type=int, default=1,
                        help='run validation every val-epoch')
    parser.add_argument('--skip-val', action='store_true', default=False,
                        help='skip validation during training')
    args = parser.parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    if num_gpus > 1 and args.local_rank == 0:
        if not os.path.exists(args.log_dir):
            os.makedirs(args.log_dir)
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)

    if args.student_backbone.startswith('resnet'):
        args.aux = False  # 本来是true
    elif args.student_backbone.startswith('mobile'):
        args.aux = False
    else:
        raise ValueError('no such network')

    return args



class Trainer(object):
    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device)
        self.num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
        # self.feature_maps = [] # ###
        # self.some_frequency = 10 # ###

        if args.dataset == 'citys':
            train_dataset = CSTrainValSet(args.data,
                                          list_path='./dataset/list/cityscapes/train.lst',
                                          max_iters=args.max_iterations * args.batch_size,
                                          crop_size=args.crop_size, scale=True, mirror=True)
            val_dataset = CSTrainValSet(args.data,
                                        list_path='./dataset/list/cityscapes/val.lst',
                                        crop_size=(1024, 2048), scale=False, mirror=False)
        elif args.dataset == 'voc':
            train_dataset = VOCDataTrainSet(args.data, './dataset/list/voc/train_aug.txt',
                                            max_iters=args.max_iterations * args.batch_size,
                                            crop_size=args.crop_size, scale=True, mirror=True)
            val_dataset = VOCDataValSet(args.data, './dataset/list/voc/val.txt')
        elif args.dataset == 'ade20k':
            train_dataset = ADETrainSet(args.data, max_iters=args.max_iterations * args.batch_size,
                                        ignore_label=args.ignore_label,
                                        crop_size=args.crop_size, scale=True, mirror=True)
            val_dataset = ADEDataValSet(args.data)
        elif args.dataset == 'camvid':
            train_dataset = CamvidTrainSet(args.data, './dataset/list/CamVid/camvid_train_list.txt',
                                           max_iters=args.max_iterations * args.batch_size,
                                           ignore_label=args.ignore_label, crop_size=args.crop_size, scale=True,
                                           mirror=True)
            val_dataset = CamvidValSet(args.data, './dataset/list/CamVid/camvid_val_list.txt')
        elif args.dataset == 'coco_stuff_164k':
            train_dataset = CocoStuff164kTrainSet(args.data, './dataset/list/coco_stuff_164k/coco_stuff_164k_train.txt',
                                                  max_iters=args.max_iterations * args.batch_size,
                                                  ignore_label=args.ignore_label,
                                                  crop_size=args.crop_size, scale=True, mirror=True)
            val_dataset = CocoStuff164kValSet(args.data, './dataset/list/coco_stuff_164k/coco_stuff_164k_val.txt')
        else:
            raise ValueError('dataset unfind')

        args.batch_size = args.batch_size // num_gpus
        train_sampler = make_data_sampler(train_dataset, shuffle=True, distributed=args.distributed)
        train_batch_sampler = make_batch_data_sampler(train_sampler, args.batch_size, args.max_iterations)
        val_sampler = make_data_sampler(val_dataset, False, args.distributed)
        val_batch_sampler = make_batch_data_sampler(val_sampler, images_per_batch=1)

        self.train_loader = data.DataLoader(dataset=train_dataset,
                                            batch_sampler=train_batch_sampler,
                                            num_workers=args.workers,
                                            pin_memory=True)
        # #print(ground_truth.shape)
        # print('##')
        # for images, labels, name in self.train_loader:
        #     print(labels.shape)
        #     break
        # print('##')
        # print(args.crop_size)
        # print('##')
        self.val_loader = data.DataLoader(dataset=val_dataset,
                                          batch_sampler=val_batch_sampler,
                                          num_workers=args.workers,
                                          pin_memory=True)

        # create network
        BatchNorm2d = nn.SyncBatchNorm if args.distributed else nn.BatchNorm2d

        self.t_model = get_segmentation_model(model=args.teacher_model,
                                              backbone=args.teacher_backbone,
                                              pretrained=args.teacher_pretrained,
                                              num_class=train_dataset.num_class,
                                              img_size=args.crop_size,
                                              batchnorm_layer=nn.BatchNorm2d
                                              ).to(self.device)

        self.s_model = get_segmentation_model(model=args.student_model,
                                              backbone=args.student_backbone,
                                              local_rank=args.local_rank,
                                              pretrained_base=args.student_pretrained_base,
                                              pretrained=args.student_pretrained,
                                              aux=args.aux,
                                              norm_layer=BatchNorm2d,
                                              num_class=train_dataset.num_class).to(self.device)
        # self.s_model.pretrained.layer3.register_forward_hook(self.get_feature_map) # ###
        # [512,512]
        # self.fam_model = get_FAM_model(batchsize=args.batch_size,
        #                                             in_channels = 64,
        #                                             out_channels = 64,
        #                                             shapes = 128,
        #                                             kernel_size = 3,
        #                                             stride = 1,
        #                                             padding = 1, 
        #                                             groups = 4,
        #                                             bias = False).to(self.device)
        # self.fam1_model = get_FAM_model(batchsize=args.batch_size,
        #                                             in_channels = 256,
        #                                             out_channels = 256,
        #                                             shapes = 64,
        #                                             kernel_size = 3,
        #                                             stride = 1,
        #                                             padding = 1, 
        #                                             groups = 4,
        #                                             bias = False).to(self.device)
        # self.cam_model = get_cross_attention_map_5_model(batchsize=args.batch_size,
        #                                                              h_t = 16,
        #                                                              w_t = 16,
        #                                                              h_s = 64,
        #                                                              w_s = 64,
        #                                                              h_gt = 512,
        #                                                              w_gt = 512).to(self.device)
        # self.grandy_edge_model = get_Grandy_edge_map_model(batchsize=args.batch_size,
        #                                                    h_t_1=128,
        #                                                    w_t_1=128,
        #                                                    h_t_2=128,
        #                                                    w_t_2=128).to(self.device)
        # self.grandy_edge2_model = get_Grandy_edge_map_model(batchsize=args.batch_size,
        #                                                    h_t_1=64,
        #                                                    w_t_1=64,
        #                                                    h_t_2=64,
        #                                                    w_t_2=64).to(self.device)
        # self.cam_model = get_cross_attention_map_5_model(batchsize=args.batch_size,
        #                                                              h_t = 16,
        #                                                              w_t = 32,
        #                                                              h_s = 64,
        #                                                              w_s = 128,
        #                                                              h_gt = 512,
        #                                                              w_gt = 1024).to(self.device) # [512,1024]
        # self.gla_model = get_GLA_model(batchsize=args.batch_size).to(self.device)
        # self.aicsd_model = get_AICSD_model(batchsize=args.batch_size).to(self.device)
        # self.dkd_model = get_DKD_model(batchsize=args.batch_size,ALPHA=3.0,BETA=1.0,T=4,WARMUP=20).to(self.device)
        # self.dtkd_model = get_DTKD_model(batchsize=args.batch_size,ALPHA=3.0,BETA=1.0,T=4,WARMUP=20).to(self.device)
        # self.hfa_model = get_heterogeneous_feature_align_model(batchsize=args.batch_size).to(self.device)
        # self.logit_model = get_logit_model(opt=[0,1,2,3],num_classes=19,in_chans=[64, 128, 256, 512]).to(self.device)
        # self.hfa1_model = get_heterogeneous_feature_align_model(batchsize=args.batch_size).to(self.device)
        # self.uncertainty_model = get_uncertainty_model(batchsize=2, num_classes=19, top_k=5,
        #                                                compensation_power=1).to(self.device)



        for t_n, t_p in self.t_model.named_parameters():
            t_p.requires_grad = False
        self.t_model.eval()
        self.s_model.eval()
    
        
        # self.D_model = Discriminator(preprocess_GAN_mode=1, input_channel=train_dataset.num_class, distributed=args.distributed).cuda()

        # resume checkpoint if needed
        if args.resume:
            if os.path.isfile(args.resume):
                name, ext = os.path.splitext(args.resume)
                assert ext == '.pkl' or '.pth', 'Sorry only .pth and .pkl files supported.'
                print('Resuming training, loading {}...'.format(args.resume))
                self.s_model.load_state_dict(torch.load(args.resume, map_location=lambda storage, loc: storage))

        # create criterion
        # x = torch.randn(1, 3, 512, 512).cuda()
        # t_y = self.t_model(x)
        #
        # s_y = self.s_model(x)
        # t_channels = t_y[-5].size(1)  # 改#_c
        # s_channels = s_y[-1].size(1)

        self.criterion = SegCrossEntropyLoss(ignore_index=args.ignore_label).to(self.device)

        params_list = nn.ModuleList([])
        params_list.append(self.s_model)
        #[512,512]
        # params_list.append(self.aicsd_model)
        # params_list.append(self.dkd_model)
        # params_list.append(self.dtkd_model)
        # params_list.append(self.fam_model)
        # params_list.append(self.fam1_model)
        # params_list.append(self.cam_model)
        # params_list.append(self.grandy_edge_model)
        # params_list.append(self.grandy_edge2_model)
        # params_list.append(self.gla_model)
        # params_list.append(self.hfa_model)
        # params_list.append(self.hfa1_model)
        # params_list.append(self.uncertainty_model)
        # params_list.append(self.logit_model)
        # params_list.append(self.criterion_cwd)
        # params_list.append(self.criterion_fitnet)
        #[512,1024]
        # params_list.append(self.cam_model)
        

        self.optimizer = torch.optim.SGD(params_list.parameters(),
                                         lr=args.lr,
                                         momentum=args.momentum,
                                         weight_decay=args.weight_decay)

        # self.D_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad,
        #                                     self.D_model.parameters()),
        #                                     4e-4, [0.9, 0.99])
        if args.distributed:
            self.s_model = nn.parallel.DistributedDataParallel(self.s_model,
                                                                device_ids=[args.local_rank],
                                                                output_device=args.local_rank, find_unused_parameters=True)
            # self.fam_model = nn.parallel.DistributedDataParallel(self.fam_model,
            #                                                    device_ids=[args.local_rank],
            #                                                    output_device=args.local_rank, find_unused_parameters=True)
            # self.fam1_model = nn.parallel.DistributedDataParallel(self.fam1_model,
            #                                                    device_ids=[args.local_rank],
            #                                                    output_device=args.local_rank, find_unused_parameters=True)
            # self.cam_model = nn.parallel.DistributedDataParallel(self.cam_model,
            #                                                    device_ids=[args.local_rank],
            #                                                    output_device=args.local_rank, find_unused_parameters=True)
            # self.grandy_edge_model = nn.parallel.DistributedDataParallel(self.grandy_edge_model,
            #                                                    device_ids=[args.local_rank],
            #                                                    output_device=args.local_rank, find_unused_parameters=True)
            # self.grandy_edge2_model = nn.parallel.DistributedDataParallel(self.grandy_edge2_model,
            #                                                    device_ids=[args.local_rank],
            #                                                    output_device=args.local_rank, find_unused_parameters=True)
            # self.gla_model = nn.parallel.DistributedDataParallel(self.gla_model,
            #                                                              device_ids=[args.local_rank],
            #                                                              output_device=args.local_rank,
            #                                                              find_unused_parameters=True)
            # self.hfa_model = nn.parallel.DistributedDataParallel(self.hfa_model,
            #                                                      device_ids=[args.local_rank],
            #                                                      output_device=args.local_rank,
            #                                                      find_unused_parameters=True)
            # self.aicsd_model = nn.parallel.DistributedDataParallel(self.aicsd_model,
            #                                                      device_ids=[args.local_rank],
            #                                                      output_device=args.local_rank,
            #                                                      find_unused_parameters=True)
            # self.dkd_model = nn.parallel.DistributedDataParallel(self.dkd_model,
            #                                                      device_ids=[args.local_rank],
            #                                                      output_device=args.local_rank,
            #                                                      find_unused_parameters=True)
            # self.dtkd_model = nn.parallel.DistributedDataParallel(self.dtkd_model,
            #                                                      device_ids=[args.local_rank],
            #                                                      output_device=args.local_rank,
            #                                                      find_unused_parameters=True)
            # self.hfa1_model = nn.parallel.DistributedDataParallel(self.hfa_model,
            #                                                      device_ids=[args.local_rank],
            #                                                      output_device=args.local_rank,
            #                                                      find_unused_parameters=True)
            # self.uncertainty_model = nn.parallel.DistributedDataParallel(self.uncertainty_model,
            #                                                    device_ids=[args.local_rank],
            #                                                    output_device=args.local_rank, find_unused_parameters=True)
            # self.logit_model = nn.parallel.DistributedDataParallel(self.logit_model,
            #                                                    device_ids=[args.local_rank],
            #                                                    output_device=args.local_rank,
            #                                                    find_unused_parameters=True)
            # self.D_model = nn.parallel.DistributedDataParallel(self.D_model, device_ids=[args.local_rank],
            #                                                     output_device=args.local_rank)
            # self.criterion_cwd = nn.parallel.DistributedDataParallel(self.criterion_cwd,
            #                                                     device_ids=[args.local_rank],
            #                                                     output_device=args.local_rank)
            # self.criterion_fitnet = nn.parallel.DistributedDataParallel(self.criterion_fitnet,
            #                                                     device_ids=[args.local_rank],
            #                                                     output_device=args.local_rank)

        # evaluation metrics
        self.metric = SegmentationMetric(train_dataset.num_class)
        self.best_pred = 0.0
#     def get_feature_map(self, module, input, output):
#         self.feature_maps.append(output)
#     def visualize_feature_maps(self,iteration):
#         import matplotlib.pyplot as plt
#         import numpy as np
#         if self.feature_maps:
#             # 取最后一次的特征图
#             feature_map = self.feature_maps[-1].detach().cpu().numpy()
#             # 选择可视化的层（假设我们只选择第一个样本的特征图）
#             feature_map = feature_map[0]  # 选择第一个样本的特征图
#             num_feature_maps = feature_map.shape[0]
#             # 创建子图来显示特征图
#             cols = 8
#             rows = (num_feature_maps // cols) + (num_feature_maps % cols > 0)  # 计算行数
#             plt.figure(figsize=(20, 20))
#             for i in range(num_feature_maps):
#                 plt.subplot(rows, cols, i + 1)  # 假设你希望展示最多 64 个特征图
#                 plt.imshow(feature_map[i], cmap='viridis')
#                 plt.axis('off')

#             plt.savefig(f'feature_maps_iter_{iteration}.png')  # 保存特征图
#             plt.close()
    def adjust_lr(self, base_lr, iter, max_iter, power):
        cur_lr = base_lr * ((1 - float(iter) / max_iter) ** (power))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = cur_lr

        return cur_lr

    #
    def reduce_tensor(self, tensor):
        rt = tensor.clone()
        dist.all_reduce(rt, op=dist.ReduceOp.SUM)
        return rt

    # def reduce_mean_tensor(self, tensor):
    #     rt = tensor.clone()
    #     dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    #     rt /= self.num_gpus
    #     return rt
    
    def reduce_mean_tensor(self,tensor):
        if isinstance(tensor, torch.Tensor):
            rt = tensor.clone()
            dist.all_reduce(rt, op=dist.ReduceOp.SUM)
            rt /= dist.get_world_size()
            return rt
        else:  # 处理标量输入
            # return torch.tensor(tensor).to(tensor.device)  # 直接返回标量值
            return torch.tensor(tensor).to('cuda')
            # 或者将其转换为 0 维张量:
            # return torch.tensor(tensor).to(tensor.device) 


    def train(self):
        save_to_disk = get_rank() == 0
        log_per_iters, val_per_iters = self.args.log_iter, self.args.val_per_iters
        save_per_iters = self.args.save_per_iters
        start_time = time.time()
        logger.info('Start training, Total Iterations {:d}'.format(args.max_iterations))

        self.s_model.train()
        # self.fam_model.train()
        # self.fam1_model.train()
        # self.cam_model.train()
#         self.gla_model.train()
        # self.dkd_model.train()
        # self.dtkd_model.train()
        # self.aicsd_model.train()
#         self.hfa_model.train()
        # self.hfa1_model.train()

        # self.grandy_edge_model.train()
        # self.grandy_edge2_model.train()
        # self.uncertainty_model.train()
        # self.logit_model.train()

        for iteration, (images, targets, _) in enumerate(self.train_loader):
            iteration = iteration + 1

            images = images.to(self.device)
            targets = targets.long().to(self.device)  # B,512,1024
            # print('targets:',targets.shape) targets: torch.Size([4, 512, 512])
            # print(targets.shape)

            with torch.no_grad():
                t_outputs = self.t_model(images) # pred _c attmap_1 attmap_2 attmap_3 attmap_4 t_outputs[0]: torch.Size([4, 19, 128, 128])

            s_outputs = self.s_model(images)# pred cnn_c1 cnn_c2 cnn_c3 cnn_c4 aspp
            # print("t_outputs:",t_outputs[4][3].shape)
            # t_outputs[0] s_outputs[0] 分别为对应的logits输出
            # print("t_outputs:",t_outputs[0].shape)
            # print("s_outputs:",s_outputs[0].shape)
            #t_outputs: torch.Size([4, 19, 128, 128])
            #s_outputs: torch.Size([4, 19, 64, 64])
            # print('t_outputs[1]:',t_outputs[1].shape) #torch.Size([4, 768, 128, 128])
            # print('s_outputs[5]:',s_outputs[5].shape) #torch.Size([4, 128, 64, 64])
            # print('###:t_outputs[0][0],t_outputs[0][1],t_outputs[0][2],t_outputs[0][3]')
            # print(t_outputs)
            # print(type(t_outputs))
            # print(t_outputs[0][0].shape)
            # print(t_outputs[0][1].shape)
            # print(t_outputs[0][2].shape)
            # print(t_outputs[0][3].shape)
            # print('###')

            # torch.save(targets, 'targets.pt')
            # if iteration % self.some_frequency == 0:  # 选择适当的频率
            #     self.visualize_feature_maps(iteration)
            # print('t_outputs[0]:',t_outputs[0].shape)
            my_target = targets
            my_target = my_target.unsqueeze(1).to(dtype=torch.float)  # unsqueeze会在原张量上改变维度 my_target: torch.Size([4, 1, 512, 512])
            # print("my_target:",my_target.shape)
            # print('my_targets:')
            # print(my_target.shape)
            # f_t=t_outputs[1].mean(dim=1).unsqueeze(1)
            # f_s=s_outputs[-1].mean(dim=1).unsqueeze(1)
            # fam_loss = self.fam_model(s_outputs[1],t_outputs[2][3]) # 实用注意力机制！
            # fam1_loss = self.fam1_model(s_outputs[3],t_outputs[4][3])
            # cam_loss = self.cam_model(sum(t_outputs[-1])/len(t_outputs[-1]),s_outputs[-1],my_target)
            # cam_loss = self.cam_model(sum(t_outputs[-1])/len(t_outputs[-1]),s_outputs[4],my_target)
            # grandy_edge_losses = self.grandy_edge_model(sum(t_outputs[2])/len(t_outputs[2]),s_outputs[1])
            # grandy_edge2_losses = self.grandy_edge2_model(sum(t_outputs[3])/len(t_outputs[3]),s_outputs[2])
            
            # u_outputs = self.uncertainty_model(t_outputs[0], s_outputs[0], my_target)
            #print(t_outputs[3][0].shape,((sum(t_outputs[3])-t_outputs[3][0])/(len(t_outputs[3])-1)).shape,t_outputs[4][0].shape,((sum(t_outputs[4])-t_outputs[4][0])/(len(t_outputs[4])-1)).shape,t_outputs[5][0].shape,((sum(t_outputs[5])-t_outputs[5][0])/(len(t_outputs[5])-1)).shape)
            #gla_loss_g,gla_loss_l=self.gla_model([t_outputs[3][0],(sum(t_outputs[3])-t_outputs[3][0])/(len(t_outputs[3])-1),t_outputs[4][0],(sum(t_outputs[4])-t_outputs[4][0])/(len(t_outputs[4])-1),t_outputs[5][0],(sum(t_outputs[5])-t_outputs[5][0])/(len(t_outputs[5])-1)],[s_outputs[2],s_outputs[3],s_outputs[4]]) # 2,3,4
            #gla_loss_g,gla_loss_l=self.gla_model([t_outputs[3][0],t_outputs[3][-1],t_outputs[4][0],t_outputs[4][-1],t_outputs[5][0],t_outputs[5][-1]],[s_outputs[2],s_outputs[3],s_outputs[4]]) # 2,3,4
            #gla_loss_g,gla_loss_l=self.gla_model([t_outputs[2][0],t_outputs[2][-1],t_outputs[3][0],t_outputs[3][-1],t_outputs[4][0],t_outputs[4][-1],t_outputs[5][0],t_outputs[5][-1]],[s_outputs[1],s_outputs[2],s_outputs[3],s_outputs[4]]) # 1,2,3,4
            # gla_loss_g,gla_loss_l=self.gla_model([t_outputs[3][-2],t_outputs[4][-2],t_outputs[5][-2],t_outputs[2][-2]],[s_outputs[2],s_outputs[3],s_outputs[4],s_outputs[1]]) # 1，2，3，4
            # gla_loss_g,gla_loss_l=self.gla_model([t_outputs[3][-2],t_outputs[2][-2]],[s_outputs[2],s_outputs[1]]) # 1，2
            # print('###')
            # print(t_outputs[3][-2].shape)
            # print(t_outputs[2][-2].shape)
            # print('###')
            # gla_loss_g,gla_loss_l=self.gla_model([t_outputs[3][-2]],[s_outputs[2]]) # 2
            # gla_loss_g,gla_loss_l=self.gla_model([(sum(t_outputs[3])-t_outputs[3][0])/(len(t_outputs[3])-1)],[s_outputs[2]])
            # gla_loss=self.gla_model(t_outputs[2][-2],s_outputs[1]) # 1
            # hfa_loss=self.hfa_model((sum(t_outputs[4])-t_outputs[4][0])/(len(t_outputs[4])-1),s_outputs[3],(sum(t_outputs[5])-t_outputs[5][0])/(len(t_outputs[5])-1),s_outputs[4]) #
            # hfa_loss=self.hfa_model((sum(t_outputs[4])-t_outputs[4][0])/(len(t_outputs[4])-1),s_outputs[3])
            # hfa1_loss=self.hfa1_model((sum(t_outputs[5])-t_outputs[5][0])/(len(t_outputs[5])-1),s_outputs[4])
            # fam_loss = self.fam_model(s_outputs[1],t_outputs[2][3]) # 实用注意力机制！
            # hfa_loss=self.hfa_model((sum(t_outputs[4])-t_outputs[4][0])/(len(t_outputs[4])-1),s_outputs[3],(sum(t_outputs[5])-t_outputs[5][0])/(len(t_outputs[5])-1),s_outputs[4])
            # hfa_loss=self.hfa_model((sum(t_outputs[4])-t_outputs[4][0])/(len(t_outputs[4])-1),s_outputs[3],(sum(t_outputs[4])-t_outputs[4][0])/(len(t_outputs[4])-1),s_outputs[1])
            # hfa_loss=self.hfa_model((sum(t_outputs[4])-t_outputs[4][0])/(len(t_outputs[4])-1),s_outputs[3])
            # l_outputs=self.logit_model(t_outputs[0],[s_outputs[1],s_outputs[2],s_outputs[3],s_outputs[4]])
            # pi_loss, lo_loss = self.aicsd_model(s_outputs[5],t_outputs[1],t_outputs[0],s_outputs[0])
            # print(targets)
            # print(s_outputs[0])
            # dkd_loss = self.dkd_model(s_outputs[0],t_outputs[0],targets,iteration)
            # dtkd_loss = self.dtkd_model(s_outputs[0],t_outputs[0],iteration)
            task_loss = self.criterion(s_outputs[0], targets)
            
            # FAM_loss=torch.tensor(0.).cuda()
            # FAM1_loss=torch.tensor(0.).cuda()
            # ATT_loss = torch.tensor(0.).cuda()
            # Edge_loss_1 = torch.tensor(0.).cuda()
            # Edge_loss_2 = torch.tensor(0.).cuda()
            # Edge_loss = torch.tensor(0.).cuda()
            # Edge2_loss_1 = torch.tensor(0.).cuda()
            # Edge2_loss_2 = torch.tensor(0.).cuda()
            # Edge2_loss = torch.tensor(0.).cuda()
            # Uncertainty_loss = torch.tensor(0.).cuda()
#             GLA_loss=torch.tensor(0.).cuda()
            # AICSD_loss1 = torch.tensor(0.).cuda()
            # AICSD_loss2 = torch.tensor(0.).cuda()
#             HFA_loss=torch.tensor(0.).cuda()
            # Logit_loss=torch.tensor(0.).cuda()
            # HFA1_loss=torch.tensor(0.).cuda()
            # DKD_loss=torch.tensor(0.).cuda()
            # DTKD_loss = torch.tensor(0.).cuda()
            

            # adv_G_loss = self.args.lambda_adv*self.criterion_adv_for_G(self.D_model(s_outputs[0]))

            # adv_D_loss = self.args.lambda_d*(self.criterion_adv(self.D_model(s_outputs[0].detach()), self.D_model(t_outputs[0].detach())))
            # AICSD_loss1 = pi_loss
            # AICSD_loss2 = lo_loss*0.1
            # DKD_loss = dkd_loss
            # DTKD_loss = dtkd_loss
            # FAM_loss = fam_loss
            # FAM1_loss = fam1_loss
            # if self.args.lambda_gla[0] != 0.:
            #     GLA_loss = self.args.lambda_gla[0]*gla_loss
            # HFA_loss = hfa_loss
            # HFA_loss = hfa1_loss
            # if self.args.lambda_ATT[4] != 0.:
            #     ATT_loss=cam_loss
            #     #print('ATT_loss:{}'.format(ATT_loss))
            # if self.args.lambda_edge[0] != 0.:
            #     Edge_loss_1 = self.args.lambda_edge[0] * grandy_edge_losses[0]
            #     Edge_loss = Edge_loss_1
            # if self.args.lambda_edge[0] != 0.:
            #     Edge2_loss_1 = self.args.lambda_edge[0] * grandy_edge2_losses[0]
            #     Edge2_loss = Edge2_loss_1
            # if self.args.lambda_gla[0] != 0.:
            #     GLA_loss = self.args.lambda_gla[0]*gla_loss_g+self.args.lambda_gla[1]*gla_loss_l
            #     #print('Edge_loss_1:{}\tEdge_loss_2:{}\tEdge_loss:{}'.format(Edge_loss_1.item(), Edge_loss_2.item(),Edge_loss.item()))
            # if self.args.lambda_Uncertainty != 0.:
            #     Uncertainty_loss = self.args.lambda_Uncertainty*(u_outputs[0]+u_outputs[1]+u_outputs[2])
            #     #print('Uncertainty_loss:{}'.format(Uncertainty_loss.item()))
            # if self.args.lambda_Logit != 0.:
            #     Logit_loss = self.args.lambda_Logit * (sum(l_outputs)) * 0.01
            # ########### uncomment lines below for ALW ##################
            # alpha = iteration/800
            # # losses = alpha * (task_loss + lo_loss) + (1-alpha) * pi_loss
            # losses = alpha * (task_loss + AICSD_loss2) + (1-alpha) * AICSD_loss1
            # if self.args.aux:
            #     task_loss = self.criterion(s_outputs[0], targets) + 0.4 * self.criterion(s_outputs[1], targets)
            # else:
            #     task_loss = self.criterion(s_outputs[0], targets)
            # #取一些特征图
            losses = task_loss
            # losses = task_loss + HFA_loss
            # losses = task_loss + FAM_loss + FAM1_loss
            # losses = task_loss + GLA_loss + HFA_loss # + Logit_loss
            # losses = task_loss+GLA_loss
            # losses = task_loss + ATT_loss + Edge_loss + Edge2_loss + Uncertainty_loss
            # losses = task_loss + Uncertainty_loss
            # losses = task_loss + ATT_loss
            # losses = task_loss + Edge_loss + Edge2_loss
            # losses = task_loss + Edge_loss + ATT_loss
            # losses = task_loss + GLA_loss + FAM_loss
            # losses = task_loss + FAM_loss + HFA_loss
            # losses = task_loss + FAM_loss + ATT_loss
            # losses = task_loss + DKD_loss
            # losses = task_loss + DTKD_loss


            lr = self.adjust_lr(base_lr=args.lr, iter=iteration - 1, max_iter=args.max_iterations, power=0.9)
            self.optimizer.zero_grad()
            losses.backward()
            self.optimizer.step()
            # #查看cam_model是否在更新
            # for name, param in self.s_model.named_parameters():
            #     if param.requires_grad:
            #         print(name, param.data)

            task_loss_reduced = self.reduce_mean_tensor(task_loss)
            # AICSD_loss1_reduced = self.reduce_mean_tensor(AICSD_loss1)
            # AICSD_loss2_reduced = self.reduce_mean_tensor(AICSD_loss2)
            # dkd_loss_reduced = self.reduce_mean_tensor(DKD_loss)
            # dtkd_loss_reduced = self.reduce_mean_tensor(DTKD_loss)
            # fam_loss_reduced = self.reduce_mean_tensor(FAM_loss)
            # fam1_loss_reduced = self.reduce_mean_tensor(FAM1_loss)
            # cam_loss_reduced = self.reduce_mean_tensor(ATT_loss)
           
            # Edge_loss_reduced = self.reduce_mean_tensor(Edge_loss)
            # Edge2_loss_reduced = self.reduce_mean_tensor(Edge2_loss)
            # GLA_loss_reduced = self.reduce_mean_tensor(GLA_loss)
            # HFA_loss_reduced = self.reduce_mean_tensor(hfa_loss)
            # HFA1_loss_reduced = self.reduce_mean_tensor(hfa_loss)
            # Uncertainty_loss_reduced = self.reduce_mean_tensor(Uncertainty_loss)
            # Logit_loss_reduced = self.reduce_mean_tensor(Logit_loss)


            eta_seconds = ((time.time() - start_time) / iteration) * (args.max_iterations - iteration)
            eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

            if iteration % log_per_iters == 0 and save_to_disk:
                # logger.info(
                #     "Iters: {:d}/{:d} || Lr: {:.6f} || Task Loss: {:.4f} || Edge Loss: {:.4f} || ATT Loss: {:.4f} || Logit Loss: {:.4f} " \
                #     "|| Cost Time: {} || Estimated Time: {}".format(
                #         iteration, args.max_iterations, self.optimizer.param_groups[0]['lr'],
                #         task_loss_reduced.item(),
                #         Edge_loss_reduced.item(),
                #         ATT_loss_reduced.item(),
                #         #Logit_loss_reduced.item(),
                #         str(datetime.timedelta(seconds=int(time.time() - start_time))),
                #         eta_string))
                logger.info(
                    "Iters: {:d}/{:d} || Lr: {:.6f} || Task Loss(OM-resnet101-0.01): {:.4f} " \
                    "|| Cost Time: {} || Estimated Time: {}".format(
                        iteration, args.max_iterations, self.optimizer.param_groups[0]['lr'],
                        task_loss_reduced.item(),
                        str(datetime.timedelta(seconds=int(time.time() - start_time))),
                        eta_string))
                # logger.info(
                #     "Iters: {:d}/{:d} || Lr: {:.6f} || Task Loss(AICSD-1): {:.4f} || AICSD_loss1: {:.4f} || AICSD_loss2: {:.4f} "\
                #     "|| Cost Time: {} || Estimated Time: {}".format(
                #         iteration, args.max_iterations, self.optimizer.param_groups[0]['lr'],
                #         task_loss_reduced.item(),
                #         AICSD_loss1_reduced.item(),
                #         AICSD_loss2_reduced.item(),
                #         str(datetime.timedelta(seconds=int(time.time() - start_time))),
                #         eta_string))
                # cam_loss_reduced

            if iteration % save_per_iters == 0 and save_to_disk:
                save_checkpoint(self.s_model, self.args, is_best=False)

            if not self.args.skip_val and iteration % val_per_iters == 0:
                self.validation()
                self.s_model.train()

        save_checkpoint(self.s_model, self.args, is_best=False)
        total_training_time = time.time() - start_time
        total_training_str = str(datetime.timedelta(seconds=total_training_time))
        logger.info(
            "Total training time: {} ({:.4f}s / it)".format(
                total_training_str, total_training_time / args.max_iterations))

    def validation(self):
        is_best = False
        self.metric.reset()
        if self.args.distributed:
            model = self.s_model.module
        else:
            model = self.s_model
        torch.cuda.empty_cache()  # TODO check if it helps
        model.eval()
        logger.info("Start validation, Total sample: {:d}".format(len(self.val_loader)))
        for i, (image, target, filename) in enumerate(self.val_loader):
            image = image.to(self.device)
            target = target.to(self.device)

            with torch.no_grad():
                outputs = model(image)

            B, H, W = target.size()
            outputs[0] = F.interpolate(outputs[0], (H, W), mode='bilinear', align_corners=True)

            self.metric.update(outputs[0], target)
            pixAcc, mIoU = self.metric.get()
            logger.info("Sample: {:d}, Validation pixAcc: {:.3f}, mIoU: {:.3f}".format(i + 1, pixAcc, mIoU))

        if self.num_gpus > 1:
            sum_total_correct = torch.tensor(self.metric.total_correct).cuda().to(args.local_rank)
            sum_total_label = torch.tensor(self.metric.total_label).cuda().to(args.local_rank)
            sum_total_inter = torch.tensor(self.metric.total_inter).cuda().to(args.local_rank)
            sum_total_union = torch.tensor(self.metric.total_union).cuda().to(args.local_rank)
            sum_total_correct = self.reduce_tensor(sum_total_correct)
            sum_total_label = self.reduce_tensor(sum_total_label)
            sum_total_inter = self.reduce_tensor(sum_total_inter)
            sum_total_union = self.reduce_tensor(sum_total_union)

            pixAcc = 1.0 * sum_total_correct / (2.220446049250313e-16 + sum_total_label)
            IoU = 1.0 * sum_total_inter / (2.220446049250313e-16 + sum_total_union)
            mIoU = IoU.mean().item()

            logger.info("Overall validation pixAcc: {:.3f}, mIoU: {:.3f}".format(
                pixAcc.item() * 100, mIoU * 100))

        new_pred = mIoU
        if new_pred > self.best_pred:
            is_best = True
            self.best_pred = new_pred
        if (args.distributed is not True) or (args.distributed and args.local_rank == 0):
            save_checkpoint(self.s_model, self.args, is_best)
        synchronize()


def save_npy(array, name):
    """Save Checkpoint"""


    if (args.distributed is not True) or (args.distributed and args.local_rank == 0):
        directory = os.path.expanduser(args.save_dir)
        np.save(os.path.join(directory, name), array)


def save_checkpoint(model, args, is_best=False):
    """Save Checkpoint"""
    directory = os.path.expanduser(args.save_dir)
    if not os.path.exists(directory):
        os.makedirs(directory)
    timestr = time.strftime("%Y%m%d-%H%M%S")
    filename = 'G12_{}_{}_{}_OM((OM-resnet101-0.01)).pth'.format(args.student_model, args.student_backbone, args.dataset) ###
    filename = os.path.join(directory, filename)

    if args.distributed:
        model = model.module

    torch.save(model.state_dict(), filename)
    if is_best:
        best_filename = 'G12_{}_{}_{}_OM((OM-resnet101-0.01))_best_model.pth'.format(args.student_model, args.student_backbone,
                                                                        args.dataset)###
        best_filename = os.path.join(directory, best_filename)
        shutil.copyfile(filename, best_filename)


if __name__ == '__main__':
    args = parse_args()

    # reference maskrcnn-benchmark
    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.num_gpus = num_gpus
    args.distributed = num_gpus > 1
    if not args.no_cuda and torch.cuda.is_available():
        cudnn.benchmark = False
        args.device = "cuda"
    else:
        args.distributed = False
        args.device = "cpu"
    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        synchronize()
    timestr = time.strftime("%Y%m%d-%H%M%S")
    logger = setup_logger("semantic_segmentation", args.log_dir, get_rank(),
                              filename='{}_G12_{}_{}_{}_OM((OM-resnet101-0.01))_log.txt'.format(
                              timestr,args.student_model, args.teacher_backbone, args.student_backbone, args.dataset))###
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(args)

    trainer = Trainer(args)
    trainer.train()
    torch.cuda.empty_cache()
