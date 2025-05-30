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
    parser.add_argument('--student-backbone', type=str, default='resnet18',
                        help='backbone name')
    parser.add_argument('--teacher-backbone', type=str, default='MiT_B4',
                        help='backbone name')
    parser.add_argument('--dataset', type=str, default='citys',
                        help='dataset name')
    parser.add_argument('--data', type=str, default='./data/cityscapes/',
                        help='dataset directory')
    # parser.add_argument('--crop-size', type=int, default=[512, 1024], nargs='+',
    #                     help='crop image size: [height, width]')
    parser.add_argument('--crop-size', type=int, default=[512, 512], nargs='+',
                        help='crop image size: [height, width]')
    parser.add_argument('--workers', '-j', type=int, default=8,
                        metavar='N', help='dataloader threads')
    parser.add_argument('--ignore-label', type=int, default=-1, metavar='N',
                        help='ignore label')

    # training hyper params
    parser.add_argument('--aux', action='store_true', default=False,
                        help='Auxiliary loss')
    parser.add_argument('--batch-size', type=int, default=8, metavar='N',
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

    parser.add_argument('--pixel-memory-size', type=int, default=20000)
    parser.add_argument('--region-memory-size', type=int, default=2000)
    parser.add_argument('--region-contrast-size', type=int, default=1024)
    parser.add_argument('--pixel-contrast-size', type=int, default=4096)
    parser.add_argument("--kd-temperature", type=float, default=1.0, help="logits KD temperature")
    parser.add_argument("--contrast-kd-temperature", type=float, default=1.0, help="similarity distribution KD temperature")
    parser.add_argument("--contrast-temperature", type=float, default=0.1, help="similarity distribution temperature")
    
    parser.add_argument("--lambda-kd", type=float, default=1., help="lambda_kd")
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
    parser.add_argument("--lambda-memory-pixel", type=float, default=0.1, help="lambda memory-based pixel")
    parser.add_argument("--lambda-memory-region", type=float, default=0.1, help="lambda memory-based region")
    


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
                        default='segformer_MiT_B4_citys_best_model.pth',
                        help='pretrained seg model')
    parser.add_argument('--student-pretrained-base', type=str, default='resnet18-imagenet.pth',
                        help='pretrained backbone')
    # CIRKD1/CIRKD1127/saves/models/G12_deeplabv3_resnet18_citys_OM(None).pth
    parser.add_argument('--student-pretrained', type=str, default='G12_deeplabv3_resnet18_citys_OM(None)_best_model.pth',
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
        x = torch.randn(1, 3, 512, 512).cuda()
        t_y = self.t_model(x)
        s_y = self.s_model(x)
        t_channels = s_y[-1].size(1)
        s_channels = s_y[-1].size(1)

        self.criterion = SegCrossEntropyLoss(ignore_index=args.ignore_label).to(self.device)
        self.criterion_kd = CriterionKD(temperature=args.kd_temperature).to(self.device)
        self.criterion_memory_contrast = StudentSegContrast(num_classes=train_dataset.num_class,
                                                     pixel_memory_size=args.pixel_memory_size,
                                                     region_memory_size=args.region_memory_size,
                                                     region_contrast_size=args.region_contrast_size//train_dataset.num_class+1,
                                                     pixel_contrast_size=args.pixel_contrast_size//train_dataset.num_class+1,
                                                     contrast_kd_temperature=args.contrast_kd_temperature,
                                                     contrast_temperature=args.contrast_temperature,
                                                     s_channels=s_channels,
                                                     t_channels=t_channels, 
                                                     ignore_label=args.ignore_label).to(self.device)
        params_list = nn.ModuleList([])
        params_list.append(self.s_model)
        #[512,512]
        # params_list.append(self.aicsd_model)
        # params_list.append(self.dkd_model)
        # params_list.append(self.dtkd_model)
        params_list.append(self.criterion_memory_contrast)
        

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
            
            self.criterion_memory_contrast = nn.parallel.DistributedDataParallel(self.criterion_memory_contrast, 
                                                             device_ids=[args.local_rank],
                                                             output_device=args.local_rank)
        # evaluation metrics
        self.metric = SegmentationMetric(train_dataset.num_class)
        self.best_pred = 0.0

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


        for iteration, (images, targets, _) in enumerate(self.train_loader):
            iteration = iteration + 1

            images = images.to(self.device)
            targets = targets.long().to(self.device)  # B,512,1024

            with torch.no_grad():
                t_outputs = self.t_model(images) # pred _c attmap_1 attmap_2 attmap_3 attmap_4 t_outputs[0]: torch.Size([4, 19, 128, 128])

            s_outputs = self.s_model(images)# pred cnn_c1 cnn_c2 cnn_c3 cnn_c4 aspp

            my_target = targets
            my_target = my_target.unsqueeze(1).to(dtype=torch.float)  # unsqueeze会在原张量上改变维度 my_target: torch.Size([4, 1, 512, 512])
            
            # s_outputs[0]: torch.Size([4, 19, 64, 64])
            # t_outputs[0]: torch.Size([4, 19, 128, 128])
            
            B, N, H, W = t_outputs[0].size()
            s_outputs1 = F.interpolate(input=s_outputs[0], size=(H, W), mode='bilinear', align_corners=True)
            
            kd_loss = self.args.lambda_kd * self.criterion_kd(s_outputs1, t_outputs[0])
            _, predict = torch.max(s_outputs[0], dim=1) 
            # B, N, H, W = s_outputs[0].size()
            s_outputs2 = F.interpolate(input=s_outputs[-1], size=(H, W), mode='bilinear', align_corners=True)
            
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Get the device
            transform_layer = nn.Conv2d(768, 128, kernel_size=1, stride=1, padding=0)
            transform_layer.to(device) # Move the transform_layer to the GPU
            t_outputs2 = transform_layer(t_outputs[1].to(device)) 

            
            # print("s_outputs[-1]:",s_outputs[-1].shape) s_outputs[-1]: torch.Size([4, 128, 64, 64])
            # print("t_outputs[1]:",t_outputs[1].shape) t_outputs[1]: torch.Size([4, 768, 128, 128])
            # print("t_outputs2:",t_outputs2.shape)
            # print("s_outputs2:",s_outputs2.shape)
            memory_pixel_contrast_loss, memory_region_contrast_loss = self.criterion_memory_contrast(s_outputs2, t_outputs2.detach(), targets, predict)
            task_loss = self.criterion(s_outputs[0], targets)
            

            # DKD_loss=torch.tensor(0.).cuda()
            # DTKD_loss = torch.tensor(0.).cuda()
            


            memory_pixel_contrast_loss = self.args.lambda_memory_pixel * memory_pixel_contrast_loss
            memory_region_contrast_loss = self.args.lambda_memory_region * memory_region_contrast_loss

            losses = task_loss + kd_loss + memory_pixel_contrast_loss + memory_region_contrast_loss



            lr = self.adjust_lr(base_lr=args.lr, iter=iteration - 1, max_iter=args.max_iterations, power=0.9)
            self.optimizer.zero_grad()
            losses.backward()
            self.optimizer.step()


            task_loss_reduced = self.reduce_mean_tensor(task_loss)

            memory_pixel_contrast_loss_reduced = self.reduce_mean_tensor(memory_pixel_contrast_loss)
            memory_region_contrast_loss_reduced = self.reduce_mean_tensor(memory_region_contrast_loss)


            eta_seconds = ((time.time() - start_time) / iteration) * (args.max_iterations - iteration)
            eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

            if iteration % log_per_iters == 0 and save_to_disk:
                # logger.info(
                #     "Iters: {:d}/{:d} || Lr: {:.6f} || Task Loss: {:.4f} || GLA Loss: {:.4f} " \
                #     "|| Cost Time: {} || Estimated Time: {}".format(
                #         iteration, args.max_iterations, self.optimizer.param_groups[0]['lr'],
                #         task_loss_reduced.item(),
                #         GLA_loss_reduced.item(),
                #         str(datetime.timedelta(seconds=int(time.time() - start_time))),
                #         eta_string))
                logger.info(
                    "Iters: {:d}/{:d} || Lr: {:.6f} || Task Loss(CIRKD): {:.4f} || memory_pixel_contrast_loss: {:.4f} "\
                    "|| memory_region_contrast_loss: {:.4f} || Cost Time: {} || Estimated Time: {}".format(
                        iteration, args.max_iterations, self.optimizer.param_groups[0]['lr'],
                        task_loss_reduced.item(),
                        memory_pixel_contrast_loss_reduced.item(),
                        memory_region_contrast_loss_reduced.item(),
                        # dkd_loss_reduced.item(),
                        # dtkd_loss_reduced.item(),
                        # Logit_loss_reduced.item(),
                        str(datetime.timedelta(seconds=int(time.time() - start_time))),
                        eta_string))
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
    filename = 'G12_{}_{}_{}_OM(CIRKD).pth'.format(args.student_model, args.student_backbone, args.dataset) ###
    filename = os.path.join(directory, filename)

    if args.distributed:
        model = model.module

    torch.save(model.state_dict(), filename)
    if is_best:
        best_filename = 'G12_{}_{}_{}_OM(CIRKD)_best_model.pth'.format(args.student_model, args.student_backbone,
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
                              filename='{}_G12_{}_{}_{}_OM(CIRKD)_log.txt'.format(
                              timestr,args.student_model, args.teacher_backbone, args.student_backbone, args.dataset))###
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(args)

    trainer = Trainer(args)
    trainer.train()
    torch.cuda.empty_cache()
