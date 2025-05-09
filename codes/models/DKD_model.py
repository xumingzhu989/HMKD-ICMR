import torch
import torch.nn as nn
import torch.nn.functional as F

#from ._base import Distiller

__all__ = ['get_DKD_model']


def dkd_loss(logits_student, logits_teacher, target, alpha=3.0, beta=1.0, temperature=4):
    gt_mask = _get_gt_mask(logits_student, target)
    other_mask = _get_other_mask(logits_student, target)
    pred_student = F.softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    pred_student = cat_mask(pred_student, gt_mask, other_mask)
    pred_teacher = cat_mask(pred_teacher, gt_mask, other_mask)
    log_pred_student = torch.log(pred_student)
    tckd_loss = (
        F.kl_div(log_pred_student, pred_teacher, size_average=False)
        * (temperature**2)
        / target.shape[0]
    )
    pred_teacher_part2 = F.softmax(
        logits_teacher / temperature - 1000.0 * gt_mask, dim=1
    )
    log_pred_student_part2 = F.log_softmax(
        logits_student / temperature - 1000.0 * gt_mask, dim=1
    )
    nckd_loss = (
        F.kl_div(log_pred_student_part2, pred_teacher_part2, size_average=False)
        * (temperature**2)
        / target.shape[0]
    )
    return alpha * tckd_loss + beta * nckd_loss

def _get_gt_mask(logits, target):
    batch_size, num_classes, height, width = logits.shape
    # 将target重塑为(batch_size, 1, height, width)，使其维度与mask相同
    target = target.reshape(batch_size, 1, height, width).long()  # 确保target是整数类型
    # 创建一个与logits形状相同的布尔型mask张量
    mask = torch.zeros_like(logits, dtype=torch.bool).cuda()
    
    # 使用scatter_操作在mask上根据target的索引设置True值
    mask.scatter_(1, target, 1)
    return mask
def _get_other_mask(logits, target):
    batch_size, num_classes, height, width = logits.shape
    # 将target重塑为(batch_size, 1, height, width)，使其维度与mask相同
    target = target.reshape(batch_size, 1, height, width).long()  # 确保target是整数类型
    # 创建一个与logits形状相同的布尔型mask张量
    mask = torch.ones_like(logits, dtype=torch.bool).cuda()
    
    # 使用scatter_操作在mask上根据target的索引设置True值
    mask.scatter_(1, target, 0)
    return mask





def cat_mask(t, mask1, mask2):
    t1 = (t * mask1).sum(dim=1, keepdims=True)
    t2 = (t * mask2).sum(1, keepdims=True)
    rt = torch.cat([t1, t2], dim=1)
    return rt


class DKD_model(nn.Module):
    """Decoupled Knowledge Distillation(CVPR 2022)"""

    def __init__(self, batchsize,ALPHA=3.0,BETA=1.0,T=4,WARMUP=20):
        super(DKD_model, self).__init__()
        self.batchsize = batchsize
        self.alpha = ALPHA
        self.beta = BETA
        self.temperature = T
        self.warmup = WARMUP
        self.fc = nn.Linear(19, 19)

    def forward(self,logits_s,logits_t,target, iteration):
        B,H,W = target.shape # torch.Size([4, 1, 512, 512])
        # target = target.reshape(-1) 
        logits_s = F.interpolate(input=logits_s, size=(H, W), mode='bilinear', align_corners=True)
        logits_t = F.interpolate(input=logits_t, size=(H, W), mode='bilinear', align_corners=True)
        target = target.long()
        
        target = torch.where(target == -1, torch.zeros_like(target), target)
        # print(logits_s.shape)
        # losses
        loss_dkd = min(iteration / self.warmup, 1.0) * dkd_loss(
            logits_s,
            logits_t,
            target,
            self.alpha,
            self.beta,
            self.temperature,
        )
        return loss_dkd*0.000001

def get_DKD_model(batchsize,ALPHA=3.0,BETA=1.0,T=4,WARMUP=20):
    model = DKD_model(batchsize,ALPHA,BETA,T,WARMUP)
    device = 'cuda'
    model.to(device)
    return model

if __name__ == '__main__':
    model = get_DKD_model(batchsize=4,ALPHA=3.0,BETA=1.0,T=4,WARMUP=20)
    # target = torch.randint(0, 19, (4, 512, 512), device='cuda')
    logit_t = torch.randn(4,19,128,128).cuda()
    logit_s = torch.randn(4,19,64,64).cuda()
    target = torch.randn(4,512,512).cuda()
    # target = (target * 18 / 2).round().long()
    iteration=10
    output = model(logit_s,logit_t,target,iteration)
    print("Number of Loss:",output)
