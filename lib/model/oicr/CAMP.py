import torch
import torch.nn as nn
import torch.nn.functional as F

def sample_gumbel(shape, device, eps=1e-20):  # 这个函数制造随机值Gi
    U = torch.rand(shape).to(device)
    return -torch.log(-torch.log(U + eps) + eps)


def gumbel_softmax(logits, tau, hard=False, eps=1e-10, dim=-1):
    # scores = gumbel_softmax(_scores.add(1e-10).log(), dim=1, tau=self.tau, hard=self.is_hard, eps=1e-10)
    # type: #(Tensor, float, bool, float, int) -> Tensor
    device = logits.device
    gumbels = sample_gumbel(logits.shape, device, eps)
    gumbels = (logits + gumbels) / tau  # ~Gumbel(logits,tau) #到这里gumbel部分就计算完成了
    # y_soft = gumbels.softmax(dim)
    y_soft = torch.exp(
        F.log_softmax(gumbels, dim))  # (bsize, 2, height, width)，注意，这里dim=1，所以gumbel-sofmaxsh是通道级别的softmax而不是空间？

    if hard:
        # Straight through.
        index = y_soft.max(dim, keepdim=True)[1]
        y_hard = torch.zeros_like(logits).scatter_(dim, index, 1.0)
        ret = y_hard - y_soft.detach() + y_soft
    else:
        # Reparametrization trick.
        ret = y_soft

    return ret


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):  # 不改变size
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)



# class BasicBlock(nn.Module):
#     def __init__(self, planes, stride=1, dilation=1, norm_layer=None):
#         super(BasicBlock, self).__init__()
#         norm_layer = nn.BatchNorm2d
#         self.conv1 = conv3x3(planes, planes, stride)
#         self.bn1 = norm_layer(planes)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv2 = conv3x3(planes, 2)
#         self.bn2 = norm_layer(2)
#         self.downsample = conv1x1(planes, 2)
#
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#             elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)
#
#     def forward(self, x, drop_prob):
#         identity = x
#
#         out = self.conv1(x)  # (bsize, channels, height, width)
#         out = self.bn1(out)
#         out = self.relu(out)
#
#         out = self.conv2(out)  # (bsize, 2, height, width)
#         out = self.bn2(out)
#
#         identity = self.downsample(x)  # (bsize, 2, height, width)
#         out += identity  # (bsize, 2, height, width)
#
#         out_mask = torch.sigmoid(out[:, 0:1]) * drop_prob  # drop_prob=0.3/9，ROI面积/未删除面积？？？？？？
#         # out[:,0:1]=out[:.0:1,:,:]这是种省略的写法，两种写法是等价的
#         # 这里只取了两个通道中的一个，虽然的确生成了两个通道（Faster-RCNN中生成前背景概率的时候也有同样的操作，可能是为了增强可解释性？）
#         # 从下面被注释掉的源代码来看他原来写的也是1个通道
#         out_bg = 1 - out_mask
#         new_out = torch.cat((out_mask, out_bg), dim=1)  # 感觉这里的两个通道可以理解为前背景的概率
#
#         return new_out

class CAM_DROPBLOCK(nn.Module):
    def __init__(self,inchannel, outchannel=20,block=3):
        super(CAM_DROPBLOCK, self).__init__()

        self.tau=0.01
        self.block_size=block
        self.drop_prob=0.3

        self.outchannel=outchannel
        norm_layer = nn.BatchNorm2d
        self.conv6 = conv1x1(inchannel, inchannel)
        self.bn1 = norm_layer(inchannel)
        self.relu = nn.ReLU(inplace=True)
        self.conv7 = conv1x1(inchannel, outchannel)
        self.bn2 = norm_layer(outchannel)
        self.bn3 = norm_layer(2)

        self.convA=conv1x1(inchannel,2)

        self.linear=nn.Linear(in_features=outchannel, out_features=outchannel, bias=True)

    def forward(self, x, drop_prob):
        out = self.conv6(x)   # (bsize, channels, height, width)
        out = self.bn1(out)
        out = self.relu(out)

        out_mask=self.convA(out.detach())
        out_mask=self.bn3(out_mask)

        self.drop_prob = drop_prob
        gamma = self._compute_gamma(x.detach())
        out_mask = torch.sigmoid(out_mask[:, 0:1]) * gamma  # drop_prob=0.3/9，ROI面积/未删除面积？？？？？？
        # out[:,0:1]=out[:.0:1,:,:]这是种省略的写法，两种写法是等价的
        # 这里只取了两个通道中的一个，虽然的确生成了两个通道（Faster-RCNN中生成前背景概率的时候也有同样的操作，可能是为了增强可解释性？）
        # 从下面被注释掉的源代码来看他原来写的也是1个通道
        out_bg = 1 - out_mask
        new_out = torch.cat((out_mask, out_bg), dim=1)  # 感觉这里的两个通道可以理解为前背景的概率

        scores = gumbel_softmax(new_out.add(1e-10).log(), dim=1, tau=self.tau, hard=True, eps=1e-10)
        mask = scores[:, 0]  # 到这里终于只取了其中一个通道，终于是只有一个通道的mask了(bsize, 1, height, width)

        # compute block mask
        block_mask = self._compute_block_mask(mask)

        out = self.conv7(out)  # (bsize, 2, height, width)
        out = self.bn2(out)
        out = self.relu(out)

        out = torch.nn.functional.adaptive_avg_pool2d(out,(1,1))

        out=out.view((out.size(0),-1))

        out=self.linear(out)


        out= F.softmax(out, dim=1)

        return block_mask,out


        # self.downsample = conv1x1(planes, 2)
    def _compute_block_mask(self, mask):
        block_mask = F.max_pool2d(input=mask[:, None, :, :],#填None多一维，input[bsize, 1,1, height, width]
                                  kernel_size=(self.block_size, self.block_size),
                                  stride=(1, 1),
                                  padding=self.block_size // 2)

        if self.block_size % 2 == 0:
            block_mask = block_mask[:, :, :-1, :-1]

        block_mask = 1 - block_mask.squeeze(1)

        return block_mask

    def _compute_gamma(self, x):
        return self.drop_prob / (self.block_size ** 2) #0.3/9

class ConvConcreteDB(torch.nn.Module):

    def __init__(self, planes):
        super(ConvConcreteDB, self).__init__()
        # self.roi_size = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        # self.drop_prob = cfg.DB.TAU  # _C.DB.TAU = 0.3
        # self.block_size = cfg.DB.SIZE  # _C.DB.SIZE = 3
        # self.tau = cfg.DB.GSM_THRES  # _C.DB.GSM_THRES = 0.01
        # self.conv = BasicBlock(planes)
        # self.is_hard = cfg.DB.IS_HARD
        self.roi_size = 14  # cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        self.drop_prob = 0.3  # _C.DB.TAU = 0.3
        self.block_size = 3  # _C.DB.SIZE = 3
        self.tau = 0.01  # _C.DB.GSM_THRES = 0.01
        self.conv = CAM_DROPBLOCK(planes)
        self.is_hard = True

    def forward(self, x):
        # shape: (bsize, channels, height, width)
        assert x.dim() == 4, "Expected input with 4 dimensions (bsize, channels, height, width)"

        if not self.training or self.drop_prob == 0.:
            return x
        else:
            # get gamma value
            gamma = self._compute_gamma(x.detach())  # 这个gamma代表的是具体到每个特征点需要丢弃的概率，假如drop=0.3,
            # 每个中心点周围3*3的区域被丢掉，那么每个特征点被丢掉的概率应该为
            # 0.3/（3*3）
            _scores,_CAM_out = self.conv(x.detach(), gamma)  # (bsize, 2, height, width)

            # creat mask
            scores = gumbel_softmax(_scores.add(1e-10).log(), dim=1, tau=self.tau, hard=self.is_hard, eps=1e-10)
            mask = scores[:, 0]  # 到这里终于只取了其中一个通道，终于是只有一个通道的mask了(bsize, 1, height, width)

            # compute block mask
            block_mask = self._compute_block_mask(mask)

            # apply block mask
            out = x * block_mask[:, None, :, :]

            # scale output
            out = out * block_mask.numel() / block_mask.sum()

            return out

    def _compute_block_mask(self, mask):
        block_mask = F.max_pool2d(input=mask[:, None, :, :],  # 填None多一维，input[bsize, 1,1, height, width]
                                  kernel_size=(self.block_size, self.block_size),
                                  stride=(1, 1),
                                  padding=self.block_size // 2)

        if self.block_size % 2 == 0:
            block_mask = block_mask[:, :, :-1, :-1]

        block_mask = 1 - block_mask.squeeze(1)

        return block_mask

9





# import numpy as np
# if __name__ == '__main__':
#     input=torch.randn(2,3,4,4)
#
#     model=CAM_DROPBLOCK(inchannel=3,block=3)
#
#     mask,out=model(input,drop_prob=0.3)
#
#     label = torch.from_numpy(np.array([[0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]]))
#     label=label.float()
#     loss1 = multi_class_cross_entropy_loss(out, label, eps=1e-6)
#
#     total_loss = 0
#     # train the model using minibatch
#
#     lr = 0.001
#     # backward and optimize
#     optimizer = torch.optim.Adam(model.parameters(), lr=lr)
#     optimizer.zero_grad()
#     total_loss = loss1
#     loss1.backward()
#     optimizer.step()
#
#
#     print('end')
