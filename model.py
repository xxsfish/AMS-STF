import copy

import torch
import torchvision
from torch import nn, Tensor
import torch.nn.functional as F
import numpy as np
from metrics import msssim
from torch.autograd import Variable
from typing import List


class SymmetricPadding2d(nn.Module):
    def __init__(self, padding):
        super(SymmetricPadding2d, self).__init__()
        self.padding = padding

    def forward(self, x):
        return F.pad(x, (self.padding, self.padding, self.padding, self.padding), mode='symmetric')


class SymmetricConv2d(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1, padding: int = 1,
                 bias: bool = True, dilation: int = 1, groups: int = 1):
        super(SymmetricConv2d, self).__init__(
            SymmetricPadding2d(padding),
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, 0,
                      bias=bias, dilation=dilation, groups=groups)
        )


class ReplicationConv2d(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1, padding: int = 1,
                 bias: bool = True, dilation: int = 1, groups: int = 1):
        super(ReplicationConv2d, self).__init__(
            nn.ReflectionPad2d(padding),
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, 0,
                      bias=bias, dilation=dilation, groups=groups)
        )


class Conv2dNormActivation(nn.Module):
    def __init__(
            self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1, padding: int = 1,
            bias: bool = True, dilation: int = 1, groups: int = 1, activation_layer: str = 'lrelu',
            norm_layer: str = '', padding_mode: str = 'replication', skip_connection: bool = False
    ) -> None:
        super().__init__()

        self.skip_connection = skip_connection

        if padding_mode == 'symmetric':
            self.conv = SymmetricConv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                                        padding=padding, bias=bias, dilation=dilation, groups=groups)
        else:
            self.conv = ReplicationConv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                                          padding=padding, bias=bias, dilation=dilation, groups=groups)

        self.active = None
        if activation_layer != '':
            if activation_layer == 'lrelu':
                self.active = nn.LeakyReLU(inplace=True)

        self.norm = None
        if norm_layer != '':
            if norm_layer == 'batch-norm':
                self.norm = nn.BatchNorm2d(out_channels)

    def forward(self, x: Tensor) -> Tensor:
        out = self.conv(x)

        if self.norm:
            out = self.norm(out)

        if self.skip_connection:
            out = out + x

        if self.active:
            out = self.active(out)

        return out


class SubExtractFeatureModule(nn.Module):

    def __init__(self, num_feat: int) -> None:
        super().__init__()

        self.conv1 = Conv2dNormActivation(num_feat, num_feat, kernel_size=3, padding=1,
                                          norm_layer='', activation_layer='', skip_connection=True)
        self.conv2 = Conv2dNormActivation(num_feat, num_feat, kernel_size=3, padding=1,
                                          norm_layer='', activation_layer='', skip_connection=True)

    def forward(self, x: Tensor) -> Tensor:
        x1 = self.conv1(x)
        out = self.conv2(x1)
        return out


class ExtractFeatureModule(nn.Module):
    def __init__(self, in_channels: int, channels: List[int]) -> None:
        super(ExtractFeatureModule, self).__init__()

        self.levels = len(channels)
        self.list = nn.ModuleList()

        num_feats = list(copy.deepcopy(channels))
        num_feats.insert(0, in_channels)

        for i in range(self.levels):
            stride = 1
            if i > 0:
                stride = 2

            self.list.append(
                nn.Sequential(
                    Conv2dNormActivation(num_feats[i], num_feats[i + 1], kernel_size=3, padding=1,
                                         stride=stride, norm_layer='batch-norm', activation_layer='lrelu'),
                    SubExtractFeatureModule(num_feats[i + 1])
                )
            )

    def forward(self, inputs):
        x = inputs
        results = []
        for i in range(self.levels):
            x = self.list[i](x)
            results.append(x)

        return results


class TAM(nn.Module):
    def __init__(self,
                 in_channels,
                 n_segment,
                 kernel_size=3,
                 stride=1,
                 padding=1):
        super(TAM, self).__init__()
        self.in_channels = in_channels
        self.n_segment = n_segment
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.G = nn.Sequential(
            nn.Linear(n_segment, n_segment * 2, bias=False),
            nn.BatchNorm1d(n_segment * 2), nn.ReLU(inplace=True),
            nn.Linear(n_segment * 2, kernel_size, bias=False), nn.Softmax(-1))

        self.L = nn.Sequential(
            nn.Conv1d(in_channels,
                      in_channels // 4,
                      kernel_size,
                      stride=1,
                      padding=kernel_size // 2,
                      bias=False), nn.BatchNorm1d(in_channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels // 4, in_channels, 1, bias=False),
            nn.Sigmoid())

    def forward(self, x):
        nt, c, h, w = x.size()
        t = self.n_segment
        n_batch = nt // t
        new_x = x.view(n_batch, t, c, h, w).permute(0, 2, 1, 3, 4).contiguous()
        out = F.adaptive_avg_pool2d(new_x.view(n_batch * c, t, h, w), (1, 1))
        out = out.view(-1, t)
        conv_kernel = self.G(out.view(-1, t)).view(n_batch * c, 1, -1, 1)
        local_activation = self.L(out.view(n_batch, c,
                                           t)).view(n_batch, c, t, 1, 1)
        new_x = new_x * local_activation
        out = F.conv2d(new_x.view(1, n_batch * c, t, h * w),
                       conv_kernel,
                       bias=None,
                       stride=(self.stride, 1),
                       padding=(self.padding, 0),
                       groups=n_batch * c)
        out = out.view(n_batch, c, t, h, w)
        out = out.permute(0, 2, 1, 3, 4).contiguous().view(nt, c, h, w)

        return out


class TemporalBottleneck(nn.Module):
    def __init__(self,
                 in_channels, out_channels,
                 n_segment=3,
                 t_kernel_size=3,
                 t_stride=1,
                 t_padding=1):
        super(TemporalBottleneck, self).__init__()
        self.n_segment = n_segment

        self.conv1 = Conv2dNormActivation(in_channels, out_channels, kernel_size=1, padding=0,
                                          norm_layer='batch-norm', activation_layer='lrelu')
        self.conv2 = Conv2dNormActivation(out_channels, out_channels, kernel_size=3, padding=1,
                                          norm_layer='batch-norm', activation_layer='lrelu')
        self.conv3 = Conv2dNormActivation(out_channels, out_channels, kernel_size=1, padding=0,
                                          norm_layer='batch-norm', activation_layer='')

        self.tam = TAM(in_channels=out_channels,
                       n_segment=n_segment,
                       kernel_size=t_kernel_size,
                       stride=t_stride,
                       padding=t_padding)

        self.lrelu = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.tam(out)

        out = self.conv2(out)
        out = self.conv3(out)

        out += identity
        out = self.lrelu(out)

        return out


class TANet(nn.Module):
    def __init__(self, num_feat: int) -> None:
        super(TANet, self).__init__()

        self.conv1 = TemporalBottleneck(num_feat, num_feat, n_segment=2)
        self.conv2 = TemporalBottleneck(num_feat, num_feat, n_segment=2)

    def forward(self, inputs):
        n, _, _, _ = inputs[0].size()
        x = torch.cat((inputs[0], inputs[1]), dim=0)

        x1 = self.conv1(x)
        x1 = self.conv2(x1)

        x = x1 + x
        return x[0:n, :, :, :], x[n:2*n, :, :, :]


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class CAM(nn.Module):
    """
    Channel Attention Module
    """

    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'], output_weight=False):
        super(CAM, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
        )
        self.pool_types = pool_types
        self.output_weight = output_weight

    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type == 'avg':
                avg_pool = F.avg_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(avg_pool)
            elif pool_type == 'max':
                max_pool = F.max_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(max_pool)
            elif pool_type == 'lp':
                lp_pool = F.lp_pool2d(x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(lp_pool)
            else:
                raise Exception("Unsupported pool type")

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = torch.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).expand_as(x)
        if self.output_weight:
            return scale
        else:
            return x * scale


class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)


class SAM(nn.Module):
    """
    Spatial Attention Module
    """

    def __init__(self, output_weight=False):
        super(SAM, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = Conv2dNormActivation(2, 1, kernel_size=kernel_size, padding=3,
                                            norm_layer='', activation_layer='')
        self.output_weight = output_weight

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid(x_out)  # broadcasting
        if self.output_weight:
            return scale
        else:
            return x * scale


class OffsetAttentionModule(nn.Sequential):
    def __init__(self, num_in_ch: int, kernel_size: int = 3):
        super(OffsetAttentionModule, self).__init__(
            torch.nn.Conv2d(in_channels=num_in_ch * 2, out_channels=num_in_ch, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(in_channels=num_in_ch, out_channels=int(num_in_ch / 2), kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(in_channels=int(num_in_ch / 2), out_channels=(2 * (kernel_size ** 2)), kernel_size=3,
                            stride=1, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(in_channels=(2 * (kernel_size ** 2)), out_channels=(2 * (kernel_size ** 2)), kernel_size=3,
                            stride=1, padding=1)
        )


class MaskEstimatorModule(nn.Module):
    def __init__(self, num_feats: int):
        super(MaskEstimatorModule, self).__init__()

        self.conv_first = Conv2dNormActivation(num_feats, num_feats, kernel_size=3, padding=1,
                                               norm_layer='batch-norm', activation_layer='lrelu')
        self.conv_combine = Conv2dNormActivation(2 * num_feats, num_feats, kernel_size=3, padding=1,
                                                 norm_layer='batch-norm', activation_layer='lrelu')

        kernel_size = 3
        self.conv = nn.Sequential(
            torch.nn.Conv2d(in_channels=num_feats, out_channels=int(num_feats / 2), kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(in_channels=int(num_feats / 2), out_channels=(2 * (kernel_size ** 2)), kernel_size=3,
                            stride=1, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(in_channels=(2 * (kernel_size ** 2)), out_channels=(kernel_size ** 2), kernel_size=3,
                            stride=1, padding=1),
            torch.nn.Sigmoid()
        )

    def forward(self, change_feats, base_feats):
        x = self.conv_first(change_feats)
        x = self.conv_combine(torch.cat([x, base_feats], dim=1))
        return self.conv(x)


class AmsStfLevel(nn.Module):
    def __init__(self, num_feats: int):
        super(AmsStfLevel, self).__init__()

        self.tanet = TANet(num_feats)

        self.cam = CAM(num_feats)
        self.sam = SAM()

        self.off_estimator = OffsetAttentionModule(num_feats, kernel_size=3)
        self.mask_estimator = MaskEstimatorModule(num_feats)

        self.regular_conv = torch.nn.Conv2d(in_channels=num_feats, out_channels=num_feats, kernel_size=3, stride=1,
                                            padding=1, bias=True)

        self.bn = nn.BatchNorm2d(num_feats)
        self.act = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, lr1_feats, lr2_feats, lr3_feats, hr1_feats, hr3_feats):
        ff_t1, ff_t3 = self.tanet([hr1_feats, hr3_feats])

        # Offset
        offset = self.off_estimator(torch.cat([hr1_feats, hr3_feats], dim=1))

        # Mask
        mask_t12 = self.mask_estimator(lr1_feats - lr2_feats, lr1_feats)
        mask_t32 = self.mask_estimator(lr3_feats - lr2_feats, lr3_feats)

        # Fusion
        out1 = torchvision.ops.deform_conv2d(input=ff_t1, offset=offset, mask=mask_t12,
                                                padding=(1, 1),
                                                weight=self.regular_conv.weight,
                                                bias=self.regular_conv.bias)

        out2 = torchvision.ops.deform_conv2d(input=ff_t3, offset=offset, mask=mask_t32,
                                                padding=(1, 1),
                                                weight=self.regular_conv.weight,
                                                bias=self.regular_conv.bias)

        # norm - activate
        out1 = self.bn(out1)
        out1 = self.act(out1)

        # norm - activate
        out2 = self.bn(out2)
        out2 = self.act(out2)

        return out1 + out2


class AmsStf(nn.Module):
    def __init__(self, num_in_ch: int, num_out_ch: int, dropout_p: float = 0.1):
        super(AmsStf, self).__init__()

        num_feats = [32, 64, 128]
        self.levels = len(num_feats)
        assert self.levels >= 1

        self.lr_encoder = ExtractFeatureModule(num_in_ch, num_feats)
        self.hr_encoder = ExtractFeatureModule(num_in_ch, num_feats)

        self.subnet_list = nn.ModuleList()
        for i in range(self.levels):
            self.subnet_list.append(AmsStfLevel(num_feats[i]))

        self.up_sample_conv_list = nn.ModuleList()
        for i in range(self.levels - 1):
            self.up_sample_conv_list.append(
                nn.Conv2d(num_feats[i + 1], num_feats[i], kernel_size=3, stride=1, padding=1))

        self.conv_out1 = nn.Conv2d(num_feats[0], 16, kernel_size=3, stride=1, padding=1)
        self.conv_out2 = nn.Conv2d(16, num_out_ch, kernel_size=3, stride=1, padding=1)

        self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, inputs):
        # Extract features
        lr1_feats = self.lr_encoder(inputs[0])
        hr1_feats = self.hr_encoder(inputs[1])

        lr3_feats = self.lr_encoder(inputs[2])
        hr3_feats = self.hr_encoder(inputs[3])

        lr2_feats = self.lr_encoder(inputs[4])

        f_levels_out = []
        for i in range(self.levels):
            f_levels_out.append(
                self.subnet_list[i](lr1_feats[i], lr2_feats[i], lr3_feats[i], hr1_feats[i], hr3_feats[i]))

        # FPN
        out = f_levels_out[self.levels - 1]
        for i in range(self.levels - 2, -1, -1):
            out = self.up_sample_conv_list[i](F.interpolate(out, scale_factor=2, mode='nearest'))
            out = out + f_levels_out[i]

        out = self.conv_out1(out)

        # conv-last dropout
        out = self.dropout(out)

        out = self.conv_out2(out)
        return out


def compute_gradient(inputs):
    kernel_v = [[-1, -2, -1],
                [0, 0, 0],
                [1, 2, 1]]
    kernel_h = [[-1, 0, 1],
                [-2, 0, 2],
                [-1, 0, 1]]
    kernel_h = torch.FloatTensor(kernel_h).unsqueeze(0).unsqueeze(0).to(inputs.device)
    kernel_v = torch.FloatTensor(kernel_v).unsqueeze(0).unsqueeze(0).to(inputs.device)
    gradients = []
    for i in range(inputs.shape[1]):
        data = inputs[:, i]
        data_v = F.conv2d(data.unsqueeze(1), kernel_v, padding=1)
        data_h = F.conv2d(data.unsqueeze(1), kernel_h, padding=1)
        data = torch.sqrt(torch.pow(data_v, 2) + torch.pow(data_h, 2) + 1e-6)
        gradients.append(data)

    result = torch.cat(gradients, dim=1)
    return result


class ReconstructionLoss(nn.Module):
    def __init__(self, weight_content_loss: float = 1.0, weight_pixel_loss: float = 1.0,
                 weight_spectral_loss: float = 1.0, weight_vision_loss: float = 1.0):
        super(ReconstructionLoss, self).__init__()

        self.weight_content_loss = weight_content_loss
        self.weight_pixel_loss = weight_pixel_loss
        self.weight_spectral_loss = weight_spectral_loss
        self.weight_vision_loss = weight_vision_loss

    @staticmethod
    def safe_arccos(inputs, epsilon=1e-6):
        return torch.acos(torch.clamp(inputs, -1 + epsilon, 1 - epsilon))

    def forward(self, prediction, target, output_details=False):
        sobel_loss = F.mse_loss(compute_gradient(prediction), compute_gradient(target))
        feature_loss = F.l1_loss(prediction, target)
        spectral_loss = torch.mean(self.safe_arccos(F.cosine_similarity(prediction, target, 1)))
        vision_loss = 1.0 - msssim(prediction, target, normalize=True)

        loss = self.weight_pixel_loss * feature_loss
        loss = loss + self.weight_spectral_loss * spectral_loss
        loss = loss + self.weight_vision_loss * vision_loss
        loss = loss + self.weight_content_loss * sobel_loss

        if output_details is False:
            return loss
        else:
            return loss, {
                "sobel_loss": sobel_loss,
                "feature_loss": feature_loss,
                "spectral_loss": spectral_loss,
                "vision_loss": vision_loss,
            }


class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.cuda.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label)
                self.real_label_var = Variable(real_tensor, requires_grad=False)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
                self.fake_label_var = Variable(fake_tensor, requires_grad=False)
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input, target_is_real):
        if isinstance(input[0], list):
            loss = 0
            for input_i in input:
                pred = input_i[-1]
                target_tensor = self.get_target_tensor(pred, target_is_real)
                loss += self.loss(pred, target_tensor)
            return loss
        else:
            target_tensor = self.get_target_tensor(input[-1], target_is_real)
            return self.loss(input[-1], target_tensor)


class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False, getIntermFeat=False):
        super(NLayerDiscriminator, self).__init__()
        self.getIntermFeat = getIntermFeat
        self.n_layers = n_layers

        kw = 4
        padw = int(np.ceil((kw - 1.0) / 2))
        sequence = [[nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]]

        nf = ndf
        for n in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            sequence += [[
                nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=2, padding=padw),
                norm_layer(nf), nn.LeakyReLU(0.2, True)
            ]]

        nf_prev = nf
        nf = min(nf * 2, 512)
        sequence += [[
            nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=1, padding=padw),
            norm_layer(nf),
            nn.LeakyReLU(0.2, True)
        ]]

        sequence += [[nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)]]

        if use_sigmoid:
            sequence += [[nn.Sigmoid()]]

        if getIntermFeat:
            for n in range(len(sequence)):
                setattr(self, 'model' + str(n), nn.Sequential(*sequence[n]))
        else:
            sequence_stream = []
            for n in range(len(sequence)):
                sequence_stream += sequence[n]
            self.model = nn.Sequential(*sequence_stream)

    def forward(self, input):
        if self.getIntermFeat:
            res = [input]
            for n in range(self.n_layers + 2):
                model = getattr(self, 'model' + str(n))
                res.append(model(res[-1]))
            return res[1:]
        else:
            return self.model(input)
