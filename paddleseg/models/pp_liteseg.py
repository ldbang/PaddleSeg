# copyright (c) 2022 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from paddleseg import utils
from paddleseg.models import layers
from paddleseg.cvlibs import manager
from paddleseg.utils import utils

@manager.MODELS.add_component
class PPLiteSeg(nn.Layer):
    """
    The PP_LiteSeg implementation based on PaddlePaddle.

    The original article refers to "Juncai Peng, Yi Liu, Shiyu Tang, Yuying Hao, Lutao Chu,
    Guowei Chen, Zewu Wu, Zeyu Chen, Zhiliang Yu, Yuning Du, Qingqing Dang,Baohua Lai,
    Qiwen Liu, Xiaoguang Hu, Dianhai Yu, Yanjun Ma. PP-LiteSeg: A Superior Real-Time Semantic
    Segmentation Model. https://arxiv.org/abs/2204.02681".

    Args:
        num_classes (int): The number of target classes.
        backbone(nn.Layer): Backbone network, such as stdc1net and resnet18. The backbone must
            has feat_channels, of which the length is 5.
        backbone_indices (List(int), optional): The values indicate the indices of output of backbone.
            Default: [2, 3, 4].
        arm_type (str, optional): The type of attention refinement module. Default: ARM_Add_SpAttenAdd3.
        cm_bin_sizes (List(int), optional): The bin size of context module. Default: [1,2,4].
        cm_out_ch (int, optional): The output channel of the last context module. Default: 128.
        arm_out_chs (List(int), optional): The out channels of each arm module. Default: [64, 96, 128].
        seg_head_inter_chs (List(int), optional): The intermediate channels of segmentation head.
            Default: [64, 64, 64].
        resize_mode (str, optional): The resize mode for the upsampling operation in decoder.
            Default: bilinear.
        pretrained (str, optional): The path or url of pretrained model. Default: None.

    """

    def __init__(self,
                 num_classes,
                 backbone,
                 backbone_indices=[2, 3, 4],
                 arm_type='UAFM_SpAtten',
                 cm_bin_sizes=[1, 2, 4,8],
                 cm_out_ch=128,
                 arm_out_chs=[64, 96, 128],
                 seg_head_inter_chs=[64, 64, 64],
                 resize_mode='bilinear',
                 pretrained=None):
        super().__init__()

        # backbone
        assert hasattr(backbone, 'feat_channels'), \
            "The backbone should has feat_channels."
        assert len(backbone.feat_channels) >= len(backbone_indices), \
            f"The length of input backbone_indices ({len(backbone_indices)}) should not be" \
            f"greater than the length of feat_channels ({len(backbone.feat_channels)})."
        assert len(backbone.feat_channels) > max(backbone_indices), \
            f"The max value ({max(backbone_indices)}) of backbone_indices should be " \
            f"less than the length of feat_channels ({len(backbone.feat_channels)})."
        self.backbone = backbone

        assert len(backbone_indices) > 1, "The lenght of backbone_indices " \
            "should be greater than 1"
        self.backbone_indices = backbone_indices  # [..., x16_id, x32_id]
        backbone_out_chs = [backbone.feat_channels[i] for i in backbone_indices]

        # head
        if len(arm_out_chs) == 1:
            arm_out_chs = arm_out_chs * len(backbone_indices)
        assert len(arm_out_chs) == len(backbone_indices), "The length of " \
            "arm_out_chs and backbone_indices should be equal"

        self.ppseg_head = PPLiteSegHead(backbone_out_chs, arm_out_chs,
                                        cm_bin_sizes, cm_out_ch, arm_type,
                                        resize_mode)

        if len(seg_head_inter_chs) == 1:
            seg_head_inter_chs = seg_head_inter_chs * len(backbone_indices)
        assert len(seg_head_inter_chs) == len(backbone_indices), "The length of " \
            "seg_head_inter_chs and backbone_indices should be equal"
        self.seg_heads = nn.LayerList()  # [..., head_16, head32]
        for in_ch, mid_ch in zip(arm_out_chs, seg_head_inter_chs):
            self.seg_heads.append(SegHead(in_ch, mid_ch, num_classes))

        # pretrained
        self.pretrained = pretrained
        self.init_weight()

    def forward(self, x):
        x_hw = paddle.shape(x)[2:]

        feats_backbone = self.backbone(x)  # [x2, x4, x8, x16, x32]
        assert len(feats_backbone) >= len(self.backbone_indices), \
            f"The nums of backbone feats ({len(feats_backbone)}) should be greater or " \
            f"equal than the nums of backbone_indices ({len(self.backbone_indices)})"

        feats_selected = [feats_backbone[i] for i in self.backbone_indices]

        feats_head = self.ppseg_head(feats_selected)  # [..., x8, x16, x32]

        if self.training:
            logit_list = []

            for x, seg_head in zip(feats_head, self.seg_heads):
                x = seg_head(x)
                logit_list.append(x)

            logit_list = [
                F.interpolate(
                    x, x_hw, mode='bilinear', align_corners=False)
                for x in logit_list
            ]
        else:
            x = self.seg_heads[0](feats_head[0])
            x = F.interpolate(x, x_hw, mode='bilinear', align_corners=False)
            logit_list = [x]

        return logit_list

    def init_weight(self):
        if self.pretrained is not None:
            utils.load_entire_model(self, self.pretrained)


class PPLiteSegHead(nn.Layer):
    """
    The head of PPLiteSeg.

    Args:
        backbone_out_chs (List(Tensor)): The channels of output tensors in the backbone.
        arm_out_chs (List(int)): The out channels of each arm module.
        cm_bin_sizes (List(int)): The bin size of context module.
        cm_out_ch (int): The output channel of the last context module.
        arm_type (str): The type of attention refinement module.
        resize_mode (str): The resize mode for the upsampling operation in decoder.
    """

    def __init__(self, backbone_out_chs, arm_out_chs, cm_bin_sizes, cm_out_ch,
                 arm_type, resize_mode):
        super().__init__()
#PPModule   ASPPModule
# [1, 2, 4,8]
#         self.cm = PPContextModule(backbone_out_chs[-1], cm_out_ch, cm_out_ch,
#                                   cm_bin_sizes)
    
#             self.cm = PPContextModule(backbone_out_chs[-1], cm_out_ch, cm_out_ch,
#                                   [1, 2, 4,8])
#             self.cm = PPModule(backbone_out_chs[-1], cm_out_ch, cm_out_ch,
#                                   [1, 2, 3,6])
            self.cm = ASPPModule(backbone_out_chs[-1], cm_out_ch, cm_out_ch,
                                  [1, 2, 3,6])
        assert hasattr(layers,arm_type), \
            "Not support arm_type ({})".format(arm_type)
        arm_class = eval("layers." + arm_type)

        self.arm_list = nn.LayerList()  # [..., arm8, arm16, arm32]
        for i in range(len(backbone_out_chs)):
            low_chs = backbone_out_chs[i]
            high_ch = cm_out_ch if i == len(
                backbone_out_chs) - 1 else arm_out_chs[i + 1]
            out_ch = arm_out_chs[i]
            arm = arm_class(
                low_chs, high_ch, out_ch, ksize=3, resize_mode=resize_mode)
            self.arm_list.append(arm)

    def forward(self, in_feat_list):
        """
        Args:
            in_feat_list (List(Tensor)): Such as [x2, x4, x8, x16, x32].
                x2, x4 and x8 are optional.
        Returns:
            out_feat_list (List(Tensor)): Such as [x2, x4, x8, x16, x32].
                x2, x4 and x8 are optional.
                The length of in_feat_list and out_feat_list are the same.
        """

        high_feat = self.cm(in_feat_list[-1])
        out_feat_list = []

        for i in reversed(range(len(in_feat_list))):
            low_feat = in_feat_list[i]
            arm = self.arm_list[i]
            high_feat = arm(low_feat, high_feat)
            out_feat_list.insert(0, high_feat)

        return out_feat_list


    
# class GhostModule(nn.Layer):
#     def __init__(self,
#                  in_channels,
#                  output_channels,
#                  kernel_size=1,
#                  ratio=2,
#                  dw_size=3,
#                  stride=1,
#                  relu=True,
#                  name=None):
#         super(GhostModule, self).__init__()
#         init_channels = int(math.ceil(output_channels / ratio))
#         new_channels = int(init_channels * (ratio - 1))
#         self.primary_conv = ConvBNLayer(
#             in_channels=in_channels,
#             out_channels=init_channels,
#             kernel_size=kernel_size,
#             stride=stride,
#             groups=1,
#             act="relu" if relu else None,
#             name=name + "_primary_conv")
#         self.cheap_operation = ConvBNLayer(
#             in_channels=init_channels,
#             out_channels=new_channels,
#             kernel_size=dw_size,
#             stride=1,
#             groups=init_channels,
#             act="relu" if relu else None,
#             name=name + "_cheap_operation")

#     def forward(self, inputs):
#         x = self.primary_conv(inputs)
#         y = self.cheap_operation(x)
#         out = paddle.concat([x, y], axis=1)
#         return out
    
    
    
    
    
    
    
    
    
    
class PPContextModule(nn.Layer):
    """
    Simple Context module.

    Args:
        in_channels (int): The number of input channels to pyramid pooling module.
        inter_channels (int): The number of inter channels to pyramid pooling module.
        out_channels (int): The number of output channels after pyramid pooling module.
        bin_sizes (tuple, optional): The out size of pooled feature maps. Default: (1, 3).
        align_corners (bool): An argument of F.interpolate. It should be set to False
            when the output size of feature is even, e.g. 1024x512, otherwise it is True, e.g. 769x769.
    """

    def __init__(self,
                 in_channels,
                 inter_channels,
                 out_channels,
                 bin_sizes,
                 align_corners=False):
        super().__init__()

        self.stages = nn.LayerList([
            self._make_stage(in_channels, inter_channels, size)
            for size in bin_sizes
        ])

        self.conv_out = layers.ConvBNReLU(
            in_channels=inter_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1)
#         self.conv_ghost = GhostModule()

        self.align_corners = align_corners

    def _make_stage(self, in_channels, out_channels, size):
        prior = nn.AdaptiveAvgPool2D(output_size=size)
        conv = layers.ConvBNReLU(
            in_channels=in_channels, out_channels=out_channels, kernel_size=1)
        return nn.Sequential(prior, conv)

    def forward(self, input):
        out = None
        input_shape = paddle.shape(input)[2:]

        for stage in self.stages:
            x = stage(input)
            x = F.interpolate(
                x,
                input_shape,
                mode='bilinear',
                align_corners=self.align_corners)
            if out is None:
                out = x
            else:
                out += x
#                 out = x+2
        out = self.conv_out(out)
        return out


class SegHead(nn.Layer):
    def __init__(self, in_chan, mid_chan, n_classes):
        super().__init__()
        self.conv = layers.ConvBNReLU(
            in_chan,
            mid_chan,
            kernel_size=3,
            stride=1,
            padding=1,
            bias_attr=False)
        self.conv_out = nn.Conv2D(
            mid_chan, n_classes, kernel_size=1, bias_attr=False)

    def forward(self, x):
        x = self.conv(x)
        x = self.conv_out(x)
        return x
    
    
    
    
    
class PPModule(nn.Layer):
    """
    Pyramid pooling module originally in PSPNet.
    Args:
        in_channels (int): The number of intput channels to pyramid pooling module.
        out_channels (int): The number of output channels after pyramid pooling module.
        bin_sizes (tuple, optional): The out size of pooled feature maps. Default: (1, 2, 3, 6).
        dim_reduction (bool, optional): A bool value represents if reducing dimension after pooling. Default: True.
        align_corners (bool): An argument of F.interpolate. It should be set to False when the output size of feature
            is even, e.g. 1024x512, otherwise it is True, e.g. 769x769.
    """
#     def __init__(self,
#                  in_channels,
#                  inter_channels,
#                  out_channels,
#                  bin_sizes,
#                  align_corners=False):
#         super().__init__()
    def __init__(self, in_channels, out_channels, bin_sizes, dim_reduction,
                 align_corners):
        super().__init__()

        self.bin_sizes = bin_sizes

        inter_channels = in_channels
        if dim_reduction:
            inter_channels = in_channels // len(bin_sizes)

        # we use dimension reduction after pooling mentioned in original implementation.
        self.stages = nn.LayerList([
            self._make_stage(in_channels, inter_channels, size)
            for size in bin_sizes
        ])

        self.conv_bn_relu2 = layers.ConvBNReLU(
            in_channels=in_channels + inter_channels * len(bin_sizes),
            out_channels=out_channels,
            kernel_size=3,
            padding=1)

        self.align_corners = align_corners

    def _make_stage(self, in_channels, out_channels, size):
        """
        Create one pooling layer.
        In our implementation, we adopt the same dimension reduction as the original paper that might be
        slightly different with other implementations.
        After pooling, the channels are reduced to 1/len(bin_sizes) immediately, while some other implementations
        keep the channels to be same.
        Args:
            in_channels (int): The number of intput channels to pyramid pooling module.
            size (int): The out size of the pooled layer.
        Returns:
            conv (Tensor): A tensor after Pyramid Pooling Module.
        """

        prior = nn.AdaptiveAvgPool2D(output_size=(size, size))
        conv = layers.ConvBNReLU(
            in_channels=in_channels, out_channels=out_channels, kernel_size=1)

        return nn.Sequential(prior, conv)

    def forward(self, input):
        cat_layers = []
        for stage in self.stages:
            x = stage(input)
            x = F.interpolate(
                x,
                paddle.shape(input)[2:],
                mode='bilinear',
                align_corners=self.align_corners)
            cat_layers.append(x)
        cat_layers = [input] + cat_layers[::-1]
        cat = paddle.concat(cat_layers, axis=1)
        out = self.conv_bn_relu2(cat)

        return out 
    
    
    
class ASPPModule(nn.Layer):
    """
    Atrous Spatial Pyramid Pooling.
    Args:
        aspp_ratios (tuple): The dilation rate using in ASSP module.
        in_channels (int): The number of input channels.
        out_channels (int): The number of output channels.
        align_corners (bool): An argument of F.interpolate. It should be set to False when the output size of feature
            is even, e.g. 1024x512, otherwise it is True, e.g. 769x769.
        use_sep_conv (bool, optional): If using separable conv in ASPP module. Default: False.
        image_pooling (bool, optional): If augmented with image-level features. Default: False
    """

    def __init__(self,
                 aspp_ratios,
                 in_channels,
                 out_channels,
                 align_corners,
                 use_sep_conv=False,
                 image_pooling=False,
                 data_format='NCHW'):
        super().__init__()

        self.align_corners = align_corners
        self.data_format = data_format
        self.aspp_blocks = nn.LayerList()

        for ratio in aspp_ratios:
            if use_sep_conv and ratio > 1:
                conv_func = layers.SeparableConvBNReLU
            else:
                conv_func = layers.ConvBNReLU

            block = conv_func(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1 if ratio == 1 else 3,
                dilation=ratio,
                padding=0 if ratio == 1 else ratio,
                data_format=data_format)
            self.aspp_blocks.append(block)

        out_size = len(self.aspp_blocks)

        if image_pooling:
            self.global_avg_pool = nn.Sequential(
                nn.AdaptiveAvgPool2D(
                    output_size=(1, 1), data_format=data_format),
                layers.ConvBNReLU(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    bias_attr=False,
                    data_format=data_format))
            out_size += 1
        self.image_pooling = image_pooling

        self.conv_bn_relu = layers.ConvBNReLU(
            in_channels=out_channels * out_size,
            out_channels=out_channels,
            kernel_size=1,
            data_format=data_format)

        self.dropout = nn.Dropout(p=0.1)  # drop rate

    def forward(self, x):
        outputs = []
        if self.data_format == 'NCHW':
            interpolate_shape = paddle.shape(x)[2:]
            axis = 1
        else:
            interpolate_shape = paddle.shape(x)[1:3]
            axis = -1
        for block in self.aspp_blocks:
            y = block(x)
            outputs.append(y)

        if self.image_pooling:
            img_avg = self.global_avg_pool(x)
            img_avg = F.interpolate(
                img_avg,
                interpolate_shape,
                mode='bilinear',
                align_corners=self.align_corners,
                data_format=self.data_format)
            outputs.append(img_avg)

        x = paddle.concat(outputs, axis=axis)
        x = self.conv_bn_relu(x)
        x = self.dropout(x)

        return x
