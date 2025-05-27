# Copyright (c) MICROSOFT.
# Licensed under the MIT License.
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from torchvision import models
from torch import autocast
from PIL import Image, ImageCms
from random import random
from torchvision.transforms import Resize, CenterCrop, ToTensor, Normalize, Compose
from unitorch.models import GenericModel
from unitorch.cli import cached_path
from unitorch.cli import (
    add_default_section_for_init,
    add_default_section_for_function,
    register_model,
    register_process,
)
from unitorch.cli import WriterMixin, WriterOutputs
from unitorch.cli.models import TensorsInputs, SegmentationOutputs, SegmentationTargets


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False,
    )


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
    ):
        super().__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class RefUnet(nn.Module):
    def __init__(self, in_ch, inc_ch):
        super().__init__()

        self.conv0 = nn.Conv2d(in_ch, inc_ch, 3, padding=1)

        self.conv1 = nn.Conv2d(inc_ch, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)

        self.pool1 = nn.MaxPool2d(2, 2, ceil_mode=True)

        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=True)

        self.pool2 = nn.MaxPool2d(2, 2, ceil_mode=True)

        self.conv3 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU(inplace=True)

        self.pool3 = nn.MaxPool2d(2, 2, ceil_mode=True)

        self.conv4 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.relu4 = nn.ReLU(inplace=True)

        self.pool4 = nn.MaxPool2d(2, 2, ceil_mode=True)

        #####

        self.conv5 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn5 = nn.BatchNorm2d(64)
        self.relu5 = nn.ReLU(inplace=True)

        #####

        self.conv_d4 = nn.Conv2d(128, 64, 3, padding=1)
        self.bn_d4 = nn.BatchNorm2d(64)
        self.relu_d4 = nn.ReLU(inplace=True)

        self.conv_d3 = nn.Conv2d(128, 64, 3, padding=1)
        self.bn_d3 = nn.BatchNorm2d(64)
        self.relu_d3 = nn.ReLU(inplace=True)

        self.conv_d2 = nn.Conv2d(128, 64, 3, padding=1)
        self.bn_d2 = nn.BatchNorm2d(64)
        self.relu_d2 = nn.ReLU(inplace=True)

        self.conv_d1 = nn.Conv2d(128, 64, 3, padding=1)
        self.bn_d1 = nn.BatchNorm2d(64)
        self.relu_d1 = nn.ReLU(inplace=True)

        self.conv_d0 = nn.Conv2d(64, 1, 3, padding=1)

        self.upscore2 = nn.Upsample(scale_factor=2, mode="bilinear")

    def forward(self, x):
        hx = x
        hx = self.conv0(hx)

        hx1 = self.relu1(self.bn1(self.conv1(hx)))
        hx = self.pool1(hx1)

        hx2 = self.relu2(self.bn2(self.conv2(hx)))
        hx = self.pool2(hx2)

        hx3 = self.relu3(self.bn3(self.conv3(hx)))
        hx = self.pool3(hx3)

        hx4 = self.relu4(self.bn4(self.conv4(hx)))
        hx = self.pool4(hx4)

        hx5 = self.relu5(self.bn5(self.conv5(hx)))

        hx = self.upscore2(hx5)

        d4 = self.relu_d4(self.bn_d4(self.conv_d4(torch.cat((hx, hx4), 1))))
        hx = self.upscore2(d4)

        d3 = self.relu_d3(self.bn_d3(self.conv_d3(torch.cat((hx, hx3), 1))))
        hx = self.upscore2(d3)

        d2 = self.relu_d2(self.bn_d2(self.conv_d2(torch.cat((hx, hx2), 1))))
        hx = self.upscore2(d2)

        d1 = self.relu_d1(self.bn_d1(self.conv_d1(torch.cat((hx, hx1), 1))))

        residual = self.conv_d0(d1)

        return x + residual


@register_model("microsoft/picasso/model/segmentation/basnet")
class BASNetForSegmentation(GenericModel):
    def __init__(
        self,
        n_channels: int,
        n_classes: int,
    ):
        super().__init__()

        resnet = models.resnet34(pretrained=True)

        ## -------------Encoder--------------

        self.inconv = nn.Conv2d(n_channels, 64, 3, padding=1)
        self.inbn = nn.BatchNorm2d(64)
        self.inrelu = nn.ReLU(inplace=True)

        # stage 1
        self.encoder1 = resnet.layer1  # 224
        # stage 2
        self.encoder2 = resnet.layer2  # 112
        # stage 3
        self.encoder3 = resnet.layer3  # 56
        # stage 4
        self.encoder4 = resnet.layer4  # 28

        self.pool4 = nn.MaxPool2d(2, 2, ceil_mode=True)

        # stage 5
        self.resb5_1 = BasicBlock(512, 512)
        self.resb5_2 = BasicBlock(512, 512)
        self.resb5_3 = BasicBlock(512, 512)  # 14

        self.pool5 = nn.MaxPool2d(2, 2, ceil_mode=True)

        # stage 6
        self.resb6_1 = BasicBlock(512, 512)
        self.resb6_2 = BasicBlock(512, 512)
        self.resb6_3 = BasicBlock(512, 512)  # 7

        ## -------------Bridge--------------

        # stage Bridge
        self.convbg_1 = nn.Conv2d(512, 512, 3, dilation=2, padding=2)  # 7
        self.bnbg_1 = nn.BatchNorm2d(512)
        self.relubg_1 = nn.ReLU(inplace=True)
        self.convbg_m = nn.Conv2d(512, 512, 3, dilation=2, padding=2)
        self.bnbg_m = nn.BatchNorm2d(512)
        self.relubg_m = nn.ReLU(inplace=True)
        self.convbg_2 = nn.Conv2d(512, 512, 3, dilation=2, padding=2)
        self.bnbg_2 = nn.BatchNorm2d(512)
        self.relubg_2 = nn.ReLU(inplace=True)

        ## -------------Decoder--------------

        # stage 6d
        self.conv6d_1 = nn.Conv2d(1024, 512, 3, padding=1)  # 16
        self.bn6d_1 = nn.BatchNorm2d(512)
        self.relu6d_1 = nn.ReLU(inplace=True)

        self.conv6d_m = nn.Conv2d(512, 512, 3, dilation=2, padding=2)  ###
        self.bn6d_m = nn.BatchNorm2d(512)
        self.relu6d_m = nn.ReLU(inplace=True)

        self.conv6d_2 = nn.Conv2d(512, 512, 3, dilation=2, padding=2)
        self.bn6d_2 = nn.BatchNorm2d(512)
        self.relu6d_2 = nn.ReLU(inplace=True)

        # stage 5d
        self.conv5d_1 = nn.Conv2d(1024, 512, 3, padding=1)  # 16
        self.bn5d_1 = nn.BatchNorm2d(512)
        self.relu5d_1 = nn.ReLU(inplace=True)

        self.conv5d_m = nn.Conv2d(512, 512, 3, padding=1)  ###
        self.bn5d_m = nn.BatchNorm2d(512)
        self.relu5d_m = nn.ReLU(inplace=True)

        self.conv5d_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn5d_2 = nn.BatchNorm2d(512)
        self.relu5d_2 = nn.ReLU(inplace=True)

        # stage 4d
        self.conv4d_1 = nn.Conv2d(1024, 512, 3, padding=1)  # 32
        self.bn4d_1 = nn.BatchNorm2d(512)
        self.relu4d_1 = nn.ReLU(inplace=True)

        self.conv4d_m = nn.Conv2d(512, 512, 3, padding=1)  ###
        self.bn4d_m = nn.BatchNorm2d(512)
        self.relu4d_m = nn.ReLU(inplace=True)

        self.conv4d_2 = nn.Conv2d(512, 256, 3, padding=1)
        self.bn4d_2 = nn.BatchNorm2d(256)
        self.relu4d_2 = nn.ReLU(inplace=True)

        # stage 3d
        self.conv3d_1 = nn.Conv2d(512, 256, 3, padding=1)  # 64
        self.bn3d_1 = nn.BatchNorm2d(256)
        self.relu3d_1 = nn.ReLU(inplace=True)

        self.conv3d_m = nn.Conv2d(256, 256, 3, padding=1)  ###
        self.bn3d_m = nn.BatchNorm2d(256)
        self.relu3d_m = nn.ReLU(inplace=True)

        self.conv3d_2 = nn.Conv2d(256, 128, 3, padding=1)
        self.bn3d_2 = nn.BatchNorm2d(128)
        self.relu3d_2 = nn.ReLU(inplace=True)

        # stage 2d

        self.conv2d_1 = nn.Conv2d(256, 128, 3, padding=1)  # 128
        self.bn2d_1 = nn.BatchNorm2d(128)
        self.relu2d_1 = nn.ReLU(inplace=True)

        self.conv2d_m = nn.Conv2d(128, 128, 3, padding=1)  ###
        self.bn2d_m = nn.BatchNorm2d(128)
        self.relu2d_m = nn.ReLU(inplace=True)

        self.conv2d_2 = nn.Conv2d(128, 64, 3, padding=1)
        self.bn2d_2 = nn.BatchNorm2d(64)
        self.relu2d_2 = nn.ReLU(inplace=True)

        # stage 1d
        self.conv1d_1 = nn.Conv2d(128, 64, 3, padding=1)  # 256
        self.bn1d_1 = nn.BatchNorm2d(64)
        self.relu1d_1 = nn.ReLU(inplace=True)

        self.conv1d_m = nn.Conv2d(64, 64, 3, padding=1)  ###
        self.bn1d_m = nn.BatchNorm2d(64)
        self.relu1d_m = nn.ReLU(inplace=True)

        self.conv1d_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn1d_2 = nn.BatchNorm2d(64)
        self.relu1d_2 = nn.ReLU(inplace=True)

        ## -------------Bilinear Upsampling--------------
        self.upscore6 = nn.Upsample(scale_factor=32, mode="bilinear")  ###
        self.upscore5 = nn.Upsample(scale_factor=16, mode="bilinear")
        self.upscore4 = nn.Upsample(scale_factor=8, mode="bilinear")
        self.upscore3 = nn.Upsample(scale_factor=4, mode="bilinear")
        self.upscore2 = nn.Upsample(scale_factor=2, mode="bilinear")

        ## -------------Side Output--------------
        self.outconvb = nn.Conv2d(512, 1, 3, padding=1)
        self.outconv6 = nn.Conv2d(512, 1, 3, padding=1)
        self.outconv5 = nn.Conv2d(512, 1, 3, padding=1)
        self.outconv4 = nn.Conv2d(256, 1, 3, padding=1)
        self.outconv3 = nn.Conv2d(128, 1, 3, padding=1)
        self.outconv2 = nn.Conv2d(64, 1, 3, padding=1)
        self.outconv1 = nn.Conv2d(64, 1, 3, padding=1)

        ## -------------Refine Module-------------
        self.refunet = RefUnet(1, 64)

    @classmethod
    @add_default_section_for_init("microsoft/picasso/model/segmentation/basnet")
    def from_core_configure(cls, config, **kwargs):
        config.set_default_section("microsoft/picasso/model/segmentation/basnet")
        n_channels = config.getoption("n_channels", 3)
        n_classes = config.getoption("n_classes", 1)

        inst = cls(
            n_channels=n_channels,
            n_classes=n_classes,
        )

        weight_path = config.getoption(
            "pretrained_weight_path",
            None,
        )
        if weight_path is not None:
            inst.from_pretrained(
                weight_path,
            )

        return inst

    def forward(
        self, image: torch.Tensor, sizes: Optional[List[Tuple[int, int]]] = None
    ):
        hx = image

        ## -------------Encoder-------------
        hx = self.inconv(hx)
        hx = self.inbn(hx)
        hx = self.inrelu(hx)

        h1 = self.encoder1(hx)  # 256
        h2 = self.encoder2(h1)  # 128
        h3 = self.encoder3(h2)  # 64
        h4 = self.encoder4(h3)  # 32

        hx = self.pool4(h4)  # 16

        hx = self.resb5_1(hx)
        hx = self.resb5_2(hx)
        h5 = self.resb5_3(hx)

        hx = self.pool5(h5)  # 8

        hx = self.resb6_1(hx)
        hx = self.resb6_2(hx)
        h6 = self.resb6_3(hx)

        ## -------------Bridge-------------
        hx = self.relubg_1(self.bnbg_1(self.convbg_1(h6)))  # 8
        hx = self.relubg_m(self.bnbg_m(self.convbg_m(hx)))
        hbg = self.relubg_2(self.bnbg_2(self.convbg_2(hx)))

        ## -------------Decoder-------------

        hx = self.relu6d_1(self.bn6d_1(self.conv6d_1(torch.cat((hbg, h6), 1))))
        hx = self.relu6d_m(self.bn6d_m(self.conv6d_m(hx)))
        hd6 = self.relu6d_2(self.bn6d_2(self.conv6d_2(hx)))

        hx = self.upscore2(hd6)  # 8 -> 16

        hx = self.relu5d_1(self.bn5d_1(self.conv5d_1(torch.cat((hx, h5), 1))))
        hx = self.relu5d_m(self.bn5d_m(self.conv5d_m(hx)))
        hd5 = self.relu5d_2(self.bn5d_2(self.conv5d_2(hx)))

        hx = self.upscore2(hd5)  # 16 -> 32

        hx = self.relu4d_1(self.bn4d_1(self.conv4d_1(torch.cat((hx, h4), 1))))
        hx = self.relu4d_m(self.bn4d_m(self.conv4d_m(hx)))
        hd4 = self.relu4d_2(self.bn4d_2(self.conv4d_2(hx)))

        hx = self.upscore2(hd4)  # 32 -> 64

        hx = self.relu3d_1(self.bn3d_1(self.conv3d_1(torch.cat((hx, h3), 1))))
        hx = self.relu3d_m(self.bn3d_m(self.conv3d_m(hx)))
        hd3 = self.relu3d_2(self.bn3d_2(self.conv3d_2(hx)))

        hx = self.upscore2(hd3)  # 64 -> 128

        hx = self.relu2d_1(self.bn2d_1(self.conv2d_1(torch.cat((hx, h2), 1))))
        hx = self.relu2d_m(self.bn2d_m(self.conv2d_m(hx)))
        hd2 = self.relu2d_2(self.bn2d_2(self.conv2d_2(hx)))

        hx = self.upscore2(hd2)  # 128 -> 256

        hx = self.relu1d_1(self.bn1d_1(self.conv1d_1(torch.cat((hx, h1), 1))))
        hx = self.relu1d_m(self.bn1d_m(self.conv1d_m(hx)))
        hd1 = self.relu1d_2(self.bn1d_2(self.conv1d_2(hx)))

        ## -------------Side Output-------------
        db = self.outconvb(hbg)
        db = self.upscore6(db)  # 8->256

        d6 = self.outconv6(hd6)
        d6 = self.upscore6(d6)  # 8->256

        d5 = self.outconv5(hd5)
        d5 = self.upscore5(d5)  # 16->256

        d4 = self.outconv4(hd4)
        d4 = self.upscore4(d4)  # 32->256

        d3 = self.outconv3(hd3)
        d3 = self.upscore3(d3)  # 64->256

        d2 = self.outconv2(hd2)
        d2 = self.upscore2(d2)  # 128->256

        d1 = self.outconv1(hd1)  # 256

        ## -------------Refine Module-------------
        dout = self.refunet(d1)  # 256

        masks = F.sigmoid(dout)
        if sizes is None:
            return SegmentationOutputs(masks=masks)
        masks = [
            F.interpolate(mask.unsqueeze(0), size=list(size), mode="bilinear").squeeze(
                0
            )
            for mask, size in zip(masks, sizes)
        ]
        masks = [m.permute(1, 2, 0) for m in masks]
        masks = [(m - m.min()) / (m.max() - m.min()) for m in masks]
        return SegmentationOutputs(
            masks=masks,
        )


class BASNetProcessor:
    def __init__(
        self,
        pixel_mean: List[float] = [0.406, 0.456, 0.485],
        pixel_std: List[float] = [0.225, 0.224, 0.229],
        resize_shape: List[int] = [256, 256],
        return_mask: Optional[bool] = False,
    ):
        self.pixel_mean = torch.tensor(pixel_mean)
        self.pixel_std = torch.tensor(pixel_std)
        self.resize_shape = tuple(resize_shape)
        self.return_mask = return_mask

    @classmethod
    @add_default_section_for_init("microsoft/picasso/process/basnet")
    def from_core_configure(cls, config, **kwargs):
        pass

    @register_process("microsoft/picasso/process/basnet/segmentation_inputs")
    def _segmentation_inputs(self, image):
        width, height = image.size
        image = np.array(image)[:, :, ::-1]
        image = cv2.resize(image, self.resize_shape, cv2.INTER_CUBIC)
        image = torch.tensor(image) / torch.tensor(image).max()  # 255
        image = (image - self.pixel_mean[[2, 1, 0]]) / self.pixel_std[[2, 1, 0]]
        image = image.permute(2, 0, 1)
        return TensorsInputs(
            image=torch.tensor(image),
            sizes=torch.tensor([height, width]),
        )

    @register_process("microsoft/picasso/postprocess/basnet/segmentation")
    def _postprocess_segmentation(self, outputs: SegmentationOutputs):
        results = outputs.to_pandas()
        masks = outputs.masks

        def process_roi(mask):
            mask = np.array(mask)
            mask1, mask2 = mask, (mask >= 0.1).astype("uint8")
            mw, mh = mask1.shape[:2]
            x, y, w, h = cv2.boundingRect(mask2)
            x1 = max(0, x)
            x2 = min(x + w, mw - 1)
            y1 = max(0, y)
            y2 = min(y + h, mh - 1)
            return f"{x1},{y1},{x2},{y2}"

        def image_to_base64(image):
            import base64

            _, buffer = cv2.imencode(".jpg", image)
            image = buffer.tobytes()
            image_base64 = base64.b64encode(image).decode("utf-8")
            return image_base64

        def process_mask(mask):
            mask = np.array(mask)
            # No resize back here. If need to use mask together with image, please resize it back.
            if np.max(mask) < 2:
                mask = (mask * 255).astype("uint8")
            mask_output = image_to_base64(mask)
            return mask_output

        results["roi"] = [process_roi(m) for m in masks]
        if self.return_mask:
            results["mask"] = [process_mask(m) for m in masks]

        return WriterOutputs(results)


class BASNetForSegmentationPipeline(BASNetForSegmentation):
    def __init__(
        self,
        n_channels: int = 3,
        n_classes: int = 1,
        device: Optional[Union[str, int]] = "cpu",
    ):
        super().__init__(
            n_channels=n_channels,
            n_classes=n_classes,
        )
        self._processor = BASNetProcessor()
        self._device = "cpu" if device == "cpu" else int(device)

        self.to(device=self._device)
        self.eval()

    @classmethod
    @add_default_section_for_init("microsoft/picasso/pipeline/basnet")
    def from_core_configure(cls, config, **kwargs):
        config.set_default_section("microsoft/picasso/pipeline/basnet")
        n_channels = config.getoption("n_channels", 3)
        n_classes = config.getoption("n_classes", 1)
        device = config.getoption("device", "cpu")

        inst = cls(
            n_channels=n_channels,
            n_classes=n_classes,
            device=device,
        )

        weight_path = config.getoption(
            "pretrained_weight_path",
            "https://huggingface.co/datasets/fuliucansheng/unitorchblobfuse/resolve/main/models/picasso/pytorch_model.basnet.jewelry.2503.bin",
        )
        if weight_path is not None:
            inst.from_pretrained(
                weight_path,
            )

        return inst

    @torch.no_grad()
    def __call__(
        self,
        image: Image.Image,
        threshold: Optional[float] = 0.1,
    ):
        inputs = self._processor._segmentation_inputs(image)
        pixel = inputs.image.unsqueeze(0).to(self._device)
        sizes = inputs.sizes.unsqueeze(0).to(self._device)

        outputs = self.forward(
            image=pixel,
            sizes=sizes,
        ).masks

        masks = [
            (mask.squeeze(0).cpu().numpy() > threshold).astype(np.uint8)
            for mask in outputs
        ][0]
        masks = masks.reshape(*masks.shape[:2])
        result_image = Image.fromarray(masks * 255, mode="L")
        result_image = result_image.resize(image.size, resample=Image.LANCZOS)

        return result_image
