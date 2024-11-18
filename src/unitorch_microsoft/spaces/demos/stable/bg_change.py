# Copyright (c) MICROSOFT.
# Licensed under the MIT License.

import os
import cv2
import gc
import torch
import numpy as np
import gradio as gr
from PIL import Image, ImageDraw, ImageOps
from transformers import AutoModelForImageSegmentation
from diffusers import (
    StableDiffusionInpaintPipeline,
    StableDiffusionControlNetInpaintPipeline,
    ControlNetModel,
    UniPCMultistepScheduler,
)
from torchvision import transforms
from unitorch import mktempfile
from unitorch.utils import read_file
from unitorch.models import GenericOutputs
from unitorch.cli import CoreConfigureParser
from unitorch.cli.pipelines.stable.interrogator import ClipInterrogatorPipeline
from unitorch.cli.pipelines.sam import SamForSegmentationPipeline
from unitorch.cli.fastapis.controlnet import ControlNetForImageInpaintingFastAPIPipeline
from unitorch.cli.pipelines.tools import depth, canny
from unitorch.cli.webuis import SimpleWebUI
from unitorch_microsoft import cached_path
import unitorch_microsoft.models.sam
from unitorch_microsoft.spaces import (
    create_element,
    create_row,
    create_column,
    create_flex_layout,
    create_blocks,
    create_toper_menus,
    create_footer,
    create_dashboard_card,
    create_card,
    create_dashboard_cards_group,
    create_cards_group,
)


class ChangeBGWebUI(SimpleWebUI):
    def __init__(self, config: CoreConfigureParser):
        self._status = getattr(self, "_status", "Stopped")
        # create elements
        toper_menus = create_toper_menus()
        footer = create_footer()
        header = create_element(
            "markdown",
            label=f"# <div style='margin-top:10px'>✨ Change Background</div>",
            interactive=False,
        )
        description = create_element(
            "markdown",
            label="description",
            interactive=False,
        )

        status = create_element("text", "Status", default="Stopped", interactive=False)
        start = create_element("button", "Start", variant="primary")
        stop = create_element("button", "Stop", variant="stop")
        input_image = create_element("image", "Image")
        prompt = create_element(
            "text", "Input Prompt", lines=3, default="simple background"
        )
        generate = create_element("button", "Generate")
        output_image = create_element("image", "Output Image")

        left = create_column(input_image, prompt, generate)
        right = create_column(output_image)
        iface = create_blocks()

        iface = create_blocks(
            toper_menus,
            create_row(
                create_column(header, description, scale=1),
                create_row(status, start, stop),
            ),
            create_row(
                left, right, elem_classes="ut-bg-transparent ut-ms-min-70-height"
            ),
            footer,
        )

        # create events
        iface.__enter__()

        start.click(
            self.start,
            outputs=[status],
            trigger_mode="once",
        )
        stop.click(
            self.stop,
            outputs=[status],
            trigger_mode="once",
        )

        generate.click(
            fn=self.serve,
            inputs=[input_image, prompt],
            outputs=[output_image],
            trigger_mode="once",
        )

        iface.load(
            fn=lambda: self._status,
            outputs=status,
        )

        iface.__exit__()

        super().__init__(config, iname="Change Background", iface=iface)

    def start(self):
        self._pipe = AutoModelForImageSegmentation.from_pretrained(
            "ZhengPeng7/BiRefNet", trust_remote_code=True, torch_dtype=torch.float32
        )
        self._pipe.to("cuda")
        self._pipe.eval()
        # self._clip_pipe = ClipInterrogatorPipeline.from_core_configure(config=self._config, pretrained_name="clip-vit-large-patch14")
        self._diff_pipe = ControlNetForImageInpaintingFastAPIPipeline.from_core_configure(
            config=self._config,
            pretrained_name="stable-v1.5-realistic-v5.1-inpainting",
            pretrained_controlnet_names=["stable-v1.5-controlnet-canny"],
            pretrained_inpainting_controlnet_name="stable-v1.5-controlnet-inpainting",
        )
        self._status = "Running"
        return self._status

    def stop(self):
        self._pipe.to("cpu")
        del self._pipe
        # self._clip_pipe.to("cpu")
        # del self._clip_pipe
        self._diff_pipe.to("cpu")
        del self._diff_pipe
        gc.collect()
        torch.cuda.empty_cache()
        self._status = "Stopped"
        return self._status

    def extract_object(self, image):
        # Data settings
        image_size = (1024, 1024)
        transform_image = transforms.Compose(
            [
                transforms.Resize(image_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

        input_images = transform_image(image).unsqueeze(0).to("cuda")

        # Prediction
        with torch.no_grad():
            preds = self._pipe(input_images)[-1].sigmoid().cpu()
        pred = preds[0].squeeze()
        pred_pil = transforms.ToPILImage()(pred)
        mask = pred_pil.resize(image.size)
        return mask

    def serve(self, image, prompt):
        mask = self.extract_object(image)
        mask = mask.convert("1").resize(image.size)
        # pos_prompt = f"cinematic photo of {prompt}, simple background, realistic, extremely detailed, photorealistic, best quality"
        pos_prompt = (
            f"{prompt}, realistic, extremely detailed, photorealistic, best quality"
        )
        neg_prompt = "nsfw, paintings, sketches, (worst quality:2), (low quality:2) lowers, normal quality, ((monochrome)), ((grayscale)), logo, word, character, nudity, naked, disfigured, nude, blurry, blurry background"
        new_image = Image.new("RGBA", image.size, (0, 0, 0, 0))
        new_image.paste(image, (0, 0), mask)
        new_image = new_image.convert("RGB")
        canny_image = canny(new_image)

        mask = ImageOps.invert(mask)

        result = self._diff_pipe(
            pos_prompt,
            image,
            mask,
            neg_text=neg_prompt,
            width=image.width,
            height=image.height,
            guidance_scale=7.5,
            strength=1.0,
            num_timesteps=25,
            seed=42,
            freeu_params=None,
            controlnet_images=[canny_image],
            controlnet_guidance_scales=[0.8],
            inpaint_controlnet_image=image,
            inpaint_controlnet_guidance_scale=0.2,
        )

        # new_image = image.resize(result.size).convert("RGB")
        # result = result.convert("RGB")
        # mask = mask.resize(result.size)
        # mask = np.array(mask.convert("1"))[:, :, None]
        # result = Image.fromarray(
        #     (np.array(result) * mask + np.array(new_image) * (1 - mask)).astype(
        #         np.uint8
        #     )
        # )
        return result
