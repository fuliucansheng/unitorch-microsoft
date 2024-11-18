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

# from diffusers import (
#     FluxInpaintPipeline,
#     FluxControlNetInpaintPipeline,
#     FluxControlNetModel,
# )
from torchvision import transforms
from unitorch import mktempfile
from unitorch.utils import read_file
from unitorch.models import GenericOutputs
from unitorch.cli import CoreConfigureParser
from unitorch.cli.pipelines.stable.interrogator import ClipInterrogatorPipeline
from unitorch.cli.pipelines.sam import SamForSegmentationPipeline

# from unitorch.cli.pipelines.stable_flux import StableFluxForImageInpaintingPipeline
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
from unitorch_microsoft.spaces.demos.flux.alimama import (
    FluxControlNetModel as AliFluxControlNetModel,
    FluxTransformer2DModel as AliFluxTransformer2DModel,
    FluxControlNetInpaintingPipeline as AliFluxControlNetInpaintingPipeline,
)


class ExpandBGWebUI(SimpleWebUI):
    def __init__(self, config: CoreConfigureParser):
        self._status = getattr(self, "_status", "Stopped")
        # create elements
        toper_menus = create_toper_menus()
        footer = create_footer()
        header = create_element(
            "markdown",
            label=f"# <div style='margin-top:10px'>🎨 Expand Background </div>",
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
        width = create_element(
            "slider", "Width", default=1024, min_value=1, max_value=2048, step=1
        )
        height = create_element(
            "slider", "Height", default=1024, min_value=1, max_value=2048, step=1
        )
        generate = create_element("button", "Generate")
        output_image = create_element("image", "Output Image")

        left = create_column(input_image, prompt, width, height, generate)
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
            inputs=[input_image, prompt, width, height],
            outputs=[output_image],
            trigger_mode="once",
        )

        input_image.upload(
            lambda x: x.size,
            inputs=[input_image],
            outputs=[width, height],
        )

        iface.load(
            fn=lambda: self._status,
            outputs=status,
        )

        iface.__exit__()

        super().__init__(config, iname="Expand Background", iface=iface)

    def start(self):
        # self._clip_pipe = ClipInterrogatorPipeline.from_core_configure(config=CoreConfigureParser(), pretrained_name="clip-vit-large-patch14")
        controlnet = AliFluxControlNetModel.from_pretrained(
            "alimama-creative/FLUX.1-dev-Controlnet-Inpainting-Beta",
            torch_dtype=torch.bfloat16,
        )
        transformer = AliFluxTransformer2DModel.from_pretrained(
            "black-forest-labs/FLUX.1-dev",
            subfolder="transformer",
            torch_dytpe=torch.bfloat16,
        )
        self._diff_pipe = AliFluxControlNetInpaintingPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-dev",
            controlnet=controlnet,
            transformer=transformer,
            torch_dtype=torch.bfloat16,
            safety_checker=None,
        )
        self._diff_pipe.transformer.to(torch.bfloat16)
        self._diff_pipe.controlnet.to(torch.bfloat16)
        self._diff_pipe.enable_model_cpu_offload("1")
        # self._diff_pipe = FluxInpaintPipeline.from_pretrained(
        #     "black-forest-labs/FLUX.1-schnell",
        #     torch_dtype=torch.bfloat16,
        #     safety_checker=None,
        # )
        # self._diff_pipe.enable_model_cpu_offload("1")
        # self._diff_pipe = StableFluxForImageInpaintingPipeline.from_core_configure(
        #     config=self._config, pretrained_name="stable-v1.5-dreamshaper-8-inpainting",
        # )
        self._status = "Running"
        return self._status

    def stop(self):
        # self._clip_pipe.to("cpu")
        # del self._clip_pipe
        self._diff_pipe.to("cpu")
        del self._diff_pipe
        gc.collect()
        torch.cuda.empty_cache()
        self._status = "Stopped"
        return self._status

    def serve(self, image, prompt, width, height):
        # prompt = self._clip_pipe.get_best_prompt(image)
        mask = Image.new("L", (width, height), 255)
        im_width, im_height = image.size
        assert width >= im_width and height >= im_height
        black = Image.new("RGB", (im_width, im_height), (0, 0, 0))
        mask.paste(black, ((width - im_width) // 2, (height - im_height) // 2))
        new_image = Image.new("RGB", (width, height), (255, 255, 255))
        new_image.paste(image, ((width - im_width) // 2, (height - im_height) // 2))
        # pos_prompt = f"cinematic photo of {prompt}, realistic, extremely detailed, photorealistic, best quality"
        pos_prompt = (
            f"{prompt}, realistic, extremely detailed, photorealistic, best quality"
        )
        neg_prompt = "nsfw, paintings, sketches, (worst quality:2), (low quality:2) lowers, normal quality, ((monochrome)), ((grayscale)), logo, word, character, nudity, naked, disfigured, nude, blurry, blurry background"
        result = self._diff_pipe(
            control_image=new_image,
            control_mask=mask,
            prompt=pos_prompt,
            # negative_prompt=neg_prompt,
            width=width // 8 * 8,
            height=height // 8 * 8,
            guidance_scale=3.5,
            controlnet_conditioning_scale=0.9,
            generator=torch.Generator(device=self._diff_pipe.device).manual_seed(42),
            num_inference_steps=28,
            num_images_per_prompt=1,
            true_guidance_scale=3.5,
        ).images[0]
        # result = self._diff_pipe(
        #     pos_prompt,
        #     new_image,
        #     mask,
        #     neg_prompt,
        #     width=new_image.width,
        #     height=new_image.height,
        #     guidance_scale=7.5,
        #     strength=1.0,
        #     num_timesteps=25,
        #     seed=42,
        #     scheduler="DEIC++",
        #     freeu_params=(0.9, 0.2, 1.2, 1.4),
        #     controlnet_checkpoints=[],
        #     controlnet_images=[],
        #     controlnet_guidance_scales=[],
        #     lora_checkpoints=[],
        #     lora_weights=[],
        #     lora_alphas=[],
        #     lora_urls=[],
        #     lora_files=[],
        # )
        # new_image = new_image.resize(result.size).convert("RGB")
        # result = result.convert("RGB")
        # mask = mask.resize(result.size)
        # mask = np.array(mask.convert("1"))[:, :, None]
        # result = Image.fromarray(
        #     (np.array(result) * mask + np.array(new_image) * (1 - mask)).astype(
        #         np.uint8
        #     )
        # )
        return result
