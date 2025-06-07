# Copyright (c) MICROSOFT.
# Licensed under the MIT License.

import os
import cv2
import gc
import torch
import numpy as np
import gradio as gr
from PIL import Image, ImageDraw, ImageOps
from torchvision import transforms
from unitorch import mktempfile
from unitorch.utils import read_file
from unitorch.models import GenericOutputs
from unitorch.cli import CoreConfigureParser
from unitorch.cli.pipelines.bria import BRIAForSegmentationPipeline
from unitorch.cli.fastapis.stable_flux import (
    StableFluxForImageInpaintingFastAPIPipeline,
    StableFluxForReduxInpaintingFastAPIPipeline,
)
from unitorch.cli.webuis import SimpleWebUI
from unitorch_microsoft import cached_path
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


class AddObjWebUI(SimpleWebUI):
    _title = "Add Object to Image"
    _description = "This is a demo for adding objects to images using FLUX. You can input an image and a prompt, and the model will generate a new image with the specified object added."

    def __init__(self, config: CoreConfigureParser):
        self._status = getattr(self, "_status", "Stopped")
        # create elements
        toper_menus = create_toper_menus()
        footer = create_footer()
        header = create_element(
            "markdown",
            label=f"# <div style='margin-top:10px'>🎨 {self._title} </div>",
            interactive=False,
        )
        description = create_element(
            "markdown",
            label=self._description,
            interactive=False,
        )

        status = create_element("text", "Status", default="Stopped", interactive=False)
        start = create_element("button", "Start", variant="primary")
        stop = create_element("button", "Stop", variant="stop")
        image = create_element("image_editor", "Input Image")
        mask_image = create_element("image", "Input Image Mask")
        refer_image = create_element("image", "Reference Image")
        generate = create_element("button", "Generate")
        output_image = create_element("image", "Output Image")

        left = create_column(create_row(image, mask_image), refer_image, generate)
        right = create_column(output_image)

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
        iface._title = self._title
        iface._description = self._description

        # create events
        iface.__enter__()

        image.change(fn=self.composite_images, inputs=[image], outputs=[mask_image])

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
            inputs=[image, mask_image, refer_image],
            outputs=[output_image],
            trigger_mode="once",
        )

        iface.load(
            fn=lambda: self._status,
            outputs=status,
        )

        iface.__exit__()

        super().__init__(config, iname=self._title, iface=iface)

    def start(self):
        self._pipe1 = BRIAForSegmentationPipeline.from_core_configure(
            config=self._config,
            pretrained_weight_path="https://huggingface.co/datasets/fuliucansheng/hubfiles/resolve/main/bria_rmbg2.0_pytorch_model.bin",
        )
        self._pipe2 = StableFluxForReduxInpaintingFastAPIPipeline.from_core_configure(
            config=self._config,
            pretrained_name="stable-flux-dev-redux-fill",
        )
        self._status = "Running"
        return self._status

    def stop(self):
        self._pipe1.to("cpu")
        del self._pipe1
        self._pipe2.to("cpu")
        del self._pipe2
        gc.collect()
        torch.cuda.empty_cache()
        self._status = "Stopped"
        return self._status

    def composite_images(self, images):
        if images is None:
            return None
        layers = images["layers"]
        if len(layers) == 0:
            return None
        image = layers[0]
        for i in range(1, len(layers)):
            image = Image.alpha_composite(image, layers[i])
        image = image.convert("L")
        image = image.point(lambda p: p < 5 and 255)
        image = ImageOps.invert(image)
        return image

    def serve(self, image, mask, refer_image):
        image = image["background"].convert("RGB")
        white = Image.new("RGB", (image.width, image.height), (255, 255, 255))
        image.paste(white, mask=mask.convert("L"))
        pos_prompt = f"realistic, extremely detailed, photorealistic, best quality"
        neg_prompt = "nsfw, paintings, sketches, (worst quality:2), (low quality:2) lowers, normal quality, ((monochrome)), ((grayscale)), logo, word, character, nudity, naked, disfigured, nude, blurry, blurry background"

        result = self._pipe2(
            pos_prompt,
            image,
            mask,
            refer_image,
            neg_prompt,
            width=image.width // 16 * 16,
            height=image.height // 16 * 16,
            guidance_scale=30,
            strength=1.0,
            num_timesteps=50,
            seed=42,
        )

        # new_image = image.resize(result.size, resample=Image.LANCZOS).convert("RGB")
        # result = result.convert("RGB")
        # mask = mask.resize(result.size, resample=Image.LANCZOS)
        # mask = np.array(mask.convert("1"))[:, :, None]
        # result = Image.fromarray(
        #     (np.array(result) * mask + np.array(new_image) * (1 - mask)).astype(
        #         np.uint8
        #     )
        # )
        return result
