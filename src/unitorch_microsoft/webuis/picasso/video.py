# Copyright (c) MICROSOFT.
# Licensed under the MIT License.
import cv2
import numpy as np
import gradio as gr
from PIL import Image, ImageDraw
from unitorch import mktempfile
from unitorch.cli import CoreConfigureParser
from unitorch.cli import register_webui
from unitorch.cli.webuis import SimpleWebUI
from unitorch.cli.webuis import (
    create_element,
    create_accordion,
    create_row,
    create_column,
    create_group,
    create_tab,
    create_tabs,
    create_blocks,
    create_pretrain_layout,
)
from unitorch_microsoft.picasso.video.opencv import zoom_in_effect


class ZoomInWebUI(SimpleWebUI):
    def __init__(self, config: CoreConfigureParser):
        # create elements
        input_image = create_element("image", "Image")
        frames = create_element(
            "slider", "Frames", min_value=1, max_value=1200, step=1, default=60
        )
        reset = create_element("button", "Reset")
        generate = create_element("button", "Generate")
        output_video = create_element("video", "Output Video")

        left = create_column(input_image, frames, create_row(reset, generate))
        right = create_column(output_video)
        iface = create_blocks(create_row(left, right))

        # create events
        iface.__enter__()
        origin_input_image = gr.State(None)
        boxes_points = gr.State([])

        input_image.upload(
            lambda image: (image.copy() if image is not None else None, []),
            inputs=[input_image],
            outputs=[origin_input_image, boxes_points],
        )
        input_image.select(
            self.add_click_points,
            [origin_input_image, boxes_points],
            [input_image, boxes_points],
        )

        reset.click(
            lambda x: (x, []),
            inputs=[origin_input_image],
            outputs=[input_image, boxes_points],
            trigger_mode="once",
        )

        generate.click(
            fn=self.serve,
            inputs=[origin_input_image, frames, boxes_points],
            outputs=[output_video],
            trigger_mode="once",
        )

        iface.__exit__()

        super().__init__(config, iname="Zoom-In", iface=iface)

    def add_click_points(self, image, click_points, evt: gr.SelectData):
        x, y = evt.index[0], evt.index[1]
        click_points = click_points + [(x, y)]
        new_image = image.copy()
        draw = ImageDraw.Draw(new_image)
        point_color = (255, 0, 0)
        radius = 3
        for point in click_points:
            x, y = point
            draw.ellipse(
                (x - radius, y - radius, x + radius, y + radius), fill=point_color
            )

        return new_image, click_points

    def serve(self, image, frames, points):
        x1 = min([point[0] for point in points])
        y1 = min([point[1] for point in points])
        x2 = max([point[0] for point in points])
        y2 = max([point[1] for point in points])

        w, h = image.size
        cv2_image = np.array(image)
        cv2_image = cv2_image[:, :, ::-1]
        new_w, new_h = x2 - x1, y2 - y1
        ratio, new_ratio = w / h, new_w / new_h
        if new_ratio > ratio:
            new_w, new_h = new_w, int(new_w / ratio)
            x1, y1, x2, y2 = (
                x1,
                max(0, y1 - (new_h - (y2 - y1)) // 2),
                x2,
                min(h, y2 + (new_h - (y2 - y1)) // 2),
            )
        else:
            new_w, new_h = int(new_h * ratio), new_h
            x1, y1, x2, y2 = (
                max(0, x1 - (new_w - (x2 - x1)) // 2),
                y1,
                min(w, x2 + (new_w - (x2 - x1)) // 2),
                y2,
            )

        start_coords, end_coords = (0, 0, w, h), (x1, y1, x2 - x1, y2 - y1)
        output_video = mktempfile(suffix=".mp4")
        zoom_in_effect(
            cv2_image, start_coords, end_coords, output_video, num_frames=frames
        )
        return output_video


class VideoWebUI(SimpleWebUI):
    def __init__(self, config: CoreConfigureParser):
        webuis = [
            ZoomInWebUI(config),
        ]
        iface = gr.TabbedInterface(
            [webui.iface for webui in webuis],
            tab_names=[webui.iname for webui in webuis],
        )
        super().__init__(config, iname="Video", iface=iface)
