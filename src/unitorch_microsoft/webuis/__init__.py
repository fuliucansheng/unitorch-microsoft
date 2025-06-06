# Copyright (c) MICROSOFT.
# Licensed under the MIT License.

import gradio as gr
import unitorch
from unitorch.utils.decorators import replace
from unitorch.utils import is_diffusers_available
from unitorch.cli import CoreConfigureParser
from unitorch.cli import register_webui
from unitorch.cli.webuis import SimpleWebUI
from unitorch_microsoft import is_faiss_available, is_openai_available
from unitorch_microsoft.webuis.detection import DetectionWebUI
from unitorch_microsoft.webuis.llm import LLMWebUI
from unitorch_microsoft.webuis.picasso import PicassoWebUI
from unitorch_microsoft.webuis.classification import ClassificationWebUI
from unitorch_microsoft.webuis.segmentation import SegmentationWebUI
from unitorch_microsoft.webuis.tools import ToolsWebUI

if is_faiss_available():
    from unitorch_microsoft.webuis.selection import SelectionWebUI

if is_openai_available():
    from unitorch_microsoft.webuis.chatgpt import ChatGPTWebUI

if is_diffusers_available():
    from unitorch_microsoft.webuis.diffusion import DiffusionWebUI


@replace(unitorch.cli.webuis.utils.layouts.create_element)
def create_element_v2(
    dtype,
    label,
    default=None,
    values=[],
    min_value=None,
    max_value=None,
    step=None,
    scale=None,
    multiselect=None,
    info=None,
    interactive=None,
    variant=None,
    lines=1,
    placeholder=None,
    show_label=True,
    elem_id=None,
    elem_classes=None,
    link=None,
):
    if dtype == "text":
        return gr.Textbox(
            value=default,
            label=label,
            scale=scale,
            info=info,
            interactive=interactive,
            lines=lines,
            placeholder=placeholder,
            show_label=show_label,
            elem_id=elem_id,
            elem_classes=elem_classes,
        )

    if dtype == "number":
        return gr.Number(
            value=default,
            label=label,
            scale=scale,
            info=info,
            interactive=interactive,
            show_label=show_label,
            elem_id=elem_id,
            elem_classes=elem_classes,
        )

    if dtype == "slider":
        return gr.Slider(
            value=default,
            label=label,
            minimum=min_value,
            maximum=max_value,
            step=step,
            scale=scale,
            info=info,
            interactive=interactive,
            elem_id=elem_id,
            elem_classes=elem_classes,
        )

    if dtype == "checkbox":
        return gr.Checkbox(
            value=default,
            label=label,
            scale=scale,
            info=info,
            interactive=interactive,
            show_label=show_label,
            elem_id=elem_id,
            elem_classes=elem_classes,
        )

    if dtype == "checkboxgroup":
        return gr.CheckboxGroup(
            value=default,
            label=label,
            choices=values,
            scale=scale,
            info=info,
            interactive=interactive,
            show_label=show_label,
            elem_id=elem_id,
            elem_classes=elem_classes,
        )

    if dtype == "radio":
        return gr.Radio(
            value=default,
            label=label,
            choices=values,
            scale=scale,
            info=info,
            interactive=interactive,
            show_label=show_label,
            elem_id=elem_id,
            elem_classes=elem_classes,
        )

    if dtype == "dropdown":
        return gr.Dropdown(
            value=default,
            label=label,
            choices=values,
            scale=scale,
            multiselect=multiselect,
            info=info,
            interactive=interactive,
            show_label=show_label,
            elem_id=elem_id,
            elem_classes=elem_classes,
        )

    if dtype == "dataframe":
        return gr.Dataframe(
            value=default,
            label=label,
            scale=scale,
            interactive=interactive,
            show_label=show_label,
            elem_id=elem_id,
            elem_classes=elem_classes,
            datatype="markdown",
            wrap=True,
        )

    if dtype == "image":
        return gr.Image(
            type="pil",
            value=default,
            label=label,
            scale=scale,
            interactive=interactive,
            show_label=show_label,
            elem_id=elem_id,
            elem_classes=elem_classes,
        )

    if dtype == "anno_image":
        return gr.AnnotatedImage(
            value=default,
            label=label,
            scale=scale,
            show_label=show_label,
            elem_id=elem_id,
            elem_classes=elem_classes,
        )

    if dtype == "image_editor":
        return gr.ImageEditor(
            type="pil",
            value=default,
            label=label,
            scale=scale,
            interactive=interactive,
            show_label=show_label,
            elem_id=elem_id,
            elem_classes=elem_classes,
            eraser=gr.Eraser(),
            brush=gr.Brush(),
            canvas_size=(1024, 1024),
        )

    if dtype == "audio":
        return gr.Audio(
            label=label,
            scale=scale,
            info=info,
            interactive=interactive,
            show_label=show_label,
            elem_id=elem_id,
            elem_classes=elem_classes,
        )

    if dtype == "video":
        return gr.Video(
            label=label,
            scale=scale,
            interactive=interactive,
            show_label=show_label,
            elem_id=elem_id,
            elem_classes=elem_classes,
        )

    if dtype == "file":
        return gr.File(
            label=label,
            scale=scale,
            interactive=interactive,
            show_label=show_label,
            elem_id=elem_id,
            elem_classes=elem_classes,
        )

    if dtype == "gallery":
        return gr.Gallery(
            label=label,
            scale=scale,
            interactive=interactive,
            show_label=show_label,
            elem_id=elem_id,
            elem_classes=elem_classes,
        )

    if dtype == "button":
        interactive = True if interactive is None else interactive
        variant = "primary" if variant is None else variant
        return gr.Button(
            value=label,
            scale=scale,
            interactive=interactive,
            variant=variant,
            elem_id=elem_id,
            elem_classes=elem_classes,
        )

    if dtype == "markdown":
        return gr.Markdown(
            value=label,
            show_label=show_label,
            elem_id=elem_id,
            elem_classes=elem_classes,
        )

    if dtype == "download_button":
        return gr.DownloadButton(
            label=label,
            value=link,
            scale=scale,
            elem_id=elem_id,
            elem_classes=elem_classes,
        )

    if dtype == "html":
        return gr.HTML(
            value=default,
            label=label,
            show_label=show_label,
            elem_id=elem_id,
            elem_classes=elem_classes,
        )

    raise ValueError(f"Unknown element type: {dtype}")
