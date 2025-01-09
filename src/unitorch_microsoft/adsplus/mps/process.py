from transformers import AutoProcessor, AutoModel
from PIL import Image
import torch
from transformers import CLIPImageProcessor, CLIPTokenizer, AutoTokenizer
import io
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from unitorch.models import (
    HfImageClassificationProcessor,
    HfTextClassificationProcessor,
    GenericOutputs,
)
from unitorch.cli import (
    add_default_section_for_init,
    add_default_section_for_function,
    register_process,
)

from unitorch.cli.models import (
    TensorsInputs,
    GenerationOutputs,
    GenerationTargets,
)

class MPSImageProcessor:
    def __init__(
        self
    ):
        super().__init__()

        processor_name_or_path = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
        self.image_processor = CLIPImageProcessor.from_pretrained(processor_name_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained(processor_name_or_path, trust_remote_code=True)

    def _process_image(self, image):
        if isinstance(image, dict):
            image = image["bytes"]
        if isinstance(image, bytes):
            image = Image.open(io.BytesIO(image))
        if isinstance(image, str):
            image = Image.open( image )
        image = image.convert("RGB")
        pixel_values = self.image_processor(image, return_tensors="pt").pixel_values
        return pixel_values

    def _tokenize(self, caption):
        input_ids = self.tokenizer(
            caption,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).input_ids
        return input_ids

    @classmethod
    @add_default_section_for_init("microsoft/process/mps")
    def from_core_configure(cls, config, **kwargs):
        pass 
    @register_process("microsoft/process/mps/classification")
    def classification(
        self,
        text: str,
        condition: str,
        image: Union[Image.Image, str]
    ):
        """
        Performs classification using text and image inputs.

        Args:
            text (str): The input text.
            image (PIL.Image.Image): The input image.

        Returns:
            GenericOutputs: An object containing the processed inputs.
        """
        text_outputs = self._tokenize(text)
        condition_outputs = self._tokenize(condition)
        pixel_outputs = self._process_image(image).squeeze(0)
        return TensorsInputs({
                "text_inputs": text_outputs,
                "condition_inputs": condition_outputs,
                "image_inputs": pixel_outputs,
            })

