# Copyright (c) MICROSOFT.
# Licensed under the MIT License.

"""
Lightweight Python wrapper for the Spaces API endpoints.

Usage:
    from unitorch_microsoft.apps.skills.spaces import SpacesClient

    client = SpacesClient("http://127.0.0.1:5000")

    # Image classification
    categories = client.classify_image("photo.jpg", topk=5)

    # Image quality scores
    blurry  = client.score_blurry("photo.jpg")
    bg      = client.score_background("photo.jpg")
    wm      = client.score_watermark("photo.jpg")
    crop    = client.score_bad_crop("photo.jpg")
    pad     = client.score_bad_padding("photo.jpg")

    # Object detection (returns image bytes with boxes drawn)
    img_bytes = client.detect_objects_basnet("photo.jpg", threshold=0.1)
    img_bytes = client.detect_objects_detr("photo.jpg", threshold=0.1)

    # Image generation / editing
    img_bytes = client.generate_image_gpt("a red house", size="1024x1024")
    img_bytes = client.edit_image_gpt(["logo.png", "bg.png"], prompt="put logo on top-right", size="1536x1024")
    img_bytes = client.generate_image_gemini("a red house")
    img_bytes = client.edit_image_gemini(["logo.png", "bg.png"], prompt="put logo on top-right")
"""

import httpx
from pathlib import Path
from typing import List, Optional


class SpacesClient:
    def __init__(self, base_url: str = "http://127.0.0.1:5000", timeout: int = 60):
        self._base = base_url.rstrip("/")
        self._timeout = timeout

    def _url(self, path: str) -> str:
        return f"{self._base}{path}"

    # ------------------------------------------------------------------
    # Image classification (Google category)
    # ------------------------------------------------------------------

    def classify_image(self, image_path: str, topk: int = 5) -> List[dict]:
        """Return top-k Google categories for an image.

        Returns a list of {"category": str, "score": float}.
        """
        with open(image_path, "rb") as f:
            resp = httpx.post(
                self._url(f"/microsoft/apps/spaces/picasso/swin/googlecate/generate?topk={topk}"),
                files={"image": (Path(image_path).name, f, "image/png")},
                timeout=self._timeout,
            )
        resp.raise_for_status()
        return resp.json()

    # ------------------------------------------------------------------
    # Image quality scores (Bletchley / SigLIP)
    # ------------------------------------------------------------------

    def score_blurry(self, image_path: str) -> float:
        """Return blurriness score (0–1, higher = more blurry)."""
        with open(image_path, "rb") as f:
            resp = httpx.post(
                self._url("/microsoft/apps/spaces/picasso/bletchley/v1/generate1"),
                files={"image": (Path(image_path).name, f, "image/png")},
                timeout=self._timeout,
            )
        resp.raise_for_status()
        return resp.json().get("Blurry", 0.0)

    def score_background(self, image_path: str) -> dict:
        """Return background complexity scores {"Complex", "Simple", "White"}."""
        with open(image_path, "rb") as f:
            resp = httpx.post(
                self._url("/microsoft/apps/spaces/picasso/bletchley/v1/generate2"),
                files={"image": (Path(image_path).name, f, "image/png")},
                timeout=self._timeout,
            )
        resp.raise_for_status()
        return resp.json()

    def score_watermark(self, image_path: str) -> float:
        """Return watermark probability (0–1)."""
        with open(image_path, "rb") as f:
            resp = httpx.post(
                self._url("/microsoft/apps/spaces/picasso/bletchley/v3/generate1"),
                files={"image": (Path(image_path).name, f, "image/png")},
                timeout=self._timeout,
            )
        resp.raise_for_status()
        return resp.json().get("Watermark", 0.0)

    def score_bad_crop(self, image_path: str) -> float:
        """Return bad-crop score (0–1, higher = worse crop)."""
        with open(image_path, "rb") as f:
            resp = httpx.post(
                self._url("/microsoft/apps/spaces/picasso/siglip2/generate1"),
                files={"image": (Path(image_path).name, f, "image/png")},
                timeout=self._timeout,
            )
        resp.raise_for_status()
        return resp.json().get("Bad Cropped", 0.0)

    def score_bad_padding(self, image_path: str) -> float:
        """Return bad-padding score (0–1, higher = worse padding)."""
        with open(image_path, "rb") as f:
            resp = httpx.post(
                self._url("/microsoft/apps/spaces/picasso/siglip2/generate2"),
                files={"image": (Path(image_path).name, f, "image/png")},
                timeout=self._timeout,
            )
        resp.raise_for_status()
        return resp.json().get("Bad Padding", 0.0)

    # ------------------------------------------------------------------
    # Object detection (returns annotated image bytes)
    # ------------------------------------------------------------------

    def detect_objects_basnet(self, image_path: str, threshold: float = 0.1) -> bytes:
        """Return image bytes with BASNet-detected bounding boxes drawn."""
        with open(image_path, "rb") as f:
            resp = httpx.post(
                self._url(f"/microsoft/apps/spaces/picasso/basnet/generate1?threshold={threshold}"),
                files={"image": (Path(image_path).name, f, "image/png")},
                timeout=self._timeout,
            )
        resp.raise_for_status()
        return resp.content

    def detect_objects_detr(self, image_path: str, threshold: float = 0.1) -> bytes:
        """Return image bytes with DETR-detected bounding boxes drawn."""
        with open(image_path, "rb") as f:
            resp = httpx.post(
                self._url(f"/microsoft/apps/spaces/picasso/detr/generate1?threshold={threshold}"),
                files={"image": (Path(image_path).name, f, "image/png")},
                timeout=self._timeout,
            )
        resp.raise_for_status()
        return resp.content

    # ------------------------------------------------------------------
    # Image generation / editing — GPT Image 1.5
    # ------------------------------------------------------------------

    def generate_image_gpt(
        self,
        prompt: str,
        size: str = "1024x1024",
        background: str = "transparent",
    ) -> bytes:
        """Generate an image from a text prompt using GPT Image 1.5.

        size: "1024x1024" | "1536x1024" | "1024x1536"
        Returns raw image bytes (PNG).
        """
        resp = httpx.post(
            self._url(
                f"/microsoft/apps/spaces/gpt/image-15/generate"
                f"?prompt={httpx.URL('', params={'prompt': prompt}).params['prompt']}"
                f"&size={size}&background={background}"
            ),
            timeout=self._timeout,
        )
        resp.raise_for_status()
        return resp.content

    def edit_image_gpt(
        self,
        image_paths: List[str],
        prompt: str,
        size: str = "1536x1024",
        background: str = "transparent",
        mask_path: Optional[str] = None,
    ) -> bytes:
        """Edit one or more images with a text prompt using GPT Image 1.5.

        Returns raw image bytes (PNG).
        """
        files = [
            ("images", (Path(p).name, open(p, "rb"), "image/png"))
            for p in image_paths
        ]
        if mask_path:
            files.append(("mask", (Path(mask_path).name, open(mask_path, "rb"), "image/png")))
        try:
            resp = httpx.post(
                self._url(
                    f"/microsoft/apps/spaces/gpt/image-15/edit"
                    f"?prompt={httpx.URL('', params={'prompt': prompt}).params['prompt']}"
                    f"&size={size}&background={background}"
                ),
                files=files,
                timeout=self._timeout,
            )
            resp.raise_for_status()
            return resp.content
        finally:
            for _, (_, fobj, _) in files:
                try:
                    fobj.close()
                except Exception:
                    pass

    # ------------------------------------------------------------------
    # Image generation / editing — Gemini
    # ------------------------------------------------------------------

    def generate_image_gemini(
        self,
        prompt: str,
        size: str = "1024x1024",
        background: str = "transparent",
    ) -> bytes:
        """Generate an image from a text prompt using Gemini.

        Returns raw image bytes (PNG).
        """
        resp = httpx.post(
            self._url(
                f"/microsoft/apps/spaces/gemini/image/generate"
                f"?prompt={httpx.URL('', params={'prompt': prompt}).params['prompt']}"
                f"&size={size}&background={background}"
            ),
            timeout=self._timeout,
        )
        resp.raise_for_status()
        return resp.content

    def edit_image_gemini(
        self,
        image_paths: List[str],
        prompt: str,
        size: str = "1536x1024",
        background: str = "transparent",
    ) -> bytes:
        """Edit one or more images with a text prompt using Gemini.

        Returns raw image bytes (PNG).
        """
        files = [
            ("images", (Path(p).name, open(p, "rb"), "image/png"))
            for p in image_paths
        ]
        try:
            resp = httpx.post(
                self._url(
                    f"/microsoft/apps/spaces/gemini/image/edit"
                    f"?prompt={httpx.URL('', params={'prompt': prompt}).params['prompt']}"
                    f"&size={size}&background={background}"
                ),
                files=files,
                timeout=self._timeout,
            )
            resp.raise_for_status()
            return resp.content
        finally:
            for _, (_, fobj, _) in files:
                try:
                    fobj.close()
                except Exception:
                    pass
