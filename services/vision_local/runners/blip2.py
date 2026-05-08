from __future__ import annotations

import re
from pathlib import Path
from typing import List

try:
    import torch  # type: ignore
    from transformers import Blip2ForConditionalGeneration, Blip2Processor  # type: ignore
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    torch = None  # type: ignore
    Blip2ForConditionalGeneration = None  # type: ignore
    Blip2Processor = None  # type: ignore
    TRANSFORMERS_AVAILABLE = False

from .base import DescribeOutput, MatchOutput, RunContext, VisionRunner


def _normalize_tokens(text: str) -> List[str]:
    tokens = re.findall(r"[a-zA-Z][a-zA-Z0-9_-]+", text.lower())
    return tokens


def _merge_tags(caption: str, detections: List[str]) -> List[str]:
    tokens = list(dict.fromkeys(detections))
    caption_tokens = _normalize_tokens(caption)
    for token in caption_tokens:
        if token not in tokens and len(token) > 2:
            tokens.append(token)
    return tokens[:12]


class Blip2Runner(VisionRunner):
    name = "blip2"

    def __init__(self, model_dir: Path, device: str = "cpu", max_new_tokens: int = 64) -> None:
        super().__init__(device=device)
        if not TRANSFORMERS_AVAILABLE:
            raise RuntimeError(
                "BLIP-2 runner unavailable. Install services/vision_local/requirements.gpu.txt "
                "or run scripts/install-local-ai-extras.ps1."
            )
        self.model_dir = Path(model_dir)
        if not self.model_dir.exists():
            raise FileNotFoundError(f"BLIP-2 model directory {self.model_dir} not found")
        torch_dtype = torch.float16 if device.startswith("cuda") else torch.float32
        self.processor = Blip2Processor.from_pretrained(self.model_dir)
        self.model = Blip2ForConditionalGeneration.from_pretrained(
            self.model_dir,
            torch_dtype=torch_dtype,
        )
        self.model.to(device)
        self.max_new_tokens = max_new_tokens

    def describe(self, image: Image.Image, context: RunContext) -> DescribeOutput:
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=context.metadata.get("max_new_tokens", self.max_new_tokens),
        )
        caption = self.processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        tags = _merge_tags(caption, context.detections)
        return DescribeOutput(
            caption=caption,
            tags=tags,
            raw={
                "prompt": context.prompt,
                "detections": context.detections,
            },
        )

    def match_criteria(self, image: Image.Image, criteria: List[str], context: RunContext) -> MatchOutput:
        description = self.describe(image, context)
        verdict = {}
        caption_tokens = set(_normalize_tokens(description.caption))
        detection_tokens = {token.lower() for token in context.detections}
        for criterion in criteria:
            crit_tokens = set(_normalize_tokens(criterion))
            verdict[criterion] = bool(crit_tokens & (caption_tokens | detection_tokens))
        reasoning = "Evaluated criteria against BLIP-2 caption tokens and detector hints."
        return MatchOutput(
            verdict=verdict,
            reasoning=reasoning,
            raw={"caption": description.caption, "tags": description.tags},
        )

