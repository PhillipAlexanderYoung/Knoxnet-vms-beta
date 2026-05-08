from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List

try:
    import torch  # type: ignore
    from transformers import AutoProcessor, LlavaForConditionalGeneration  # type: ignore
    LLAVA_STACK_AVAILABLE = True
except ImportError:
    torch = None  # type: ignore
    AutoProcessor = None  # type: ignore
    LlavaForConditionalGeneration = None  # type: ignore
    LLAVA_STACK_AVAILABLE = False

from .base import DescribeOutput, MatchOutput, RunContext, VisionRunner


def _to_device(batch: Dict[str, torch.Tensor], device: str) -> Dict[str, torch.Tensor]:
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}


def _parse_tags(text: str) -> List[str]:
    matches = re.findall(r"[a-zA-Z][a-zA-Z0-9_-]+", text.lower())
    seen = []
    for token in matches:
        if token not in seen and len(token) > 2:
            seen.append(token)
    return seen[:12]


def _parse_boolean_lines(text: str, criteria: List[str]) -> Dict[str, bool]:
    verdict = {}
    for criterion in criteria:
        verdict[criterion] = False
    for line in text.splitlines():
        line_lower = line.lower()
        for criterion in criteria:
            key = criterion.lower()
            if key in line_lower:
                verdict[criterion] = "yes" in line_lower or "true" in line_lower
    return verdict


class LlavaRunner(VisionRunner):
    name = "llava"

    def __init__(self, model_dir: Path, device: str = "cpu", max_new_tokens: int = 128) -> None:
        super().__init__(device=device)
        if not LLAVA_STACK_AVAILABLE:
            raise RuntimeError(
                "LLaVA runner unavailable. Install services/vision_local/requirements.gpu.txt "
                "or run scripts/install-local-ai-extras.ps1."
            )
        self.model_dir = Path(model_dir)
        if not self.model_dir.exists():
            raise FileNotFoundError(f"LLaVA model directory {self.model_dir} not found")
        torch_dtype = torch.float16 if device.startswith("cuda") else torch.float32
        self.processor = AutoProcessor.from_pretrained(self.model_dir)
        self.model = LlavaForConditionalGeneration.from_pretrained(
            self.model_dir,
            torch_dtype=torch_dtype,
        )
        self.model.to(device)
        self.max_new_tokens = max_new_tokens

    def describe(self, image: Image.Image, context: RunContext) -> DescribeOutput:
        prompt = context.prompt or "Provide a concise caption for the image and list prominent objects."
        if context.detections:
            prompt += " Detected objects: " + ", ".join(context.detections) + "."
        inputs = self.processor(images=image, text=prompt, return_tensors="pt")
        inputs = _to_device(inputs, self.device)
        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=context.metadata.get("max_new_tokens", self.max_new_tokens),
            do_sample=False,
        )
        caption = self.processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        tags = list(dict.fromkeys(context.detections + _parse_tags(caption)))[:12]
        return DescribeOutput(
            caption=caption,
            tags=tags,
            raw={
                "prompt": prompt,
                "detections": context.detections,
            },
        )

    def match_criteria(self, image: Image.Image, criteria: List[str], context: RunContext) -> MatchOutput:
        criteria_prompt = "\n".join(f"- {item}" for item in criteria)
        prompt = (
            "Evaluate each criterion with a clear YES or NO. "
            "Respond using 'Criterion: YES/NO - brief reason'.\n"
            f"Criteria:\n{criteria_prompt}\n"
        )
        if context.detections:
            prompt += f"Detector hints: {', '.join(context.detections)}.\n"
        inputs = self.processor(images=image, text=prompt, return_tensors="pt")
        inputs = _to_device(inputs, self.device)
        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=context.metadata.get("max_new_tokens", self.max_new_tokens),
            do_sample=False,
        )
        response = self.processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        verdict = _parse_boolean_lines(response, criteria)
        return MatchOutput(
            verdict=verdict,
            reasoning=response,
            raw={"prompt": prompt},
        )

