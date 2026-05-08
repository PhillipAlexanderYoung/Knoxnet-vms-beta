from __future__ import annotations

from pathlib import Path
from typing import List

try:
    import torch  # type: ignore
    from transformers import AutoModel, AutoTokenizer  # type: ignore
    MINICPM_STACK_AVAILABLE = True
except ImportError:
    torch = None  # type: ignore
    AutoModel = None  # type: ignore
    AutoTokenizer = None  # type: ignore
    MINICPM_STACK_AVAILABLE = False

from .base import DescribeOutput, MatchOutput, RunContext, VisionRunner


class MiniCpmRunner(VisionRunner):
    name = "minicpm"

    def __init__(self, model_dir: Path, device: str = "cpu", max_new_tokens: int = 128) -> None:
        super().__init__(device=device)
        if not MINICPM_STACK_AVAILABLE:
            raise RuntimeError(
                "MiniCPM runner unavailable. Install services/vision_local/requirements.gpu.txt "
                "or run scripts/install-local-ai-extras.ps1."
            )
        self.model_dir = Path(model_dir)
        if not self.model_dir.exists():
            raise FileNotFoundError(f"MiniCPM-V directory {self.model_dir} not found")
        torch_dtype = torch.float16 if device.startswith("cuda") else torch.float32
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            self.model_dir,
            trust_remote_code=True,
            torch_dtype=torch_dtype,
        ).to(device)
        self.model.eval()
        self.max_new_tokens = max_new_tokens

    def describe(self, image: Image.Image, context: RunContext) -> DescribeOutput:
        prompt = context.prompt or "Describe the image in detail and list key objects."
        if context.detections:
            prompt += " Detected objects: " + ", ".join(context.detections) + "."
        response, _ = self.model.chat(
            self.tokenizer,
            image=image.convert("RGB"),
            question=prompt,
            history=[],
            max_new_tokens=context.metadata.get("max_new_tokens", self.max_new_tokens),
        )
        tags = list(dict.fromkeys(context.detections + response.lower().split()))[:12]
        return DescribeOutput(
            caption=response.strip(),
            tags=tags,
            raw={"prompt": prompt, "detections": context.detections},
        )

    def match_criteria(self, image: Image.Image, criteria: List[str], context: RunContext) -> MatchOutput:
        prompt = (
            "Evaluate the image for the following criteria. "
            "Respond with 'criterion - YES/NO - short reason' per line.\n"
        )
        prompt += "\n".join(f"- {criterion}" for criterion in criteria)
        response, _ = self.model.chat(
            self.tokenizer,
            image=image.convert("RGB"),
            question=prompt,
            history=[],
            max_new_tokens=context.metadata.get("max_new_tokens", self.max_new_tokens),
        )
        verdict = {}
        for criterion in criteria:
            verdict[criterion] = criterion.lower() in response.lower() and "yes" in response.lower()
        return MatchOutput(
            verdict=verdict,
            reasoning=response,
            raw={"prompt": prompt},
        )

