from torch import Tensor, nn
from transformers import CLIPModel, CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5Tokenizer


class HFEmbedder(nn.Module):
    def __init__(self, version: str, max_length: int, **hf_kwargs):
        super().__init__()
        self.is_clip = version.startswith("openai")
        self.max_length = max_length
        self.output_key = "pooler_output" if self.is_clip else "last_hidden_state"

        if self.is_clip:
            self.tokenizer: CLIPTokenizer = CLIPTokenizer.from_pretrained(version, max_length=max_length)
            # Some CLIP checkpoints use a full CLIP config (CLIPConfig) that wraps
            # text and vision configs. Loading the full CLIPModel and extracting
            # its text_model avoids passing a CLIPConfig into CLIPTextModel
            # (which expects CLIPTextConfig) and prevents the AttributeError for
            # missing `hidden_size` on CLIPConfig.
            clip_full: CLIPModel = CLIPModel.from_pretrained(version, **hf_kwargs)
            self.hf_module: CLIPTextModel = clip_full.text_model
        else:
            self.tokenizer: T5Tokenizer = T5Tokenizer.from_pretrained(version, max_length=max_length)
            self.hf_module: T5EncoderModel = T5EncoderModel.from_pretrained(version, **hf_kwargs)

        self.hf_module = self.hf_module.eval().requires_grad_(False)

    def forward(self, text: list[str]) -> Tensor:
        batch_encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_length=False,
            return_overflowing_tokens=False,
            padding="max_length",
            return_tensors="pt",
        )
        device = getattr(self.hf_module, "device", None)
        if device is None:
            try:
                device = next(self.hf_module.parameters()).device
            except StopIteration:
                device = torch.device("cpu")

        outputs = self.hf_module(
            input_ids=batch_encoding["input_ids"].to(device),
            attention_mask=None,
            output_hidden_states=False,
        )
        return outputs[self.output_key].bfloat16()