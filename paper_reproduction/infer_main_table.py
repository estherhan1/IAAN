#!/usr/bin/env python3
"""Run baseline and IAAN on VoxParadox with each model's generation API.

The model adapters follow the public inference APIs and implement the paper's
signed real-minus-noise encoder-FFN intervention. Exact score reproduction
requires the original model revisions, prompts, preprocessing and RNG seeds;
this code has NOT been GPU-validated against the published Table I.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import tempfile
import sys

import numpy as np
import torch
from tqdm import tqdm

from iaan_core import select_topk, amplify_fc2_inputs


def question(sample):
    choices = " ".join(
        f"({letter}): {sample['choice_'+letter.lower()]}."
        for letter in "ABCD" if sample.get("choice_"+letter.lower())
    )
    return f"{sample['question']} {choices}"


def audio_read(path):
    import librosa
    samples, _ = librosa.load(str(path), sr=16000, mono=True)
    return samples.astype("float32")


def gaussian_noise(wav: np.ndarray, index: int, model: str):
    """Match the Gaussian-reference convention in existing project code."""
    if model == "qwen":
        gen = torch.Generator(device="cpu").manual_seed(2000 + index)
        noise = torch.randn(wav.shape, generator=gen).float()
        std = torch.tensor(wav).std(unbiased=True).clamp_min(1e-6)
        noise = noise / noise.std(unbiased=True).clamp_min(1e-6) * std
        return noise.numpy()
    rng = np.random.default_rng(1000 + index)
    return rng.standard_normal(len(wav)).astype("float32") * (float(wav.std()) or 1.0)


@contextmanager
def as_audio_path(wav: np.ndarray):
    import soundfile as sf
    with tempfile.TemporaryDirectory(prefix="iaan_audio_") as tmp:
        path = Path(tmp) / "input.wav"
        sf.write(str(path), wav, 16000)
        yield path


def capture_answer(encoder_layers, generate):
    """Capture mean fc2 input at every encoder layer during one model call."""
    values = {}
    handles = []

    def register(index, layer):
        def hook(_layer, inputs):
            x = inputs[0].detach().float()
            values[index] = x.reshape(-1, x.shape[-1]).mean(dim=0).cpu()
        handles.append(layer.register_forward_pre_hook(hook))

    for i, layer in enumerate(encoder_layers):
        register(i, layer)
    try:
        response = generate()
    finally:
        for handle in handles:
            handle.remove()
    missing = set(range(len(encoder_layers))) - set(values)
    if missing:
        raise RuntimeError(f"encoder FFN hooks were not invoked: {sorted(missing)}")
    return response, [values[i] for i in range(len(encoder_layers))]


class AF3:
    def __init__(self, checkpoint, max_new_tokens):
        from transformers import AutoProcessor, AudioFlamingo3ForConditionalGeneration
        self.processor = AutoProcessor.from_pretrained(checkpoint)
        self.model = AudioFlamingo3ForConditionalGeneration.from_pretrained(
            checkpoint, torch_dtype=torch.bfloat16, device_map="cuda"
        ).eval()
        self.layers = [layer.fc2 for layer in self.model.audio_tower.layers]
        self.max_new_tokens = max_new_tokens

    @torch.inference_mode()
    def generate(self, wav, sample):
        conv = [[{"role": "user", "content": [
            {"type": "text", "text": "\n" + question(sample)},
            {"type": "audio", "audio": wav},
        ]}]]
        inputs = self.processor.apply_chat_template(
            conv, tokenize=True, add_generation_prompt=True,
            return_dict=True, sampling_rate=16000
        ).to(self.model.device)
        for key, val in list(inputs.items()):
            if torch.is_tensor(val) and val.is_floating_point():
                inputs[key] = val.to(self.model.dtype)
        result = self.model.generate(
            **inputs, do_sample=False, max_new_tokens=self.max_new_tokens
        )
        prefix = inputs["input_ids"].shape[1]
        seq = result[0] if result.ndim == 2 else result
        return self.processor.tokenizer.decode(seq[prefix:], skip_special_tokens=True).strip()


class Qwen:
    def __init__(self, checkpoint, max_new_tokens):
        from transformers import (Qwen2_5OmniForConditionalGeneration,
                                  Qwen2_5OmniProcessor, Qwen2_5OmniConfig)
        from qwen_omni_utils import process_mm_info
        self.process_mm_info = process_mm_info
        self.processor = Qwen2_5OmniProcessor.from_pretrained(checkpoint)
        config = Qwen2_5OmniConfig.from_pretrained(checkpoint)
        config.enable_audio_output = False
        self.model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            checkpoint, config=config, torch_dtype=torch.bfloat16,
            device_map="cuda"
        ).eval()
        self.layers = [layer.fc2 for layer in self.model.thinker.audio_tower.layers]
        self.max_new_tokens = max_new_tokens

    @torch.inference_mode()
    def generate(self, wav, sample):
        with as_audio_path(wav) as path:
            conv = [
                {"role": "system", "content": [{"type": "text",
                  "text": "Listen carefully to the audio and answer the question. Respond with only the letter of the correct option."}]},
                {"role": "user", "content": [
                    {"type": "audio", "audio": str(path)},
                    {"type": "text", "text": question(sample) + "\nAnswer with only A, B, C, or D."},
                ]},
            ]
            prompt = self.processor.apply_chat_template(
                conv, add_generation_prompt=True, tokenize=False
            )
            audios, images, videos = self.process_mm_info(
                conv, use_audio_in_video=False
            )
            inputs = self.processor(
                text=prompt, audio=audios, images=images, videos=videos,
                return_tensors="pt", padding=True, use_audio_in_video=False
            ).to(self.model.device, self.model.dtype)
            prefix = inputs.input_ids.shape[1]
            gen = self.model.generate(
                **inputs, use_audio_in_video=False, return_audio=False,
                thinker_max_new_tokens=self.max_new_tokens, thinker_do_sample=False
            )
            seq = gen[0] if isinstance(gen, (tuple, list)) else gen
            seq = seq[0] if seq.ndim == 2 else seq
            return self.processor.tokenizer.decode(
                seq[prefix:], skip_special_tokens=True
            ).strip()


class Kimi:
    def __init__(self, checkpoint, max_new_tokens):
        from kimia_infer.api.kimia import KimiAudio
        self.model = KimiAudio(model_path=checkpoint, load_detokenizer=False)
        enc = self.model.prompt_manager.whisper_model.speech_encoder
        self.layers = [layer.fc2 for layer in enc.layers]
        self.max_new_tokens = max_new_tokens

    @torch.inference_mode()
    def generate(self, wav, sample):
        with as_audio_path(wav) as path:
            msgs = [
                {"role": "user", "message_type": "text",
                 "content": question(sample) +
                     "\nReply with only the letter of the correct option: (A), (B), (C), or (D)."},
                {"role": "user", "message_type": "audio", "content": str(path)},
            ]
            _, answer = self.model.generate(
                msgs, output_type="text",
                text_temperature=0.0, text_top_k=5
            )
            return str(answer).strip()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", choices=["af3", "qwen", "kimi"], required=True)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--audio-base", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--k", type=int, required=True)
    ap.add_argument("--gain", type=float, required=True)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    args = ap.parse_args()

    checkpoints = {
        "af3": "nvidia/audio-flamingo-3-hf",
        "qwen": "Qwen/Qwen2.5-Omni-7B",
        "kimi": "moonshotai/Kimi-Audio-7B-Instruct",
    }
    if not torch.cuda.is_available():
        ap.error("A CUDA GPU is required to run the 7B audio-language models")
    if args.k < 1 or args.gain <= 1:
        ap.error("Use positive K and gain > 1")
    klass = {"af3": AF3, "qwen": Qwen, "kimi": Kimi}[args.model]
    model = klass(args.checkpoint or checkpoints[args.model], args.max_new_tokens)
    data = json.loads(args.data.read_text(encoding="utf-8"))
    args.out.mkdir(parents=True, exist_ok=True)
    paths = {mode: args.out / f"{mode}_predictions.jsonl"
             for mode in ("baseline", "iaan")}

    with paths["baseline"].open("w", encoding="utf-8") as fb, \
         paths["iaan"].open("w", encoding="utf-8") as fi:
        for j, item in enumerate(tqdm(data, desc=args.model)):
            audio = (args.audio_base / item["audio_path"]).resolve()
            if not audio.is_file():
                raise FileNotFoundError(f"Missing waveform for {item['id']}: {audio}")
            index = int(item.get("_iaan_index", j))
            wav = audio_read(audio)
            baseline, real_means = capture_answer(
                model.layers, lambda: model.generate(wav, item)
            )
            ref = gaussian_noise(wav, index, args.model)
            _, ref_means = capture_answer(
                model.layers, lambda: model.generate(ref, item)
            )
            selected = select_topk(real_means, ref_means, args.k)
            with amplify_fc2_inputs(model.layers, selected, args.gain):
                steered = model.generate(wav, item)
            fb.write(json.dumps({"id": item["id"], "response": baseline},
                                ensure_ascii=False) + "\n")
            fi.write(json.dumps({"id": item["id"], "response": steered},
                                ensure_ascii=False) + "\n")
            fb.flush()
            fi.flush()
    print("Saved predictions:", *paths.values())


if __name__ == "__main__":
    main()
