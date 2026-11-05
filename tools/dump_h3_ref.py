#!/usr/bin/env python3
# Reference tensors for mako-video's MiniMax-H3 tests, from the Diffusers implementation at a small canvas:
#   .venv/bin/python tools/dump_h3_ref.py ~/models/minimax-h3/FL2VA <out dir> [--stage text|dit|vae|fl2va|ref2va|all]
#                                        [--ref2va-model ~/models/minimax-h3/Ref2VA]
# The fl2va stage (files prefixed fl_) conditions on two synthetic keyframes: their resize onto the canvas, the vision
# tower and text encoder over the keyframe presentation, the VAE encoder's posteriors, and two transformer steps with
# the conditioning rows in the sequence.
# Writes manifest.json and raw float32 files (row-major, shapes in the manifest). The text encoder runs in fp32 on the
# CPU (each layer upcast for its forward), the transformer on MPS with its BF16 weights upcast to fp32 per module (what
# mako computes: fp32 activations, BF16 weights), the VAEs in fp32 on MPS. Every stage reads the earlier ones' files,
# so a stage can be rerun alone.
import gc
import json
import math
import os
import sys
import time

import numpy as np
import torch

PROMPT = ("A red fox trots through fresh snow in a pine forest at sunrise, its breath visible in the cold air; "
          "soft crunching footsteps and distant birdsong.")
SIZE = 256         # a 256x256 canvas: 16x16 latents, 8x8 = 64 transformer rows a latent frame
NUM_FRAMES = 124   # the shortest the model generates (5.17 s): 37 latent frames, 207 audio latents a channel
STEPS = 3          # sigma grid points, terminal 0 included: 2 model evaluations
SEED = 42
TEXT_LAYER = 50
PROBE_BLOCKS = (0, 1, 24, 49)
TOKEN_CASES = ["", "Hello, world!", PROMPT, "<d>spoken line</d> and <|lyrics_start|>la la<|lyrics_end|>",
               "日本語のテスト 123 456", "émigré café ÄÖÜ ß", "tabs\tand\nnewlines  double  spaces",
               "<|vision_start|><|image_pad|><|vision_end|> text after", "😀 emoji 🎬"]

model_dir, out = sys.argv[1], sys.argv[2]
stage = sys.argv[4] if len(sys.argv) > 4 and sys.argv[3] == "--stage" else "all"
ref2va_dir = sys.argv[sys.argv.index("--ref2va-model") + 1] if "--ref2va-model" in sys.argv else os.path.join(
    os.path.dirname(model_dir.rstrip("/")), "Ref2VA")
REF_SHORT_EDGE = 1024  # the released pipeline encodes image references at a 2048 short edge; the test case at 1024
os.makedirs(out, exist_ok=True)
manifest_path = os.path.join(out, "manifest.json")
manifest = json.load(open(manifest_path)) if os.path.exists(manifest_path) else {"tensors": {}}


def save(name, t):
    t = t.detach().float().contiguous().cpu()
    t.numpy().tofile(os.path.join(out, name + ".f32"))
    manifest["tensors"][name] = list(t.shape)


def save_f64(name, t):
    t = t.detach().double().contiguous().cpu()
    t.numpy().tofile(os.path.join(out, name + ".f64"))
    manifest["tensors"][name] = list(t.shape)


def load(name):
    return torch.from_numpy(np.fromfile(os.path.join(out, name + ".f32"), dtype=np.float32).reshape(
        manifest["tensors"][name]))


def write_manifest():
    json.dump(manifest, open(manifest_path, "w"), indent=1)


def log(*a):
    print(time.strftime("[%H:%M:%S]"), *a, flush=True)


# ---------------------------------------------------------------------------------------------------- text encoder
if stage in ("text", "all"):
    from transformers import Qwen2TokenizerFast, Qwen3VLForConditionalGeneration

    tok = Qwen2TokenizerFast.from_pretrained(os.path.join(model_dir, "tokenizer"))
    ids = tok(PROMPT, add_special_tokens=False)["input_ids"]
    manifest["prompt"] = PROMPT
    manifest["ids"] = ids
    manifest["token_cases"] = [{"text": t, "ids": tok(t, add_special_tokens=False)["input_ids"]} for t in TOKEN_CASES]
    log(f"{len(ids)} prompt tokens")

    log("loading the text encoder (BF16, CPU)")
    te = Qwen3VLForConditionalGeneration.from_pretrained(os.path.join(model_dir, "text_encoder"), dtype=torch.bfloat16,
                                                         device_map="cpu")
    lm = te.model.language_model
    # hidden_states[50] is the input of layer 50 (the output of 50 layers, before any norm): layers past 50 are not
    # needed, but the last kept layer's output would otherwise be returned normed, so keep 51.
    lm.layers = lm.layers[:TEXT_LAYER + 1]
    lm.config.num_hidden_layers = TEXT_LAYER + 1
    te.config.text_config.num_hidden_layers = TEXT_LAYER + 1
    gc.collect()
    lm.embed_tokens.float()
    lm.norm.float()
    lm.rotary_emb.float()

    def upcast_pre(module, args, kwargs):
        module.float()
        return args, kwargs

    def downcast_post(module, args, output):
        module.to(torch.bfloat16)
        return output

    for layer in lm.layers:
        layer.register_forward_pre_hook(upcast_pre, with_kwargs=True)
        layer.register_forward_hook(downcast_post)
    input_ids = torch.tensor([ids], dtype=torch.long)
    log("encoding in fp32")
    with torch.no_grad():
        o = te.model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids),
                     mm_token_type_ids=torch.zeros_like(input_ids), use_cache=False, output_hidden_states=True)
    for i in (0, 1, TEXT_LAYER):
        save(f"text_hidden{i}", o.hidden_states[i][0])
    emb = o.hidden_states[TEXT_LAYER][0]
    save("text_embeds", emb)
    save("text_embeds_bf16", emb.to(torch.bfloat16))
    log("text_embeds", tuple(emb.shape), "rms", float(emb.float().pow(2).mean().sqrt()))
    del te, lm, o
    gc.collect()
    write_manifest()

# ----------------------------------------------------------------------------------------------------- transformer
def load_dit(dev, tdir=None):
    """The transformer on MPS: BF16 shards, every BF16 leaf upcast to fp32 for its forward (its float inputs too)."""
    from safetensors.torch import load_file
    from diffusers import MiniMaxH3Transformer3DModel
    from diffusers.loaders.single_file_utils import convert_minimax_h3_transformer_checkpoint_to_diffusers

    tdir = tdir or os.path.join(model_dir, "transformer")
    log(f"loading the transformer (BF16 shards onto MPS) from {tdir}")
    sd = {}
    for f in sorted(f for f in os.listdir(tdir) if f.endswith(".safetensors")):
        sd.update(load_file(os.path.join(tdir, f), device="mps"))
    config = json.load(open(os.path.join(tdir, "config.json")))
    sd = convert_minimax_h3_transformer_checkpoint_to_diffusers(sd, config, qkv_layout="interleaved")
    with torch.device("meta"):
        dit = MiniMaxH3Transformer3DModel()
    missing, unexpected = dit.load_state_dict(sd, strict=False, assign=True)
    missing = [k for k in missing if not k.endswith("rope.inv_freq")]
    if missing or unexpected:
        raise RuntimeError(f"state dict mismatch: missing {missing[:5]}, unexpected {unexpected[:5]}")
    dit.rope.inv_freq = dit.rope.inv_freq.to(dev) if dit.rope.inv_freq.device.type != "meta" else (
        1.0 / (10000.0 ** (torch.arange(0, 32, 2, dtype=torch.float32, device=dev) / 32)))
    dit.eval()
    del sd
    gc.collect()
    dtypes = {}
    for n, p in dit.named_parameters():
        dtypes[str(p.dtype)] = dtypes.get(str(p.dtype), 0) + p.numel()
    log("parameters by dtype", dtypes)

    # The reference's own rounding of silu(temb) to the AdaLN projection's dtype stays, since it reads the dtype
    # before the projection runs.
    def pre(module, args, kwargs):
        module.float()
        args = tuple(a.float() if torch.is_tensor(a) and a.is_floating_point() else a for a in args)
        kwargs = {k: (v.float() if torch.is_tensor(v) and v.is_floating_point() else v) for k, v in kwargs.items()}
        return args, kwargs

    def post(module, args, output):
        module.to(torch.bfloat16)
        return output

    for name, m in dit.named_modules():
        ps = list(m.parameters(recurse=False))
        if ps and all(p.dtype == torch.bfloat16 for p in ps):
            m.register_forward_pre_hook(pre, with_kwargs=True)
            m.register_forward_hook(post)
    return dit


def run_dit(prefix, text_embeds, text_tags, anchors, condition_latents):
    """Two steps of the transformer over [text | conditions | audio | video] from the seed's noise; the layout,
    schedule, noise, per-step rows, probes and velocities saved under `prefix`. condition_latents: the normalized
    [1][24][1][h][w] latents of each keyframe, noised to t = 0.999 with the request's generator as the pipeline does."""
    from diffusers import MiniMaxH3Scheduler
    from diffusers.modular_pipelines.minimax_h3.before_denoise import (MiniMaxH3PrepareLayoutStep,
                                                                       MiniMaxH3SetTimestepsStep,
                                                                       patchify_video_latents)
    from diffusers.modular_pipelines.minimax_h3.modular_pipeline import (audio_latent_num_frames,
                                                                         video_latent_num_frames)

    dev = torch.device("mps")
    T = text_embeds.shape[0]
    # The transformer's text input is the conditioner's own dtype (BF16) in the reference.
    text_embeds = text_embeds.to(dev, torch.bfloat16)

    # The layout: [text | conditions | audio (channel-major) | video (frame-major)], its fp64 rotary grid, tags.
    lat_f = video_latent_num_frames(NUM_FRAMES, 17, 5)
    lat_h = lat_w = SIZE // 16
    n_audio = audio_latent_num_frames(NUM_FRAMES)
    pos, tags, vidx, aidx, tidx, n_cond_v, n_cond_a = MiniMaxH3PrepareLayoutStep.build_packed_sequence(
        torch.tensor(text_tags, dtype=torch.long), lat_f, lat_h, lat_w, n_audio, (1, 2, 2), 2, 2, 0, anchors)
    S = pos.shape[0]
    manifest.update({"size": SIZE, "num_frames": NUM_FRAMES, "latent_frames": lat_f, "latent_height": lat_h,
                     "latent_width": lat_w, "num_audio_latents": n_audio, "steps": STEPS, "seed": SEED,
                     "text_layer": TEXT_LAYER})
    manifest.update({prefix + "text_rows": T, prefix + "seq_len": S, prefix + "condition_rows": n_cond_v,
                     prefix + "anchors": list(anchors)})
    save_f64(prefix + "position_ids", pos)
    manifest[prefix + "token_tags"] = tags.tolist()
    log(f"sequence {S} rows: {T} text, {n_cond_v} conditioning, {aidx.numel()} audio, {vidx.numel() - n_cond_v} video")

    sched_v, sched_a = MiniMaxH3Scheduler(shift=12.0), MiniMaxH3Scheduler(shift=3.0)
    sched_v.set_timesteps(STEPS)
    sched_a.set_timesteps(STEPS)
    manifest["video_sigmas"] = sched_v.sigmas.tolist()
    manifest["audio_sigmas"] = sched_a.sigmas.tolist()
    manifest["video_timesteps"] = sched_v.timesteps.tolist()
    manifest["audio_timesteps"] = sched_a.timesteps.tolist()
    plan = [MiniMaxH3SetTimestepsStep.build_row_timesteps(vidx, aidx, n_cond_v, n_cond_a, T, float(tv), float(ta),
                                                          max(float(tv), 0.999), 1.0)
            for tv, ta in zip(sched_v.timesteps, sched_a.timesteps)]
    manifest[prefix + "step_timesteps"] = [u.tolist() for u, _ in plan]
    manifest[prefix + "step_timestep_indices"] = [i.tolist() for _, i in plan]

    g = torch.Generator().manual_seed(SEED)
    # The conditioning rows first (one draw per keyframe), then the generated rows' noise.
    cond_rows = []
    for i, z in enumerate(condition_latents):
        noise = torch.randn(z.shape, generator=g, dtype=torch.float32)
        save(f"{prefix}cond_noise{i}", noise)
        noised = 0.999 * z + (1.0 - 0.999) * noise  # the scheduler's scale_noise with a Python float timestep
        cond_rows.append(patchify_video_latents(noised, (1, 2, 2)))
    video = torch.randn((1, 24, lat_f, lat_h, lat_w), generator=g, dtype=torch.float32)
    video_rows = patchify_video_latents(video, (1, 2, 2))
    audio_rows = torch.randn((n_audio * 2, 32), generator=g, dtype=torch.float32)
    save(prefix + "noise_video_rows", video_rows)
    save(prefix + "noise_audio_rows", audio_rows)
    if cond_rows:
        cond_rows = torch.cat(cond_rows)
        assert cond_rows.shape[0] == n_cond_v
        save(prefix + "condition_rows", cond_rows)
        video_rows = torch.cat([cond_rows, video_rows])

    dit = load_dit(dev)
    probes = {}

    def grab(name):
        def f(module, args, output):
            probes[name] = output
        return f

    def grab_in(name):
        def f(module, args):
            probes[name] = args[0]
        return f

    dit.token_refiner.register_forward_hook(grab("txt"))
    dit.time_embedder.register_forward_hook(grab("temb"))
    dit.transformer_blocks[0].register_forward_pre_hook(grab_in("x0"))
    dit.transformer_blocks[0].adaln_proj.register_forward_hook(grab("mod0"))
    for b in PROBE_BLOCKS:
        dit.transformer_blocks[b].register_forward_hook(grab(f"block{b}"))
    dit.norm_out.register_forward_hook(grab("norm_out"))

    v_rows, a_rows = video_rows.clone(), audio_rows.clone()
    for i, (tv, ta) in enumerate(zip(sched_v.timesteps, sched_a.timesteps)):
        uniq, tsi = plan[i]
        save(f"{prefix}step{i}_video_rows", v_rows)
        save(f"{prefix}step{i}_audio_rows", a_rows)
        t0 = time.time()
        with torch.no_grad():
            pred_v, pred_a = dit(hidden_states=v_rows[None].to(dev), audio_hidden_states=a_rows[None].to(dev),
                                 encoder_hidden_states=text_embeds[None], timestep=uniq.to(dev),
                                 timestep_indices=tsi.to(dev), token_tags=tags.to(dev),
                                 position_ids=pos.to(dev, torch.float32), video_indices=vidx.to(dev),
                                 audio_indices=aidx.to(dev), text_indices=tidx.to(dev), return_dict=False)
        torch.mps.synchronize()
        log(f"step {i}: forward {time.time() - t0:.1f} s, timesteps {uniq.tolist()}")
        save(f"{prefix}s{i}_txt", probes["txt"][0])
        save(f"{prefix}s{i}_temb", probes["temb"])
        save(f"{prefix}s{i}_x0", probes["x0"][0])
        save(f"{prefix}s{i}_mod0", torch.stack(list(probes["mod0"])))  # [6][timesteps * 3][5376]
        for b in PROBE_BLOCKS:
            save(f"{prefix}s{i}_block{b}", probes[f"block{b}"][0])
        save(f"{prefix}s{i}_norm_out", probes["norm_out"][0])
        save(f"{prefix}step{i}_pred_video", pred_v[0])
        save(f"{prefix}step{i}_pred_audio", pred_a[0])
        # The conditioning rows are re-imposed: only the generated rows step.
        v_rows[n_cond_v:] = sched_v.step(pred_v[0, n_cond_v:].float().cpu(), tv, v_rows[n_cond_v:],
                                         return_dict=False)[0]
        a_rows = sched_a.step(pred_a[0].float().cpu(), ta, a_rows, return_dict=False)[0]
    save(prefix + "final_video_rows", v_rows)
    save(prefix + "final_audio_rows", a_rows)
    del dit
    gc.collect()
    torch.mps.empty_cache()
    write_manifest()


if stage in ("dit", "all"):
    ids = manifest["ids"]
    run_dit("", load("text_embeds"), [1] * len(ids), (), [])

# ----------------------------------------------------------------------------------------------------------- fl2va
if stage in ("fl2va", "all"):
    from PIL import Image
    from transformers import Qwen2TokenizerFast, Qwen3VLForConditionalGeneration, Qwen3VLProcessor

    def synth(w, h, seed):
        """A smooth test picture: colour gradients, a disc and a bar, so resizing and encoding have structure."""
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
        rng = np.random.default_rng(seed)
        cx, cy, r = rng.uniform(0.3, 0.7) * w, rng.uniform(0.3, 0.7) * h, 0.2 * min(w, h)
        img = np.stack([xx / w, yy / h, 0.5 + 0.5 * np.sin(6.28 * (xx / w + yy / h))], axis=-1)
        disc = (xx - cx) ** 2 + (yy - cy) ** 2 < r * r
        img[disc] = rng.uniform(0, 1, 3)
        img[int(0.1 * h):int(0.15 * h), int(0.2 * w):int(0.8 * w)] = (0.9, 0.9, 0.1)
        return Image.fromarray(np.clip(img * 255 + 0.5, 0, 255).astype(np.uint8), "RGB")

    first, last = synth(320, 200, 1), synth(200, 320, 2)
    first.save(os.path.join(out, "fl_first.png"))
    last.save(os.path.join(out, "fl_last.png"))
    np.array(first).tofile(os.path.join(out, "fl_first.u8"))  # [h][w][3]
    np.array(last).tofile(os.path.join(out, "fl_last.u8"))
    manifest["fl_first_size"] = list(first.size)
    manifest["fl_last_size"] = list(last.size)
    # Onto the SIZE x SIZE canvas as the pipeline's resize step does: the first keyframe stretched (PIL LANCZOS), the
    # second cover-cropped with the released model's rounding.
    prepared_first = first.resize((SIZE, SIZE), Image.Resampling.LANCZOS)
    scale = max(SIZE / last.size[0], SIZE / last.size[1])
    resized_size = (max(SIZE, round(last.size[0] * scale)), max(SIZE, round(last.size[1] * scale)))
    left, top = max(0, (resized_size[0] - SIZE) // 2), max(0, (resized_size[1] - SIZE) // 2)
    prepared_last = last.resize(resized_size, Image.Resampling.LANCZOS).crop((left, top, left + SIZE, top + SIZE))
    keyframes = [prepared_first, prepared_last]
    for i, k in enumerate(keyframes):
        k.save(os.path.join(out, f"fl_keyframe{i}.png"))
        np.array(k).tofile(os.path.join(out, f"fl_keyframe{i}.u8"))

    # The presentation: a "<Picture i>: " label and a vision block per keyframe, then the prompt; vision rows tagged
    # as video (0), text rows 1.
    tok = Qwen2TokenizerFast.from_pretrained(os.path.join(model_dir, "tokenizer"))
    processor = Qwen3VLProcessor.from_pretrained(os.path.join(model_dir, "processor"))
    vision = processor.image_processor(images=keyframes, return_tensors="pt")
    grid = vision["image_grid_thw"]
    save("fl_pixel_values", vision["pixel_values"])
    manifest["fl_image_grid_thw"] = grid.tolist()
    merge = processor.image_processor.merge_size ** 2
    ids, tags = [], []
    for i in range(len(keyframes)):
        n_tokens = int(grid[i].prod()) // merge
        label = tok(f"<Picture {i + 1}>: ", add_special_tokens=False)["input_ids"]
        block = ([tok.convert_tokens_to_ids("<|vision_start|>")] + [tok.convert_tokens_to_ids("<|image_pad|>")] * n_tokens
                 + [tok.convert_tokens_to_ids("<|vision_end|>")])
        ids += label + block
        tags += [1] * len(label) + [0] * len(block)
    prompt_ids = tok(PROMPT, add_special_tokens=False)["input_ids"]
    ids += prompt_ids
    tags += [1] * len(prompt_ids)
    manifest["fl_ids"], manifest["fl_tags"] = ids, tags
    mm_types = processor.create_mm_token_type_ids([ids])
    manifest["fl_mm_token_type_ids"] = mm_types[0]
    log(f"fl2va presentation: {len(ids)} tokens, grids {grid.tolist()}")

    log("loading the text encoder (BF16, CPU), the vision tower in fp32")
    te = Qwen3VLForConditionalGeneration.from_pretrained(os.path.join(model_dir, "text_encoder"), dtype=torch.bfloat16,
                                                         device_map="cpu")
    lm = te.model.language_model
    lm.layers = lm.layers[:TEXT_LAYER + 1]
    lm.config.num_hidden_layers = TEXT_LAYER + 1
    te.config.text_config.num_hidden_layers = TEXT_LAYER + 1
    gc.collect()
    lm.embed_tokens.float()
    lm.norm.float()
    lm.rotary_emb.float()
    te.model.visual.float()

    def upcast_pre(module, args, kwargs):
        module.float()
        return args, kwargs

    def downcast_post(module, args, output):
        module.to(torch.bfloat16)
        return output

    for layer in lm.layers:
        layer.register_forward_pre_hook(upcast_pre, with_kwargs=True)
        layer.register_forward_hook(downcast_post)
    vis = te.model.visual
    vprobes = {}

    def vgrab(name):
        def f(module, args, output):
            vprobes[name] = output
        return f

    vis.patch_embed.register_forward_hook(vgrab("patch_embed"))
    vis.blocks[0].register_forward_pre_hook(lambda m, a: vprobes.__setitem__("block_in", a[0]))
    vis.blocks[0].register_forward_hook(vgrab("block0"))
    vis.blocks[-1].register_forward_hook(vgrab("block_last"))
    vis.merger.register_forward_hook(vgrab("merger"))
    for i, m in enumerate(vis.deepstack_merger_list):
        m.register_forward_hook(vgrab(f"deepstack{i}"))
    save("fl_vision_inv_freq", vis.rotary_pos_emb.inv_freq)
    input_ids = torch.tensor([ids], dtype=torch.long)
    log("encoding in fp32")
    with torch.no_grad():
        o = te.model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids),
                     mm_token_type_ids=torch.tensor(mm_types, dtype=torch.long), pixel_values=vision["pixel_values"].float(),
                     image_grid_thw=grid, use_cache=False, output_hidden_states=True)
    for name in ("patch_embed", "block_in", "block0", "block_last", "merger"):
        save("fl_vis_" + name, vprobes[name])
    for i in range(len(vis.deepstack_merger_list)):
        save(f"fl_vis_deepstack{i}", vprobes[f"deepstack{i}"])
    for i in (0, 1, TEXT_LAYER):
        save(f"fl_text_hidden{i}", o.hidden_states[i][0])
    fl_text = o.hidden_states[TEXT_LAYER][0]
    save("fl_text_embeds", fl_text)
    log("fl_text_embeds", tuple(fl_text.shape), "rms", float(fl_text.float().pow(2).mean().sqrt()))
    del te, lm, o, vis
    gc.collect()
    write_manifest()

    # The keyframes through the VAE encoder: ImageNet-normalized pixels, the posterior sampled under seed 42, the
    # sample rounded to fp16 and normalized, as the pipeline's encode_vae_condition does.
    sys.path.insert(0, model_dir)
    from video_vae.minimax_h3_video_vae import MiniMaxH3VideoVAE
    from video_vae.vae_module import DiagonalGaussianDistribution

    dev = torch.device("mps")
    vcfg = json.load(open(os.path.join(model_dir, "video_vae", "config.json")))
    lmean = torch.tensor(vcfg["latents_mean"]).view(1, -1, 1, 1, 1)
    lstd = torch.tensor(vcfg["latents_std"]).view(1, -1, 1, 1, 1)
    pm = torch.tensor((0.485, 0.456, 0.406)).view(1, -1, 1, 1, 1)
    ps = torch.tensor((0.229, 0.224, 0.225)).view(1, -1, 1, 1, 1)
    log("loading the video VAE (fp32, MPS)")
    vae = MiniMaxH3VideoVAE.from_pretrained(os.path.join(model_dir, "video_vae")).to(dev)
    cond_latents = []
    for i, k in enumerate(keyframes):
        pixels = torch.from_numpy(np.array(k)).permute(2, 0, 1)[None, :, None].float()
        pixels = (pixels / 255.0 - pm) / ps
        with torch.no_grad():
            moments = vae._adaptive_encode(pixels.to(dev)).float().cpu()  # tiled where the config says so
        post = DiagonalGaussianDistribution(moments)
        noise = torch.randn(post.mean.shape, generator=torch.Generator().manual_seed(42))
        z = post.mean + post.std * noise
        z = z.to(torch.float16).float()
        zn = (z - lmean) / lstd
        save(f"fl_cond_mean{i}", post.mean)
        save(f"fl_cond_std{i}", post.std)
        save(f"fl_cond_sample_noise{i}", noise)
        save(f"fl_cond_latents{i}", zn)
        cond_latents.append(zn)
        log(f"keyframe {i}: moments {tuple(moments.shape)}, std mean {float(post.std.mean()):.4f}")
    del vae
    gc.collect()
    torch.mps.empty_cache()
    write_manifest()

    run_dit("fl_", fl_text, tags, ("first", "last"), cond_latents)

# ----------------------------------------------------------------------------------------------------------- ref2va
if stage in ("ref2va", "all"):
    from PIL import Image
    from transformers import Qwen2TokenizerFast, Qwen3VLForConditionalGeneration, Qwen3VLProcessor
    from diffusers.modular_pipelines.minimax_h3 import (MiniMaxH3AudioReference, MiniMaxH3ImageReference,
                                                        MiniMaxH3VideoReference)
    from diffusers.modular_pipelines.minimax_h3.encoders import MiniMaxH3Ref2VATextEncoderStep
    from diffusers.modular_pipelines.minimax_h3.before_denoise import (MiniMaxH3Ref2VAPrepareLayoutStep,
                                                                       MiniMaxH3SetTimestepsStep,
                                                                       patchify_video_latents)
    from diffusers.modular_pipelines.minimax_h3.modular_pipeline import (audio_latent_num_frames,
                                                                         resolve_canvas_size,
                                                                         video_latent_num_frames)
    from diffusers import MiniMaxH3Scheduler

    # Three synthetic references in this order: a square picture, a one-second stereo tone, and a 22-frame square
    # video with its own 0.92 s soundtrack. The picture goes to a REF_SHORT_EDGE short edge, the video onto the canvas
    # its own aspect ratio resolves to (768x768), the soundtracks stay at 32 kHz.
    rng = np.random.default_rng(3)
    yy, xx = np.mgrid[0:400, 0:400].astype(np.float64)
    pic = np.stack([xx / 400, (yy / 400) ** 2, 0.5 + 0.5 * np.cos(9.0 * xx / 400 + 5.0 * yy / 400)], axis=-1)
    pic[((xx - 260) ** 2 + (yy - 150) ** 2) < 60 ** 2] = (0.95, 0.2, 0.1)
    ref_image = Image.fromarray(np.clip(pic * 255 + 0.5, 0, 255).astype(np.uint8), "RGB")
    frames = []
    for i in range(22):
        yy, xx = np.mgrid[0:320, 0:320].astype(np.float64)
        img = np.stack([0.3 + 0.4 * xx / 320, 0.2 + 0.5 * yy / 320, 0.6 - 0.3 * xx / 320], axis=-1)
        cx, cy = 60 + 9 * i, 160 + 20 * np.sin(i / 3.0)
        img[((xx - cx) ** 2 + (yy - cy) ** 2) < 35 ** 2] = (1.0, 0.9, 0.2)
        frames.append(np.clip(img * 255 + 0.5, 0, 255).astype(np.uint8))
    ref_frames = np.stack(frames)
    t1 = np.arange(32000) / 32000.0
    audio1 = torch.from_numpy(np.stack([0.3 * np.sin(2 * np.pi * 440 * t1), 0.3 * np.sin(2 * np.pi * 660 * t1)])).float()
    t2 = np.arange(int(round(22 / 24 * 32000))) / 32000.0
    audio2 = torch.from_numpy(np.stack([0.25 * np.sin(2 * np.pi * 220 * t2) * (1 + 0.5 * np.sin(2 * np.pi * 3 * t2)),
                                        0.25 * np.sin(2 * np.pi * 330 * t2)])).float()
    np.array(ref_image).tofile(os.path.join(out, "rf_image.u8"))
    ref_frames.tofile(os.path.join(out, "rf_video.u8"))
    save("rf_audio1", audio1)
    save("rf_audio2", audio2)
    manifest["rf_image_size"] = list(ref_image.size)
    manifest["rf_video_shape"] = list(ref_frames.shape)
    manifest["rf_ref_short_edge"] = REF_SHORT_EDGE

    # Normalized as the setup step does: the picture's short edge to REF_SHORT_EDGE in multiples of 32 (PIL LANCZOS),
    # the video's frames onto the canvas of their aspect ratio, soundtracks truncated to the generated duration.
    w, h = ref_image.size
    scale = REF_SHORT_EDGE / min(w, h)
    th, tw = max(32, round(h * scale / 32) * 32), max(32, round(w * scale / 32) * 32)
    norm_image = ref_image.resize((tw, th), Image.Resampling.LANCZOS)
    ch, cw = resolve_canvas_size(ref_frames.shape[2], ref_frames.shape[1], 32, 768, 768 * 1344)
    norm_frames = np.stack([np.asarray(Image.fromarray(f).resize((cw, ch), Image.Resampling.LANCZOS)) for f in ref_frames])
    np.array(norm_image).tofile(os.path.join(out, "rf_image_norm.u8"))
    norm_frames.tofile(os.path.join(out, "rf_video_norm.u8"))
    manifest["rf_image_norm_size"] = [tw, th]
    manifest["rf_video_norm_shape"] = list(norm_frames.shape)
    references = [MiniMaxH3ImageReference(image=norm_image), MiniMaxH3AudioReference(audio=audio1, sample_rate=32000),
                  MiniMaxH3VideoReference(frames=norm_frames, fps=24.0, audio=audio2, sample_rate=32000)]

    tok = Qwen2TokenizerFast.from_pretrained(os.path.join(model_dir, "tokenizer"))
    processor = Qwen3VLProcessor.from_pretrained(os.path.join(model_dir, "processor"))
    step = MiniMaxH3Ref2VATextEncoderStep()
    vision_inputs, image_counts, video_counts, video_timestamps = step._gather_vision_features(processor, references, 24.0)
    ids, tags = step._build_presentation(tok, PROMPT, references, image_counts, video_counts, video_timestamps)
    manifest["rf_ids"], manifest["rf_tags"] = ids, tags
    manifest["rf_image_grid_thw"] = vision_inputs["image_grid_thw"].tolist()
    manifest["rf_video_grid_thw"] = vision_inputs["video_grid_thw"].tolist()
    manifest["rf_video_timestamps"] = video_timestamps
    save("rf_pixel_values", vision_inputs["pixel_values"])
    save("rf_pixel_values_videos", vision_inputs["pixel_values_videos"])
    mm_types = processor.create_mm_token_type_ids([ids])
    manifest["rf_mm_token_type_ids"] = mm_types[0]
    log(f"ref2va presentation: {len(ids)} tokens, image grid {manifest['rf_image_grid_thw']}, video grid "
        f"{manifest['rf_video_grid_thw']}, timestamps {video_timestamps}")

    log("loading the text encoder (BF16, CPU), the vision tower in fp32")
    te = Qwen3VLForConditionalGeneration.from_pretrained(os.path.join(model_dir, "text_encoder"), dtype=torch.bfloat16,
                                                         device_map="cpu")
    lm = te.model.language_model
    lm.layers = lm.layers[:TEXT_LAYER + 1]
    lm.config.num_hidden_layers = TEXT_LAYER + 1
    te.config.text_config.num_hidden_layers = TEXT_LAYER + 1
    gc.collect()
    lm.embed_tokens.float()
    lm.norm.float()
    lm.rotary_emb.float()
    te.model.visual.float()

    def upcast_pre(module, args, kwargs):
        module.float()
        return args, kwargs

    def downcast_post(module, args, output):
        module.to(torch.bfloat16)
        return output

    for layer in lm.layers:
        layer.register_forward_pre_hook(upcast_pre, with_kwargs=True)
        layer.register_forward_hook(downcast_post)
    vis = te.model.visual
    vprobes = {}
    vis.merger.register_forward_hook(lambda m, a, o: vprobes.setdefault("merger", []).append(o))
    for i, m in enumerate(vis.deepstack_merger_list):
        m.register_forward_hook(lambda mod, a, o, i=i: vprobes.setdefault(f"deepstack{i}", []).append(o))
    input_ids = torch.tensor([ids], dtype=torch.long)
    log("encoding in fp32")
    with torch.no_grad():
        o = te.model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids),
                     mm_token_type_ids=torch.tensor(mm_types, dtype=torch.long),
                     pixel_values=vision_inputs["pixel_values"].float(), image_grid_thw=vision_inputs["image_grid_thw"],
                     pixel_values_videos=vision_inputs["pixel_values_videos"].float(),
                     video_grid_thw=vision_inputs["video_grid_thw"], use_cache=False, output_hidden_states=True)
    # The tower ran once per modality: the image's call first, the video's second.
    for j, name in enumerate(("image", "video")):
        save(f"rf_vis_{name}_merger", vprobes["merger"][j])
        for i in range(len(vis.deepstack_merger_list)):
            save(f"rf_vis_{name}_deepstack{i}", vprobes[f"deepstack{i}"][j])
    rf_text = o.hidden_states[TEXT_LAYER][0]
    save("rf_text_embeds", rf_text)
    log("rf_text_embeds", tuple(rf_text.shape))
    del te, lm, o, vis
    gc.collect()
    write_manifest()

    # The references through the VAEs: the picture as one frame, the video in 17-frame chunks (padded by repeating
    # its last frame, the last three latents dropped), the posterior sampled under seed 42 per reference; the
    # soundtracks' posterior means, channel-major rows.
    sys.path.insert(0, model_dir)
    from video_vae.minimax_h3_video_vae import MiniMaxH3VideoVAE
    from video_vae.vae_module import DiagonalGaussianDistribution
    from audio_vae.minimax_h3_audio_vae import MiniMaxH3AudioVAE

    dev = torch.device("mps")
    vcfg = json.load(open(os.path.join(model_dir, "video_vae", "config.json")))
    lmean = torch.tensor(vcfg["latents_mean"]).view(1, -1, 1, 1, 1)
    lstd = torch.tensor(vcfg["latents_std"]).view(1, -1, 1, 1, 1)
    pm = torch.tensor((0.485, 0.456, 0.406)).view(1, -1, 1, 1, 1)
    ps = torch.tensor((0.229, 0.224, 0.225)).view(1, -1, 1, 1, 1)
    log("loading the video VAE (fp32, MPS)")
    vae = MiniMaxH3VideoVAE.from_pretrained(os.path.join(model_dir, "video_vae")).to(dev)

    def encode_pixels(pixels):  # [1][3][F][H][W] uint8 -> moments [1][48][F'][h][w], the diffusers _encode recipe
        x = (pixels.float() / 255.0 - pm) / ps
        F = x.shape[2]
        with torch.no_grad():
            if F == 1:
                return vae._adaptive_encode(x.to(dev)).float().cpu()
            pad = (-F) % 17
            if pad:
                x = torch.cat([x, x[:, :, -1:].repeat(1, 1, pad, 1, 1)], dim=2)
            parts = [vae._adaptive_encode(x[:, :, i * 17:(i + 1) * 17].to(dev)).float().cpu() for i in range(x.shape[2] // 17)]
            m = torch.cat(parts, dim=2)
            return m[:, :, :-3]

    cond_latents = []
    pix_image = torch.from_numpy(np.array(norm_image)).permute(2, 0, 1)[None, :, None]
    nf = max(1, (norm_frames.shape[0] - 5) // 17) * 17 + 5
    pix_video = torch.from_numpy(norm_frames[:nf].copy()).permute(3, 0, 1, 2)[None]
    manifest["rf_video_frames_encoded"] = int(nf)
    for name, pixels in (("image", pix_image), ("video", pix_video)):
        t0 = time.time()
        moments = encode_pixels(pixels)
        post = DiagonalGaussianDistribution(moments)
        noise = torch.randn(post.mean.shape, generator=torch.Generator().manual_seed(42))
        z = (post.mean + post.std * noise).to(torch.float16).float()
        zn = (z - lmean) / lstd
        save(f"rf_cond_{name}_mean", post.mean)
        save(f"rf_cond_{name}_std", post.std)
        save(f"rf_cond_{name}_latents", zn)
        cond_latents.append(zn)
        log(f"{name} reference: latents {tuple(zn.shape)} in {time.time() - t0:.1f} s")
    del vae
    gc.collect()
    torch.mps.empty_cache()

    acfg = json.load(open(os.path.join(model_dir, "audio_vae", "config.json")))
    amean = torch.tensor(acfg["latents_mean"]).view(1, 1, -1)
    astd = torch.tensor(acfg["latents_std"]).view(1, 1, -1)
    log("loading the audio VAE (fp32, MPS)")
    avae = MiniMaxH3AudioVAE.from_pretrained(os.path.join(model_dir, "audio_vae")).to(dev)
    audio_rows = []
    for name, wave in (("audio1", audio1), ("audio2", audio2)):
        x = wave[:, None].to(dev)  # [2][1][samples]: the stereo channels as a batch
        pad = -x.shape[-1] % 800
        if pad:
            x = torch.nn.functional.pad(x, (0, pad))
        with torch.no_grad():
            hcur = avae.encoder(x)
            hcur = avae.pre_block(hcur.transpose(1, 2)).transpose(1, 2)
            mean = avae.mean_proj(hcur)
        lat = mean.float().cpu().transpose(1, 2)  # [2][n][32]
        rows = ((lat - amean) / astd).reshape(-1, 32)
        save(f"rf_{name}_mean", mean.float().cpu())
        save(f"rf_{name}_rows", rows)
        audio_rows.append(rows)
        log(f"{name}: {rows.shape[0]} rows")
    del avae
    gc.collect()
    torch.mps.empty_cache()
    write_manifest()

    # The layout: [text | picture rows | audio 1 rows | audio 2 rows, video rows | target audio | target video].
    lat_f = video_latent_num_frames(NUM_FRAMES, 17, 5)
    lat_h = lat_w = SIZE // 16
    n_audio = audio_latent_num_frames(NUM_FRAMES)
    pos, ptags, vidx, aidx, tidx, n_cond_v, n_cond_a = MiniMaxH3Ref2VAPrepareLayoutStep.build_ref2va_packed_sequence(
        torch.tensor(tags, dtype=torch.long), references, cond_latents, audio_rows, lat_f, lat_h, lat_w, n_audio,
        (1, 2, 2), 2, 2, 0)
    S = pos.shape[0]
    T = len(ids)
    manifest.update({"rf_text_rows": T, "rf_seq_len": S, "rf_condition_rows": n_cond_v, "rf_condition_audio_rows": n_cond_a})
    save_f64("rf_position_ids", pos)
    manifest["rf_token_tags"] = ptags.tolist()
    log(f"sequence {S} rows: {T} text, {n_cond_v} reference video, {n_cond_a} reference audio")

    sched_v, sched_a = MiniMaxH3Scheduler(shift=12.0), MiniMaxH3Scheduler(shift=3.0)
    sched_v.set_timesteps(STEPS)
    sched_a.set_timesteps(STEPS)
    plan = [MiniMaxH3SetTimestepsStep.build_row_timesteps(vidx, aidx, n_cond_v, n_cond_a, T, float(tv), float(ta),
                                                          max(float(tv), 0.999), 1.0)
            for tv, ta in zip(sched_v.timesteps, sched_a.timesteps)]
    manifest["rf_step_timesteps"] = [u.tolist() for u, _ in plan]
    g = torch.Generator().manual_seed(SEED)
    cond_rows = []
    for i, z in enumerate(cond_latents):
        noise = torch.randn(z.shape, generator=g, dtype=torch.float32)
        save(f"rf_cond_noise{i}", noise)
        cond_rows.append(patchify_video_latents(0.999 * z + (1.0 - 0.999) * noise, (1, 2, 2)))
    video = torch.randn((1, 24, lat_f, lat_h, lat_w), generator=g, dtype=torch.float32)
    video_rows = patchify_video_latents(video, (1, 2, 2))
    gen_audio = torch.randn((n_audio * 2, 32), generator=g, dtype=torch.float32)
    save("rf_noise_video_rows", video_rows)
    save("rf_noise_audio_rows", gen_audio)
    cond_rows = torch.cat(cond_rows)
    assert cond_rows.shape[0] == n_cond_v
    save("rf_condition_rows", cond_rows)
    v_rows = torch.cat([cond_rows, video_rows])
    a_rows = torch.cat(audio_rows + [gen_audio])
    assert a_rows.shape[0] == n_cond_a + n_audio * 2
    text_embeds = rf_text.to(dev, torch.bfloat16)

    dit = load_dit(dev, os.path.join(ref2va_dir, "transformer"))
    probes = {}
    dit.transformer_blocks[0].register_forward_pre_hook(lambda m, a: probes.__setitem__("x0", a[0]))
    for b in PROBE_BLOCKS:
        dit.transformer_blocks[b].register_forward_hook(lambda m, a, o, b=b: probes.__setitem__(f"block{b}", o))
    dit.norm_out.register_forward_hook(lambda m, a, o: probes.__setitem__("norm_out", o))
    for i, (tv, ta) in enumerate(zip(sched_v.timesteps, sched_a.timesteps)):
        uniq, tsi = plan[i]
        save(f"rf_step{i}_video_rows", v_rows)
        save(f"rf_step{i}_audio_rows", a_rows)
        t0 = time.time()
        with torch.no_grad():
            pred_v, pred_a = dit(hidden_states=v_rows[None].to(dev), audio_hidden_states=a_rows[None].to(dev),
                                 encoder_hidden_states=text_embeds[None], timestep=uniq.to(dev),
                                 timestep_indices=tsi.to(dev), token_tags=ptags.to(dev),
                                 position_ids=pos.to(dev, torch.float32), video_indices=vidx.to(dev),
                                 audio_indices=aidx.to(dev), text_indices=tidx.to(dev), return_dict=False)
        torch.mps.synchronize()
        log(f"step {i}: forward {time.time() - t0:.1f} s, timesteps {uniq.tolist()}")
        save(f"rf_s{i}_x0", probes["x0"][0])
        for b in PROBE_BLOCKS:
            save(f"rf_s{i}_block{b}", probes[f"block{b}"][0])
        save(f"rf_s{i}_norm_out", probes["norm_out"][0])
        save(f"rf_step{i}_pred_video", pred_v[0])
        save(f"rf_step{i}_pred_audio", pred_a[0])
        v_rows[n_cond_v:] = sched_v.step(pred_v[0, n_cond_v:].float().cpu(), tv, v_rows[n_cond_v:], return_dict=False)[0]
        a_rows[n_cond_a:] = sched_a.step(pred_a[0, n_cond_a:].float().cpu(), ta, a_rows[n_cond_a:], return_dict=False)[0]
    save("rf_final_video_rows", v_rows)
    save("rf_final_audio_rows", a_rows)
    del dit
    gc.collect()
    torch.mps.empty_cache()
    write_manifest()

# ------------------------------------------------------------------------------------------------------------ VAEs
if stage in ("vae", "all"):
    # The checkpoint's own VAE modules (video_vae/ and audio_vae/ as namespace packages).
    sys.path.insert(0, model_dir)
    from video_vae.minimax_h3_video_vae import MiniMaxH3VideoVAE
    from audio_vae.minimax_h3_audio_vae import MiniMaxH3AudioVAE

    dev = torch.device("mps")
    lat_f, lat_h, lat_w = manifest["latent_frames"], manifest["latent_height"], manifest["latent_width"]
    n_audio = manifest["num_audio_latents"]
    v_rows = load("final_video_rows")
    a_rows = load("final_audio_rows")
    # Rows back to latents: [1][24][F][H][W] from frame-major rows of [c][ph][pw].
    lat = v_rows.reshape(1, lat_f, lat_h // 2, lat_w // 2, 24, 1, 2, 2).permute(0, 4, 1, 5, 2, 6, 3, 7)
    lat = lat.reshape(1, 24, lat_f, lat_h, lat_w).contiguous()
    vcfg = json.load(open(os.path.join(model_dir, "video_vae", "config.json")))
    mean = torch.tensor(vcfg["latents_mean"]).view(1, -1, 1, 1, 1)
    std = torch.tensor(vcfg["latents_std"]).view(1, -1, 1, 1, 1)
    vae_in = lat * std + mean
    save("vae_in", vae_in)

    log("loading the video VAE (fp32, MPS)")
    vae = MiniMaxH3VideoVAE.from_pretrained(os.path.join(model_dir, "video_vae")).to(dev)
    t0 = time.time()
    with torch.no_grad():
        dec = vae.decode_base(vae_in.to(dev))  # temporal chunking (17 frames a clip) and the pipeline's tiling
    video = dec[0] if isinstance(dec, (tuple, list)) else getattr(dec, "sample", dec)
    torch.mps.synchronize()
    log(f"video decoded {tuple(video.shape)} in {time.time() - t0:.1f} s")
    pm = torch.tensor((0.485, 0.456, 0.406), device=video.device).view(1, -1, 1, 1, 1)
    ps = torch.tensor((0.229, 0.224, 0.225), device=video.device).view(1, -1, 1, 1, 1)
    frames = (video.float() * ps + pm).clamp(0, 1)[0].permute(1, 2, 3, 0)  # [F][H][W][3]
    save("video", frames)
    (frames.cpu().numpy() * 255.0).round().astype(np.uint8).tofile(os.path.join(out, "video.u8"))
    save("video_raw", video[0].permute(1, 2, 3, 0))  # the decoder's own output (ImageNet-normalized)
    del vae
    gc.collect()
    torch.mps.empty_cache()

    acfg = json.load(open(os.path.join(model_dir, "audio_vae", "config.json")))
    amean = torch.tensor(acfg["latents_mean"]).view(1, -1, 1)
    astd = torch.tensor(acfg["latents_std"]).view(1, -1, 1)
    audio_lat = a_rows.reshape(2, n_audio, 32).permute(0, 2, 1).contiguous() * astd + amean  # [2][32][n]
    save("audio_vae_in", audio_lat)
    log("loading the audio VAE (fp32, MPS)")
    avae = MiniMaxH3AudioVAE.from_pretrained(os.path.join(model_dir, "audio_vae")).to(dev)
    with torch.no_grad():
        audio = avae.decode(audio_lat.to(dev))
    audio = audio[0] if isinstance(audio, (tuple, list)) else getattr(audio, "sample", audio)
    log(f"audio decoded {tuple(audio.shape)}")
    save("audio", audio.float().reshape(2, -1))
    write_manifest()

log("done")
