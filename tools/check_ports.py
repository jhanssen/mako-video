# Reference values for the ports: torchvision's antialiased bicubic on 8-bit frames, torchaudio's resampler, and
# Python's rendering of the block timestamps.
import numpy as np, torch, torchaudio, json, sys
from torchvision.transforms import v2 as tv
import torchvision.transforms.v2.functional as tvF
out = sys.argv[1]
rng = np.random.default_rng(7)
# A smooth-ish 8-bit frame 1344x768 -> the processor's size for 30 sampled frames (1184x672), and an upscale case.
yy, xx = np.mgrid[0:768, 0:1344].astype(np.float64)
img = np.stack([xx / 1344, (yy / 768) ** 1.5, 0.5 + 0.5 * np.cos(7.0 * xx / 1344 + 3.0 * yy / 768)], -1)
img = np.clip(img * 255 + rng.normal(0, 8, img.shape), 0, 255).astype(np.uint8)
img.tofile(f"{out}/bc_in.u8")
t = torch.from_numpy(img).permute(2, 0, 1)
for name, (h, w) in {"down": (672, 1184), "up": (800, 1408), "mixed": (704, 1344)}.items():
    r = tvF.resize(t, [h, w], interpolation=tvF.InterpolationMode.BICUBIC, antialias=True)
    r.permute(1, 2, 0).contiguous().numpy().tofile(f"{out}/bc_{name}.u8")
# Resampling: a 48 kHz and a 44.1 kHz stereo second to 32 kHz.
for rate in (48000, 44100):
    n = rate * 3 // 2
    tt = np.arange(n) / rate
    w = np.stack([0.5 * np.sin(2 * np.pi * 440 * tt) + 0.2 * np.sin(2 * np.pi * 7000 * tt), rng.normal(0, 0.1, n)]).astype(np.float32)
    w.tofile(f"{out}/rs_{rate}_in.f32")
    y = torchaudio.transforms.Resample(rate, 32000)(torch.from_numpy(w))
    y.numpy().tofile(f"{out}/rs_{rate}_out.f32")
    print(rate, w.shape, "->", tuple(y.shape))
print(json.dumps({"stamps": [f"<{x:.1f} seconds>" for x in (0.25, 0.75, 1.25, 2.5, 0.0, 10.25)]}))
