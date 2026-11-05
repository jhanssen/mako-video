// MiniMax-H3's transformer: RMS norms modulated per row of the packed sequence (each row its own (timestep,
// modality) modulation), gated residuals per row. Activations are fp32 rows [row][width].
#include <metal_stdlib>
using namespace metal;

#include "rows.h"

struct RowModArgs {
    uint n;         // row width
    float eps;
    uint shift_off; // of the shift within a row's modulation, in floats
    uint scale_off; // of the scale
};

// y = x / rms(x) * w * (1 + scale) + shift over each row, the row's modulation at mods + off[row] (shift and scale at
// shift_off and scale_off of it); one threadgroup per row.
kernel void rms_mod_rows(device const float* x [[buffer(0)]], device const float* w [[buffer(1)]],
                         device const float* mods [[buffer(2)]], device const uint* off [[buffer(3)]],
                         device float* y [[buffer(4)]], constant RowModArgs& a [[buffer(5)]],
                         uint row [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]])
{
    threadgroup float scratch[TG / 32];
    device const float4* xr = reinterpret_cast<device const float4*>(x + uint64_t(row) * a.n);
    device float4* yr = reinterpret_cast<device float4*>(y + uint64_t(row) * a.n);
    device const float4* w4 = reinterpret_cast<device const float4*>(w);
    device const float4* sh = reinterpret_cast<device const float4*>(mods + off[row] + a.shift_off);
    device const float4* sc = reinterpret_cast<device const float4*>(mods + off[row] + a.scale_off);
    const uint n4 = a.n / 4;
    float s = 0.0f;
    for (uint i = tid; i < n4; i += TG) {
        const float4 v = xr[i];
        s += dot(v, v);
    }
    const float inv = rsqrt(tg_sum(s, scratch, tid) / float(a.n) + a.eps);
    for (uint i = tid; i < n4; i += TG)
        yr[i] = xr[i] * inv * w4[i] * (1.0f + sc[i]) + sh[i];
}

struct GateRowsArgs {
    uint n;
    uint gate_off; // of the gate within a row's modulation, in floats
};

// x[row] += gate[row] * y[row], the row's gate at mods + off[row] + gate_off. Grid (n, rows).
kernel void gate_add_rows(device float* x [[buffer(0)]], device const float* y [[buffer(1)]],
                          device const float* mods [[buffer(2)]], device const uint* off [[buffer(3)]],
                          constant GateRowsArgs& a [[buffer(4)]], uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= a.n)
        return;
    const uint64_t o = uint64_t(gid.y) * a.n + gid.x;
    x[o] += mods[off[gid.y] + a.gate_off + gid.x] * y[o];
}

// The audio VAE decoder's one-dimensional pieces over [channels][time] floats.

struct Conv1dArgs {
    uint cin, cout, tin, tout, k, stride, pad, dil;
};

// y[co][t] = b[co] + sum over ci, j of w[co][ci][j] * x[ci][t * stride - pad + j * dil] (0 outside). Grid (tout, cout).
kernel void conv1d(device const float* x [[buffer(0)]], device const float* w [[buffer(1)]],
                   device const float* b [[buffer(2)]], device float* y [[buffer(3)]],
                   constant Conv1dArgs& a [[buffer(4)]], constant uint& has_bias [[buffer(5)]],
                   uint2 gid [[thread_position_in_grid]])
{
    const uint t = gid.x, co = gid.y;
    if (t >= a.tout || co >= a.cout)
        return;
    float s = has_bias ? b[co] : 0.0f;
    const int base = int(t * a.stride) - int(a.pad);
    device const float* wc = w + uint64_t(co) * a.cin * a.k;
    for (uint ci = 0; ci < a.cin; ++ci) {
        device const float* xc = x + uint64_t(ci) * a.tin;
        device const float* wk = wc + uint64_t(ci) * a.k;
        for (uint j = 0; j < a.k; ++j) {
            const int i = base + int(j * a.dil);
            if (i >= 0 && i < int(a.tin))
                s += wk[j] * xc[i];
        }
    }
    y[uint64_t(co) * a.tout + t] = s;
}

// y[co][t] = b[co] + sum over ci, j of w[ci][co][j] * x[ci][(t + pad - j) / stride] where it divides. Grid (tout, cout).
kernel void conv1d_transposed(device const float* x [[buffer(0)]], device const float* w [[buffer(1)]],
                              device const float* b [[buffer(2)]], device float* y [[buffer(3)]],
                              constant Conv1dArgs& a [[buffer(4)]], constant uint& has_bias [[buffer(5)]],
                              uint2 gid [[thread_position_in_grid]])
{
    const uint t = gid.x, co = gid.y;
    if (t >= a.tout || co >= a.cout)
        return;
    float s = has_bias ? b[co] : 0.0f;
    for (uint j = 0; j < a.k; ++j) {
        const int num = int(t + a.pad) - int(j);
        if (num < 0 || num % int(a.stride))
            continue;
        const uint i = uint(num) / a.stride;
        if (i >= a.tin)
            continue;
        for (uint ci = 0; ci < a.cin; ++ci)
            s += w[(uint64_t(ci) * a.cout + co) * a.k + j] * x[uint64_t(ci) * a.tin + i];
    }
    y[uint64_t(co) * a.tout + t] = s;
}

// x += sin(alpha x)^2 / (beta + 1e-9) per channel (alpha, beta already exponentiated). Grid (t, channels).
kernel void snake_beta(device float* x [[buffer(0)]], device const float* alpha [[buffer(1)]],
                       device const float* beta [[buffer(2)]], constant uint& t [[buffer(3)]],
                       uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= t)
        return;
    const uint64_t o = uint64_t(gid.y) * t + gid.x;
    const float v = x[o], s = sin(alpha[gid.y] * v);
    x[o] = v + s * s / (beta[gid.y] + 1e-9f);
}

struct ResampleArgs {
    uint tin, tout, taps;
};

// 2x upsampling by a depthwise transposed convolution with a `taps`-tap filter (stride 2) over the input padded by
// replication (taps / 2 - 1 each side), scaled by 2 and cropped to 2 tin. Grid (tout, channels).
kernel void upsample2(device const float* x [[buffer(0)]], device const float* f [[buffer(1)]],
                      device float* y [[buffer(2)]], constant ResampleArgs& a [[buffer(3)]],
                      uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= a.tout)
        return;
    const int pad = int(a.taps / 2) - 1, crop = pad * 2 + int(a.taps - 2) / 2, padded = int(a.tin) + 2 * pad;
    device const float* xc = x + uint64_t(gid.y) * a.tin;
    const int u = int(gid.x) + crop;
    float s = 0.0f;
    for (int j = 0; j < int(a.taps); ++j) {
        const int num = u - j;
        if (num < 0 || num % 2)
            continue;
        const int i = num / 2;
        if (i >= padded)
            continue;
        s += f[j] * xc[clamp(i - pad, 0, int(a.tin) - 1)];
    }
    y[uint64_t(gid.y) * a.tout + gid.x] = 2.0f * s;
}

// 2x downsampling: the filter over the input padded by replication (taps / 2 - 1 left, taps / 2 right), stride 2.
// Grid (tout, channels).
kernel void downsample2(device const float* x [[buffer(0)]], device const float* f [[buffer(1)]],
                        device float* y [[buffer(2)]], constant ResampleArgs& a [[buffer(3)]],
                        uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= a.tout)
        return;
    const int pad = int(a.taps / 2) - 1;
    device const float* xc = x + uint64_t(gid.y) * a.tin;
    float s = 0.0f;
    for (int j = 0; j < int(a.taps); ++j)
        s += f[j] * xc[clamp(int(gid.x) * 2 + j - pad, 0, int(a.tin) - 1)];
    y[uint64_t(gid.y) * a.tout + gid.x] = s;
}

struct ScaleArgs {
    uint n;
    float s;
};

// y = s * x.
kernel void scale_f32(device const float* x [[buffer(0)]], device float* y [[buffer(1)]],
                      constant ScaleArgs& a [[buffer(2)]], uint i [[thread_position_in_grid]])
{
    if (i < a.n)
        y[i] = a.s * x[i];
}

// A pruned checkpoint's modulation: out[u][o] = b[o] + sum_j w[o][j] c[u][j] over 8 curve coordinates a timestep,
// in float. Grid (n, U).
kernel void curve_mods(device const bfloat* w [[buffer(0)]], device const float* b [[buffer(1)]],
                       device const float* c [[buffer(2)]], device float* out [[buffer(3)]],
                       constant uint2& a [[buffer(4)]], uint2 gid [[thread_position_in_grid]])
{
    const uint o = gid.x, u = gid.y;
    if (o >= a.x || u >= a.y)
        return;
    float s = b[o];
    for (uint j = 0; j < 8; ++j)
        s += float(w[uint64_t(o) * 8 + j]) * c[u * 8 + j];
    out[uint64_t(u) * a.x + o] = s;
}
