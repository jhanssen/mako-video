// The frame, latent and canvas arithmetic and the schedules, against the reference's numbers (no model).
#include "video/engine.hpp"

#include <cmath>
#include <print>

namespace {

int failures = 0;

void check(bool ok, const std::string& what)
{
    std::println("  {:4} {}", ok ? "ok" : "FAIL", what);
    failures += !ok;
}

} // namespace

int main()
{
    using mako::video::Engine;
    check(Engine::align_frames(124) == 124 && Engine::align_frames(100) == 107 && Engine::align_frames(346) == 362,
          "frames snap up to 17 n + 5");
    check(Engine::latent_frames(124) == 37 && Engine::latent_frames(362) == 107 && Engine::latent_frames(22) == 7,
          "latent frames 5 n + 2");
    check(Engine::audio_latents(124) == 207 && Engine::audio_latents(362) == 603, "audio latents at 40 a second");
    const auto [w, h] = Engine::canvas(16, 9);
    check(w == 1344 && h == 768, std::format("the 16:9 canvas is {}x{}", w, h));
    const auto [pw, ph] = Engine::canvas(9, 16);
    check(pw == 768 && ph == 1344, std::format("the 9:16 canvas is {}x{}", pw, ph));
    const auto [sw, sh] = Engine::canvas(1, 1);
    check(sw == 768 && sh == 768, std::format("the square canvas is {}x{}", sw, sh));
    // 3 grid points: 1, 0.5 and 0 shifted by 12 and 3.
    const auto v = Engine::sigmas(3, 12.0f), a = Engine::sigmas(3, 3.0f);
    check(v.size() == 3 && std::fabs(v[1] - 12.0f * 0.5f / (1.0f + 11.0f * 0.5f)) < 1e-7f && v[2] == 0.0f &&
              v[0] == 1.0f,
          std::format("video sigmas {} {} {}", v[0], v[1], v[2]));
    check(a.size() == 3 && std::fabs(a[1] - 0.75f) < 1e-7f, std::format("audio sigmas {} {} {}", a[0], a[1], a[2]));
    const auto s50 = Engine::sigmas(50, 12.0f);
    bool decreasing = true;
    for (size_t i = 1; i < s50.size(); ++i)
        decreasing = decreasing && s50[i] < s50[i - 1];
    check(s50.size() == 50 && decreasing && s50.back() == 0.0f, "50 points, strictly decreasing to 0");
    return failures ? 1 : 0;
}
