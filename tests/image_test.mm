// read_image applies the file's orientation tag: a small picture with a marked corner is written as a PNG with each
// EXIF orientation (ImageIO's writer takes the tag), read back, and the marked corner must land where a viewer would
// show it. Also resize_lanczos's identity on a same-size image.
//   image_test <scratch dir>
#include "video/image.hpp"

#import <Foundation/Foundation.h>
#import <ImageIO/ImageIO.h>

#include <cstdio>
#include <format>
#include <print>
#include <string>

namespace {

int failures = 0;
void check(bool ok, const std::string& what)
{
    std::println("  {}  {}", ok ? "ok  " : "FAIL", what);
    failures += !ok;
}

// A 6 x 4 picture: grey, its top-left pixel red and its top-right green.
mako::video::Image picture()
{
    mako::video::Image p{6, 4, std::vector<uint8_t>(6 * 4 * 3, 128)};
    p.rgb[0] = 255; p.rgb[1] = 0; p.rgb[2] = 0;
    p.rgb[5 * 3] = 0; p.rgb[5 * 3 + 1] = 255; p.rgb[5 * 3 + 2] = 0;
    return p;
}

void write_with_orientation(const std::string& path, const mako::video::Image& img, int orientation)
{
    CGColorSpaceRef space = CGColorSpaceCreateDeviceRGB();
    CGDataProviderRef provider = CGDataProviderCreateWithData(nullptr, img.rgb.data(), img.rgb.size(), nullptr);
    CGImageRef image = CGImageCreate(img.width, img.height, 8, 24, img.width * 3, space, kCGBitmapByteOrderDefault,
                                     provider, nullptr, false, kCGRenderingIntentDefault);
    CGDataProviderRelease(provider);
    CGColorSpaceRelease(space);
    NSURL* url = [NSURL fileURLWithPath:[NSString stringWithUTF8String:path.c_str()]];
    CGImageDestinationRef dest = CGImageDestinationCreateWithURL((__bridge CFURLRef)url, CFSTR("public.png"), 1, nullptr);
    NSDictionary* props = @{(__bridge NSString*)kCGImagePropertyOrientation: @(orientation)};
    CGImageDestinationAddImage(dest, image, (__bridge CFDictionaryRef)props);
    CGImageDestinationFinalize(dest);
    CFRelease(dest);
    CGImageRelease(image);
}

} // namespace

int main(int argc, char** argv)
{
    if (argc < 2) {
        std::println(stderr, "image_test <scratch dir>");
        return 2;
    }
    const std::string dir = argv[1];
    const auto pic = picture();
    // Where the red (top-left) and green (top-right) pixels show for each orientation, and the shown size.
    struct Expect {
        int o;
        uint32_t w, h, rx, ry, gx, gy;
    };
    const Expect cases[] = {{1, 6, 4, 0, 0, 5, 0}, {2, 6, 4, 5, 0, 0, 0}, {3, 6, 4, 5, 3, 0, 3}, {4, 6, 4, 0, 3, 5, 3},
                            {5, 4, 6, 0, 0, 0, 5}, {6, 4, 6, 3, 0, 3, 5}, {7, 4, 6, 3, 5, 3, 0}, {8, 4, 6, 0, 5, 0, 0}};
    for (const Expect& e : cases) {
        const std::string path = std::format("{}/orientation{}.png", dir, e.o);
        write_with_orientation(path, pic, e.o);
        const auto got = mako::video::read_image(path);
        auto at = [&](uint32_t x, uint32_t y) { return &got.rgb[(uint64_t(y) * got.width + x) * 3]; };
        const bool size_ok = got.width == e.w && got.height == e.h;
        const bool red = size_ok && at(e.rx, e.ry)[0] == 255 && at(e.rx, e.ry)[1] == 0;
        const bool green = size_ok && at(e.gx, e.gy)[1] == 255 && at(e.gx, e.gy)[0] == 0;
        check(size_ok && red && green, std::format("orientation {}: {}x{}, red at ({}, {}), green at ({}, {})", e.o,
                                                    got.width, got.height, e.rx, e.ry, e.gx, e.gy));
    }
    const auto same = mako::video::resize_lanczos(pic, pic.width, pic.height);
    check(same.rgb == pic.rgb, "resize to the same size is the identity");
    std::println("{}", failures ? "FAILED" : "all passed");
    return failures ? 1 : 0;
}
