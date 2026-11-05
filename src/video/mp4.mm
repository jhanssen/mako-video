#include "video/mp4.hpp"

#include "video/image.hpp"

#import <AVFoundation/AVFoundation.h>
#import <CoreMedia/CoreMedia.h>
#import <CoreVideo/CoreVideo.h>
#import <Foundation/Foundation.h>
#import <ImageIO/ImageIO.h>

#include <algorithm>
#include <cmath>
#include <format>
#include <mutex>
#include <stdexcept>
#include <vector>

namespace mako::video {

namespace {

std::string describe(NSError* e) { return e ? std::string(e.localizedDescription.UTF8String) : "unknown error"; }

// What the two tracks' feeders share: the data, where each is, and the first failure.
struct Feed {
    uint32_t width = 0, height = 0, fps = 0, count = 0;
    std::span<const uint8_t> rgb;
    std::span<const float> audio;
    uint32_t sample_rate = 0;
    uint32_t next_frame = 0;
    uint64_t next_sample = 0, samples = 0;
    std::mutex mutex;
    std::string failure;
    void fail(const std::string& what)
    {
        std::lock_guard lock(mutex);
        if (failure.empty())
            failure = what;
    }
    bool failed()
    {
        std::lock_guard lock(mutex);
        return !failure.empty();
    }
};

// One frame as a BGRA pixel buffer from the adaptor's pool.
CVPixelBufferRef frame_buffer(Feed& f, AVAssetWriterInputPixelBufferAdaptor* adaptor, uint32_t i)
{
    CVPixelBufferRef pb = nullptr;
    if (CVPixelBufferPoolCreatePixelBuffer(kCFAllocatorDefault, adaptor.pixelBufferPool, &pb) != kCVReturnSuccess)
        return nullptr;
    CVPixelBufferLockBaseAddress(pb, 0);
    auto* base = static_cast<uint8_t*>(CVPixelBufferGetBaseAddress(pb));
    const size_t stride = CVPixelBufferGetBytesPerRow(pb);
    const uint8_t* src = f.rgb.data() + uint64_t(i) * f.width * f.height * 3;
    for (uint32_t y = 0; y < f.height; ++y) {
        uint8_t* row = base + y * stride;
        const uint8_t* s = src + uint64_t(y) * f.width * 3;
        for (uint32_t x = 0; x < f.width; ++x) {
            row[4 * x] = s[3 * x + 2];
            row[4 * x + 1] = s[3 * x + 1];
            row[4 * x + 2] = s[3 * x];
            row[4 * x + 3] = 255;
        }
    }
    CVPixelBufferUnlockBaseAddress(pb, 0);
    return pb;
}

// Up to a second of interleaved float PCM from `at`.
CMSampleBufferRef audio_buffer(Feed& f, CMAudioFormatDescriptionRef fmt, uint64_t at)
{
    const uint64_t frames = std::min<uint64_t>(f.sample_rate, f.samples - at);
    std::vector<float> chunk(frames * 2);
    for (uint64_t i = 0; i < frames; ++i) {
        chunk[2 * i] = f.audio[at + i];
        chunk[2 * i + 1] = f.audio[f.samples + at + i];
    }
    CMBlockBufferRef block = nullptr;
    const size_t bytes = chunk.size() * 4;
    if (CMBlockBufferCreateWithMemoryBlock(kCFAllocatorDefault, nullptr, bytes, kCFAllocatorDefault, nullptr, 0, bytes,
                                           0, &block) != kCMBlockBufferNoErr)
        return nullptr;
    CMBlockBufferReplaceDataBytes(chunk.data(), block, 0, bytes);
    CMSampleTimingInfo timing{CMTimeMake(1, int32_t(f.sample_rate)), CMTimeMake(int64_t(at), int32_t(f.sample_rate)),
                              kCMTimeInvalid};
    const size_t size = 8;
    CMSampleBufferRef sb = nullptr;
    const OSStatus st = CMSampleBufferCreate(kCFAllocatorDefault, block, true, nullptr, nullptr, fmt, CMItemCount(frames),
                                             1, &timing, 1, &size, &sb);
    CFRelease(block);
    return st == noErr ? sb : nullptr;
}

} // namespace

void write_mp4(const std::string& path, uint32_t width, uint32_t height, uint32_t fps, uint32_t count,
               std::span<const uint8_t> rgb, std::span<const float> audio, uint32_t sample_rate)
{
    if (rgb.size() != uint64_t(count) * width * height * 3)
        throw std::runtime_error("mp4: frames of [height][width][3]");
    if (!audio.empty() && audio.size() % 2)
        throw std::runtime_error("mp4: stereo samples [2][n]");
    @autoreleasepool {
        NSString* p = [NSString stringWithUTF8String:path.c_str()];
        [[NSFileManager defaultManager] removeItemAtPath:p error:nil];
        NSError* err = nil;
        AVAssetWriter* writer = [[AVAssetWriter alloc] initWithURL:[NSURL fileURLWithPath:p]
                                                         fileType:AVFileTypeMPEG4
                                                            error:&err];
        if (!writer)
            throw std::runtime_error("mp4: " + describe(err));
        // About 0.15 bits a pixel a frame at 24 fps: 1344x768 gets ~3.7 Mbit/s.
        const int64_t bitrate = std::max<int64_t>(2'000'000, int64_t(0.15 * double(width) * height * fps));
        NSDictionary* video_settings = @{
            AVVideoCodecKey : AVVideoCodecTypeH264,
            AVVideoWidthKey : @(width),
            AVVideoHeightKey : @(height),
            AVVideoCompressionPropertiesKey : @{
                AVVideoAverageBitRateKey : @(bitrate),
                AVVideoProfileLevelKey : AVVideoProfileLevelH264HighAutoLevel,
                AVVideoExpectedSourceFrameRateKey : @(fps),
                AVVideoMaxKeyFrameIntervalKey : @(fps * 2)
            }
        };
        AVAssetWriterInput* vin = [AVAssetWriterInput assetWriterInputWithMediaType:AVMediaTypeVideo
                                                                    outputSettings:video_settings];
        vin.expectsMediaDataInRealTime = NO;
        AVAssetWriterInputPixelBufferAdaptor* adaptor = [AVAssetWriterInputPixelBufferAdaptor
            assetWriterInputPixelBufferAdaptorWithAssetWriterInput:vin
                                       sourcePixelBufferAttributes:@{
                                           (id)kCVPixelBufferPixelFormatTypeKey : @(kCVPixelFormatType_32BGRA),
                                           (id)kCVPixelBufferWidthKey : @(width),
                                           (id)kCVPixelBufferHeightKey : @(height)
                                       }];
        if (![writer canAddInput:vin])
            throw std::runtime_error("mp4: cannot add the video track");
        [writer addInput:vin];
        AVAssetWriterInput* ain = nil;
        CMAudioFormatDescriptionRef fmt = nullptr;
        if (!audio.empty()) {
            NSDictionary* audio_settings = @{
                AVFormatIDKey : @(kAudioFormatMPEG4AAC),
                AVSampleRateKey : @(sample_rate),
                AVNumberOfChannelsKey : @2,
                AVEncoderBitRateKey : @192000
            };
            ain = [AVAssetWriterInput assetWriterInputWithMediaType:AVMediaTypeAudio outputSettings:audio_settings];
            ain.expectsMediaDataInRealTime = NO;
            if (![writer canAddInput:ain])
                throw std::runtime_error("mp4: cannot add the audio track");
            [writer addInput:ain];
            AudioStreamBasicDescription asbd{};
            asbd.mSampleRate = sample_rate;
            asbd.mFormatID = kAudioFormatLinearPCM;
            asbd.mFormatFlags = kAudioFormatFlagIsFloat | kAudioFormatFlagIsPacked;
            asbd.mBytesPerPacket = 8;
            asbd.mFramesPerPacket = 1;
            asbd.mBytesPerFrame = 8;
            asbd.mChannelsPerFrame = 2;
            asbd.mBitsPerChannel = 32;
            if (CMAudioFormatDescriptionCreate(kCFAllocatorDefault, &asbd, 0, nullptr, 0, nullptr, nullptr, &fmt) != noErr)
                throw std::runtime_error("mp4: audio format");
        }
        if (![writer startWriting])
            throw std::runtime_error("mp4: " + describe(writer.error));
        [writer startSessionAtSourceTime:kCMTimeZero];

        // Each track pulls its data when the writer is ready for it (the writer interleaves the two and holds one
        // until the other has caught up, so pushing frames and samples by turns can wait forever).
        auto* feed = new Feed;
        feed->width = width;
        feed->height = height;
        feed->fps = fps;
        feed->count = count;
        feed->rgb = rgb;
        feed->audio = audio;
        feed->sample_rate = sample_rate;
        feed->samples = audio.size() / 2;
        dispatch_group_t group = dispatch_group_create();
        dispatch_queue_t vq = dispatch_queue_create("mako.video.frames", DISPATCH_QUEUE_SERIAL);
        dispatch_group_enter(group);
        [vin requestMediaDataWhenReadyOnQueue:vq
                                   usingBlock:^{
                                     while (vin.readyForMoreMediaData) {
                                         if (feed->next_frame >= feed->count || feed->failed()) {
                                             [vin markAsFinished];
                                             dispatch_group_leave(group);
                                             return;
                                         }
                                         CVPixelBufferRef pb = frame_buffer(*feed, adaptor, feed->next_frame);
                                         if (!pb) {
                                             feed->fail("mp4: no pixel buffer");
                                             continue;
                                         }
                                         const BOOL ok = [adaptor
                                                 appendPixelBuffer:pb
                                              withPresentationTime:CMTimeMake(feed->next_frame, int32_t(feed->fps))];
                                         CVPixelBufferRelease(pb);
                                         if (!ok)
                                             feed->fail("mp4: " + describe(writer.error));
                                         ++feed->next_frame;
                                     }
                                   }];
        dispatch_queue_t aq = nil;
        if (ain) {
            aq = dispatch_queue_create("mako.video.audio", DISPATCH_QUEUE_SERIAL);
            dispatch_group_enter(group);
            [ain requestMediaDataWhenReadyOnQueue:aq
                                       usingBlock:^{
                                         while (ain.readyForMoreMediaData) {
                                             if (feed->next_sample >= feed->samples || feed->failed()) {
                                                 [ain markAsFinished];
                                                 dispatch_group_leave(group);
                                                 return;
                                             }
                                             CMSampleBufferRef sb = audio_buffer(*feed, fmt, feed->next_sample);
                                             if (!sb) {
                                                 feed->fail("mp4: audio sample buffer");
                                                 continue;
                                             }
                                             const BOOL ok = [ain appendSampleBuffer:sb];
                                             const uint64_t n = uint64_t(CMSampleBufferGetNumSamples(sb));
                                             CFRelease(sb);
                                             if (!ok)
                                                 feed->fail("mp4: " + describe(writer.error));
                                             feed->next_sample += n;
                                         }
                                       }];
        }
        dispatch_group_wait(group, DISPATCH_TIME_FOREVER);
        const std::string failure = feed->failure;
        delete feed;
        if (fmt)
            CFRelease(fmt);
        if (!failure.empty()) {
            [writer cancelWriting];
            throw std::runtime_error(failure);
        }
        dispatch_semaphore_t done = dispatch_semaphore_create(0);
        [writer finishWritingWithCompletionHandler:^{
          dispatch_semaphore_signal(done);
        }];
        dispatch_semaphore_wait(done, DISPATCH_TIME_FOREVER);
        if (writer.status != AVAssetWriterStatusCompleted)
            throw std::runtime_error("mp4: " + describe(writer.error));
    }
}

void write_png(const std::string& path, uint32_t width, uint32_t height, std::span<const uint8_t> rgb)
{
    if (rgb.size() != uint64_t(width) * height * 3)
        throw std::runtime_error("png: a frame of [height][width][3]");
    CGColorSpaceRef space = CGColorSpaceCreateDeviceRGB();
    CGDataProviderRef provider = CGDataProviderCreateWithData(nullptr, rgb.data(), rgb.size(), nullptr);
    CGImageRef image = CGImageCreate(width, height, 8, 24, width * 3, space, kCGBitmapByteOrderDefault, provider, nullptr,
                                     false, kCGRenderingIntentDefault);
    CGDataProviderRelease(provider);
    CGColorSpaceRelease(space);
    if (!image)
        throw std::runtime_error("png: cannot make the image");
    NSURL* url = [NSURL fileURLWithPath:[NSString stringWithUTF8String:path.c_str()]];
    CGImageDestinationRef dest = CGImageDestinationCreateWithURL((__bridge CFURLRef)url, CFSTR("public.png"), 1, nullptr);
    if (!dest) {
        CGImageRelease(image);
        throw std::runtime_error("png: cannot write " + path);
    }
    CGImageDestinationAddImage(dest, image, nullptr);
    const bool ok = CGImageDestinationFinalize(dest);
    CFRelease(dest);
    CGImageRelease(image);
    if (!ok)
        throw std::runtime_error("png: cannot write " + path);
}

Image oriented(std::span<const uint8_t> rgba, uint32_t W, uint32_t H, uint32_t orientation)
{
    // Orientations 5 to 8 swap the axes. For an output pixel (x, y): where in the stored image it comes from.
    const bool swap = orientation >= 5;
    Image out;
    out.width = swap ? H : W;
    out.height = swap ? W : H;
    out.rgb.resize(uint64_t(out.width) * out.height * 3);
    for (uint32_t y = 0; y < out.height; ++y)
        for (uint32_t x = 0; x < out.width; ++x) {
            uint32_t sx = x, sy = y;
            switch (orientation) {
            case 2: sx = W - 1 - x; break;                     // mirrored
            case 3: sx = W - 1 - x; sy = H - 1 - y; break;     // upside down
            case 4: sy = H - 1 - y; break;                     // flipped
            case 5: sx = y; sy = x; break;                     // transposed
            case 6: sx = y; sy = H - 1 - x; break;             // turned 90 degrees clockwise
            case 7: sx = W - 1 - y; sy = H - 1 - x; break;     // transposed the other way
            case 8: sx = W - 1 - y; sy = x; break;             // turned 90 degrees counterclockwise
            default: break;
            }
            const uint8_t* src = &rgba[(uint64_t(sy) * W + sx) * 4];
            uint8_t* dst = &out.rgb[(uint64_t(y) * out.width + x) * 3];
            dst[0] = src[0];
            dst[1] = src[1];
            dst[2] = src[2];
        }
    return out;
}

Image read_image(const std::string& path)
{
    NSURL* url = [NSURL fileURLWithPath:[NSString stringWithUTF8String:path.c_str()]];
    CGImageSourceRef src = CGImageSourceCreateWithURL((__bridge CFURLRef)url, nullptr);
    if (!src)
        throw std::runtime_error("cannot open the image " + path);
    CGImageRef image = CGImageSourceCreateImageAtIndex(src, 0, nullptr);
    // The file's orientation tag (EXIF 1 to 8), applied below so the pixels come out the way the picture is shown.
    uint32_t orientation = 1;
    if (CFDictionaryRef props = CGImageSourceCopyPropertiesAtIndex(src, 0, nullptr)) {
        if (auto* o = static_cast<CFNumberRef>(CFDictionaryGetValue(props, kCGImagePropertyOrientation))) {
            int v = 1;
            if (CFNumberGetValue(o, kCFNumberIntType, &v) && v >= 1 && v <= 8)
                orientation = uint32_t(v);
        }
        CFRelease(props);
    }
    CFRelease(src);
    if (!image)
        throw std::runtime_error(path + " is not an image ImageIO reads");
    const uint32_t W = uint32_t(CGImageGetWidth(image)), H = uint32_t(CGImageGetHeight(image));
    std::vector<uint8_t> rgba(uint64_t(W) * H * 4);
    CGColorSpaceRef space = CGColorSpaceCreateDeviceRGB();
    CGContextRef ctx = CGBitmapContextCreate(rgba.data(), W, H, 8, uint64_t(W) * 4, space,
                                             CGBitmapInfo(kCGImageAlphaPremultipliedLast) | CGBitmapInfo(kCGBitmapByteOrder32Big));
    if (!ctx) {
        CGColorSpaceRelease(space);
        CGImageRelease(image);
        throw std::runtime_error("cannot decode " + path);
    }
    CGContextSetRGBFillColor(ctx, 1, 1, 1, 1); // transparency over white
    CGContextFillRect(ctx, CGRectMake(0, 0, W, H));
    CGContextDrawImage(ctx, CGRectMake(0, 0, W, H), image);
    CGContextRelease(ctx);
    CGColorSpaceRelease(space);
    CGImageRelease(image);
    return oriented(rgba, W, H, orientation);
}

} // namespace mako::video
