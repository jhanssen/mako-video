#include "video/media.hpp"

#import <AVFoundation/AVFoundation.h>
#import <CoreMedia/CoreMedia.h>
#import <CoreVideo/CoreVideo.h>
#import <Foundation/Foundation.h>

#include <cmath>
#include <format>
#include <stdexcept>

namespace mako::video {

namespace {

std::string describe(NSError* e) { return e ? std::string(e.localizedDescription.UTF8String) : "unknown error"; }

// The asset's first track of a kind, its properties loaded (none: nil).
AVAssetTrack* first_track(AVURLAsset* asset, AVMediaType type, const std::string& path)
{
    __block AVAssetTrack* found = nil;
    __block NSError* failure = nil;
    dispatch_semaphore_t done = dispatch_semaphore_create(0);
    [asset loadTracksWithMediaType:type
                 completionHandler:^(NSArray<AVAssetTrack*>* tracks, NSError* error) {
                     failure = error;
                     found = tracks.count ? tracks[0] : nil;
                     dispatch_semaphore_signal(done);
                 }];
    dispatch_semaphore_wait(done, DISPATCH_TIME_FOREVER);
    if (failure)
        throw std::runtime_error(std::format("{}: {}", path, describe(failure)));
    if (!found)
        return nil;
    [found loadValuesAsynchronouslyForKeys:@[ @"nominalFrameRate", @"naturalSize", @"preferredTransform" ]
                         completionHandler:^{
                             dispatch_semaphore_signal(done);
                         }];
    dispatch_semaphore_wait(done, DISPATCH_TIME_FOREVER);
    return found;
}

// Every sample buffer of a track through `sink` (false stops), with the output settings given.
void read_track(AVURLAsset* asset, AVAssetTrack* track, NSDictionary* settings, const std::string& path,
                bool (^sink)(CMSampleBufferRef))
{
    NSError* error = nil;
    AVAssetReader* reader = [AVAssetReader assetReaderWithAsset:asset error:&error];
    if (!reader)
        throw std::runtime_error(std::format("{}: {}", path, describe(error)));
    AVAssetReaderTrackOutput* out = [AVAssetReaderTrackOutput assetReaderTrackOutputWithTrack:track
                                                                               outputSettings:settings];
    out.alwaysCopiesSampleData = NO;
    if (![reader canAddOutput:out])
        throw std::runtime_error(path + ": the track cannot be decoded as asked");
    [reader addOutput:out];
    if (![reader startReading])
        throw std::runtime_error(std::format("{}: {}", path, describe(reader.error)));
    while (true) {
        CMSampleBufferRef sb = [out copyNextSampleBuffer];
        if (!sb)
            break;
        const bool more = sink(sb);
        CFRelease(sb);
        if (!more) {
            [reader cancelReading];
            return;
        }
    }
    if (reader.status == AVAssetReaderStatusFailed)
        throw std::runtime_error(std::format("{}: {}", path, describe(reader.error)));
}

} // namespace

Media read_media(const std::string& path, bool video, bool audio, double max_seconds)
{
    Media result;
    Media* const mp = &result; // the blocks below fill it through the pointer (a block's captures are const)
    @autoreleasepool {
        NSURL* url = [NSURL fileURLWithPath:[NSString stringWithUTF8String:path.c_str()]];
        AVURLAsset* asset = [AVURLAsset URLAssetWithURL:url options:nil];
        if (video) {
            AVAssetTrack* track = first_track(asset, AVMediaTypeVideo, path);
            if (track) {
                mp->fps = track.nominalFrameRate;
                if (!(mp->fps > 0))
                    throw std::runtime_error(path + ": the video track has no frame rate");
                // The rotation the track asks for (a phone's portrait video is stored sideways), as a quarter turn
                // count and a flip from its preferred transform.
                const CGAffineTransform tf = track.preferredTransform;
                const int rot = (std::abs(tf.a) < 0.5) ? (tf.b > 0 ? 1 : 3) : (tf.a > 0 ? 0 : 2);
                const uint64_t limit = max_seconds > 0 ? uint64_t(std::ceil(max_seconds * mp->fps)) + 1 : UINT64_MAX;
                NSDictionary* settings = @{(NSString*)kCVPixelBufferPixelFormatTypeKey : @(kCVPixelFormatType_32BGRA)};
                read_track(asset, track, settings, path, ^bool(CMSampleBufferRef sb) {
                    Media& m = *mp;
                    CVImageBufferRef img = CMSampleBufferGetImageBuffer(sb);
                    if (!img)
                        return true;
                    CVPixelBufferLockBaseAddress(img, kCVPixelBufferLock_ReadOnly);
                    const uint32_t W = uint32_t(CVPixelBufferGetWidth(img)), H = uint32_t(CVPixelBufferGetHeight(img));
                    const size_t stride = CVPixelBufferGetBytesPerRow(img);
                    const auto* base = static_cast<const uint8_t*>(CVPixelBufferGetBaseAddress(img));
                    const uint32_t ow = rot % 2 ? H : W, oh = rot % 2 ? W : H;
                    if (!m.frames) {
                        m.width = ow;
                        m.height = oh;
                    } else if (m.width != ow || m.height != oh) {
                        CVPixelBufferUnlockBaseAddress(img, kCVPixelBufferLock_ReadOnly);
                        throw std::runtime_error(path + ": the frames change size");
                    }
                    const size_t at = m.rgb.size();
                    m.rgb.resize(at + uint64_t(ow) * oh * 3);
                    for (uint32_t y = 0; y < H; ++y) {
                        const uint8_t* row = base + y * stride;
                        for (uint32_t x = 0; x < W; ++x) {
                            uint32_t ox = x, oy = y;
                            if (rot == 1) {
                                ox = H - 1 - y;
                                oy = x;
                            } else if (rot == 2) {
                                ox = W - 1 - x;
                                oy = H - 1 - y;
                            } else if (rot == 3) {
                                ox = y;
                                oy = W - 1 - x;
                            }
                            uint8_t* dst = &m.rgb[at + (uint64_t(oy) * ow + ox) * 3];
                            dst[0] = row[x * 4 + 2];
                            dst[1] = row[x * 4 + 1];
                            dst[2] = row[x * 4];
                        }
                    }
                    CVPixelBufferUnlockBaseAddress(img, kCVPixelBufferLock_ReadOnly);
                    return ++m.frames < limit;
                });
            }
        }
        if (audio) {
            AVAssetTrack* track = first_track(asset, AVMediaTypeAudio, path);
            if (track) {
                NSDictionary* settings = @{
                    AVFormatIDKey : @(kAudioFormatLinearPCM),
                    AVLinearPCMBitDepthKey : @32,
                    AVLinearPCMIsFloatKey : @YES,
                    AVLinearPCMIsNonInterleaved : @NO,
                    AVLinearPCMIsBigEndianKey : @NO
                };
                std::vector<float> interleaved;
                std::vector<float>* const ip = &interleaved;
                read_track(asset, track, settings, path, ^bool(CMSampleBufferRef sb) {
                    Media& m = *mp;
                    std::vector<float>& interleaved = *ip;
                    if (!m.sample_rate) {
                        const auto* asbd = CMAudioFormatDescriptionGetStreamBasicDescription(CMSampleBufferGetFormatDescription(sb));
                        if (!asbd)
                            throw std::runtime_error(path + ": the soundtrack has no format");
                        m.sample_rate = uint32_t(std::lround(asbd->mSampleRate));
                        m.channels = asbd->mChannelsPerFrame;
                    }
                    CMBlockBufferRef block = CMSampleBufferGetDataBuffer(sb);
                    if (!block)
                        return true;
                    const size_t bytes = CMBlockBufferGetDataLength(block), at = interleaved.size();
                    interleaved.resize(at + bytes / 4);
                    if (CMBlockBufferCopyDataBytes(block, 0, bytes, interleaved.data() + at) != kCMBlockBufferNoErr)
                        throw std::runtime_error(path + ": the soundtrack could not be read");
                    return max_seconds <= 0 || interleaved.size() < uint64_t(max_seconds * m.sample_rate + 1) * m.channels;
                });
                if (result.channels) {
                    const uint64_t n = interleaved.size() / result.channels;
                    result.audio.resize(uint64_t(result.channels) * n);
                    for (uint64_t i = 0; i < n; ++i)
                        for (uint32_t c = 0; c < result.channels; ++c)
                            result.audio[uint64_t(c) * n + i] = interleaved[i * result.channels + c];
                }
            }
        }
    }
    return result;
}

} // namespace mako::video
