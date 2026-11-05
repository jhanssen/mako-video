#pragma once

#include "memory/residency.hpp"
#include "server/video_models.hpp"
#include "video/engine.hpp"

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace mako::server {

using VideoResidency = memory::Residency<video::Engine>;
// Loads a model with LoRAs to merge into its transformer.
using LoadVideo = std::function<std::unique_ptr<video::Engine>(const VideoModel&, const std::vector<video::Lora>&)>;

struct VideoServerOptions {
    std::string host = "127.0.0.1";
    int port = 8083;
    std::vector<std::string> api_keys; // empty: requests only from this machine (and no other web page)
    // A directory of model directories (read on each request; see VideoModel), and the one for requests that name
    // none (a name there, or a model directory alone; default: the first).
    std::string video_models, video_model;
    std::string output_dir;   // where the videos are written, NAME.mp4 and NAME.png (the middle frame)
    uint32_t max_waiting = 4; // jobs queued beyond the one generating; more get HTTP 503
};

// OpenAI's video API, as jobs: POST /v1/videos queues one and returns it at once; GET /v1/videos/{id} follows it
// (status queued, in_progress, completed or failed; progress; our stage); GET /v1/videos/{id}/content downloads the
// MP4 (variant=thumbnail: the middle frame as a PNG); GET /v1/videos lists them; DELETE /v1/videos/{id} drops one, a
// running one stopped at its next step. GET /v1/videos/models and GET /v1/videos/loras list what a request can name;
// /health and /v1/status.
class VideoServer {
public:
    VideoServer(VideoResidency& residency, LoadVideo load, VideoServerOptions opts);
    ~VideoServer();
    // Blocks until stop(); false if it could not listen.
    bool run();
    void stop();
    // The port listened on (after run() started; the given one, or the one chosen for port 0).
    int port() const;

    // The model a request names ("" for the default). Throws std::invalid_argument when there is none such.
    VideoModel model(const std::string& name) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace mako::server
