// OpenAI's video API for MiniMax-H3, as jobs (POST /v1/videos, GET /v1/videos/{id}, GET /v1/videos/{id}/content).
//   mako_video_serve [--video-models DIR] [--video-model NAME|DIR] [--output-dir DIR] [--host H] [--port N]
//                    [--api-key KEY[,KEY...]] [--api-key-file FILE] [--max-waiting N] [--llm URL] [--llm-api-key KEY]
//                    [--keep-text-encoder] [--telemetry LEVEL]
// --video-models: a directory of model directories (MiniMax-H3's files as published, a mako.json pointing at them with
// its settings and LoRAs, or a single transformer file beside such a directory: server::VideoModel); a request's
// "model" picks one (GET /v1/videos/models lists them). --video-model: the one for requests that name none (a name
// there, or a model directory; default: the first). A model loads with the first job for it and stays loaded; another
// model or other LoRAs replace it. --output-dir: where the videos go (default ~/Movies/mako), NAME.mp4 with NAME.png
// (the middle frame). --llm: a mako_serve on this machine whose memory the models borrow (its URL, e.g.
// http://127.0.0.1:8080): before a model loads the LLM is unloaded, and when a chat request comes the model is freed
// and the LLM loads again. Without it the models' memory is their own. --keep-text-encoder keeps the text encoder
// loaded beside the transformer (49 GB more) instead of loading it for each job's prompt; --keep-transformer whether
// the transformer stays loaded while a prompt is encoded (default auto: only on a machine with room for both, 1.35 x
// their files; else it is freed for the prompt and loaded again, ~15 s, which beats swapping).
// Environment variables MAKO_VIDEO_<OPTION> (e.g. MAKO_VIDEO_PORT) stand for the options.
#include "app/cli.hpp"
#include "log/log.hpp"
#include "memory/borrower.hpp"
#include "mtl/context.hpp"
#include "server/video_models.hpp"
#include "server/video_server.hpp"
#include "video/engine.hpp"

#include <cctype>
#include <csignal>
#include <filesystem>
#include <fstream>
#include <memory>
#include <optional>
#include <print>
#include <sstream>
#include <string>
#include <thread>

#include <pthread.h>
#include <unistd.h>

int main(int argc, char** argv)
{
    mako::server::VideoServerOptions so;
    std::string llm, llm_key;
    bool keep_text = false;
    std::optional<bool> keep_transformer;
    mako::gpu::Telemetry telemetry = mako::gpu::Telemetry::off;
    auto add_keys = [&](std::string_view list, char sep) {
        const size_t before = so.api_keys.size();
        for (size_t at = 0; at <= list.size();) {
            const size_t end = std::min(list.find(sep, at), list.size());
            std::string_view k = list.substr(at, end - at);
            while (!k.empty() && std::isspace(static_cast<unsigned char>(k.front())))
                k.remove_prefix(1);
            while (!k.empty() && std::isspace(static_cast<unsigned char>(k.back())))
                k.remove_suffix(1);
            if (!k.empty())
                so.api_keys.emplace_back(k);
            at = end + 1;
        }
        return so.api_keys.size() > before;
    };
    mako::app::Cli("mako_video_serve", "", 0, 0)
        .env_prefix("MAKO_VIDEO_")
        .option("--video-models", "DIR", "a directory of video model directories", so.video_models)
        .option("--video-model", "NAME|DIR", "the model for requests that name none (default: the first)", so.video_model)
        .option("--output-dir", "DIR", "where the videos are written (default ~/Movies/mako)", so.output_dir)
        .option("--host", "H", "address to listen on (default 127.0.0.1; others need an API key)", so.host)
        .option("--port", "N", "port to listen on (default 8083)", so.port)
        .option("--api-key", "KEY[,KEY...]", "keys requests must carry (Bearer or x-api-key)",
                [&](const std::string& v) { add_keys(v, ','); })
        .secret()
        .option("--api-key-file", "FILE", "keys requests must carry, one a line",
                [&](const std::string& v) {
                    std::ifstream in(v);
                    std::stringstream text;
                    text << in.rdbuf();
                    if (!in || !add_keys(text.str(), '\n'))
                        throw std::invalid_argument("names no readable key file");
                })
        .shared_env("--api-key")
        .option("--max-waiting", "N", "videos that may wait in the queue before new ones are refused", so.max_waiting)
        .option("--llm", "URL", "a mako_serve on this machine to borrow memory from (its LLM unloads meanwhile)", llm)
        .option("--llm-api-key", "KEY", "the key that mako_serve wants", llm_key)
        .secret()
        .flag("--keep-text-encoder", "keep the text encoder loaded beside the transformer", keep_text)
        .option("--keep-transformer", "auto|on|off", "keep the transformer loaded while a prompt is encoded (default auto: when both fit with room)",
                [&](const std::string& v) {
                    if (v == "on")
                        keep_transformer = true;
                    else if (v == "off")
                        keep_transformer = false;
                    else if (v != "auto")
                        throw std::invalid_argument("must be auto, on or off");
                })
        .option("--telemetry", "LEVEL", "off, summary, steps, layers or kernels: the transformer's GPU time per kernel",
                [&](const std::string& v) {
                    const auto t = mako::gpu::parse_telemetry(v);
                    if (!t)
                        throw std::invalid_argument("must be off, summary, steps, layers or kernels");
                    telemetry = *t;
                })
        .parse(argc, argv);

    sigset_t stop_signals;
    sigemptyset(&stop_signals);
    sigaddset(&stop_signals, SIGINT);
    sigaddset(&stop_signals, SIGTERM);
    pthread_sigmask(SIG_BLOCK, &stop_signals, nullptr);
    for (const auto& k : so.api_keys)
        if (k.size() < 16)
            mako::log::warn("an API key of {} characters is easy to guess (16 or more; e.g. openssl rand -hex 24)", k.size());
    if (so.host != "127.0.0.1" && so.host != "localhost" && so.host != "::1" && so.api_keys.empty()) {
        std::println(stderr, "--host {} serves beyond this machine: give an --api-key or --api-key-file", so.host);
        return 2;
    }
    if (so.video_models.empty() && so.video_model.empty()) {
        std::println(stderr, "give --video-models DIR or --video-model DIR");
        return 2;
    }
    if (!so.video_models.empty() && !std::filesystem::is_directory(so.video_models)) {
        std::println(stderr, "--video-models {}: not a directory", so.video_models);
        return 2;
    }
    if (so.video_models.empty() && !std::filesystem::is_directory(so.video_model)) {
        std::println(stderr, "--video-model {}: not a directory (or give --video-models)", so.video_model);
        return 2;
    }
    if (so.output_dir.empty()) {
        const char* home = std::getenv("HOME");
        so.output_dir = (std::filesystem::path(home ? home : ".") / "Movies" / "mako").string();
    }
    std::error_code ec;
    std::filesystem::create_directories(so.output_dir, ec);
    if (!std::filesystem::is_directory(so.output_dir)) {
        std::println(stderr, "--output-dir {}: cannot be made", so.output_dir);
        return 2;
    }

    mako::mtl::Context ctx;
    std::unique_ptr<mako::memory::Borrower> borrower;
    if (!llm.empty())
        borrower = std::make_unique<mako::memory::Borrower>(llm, llm_key, std::format("mako_video_serve :{}", so.port));
    mako::server::VideoResidency residency(std::move(borrower));
    const auto load = [&](const mako::server::VideoModel& m, const std::vector<mako::video::Lora>& loras) {
        mako::video::Options o = m.options;
        o.keep_text_encoder = keep_text;
        o.keep_transformer = keep_transformer;
        o.telemetry = telemetry;
        auto e = std::make_unique<mako::video::Engine>(m.dir, ctx, o);
        for (const auto& l : loras)
            e->add_lora(l);
        return e;
    };
    if (!so.video_models.empty())
        for (const auto& m : mako::server::video_models(so.video_models))
            mako::log::info("video model {}: {}{}{}{}", m.name, m.dir,
                            m.options.transformer.empty() ? "" : ", transformer " + m.options.transformer,
                            m.references ? " (takes references)" : "", m.loras.empty() ? "" : " (with LoRAs)");
    mako::log::info("models load with the first video for them{}; videos go to {}",
                    llm.empty() ? "" : std::format("; memory borrowed from the LLM at {}", llm), so.output_dir);
    mako::server::VideoServer server(residency, load, so);
    std::jthread stopper([&] {
        int sig = 0;
        sigwait(&stop_signals, &sig);
        server.stop();
    });
    mako::log::info("serving videos on http://{}:{}/v1", so.host, so.port);
    const bool listened = server.run();
    if (!listened)
        mako::log::error("could not listen on {}:{}", so.host, so.port);
    ::kill(::getpid(), SIGTERM); // wakes the stopper if the server ended on its own
    return listened ? 0 : 1;
}
