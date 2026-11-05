// The video server's HTTP side without a model: health and status, access checks, request validation against a
// stand-in model directory, the job queue (a job that fails to load, its retrieval, content before completion, its
// deletion), the model and LoRA listings.
//   video_server_test <scratch dir>
#include "server/video_server.hpp"

#include <httplib.h>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <print>
#include <stdexcept>
#include <string>
#include <thread>

using namespace std::chrono_literals;
namespace fs = std::filesystem;

namespace {

int failures = 0;

void check(bool ok, const std::string& what)
{
    std::println("  {}  {}", ok ? "ok  " : "FAIL", what);
    failures += !ok;
}

void write(const fs::path& p, const std::string& text)
{
    fs::create_directories(p.parent_path());
    std::ofstream(p) << text;
}

std::string field(const std::string& json, const std::string& key)
{
    const auto at = json.find("\"" + key + "\":");
    if (at == std::string::npos)
        return {};
    size_t s = at + key.size() + 3;
    if (json[s] == '"') {
        const auto e = json.find('"', s + 1);
        return json.substr(s + 1, e - s - 1);
    }
    const auto e = json.find_first_of(",}", s);
    return json.substr(s, e - s);
}

} // namespace

int main(int argc, char** argv)
{
    if (argc < 2) {
        std::println(stderr, "video_server_test <scratch dir>");
        return 2;
    }
    const fs::path root = fs::path(argv[1]) / "video_server_scratch";
    fs::remove_all(root);
    for (const char* name : {"FL2VA", "Ref2VA"}) {
        write(root / "models" / name / "model_index.json", "{}");
        for (const char* sub : {"text_encoder", "video_vae", "audio_vae", "transformer"})
            fs::create_directories(root / "models" / name / sub);
    }
    write(root / "models" / "loras" / "turbo_8step.safetensors", "");

    mako::server::VideoResidency residency(nullptr);
    mako::server::VideoServerOptions opts;
    opts.port = 0;
    opts.video_models = (root / "models").string();
    opts.output_dir = (root / "out").string();
    opts.max_waiting = 1;
    int loads = 0;
    mako::server::VideoServer server(
        residency,
        [&](const mako::server::VideoModel&, const std::vector<mako::video::Lora>&) -> std::unique_ptr<mako::video::Engine> {
            ++loads;
            throw std::runtime_error("no models in this test");
        },
        opts);
    std::thread t([&] { server.run(); });
    for (int i = 0; i < 200 && !server.port(); ++i)
        std::this_thread::sleep_for(10ms);
    httplib::Client c("127.0.0.1", server.port());

    auto r = c.Get("/health");
    check(r && r->status == 200, "/health");
    r = c.Get("/v1/status");
    check(r && r->status == 200 && r->body.contains(R"("state":"none")") && r->body.contains(R"("queued":0)"), "/v1/status: nothing loaded, nothing queued");
    r = c.Get("/v1/videos/models");
    check(r && r->status == 200 && r->body.contains(R"("id":"FL2VA")") && r->body.contains(R"("id":"Ref2VA","family":"MiniMax-H3","references":true)") &&
              r->body.contains(R"("default":true)"),
          "/v1/videos/models lists both, FL2VA the default, Ref2VA taking references");
    r = c.Get("/v1/videos/loras");
    check(r && r->status == 200 && r->body.contains(R"("name":"turbo_8step","steps":8)"), "/v1/videos/loras: the directory's LoRA with its step count");
    r = c.Get("/v1/status", httplib::Headers{{"Host", "evil.example.com"}});
    check(r && r->status == 403, "another Host is refused without a key");
    r = c.Get("/v1/status", httplib::Headers{{"Origin", "https://evil.example.com"}});
    check(r && r->status == 403, "another web page is refused without a key");
    const auto post = [&](const std::string& body) { return c.Post("/v1/videos", body, "application/json"); };
    r = post("not json");
    check(r && r->status == 400, "a body that is not JSON");
    r = post(R"({"size": "1344x768"})");
    check(r && r->status == 400 && r->body.contains("prompt is required"), "no prompt");
    r = post(R"({"prompt": "a fox", "size": "100x100"})");
    check(r && r->status == 400 && r->body.contains("size must be"), "a size the model cannot make");
    r = post(R"({"prompt": "a fox", "seconds": 3})");
    check(r && r->status == 400 && r->body.contains("seconds must be"), "too short");
    r = post(R"({"prompt": "a fox", "seconds": "20"})");
    check(r && r->status == 400 && r->body.contains("seconds must be"), "too long (as OpenAI's string)");
    r = post(R"({"prompt": "a fox", "steps": 1})");
    check(r && r->status == 400 && r->body.contains("at least 2"), "one step");
    r = post(R"({"prompt": "a fox", "model": "nope"})");
    check(r && r->status == 400 && r->body.contains("no video model"), "an unknown model");
    r = post(R"({"prompt": "a fox", "loras": [{"name": "nope"}]})");
    check(r && r->status == 400 && r->body.contains("no LoRA named"), "an unknown LoRA");
    r = post(R"({"prompt": "a fox", "input_reference": "not base64!"})");
    check(r && r->status == 400 && r->body.contains("not base64"), "a keyframe that is not base64");
    r = post(R"({"prompt": "a fox", "input_reference": {"image_url": "https://example.com/a.png"}})");
    check(r && r->status == 400 && r->body.contains("not fetched"), "a keyframe by web URL");
    r = post(R"({"prompt": "a fox", "references": [{"type": "image", "data": "aGVsbG8="}]})");
    check(r && r->status == 400 && r->body.contains("does not take references"), "references for the FL2VA model");
    r = post(R"({"prompt": "a fox", "model": "Ref2VA", "references": [{"type": "audio", "data": "aGVsbG8="}]})");
    check(r && r->status == 400 && r->body.contains("references[0]"), "a soundtrack that is not a media file");
    r = post(R"({"prompt": "a fox", "frames": 124})");
    check(r && r->status == 400 && r->body.contains("unknown field frames"), "a stray field");
    r = post(std::string(R"({"prompt": ")") + std::string(20000, 'a') + R"("})");
    check(r && r->status == 400 && r->body.contains("longer than"), "a prompt too long");

    // A job: queued at once, failing when its model cannot load, then deleted.
    r = post(R"({"prompt": "a fox", "seconds": "5", "size": "512x288", "steps": 9, "seed": 7, "loras": [{"name": "turbo_8step", "scale": 0.9}]})");
    check(r && r->status == 200 && (field(r->body, "status") == "queued" || field(r->body, "status") == "in_progress") &&
              field(r->body, "size") == "512x288" && field(r->body, "frames") == "124" && field(r->body, "seconds") == "5.16667" &&
              field(r->body, "seed") == "7" && r->body.contains(R"("loras":[{"name":"turbo_8step","scale":0.9}])"),
          "a job is accepted and described");
    const std::string id = r ? field(r->body, "id") : "";
    check(id.starts_with("video_"), "its id: " + id);
    std::string status;
    for (int i = 0; i < 300 && status != "failed"; ++i) {
        std::this_thread::sleep_for(10ms);
        r = c.Get("/v1/videos/" + id);
        status = r ? field(r->body, "status") : "";
    }
    check(status == "failed" && r->body.contains("no models in this test") && loads == 1, "the job failed on its model's load, with the error");
    r = c.Get("/v1/videos/" + id + "/content");
    check(r && r->status == 409, "no content for a failed job");
    r = c.Get("/v1/videos");
    check(r && r->status == 200 && r->body.contains(id) && r->body.contains(R"("has_more":false)"), "the list has it");
    r = c.Get("/v1/videos/nope");
    check(r && r->status == 404, "an unknown id");
    r = c.Delete("/v1/videos/" + id);
    check(r && r->status == 200 && r->body.contains(R"("deleted":true)"), "deleted");
    r = c.Get("/v1/videos/" + id);
    check(r && r->status == 404, "gone after the delete");
    r = c.Get("/v1/videos");
    check(r && r->status == 200 && r->body.contains(R"("data":[])"), "the list is empty again");

    server.stop();
    t.join();
    std::println("{}", failures ? "FAILED" : "all passed");
    return failures ? 1 : 0;
}
