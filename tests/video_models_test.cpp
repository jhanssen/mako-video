// Video model directories (server/video_models.hpp) in a scratch tree: MiniMax-H3's layout, a Ref2VA beside it, a
// single transformer file lent the first's encoders, a mako.json with settings and LoRAs, a broken one left out;
// LoRA listing, notes and a request's LoRA resolution.
//   video_models_test <scratch dir>
#include "server/video_models.hpp"

#include <filesystem>
#include <fstream>
#include <print>
#include <stdexcept>
#include <string>

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

// MiniMax-H3's parts as empty stand-ins (the listing reads the layout, not the weights).
void parts(const fs::path& d)
{
    write(d / "model_index.json", R"({"_class_name": "MiniMaxH3Pipeline"})");
    for (const char* sub : {"text_encoder", "video_vae", "audio_vae", "transformer"})
        fs::create_directories(d / sub);
}

} // namespace

int main(int argc, char** argv)
{
    if (argc < 2) {
        std::println(stderr, "video_models_test <scratch dir>");
        return 2;
    }
    const fs::path root = fs::path(argv[1]) / "video_models_scratch";
    fs::remove_all(root);
    parts(root / "FL2VA");
    parts(root / "Ref2VA");
    write(root / "eros-max" / "h3ErosMax.safetensors", "");
    write(root / "loras" / "minimax_h3_fl2v_turbo_8step_v1.1_768p_bf16.safetensors", "");
    write(root / "loras" / "style.safetensors", "");
    write(root / "loras" / "style.json", R"({"trigger": "stylized", "description": "a look", "steps": 0})");
    write(root / "turbo" / "mako.json",
          R"({"model": "../FL2VA", "transformer": "../eros-max/h3ErosMax.safetensors", "steps": 9, "sparse": 0.1,
              "loras": {"style": 0.8}, "references": false, "description": "a merge", "prompting": "plain sentences"})");
    write(root / "broken" / "mako.json", R"({"model": "../nowhere"})");
    write(root / "notes" / "readme.txt", "not a model");

    using mako::server::video_models;
    const auto list = video_models(root.string());
    std::string names;
    for (const auto& m : list)
        names += (names.empty() ? "" : ",") + m.name;
    check(names == "FL2VA,Ref2VA,eros-max,turbo", "the models listed, by name: " + names);
    auto find = [&](const std::string& n) -> const mako::server::VideoModel* {
        for (const auto& m : list)
            if (m.name == n)
                return &m;
        return nullptr;
    };
    if (const auto* m = find("FL2VA")) {
        check(m->dir == fs::absolute(root / "FL2VA").lexically_normal().string() && m->options.transformer.empty() && !m->references,
              "FL2VA: its own directory, no references");
        check(m->lora_dir == (fs::absolute(root).lexically_normal() / "loras").string(), "FL2VA: the parent's loras/");
    }
    if (const auto* m = find("Ref2VA"))
        check(m->references, "Ref2VA: takes references (by its name)");
    if (const auto* m = find("eros-max"))
        check(m->dir == fs::absolute(root / "FL2VA").lexically_normal().string() &&
                  m->options.transformer == fs::absolute(root / "eros-max" / "h3ErosMax.safetensors").lexically_normal().string(),
              "a transformer file alone: FL2VA's encoders and VAEs with that transformer");
    if (const auto* m = find("turbo")) {
        check(m->steps == 9 && m->options.sparse == 0.1f && !m->references && m->description == "a merge" && m->prompting == "plain sentences",
              "mako.json: steps, sparse, references, description, prompting");
        check(m->loras.size() == 1 && m->loras[0].scale == 0.8f && m->loras[0].path.ends_with("style.safetensors"),
              "mako.json: the default LoRA from the parent's loras/");
        check(m->key() != find("eros-max")->key(), "another sparse setting is another key");
    }

    // LoRA choices and notes.
    const auto choices = mako::server::lora_choices(find("turbo")->loras, find("turbo")->lora_dir);
    check(choices.size() == 2 && choices[0].name == "style" && choices[0].scale == 0.8f &&
              choices[1].name == "minimax_h3_fl2v_turbo_8step_v1.1_768p_bf16" && !choices[1].scale,
          "LoRA choices: the default first with its scale, then the directory's");
    const auto notes = mako::server::lora_notes(choices[1].path);
    check(notes.steps == 8, "the turbo LoRA's step count from its file name");
    const auto style = mako::server::lora_notes(choices[0].path);
    check(style.trigger == "stylized" && style.description == "a look" && style.steps == 0, "notes from the sidecar JSON");
    const auto picked = mako::server::resolve_loras(R"([{"name": "style.safetensors", "scale": 0.5}, {"name": "minimax_h3_fl2v_turbo_8step_v1.1_768p_bf16"}])", choices);
    check(picked.size() == 2 && picked[0].scale == 0.5f && picked[1].scale == 1.0f, "a request's LoRAs resolved");
    for (const auto& [json, why] : {std::pair{R"([{"name": "nope"}])", "no LoRA named"}, {R"([{"name": "style"}, {"name": "style"}])", "named twice"},
                                    {R"({"name": "style"})", "an array"}, {R"([{"name": "style", "scale": 99}])", "scale must be"}}) {
        bool threw = false;
        try {
            (void)mako::server::resolve_loras(json, choices);
        } catch (const std::invalid_argument& e) {
            threw = std::string(e.what()).contains(why);
        }
        check(threw, std::format("rejected: {}", why));
    }
    std::println("{}", failures ? "FAILED" : "all passed");
    return failures ? 1 : 0;
}
