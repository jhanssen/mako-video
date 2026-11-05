#pragma once

#include "video/engine.hpp"

#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace mako::server {

// A video model as a directory: MiniMax-H3's files as published (model_index.json, text_encoder/, transformer/,
// video_vae/, audio_vae/), or a mako.json pointing at them with its settings (every field optional, paths relative to
// its directory or from ~/):
//   {"model": DIR, "transformer": PATH, "text_encoder": DIR, "lora_dir": DIR, "loras": {NAME: SCALE, ...},
//    "steps": N, "sparse": F, "sparse_local": N, "int8": "none|all|qkv,out,fc1,fc2", "references": BOOL,
//    "description": TEXT, "prompting": TEXT}
// `transformer` takes another directory or a single file of the transformer's weights (Ref2VA's, a pruned merge);
// `references` says the transformer is a Ref2VA one that takes references (default: the directory's or the file's
// name says ref2va). LoRAs are named by file name without .safetensors, from lora_dir (default: the directory's
// loras/, else the parent's loras/); "loras" are the ones merged for requests that name none.
struct VideoModel {
    std::string name, dir;
    std::string description, prompting;
    video::Options options; // transformer, text_encoder, int8, sparse, sparse_local
    std::string lora_dir;   // empty: none
    std::vector<video::Lora> loras;
    uint32_t steps = 0;     // 0: the model's 50
    bool references = false;
    // What decides the loaded weights besides LoRAs.
    std::string key() const
    {
        return dir + "|" + options.transformer + "|" + options.text_encoder + "|" + std::to_string(options.int8) + "|" +
               std::to_string(options.sparse) + "|" + std::to_string(options.sparse_local);
    }
};

// The model in `dir` under `name`. Throws std::runtime_error naming the problem.
VideoModel load_video_model(const std::string& dir, const std::string& name);
// The subdirectories of `root` that hold a model (model_index.json or mako.json), by name, and a directory holding a
// single .safetensors transformer file when `root` has a model directory to lend it the encoders and VAEs (the first by
// name whose model_index.json names MiniMax-H3's parts); read now; ones that fail to load are left out with a
// warning.
std::vector<VideoModel> video_models(const std::string& root);

// A LoRA a request can name: its file name without .safetensors, its path, and for one given at start-up its scale.
struct LoraChoice {
    std::string name, path;
    std::optional<float> scale;
};
// The start-up LoRAs (in their order), then the .safetensors files in `dir` by name (read now; empty: none).
std::vector<LoraChoice> lora_choices(const std::vector<video::Lora>& defaults, const std::string& dir);
// What a LoRA is for, from NAME.json beside the file ({"trigger", "description", "steps"}): its trigger words, a
// description, and the step count it was made for (0: none given); the description else from the file's metadata.
struct LoraNotes {
    std::string trigger, description;
    uint32_t steps = 0;
};
LoraNotes lora_notes(const std::string& path);
// A request's "loras" value (JSON: an array of {"name" (with or without .safetensors), "scale" (default 1)}) as files
// and scales. Throws std::invalid_argument naming the problem.
std::vector<video::Lora> resolve_loras(std::string_view json, const std::vector<LoraChoice>& choices);

} // namespace mako::server
