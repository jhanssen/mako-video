#include "server/video_models.hpp"

#include "log/log.hpp"
#include "safetensors/safetensors.hpp"

#include <glaze/glaze.hpp>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <filesystem>
#include <format>
#include <fstream>
#include <iterator>
#include <stdexcept>

namespace mako::server {

namespace {

namespace fs = std::filesystem;

std::string resolve(const fs::path& base, const std::string& p)
{
    if (p.starts_with("~/"))
        if (const char* home = std::getenv("HOME"))
            return (fs::path(home) / p.substr(2)).lexically_normal().string();
    const fs::path q(p);
    return (q.is_absolute() ? q : base / q).lexically_normal().string();
}

std::string lower(std::string s)
{
    for (char& c : s)
        c = char(std::tolower(static_cast<unsigned char>(c)));
    return s;
}

bool names_ref2va(const std::string& path)
{
    const std::string l = lower(path);
    return l.contains("ref2va") || l.contains("ref2v");
}

// A directory holding MiniMax-H3's parts (its model_index.json names the text encoder and the two VAEs).
bool holds_h3(const fs::path& d)
{
    return fs::exists(d / "model_index.json") && fs::is_directory(d / "text_encoder") && fs::is_directory(d / "video_vae") &&
           fs::is_directory(d / "audio_vae");
}

// The single .safetensors file in a directory ("" when there is not exactly one).
std::string single_file(const fs::path& d)
{
    std::string found;
    std::error_code ec;
    for (const auto& e : fs::directory_iterator(d, ec))
        if (e.path().extension() == ".safetensors" && e.is_regular_file(ec)) {
            if (!found.empty())
                return {};
            found = e.path().string();
        }
    return found;
}

VideoModel load_model(const std::string& dir, const std::string& name, const std::string& parts_dir)
{
    const fs::path base = fs::absolute(fs::path(dir)).lexically_normal();
    VideoModel m;
    m.name = name;
    m.dir = base.string();
    std::string lora_dir = fs::is_directory(base / "loras") ? (base / "loras").string() : (base.parent_path() / "loras").string();
    const fs::path cfg = base / "mako.json";
    std::optional<bool> references;
    if (fs::exists(cfg)) {
        std::ifstream in(cfg, std::ios::binary);
        const std::string text{std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
        glz::generic j;
        if (glz::read_json(j, text) || !j.is_object())
            throw std::runtime_error(cfg.string() + ": not a JSON object");
        auto str = [&](const char* key) -> std::optional<std::string> {
            if (!j.contains(key))
                return std::nullopt;
            if (!j[key].is_string())
                throw std::runtime_error(std::format("{}: {} must be a string", cfg.string(), key));
            return j[key].get_string();
        };
        auto num = [&](const char* key) -> std::optional<double> {
            if (!j.contains(key))
                return std::nullopt;
            if (!j[key].is_number())
                throw std::runtime_error(std::format("{}: {} must be a number", cfg.string(), key));
            return j[key].get_number();
        };
        for (const auto& [k, v] : j.get_object())
            if (k != "model" && k != "transformer" && k != "text_encoder" && k != "lora_dir" && k != "loras" &&
                k != "steps" && k != "sparse" && k != "sparse_local" && k != "int8" && k != "references" &&
                k != "description" && k != "prompting")
                throw std::runtime_error(std::format("{}: unknown field {}", cfg.string(), k));
        if (auto v = str("model"))
            m.dir = resolve(base, *v);
        if (auto v = str("transformer"))
            m.options.transformer = resolve(base, *v);
        if (auto v = str("text_encoder"))
            m.options.text_encoder = resolve(base, *v);
        if (auto v = str("description"))
            m.description = *v;
        if (auto v = str("prompting"))
            m.prompting = *v;
        if (auto v = str("lora_dir"))
            lora_dir = resolve(base, *v);
        if (auto v = num("steps")) {
            if (*v < 0 || *v > 100 || *v != double(uint32_t(*v)))
                throw std::runtime_error(cfg.string() + ": steps must be a whole number from 0 to 100");
            m.steps = uint32_t(*v);
        }
        if (auto v = num("sparse")) {
            if (*v < 0 || *v > 1)
                throw std::runtime_error(cfg.string() + ": sparse must be a number from 0 to 1");
            m.options.sparse = float(*v);
        }
        if (auto v = num("sparse_local")) {
            if (*v != 0 && *v != 1 && *v != 3)
                throw std::runtime_error(cfg.string() + ": sparse_local must be 0, 1 or 3");
            m.options.sparse_local = uint32_t(*v);
        }
        if (auto v = str("int8")) {
            const auto mask = video::H3::parse_int8(*v);
            if (!mask)
                throw std::runtime_error(cfg.string() + ": int8 must be none, all, or some of qkv, out, fc1, fc2");
            m.options.int8 = *mask;
        }
        if (j.contains("references")) {
            if (!j["references"].is_boolean())
                throw std::runtime_error(cfg.string() + ": references must be true or false");
            references = j["references"].get_boolean();
        }
        if (j.contains("loras")) {
            if (!j["loras"].is_object())
                throw std::runtime_error(cfg.string() + ": loras must be an object of name: scale");
            for (const auto& [n, s] : j["loras"].get_object()) {
                if (!s.is_number())
                    throw std::runtime_error(std::format("{}: LoRA {} must map to its scale", cfg.string(), n));
                const fs::path f = fs::path(lora_dir) / (n.ends_with(".safetensors") ? n : n + ".safetensors");
                if (!fs::exists(f))
                    throw std::runtime_error(std::format("{}: no LoRA {}", cfg.string(), f.string()));
                m.loras.push_back({f.string(), float(s.get_number())});
            }
        }
    } else if (!holds_h3(base)) {
        // A directory of one transformer file, with another directory's encoders and VAEs.
        const std::string file = single_file(base);
        if (file.empty() || parts_dir.empty())
            throw std::runtime_error(std::format("{}: no model_index.json or mako.json{}", base.string(),
                                                 file.empty() ? "" : " (a transformer file alone wants a model directory beside it)"));
        m.options.transformer = fs::absolute(file).lexically_normal().string();
        m.dir = fs::absolute(parts_dir).lexically_normal().string();
    }
    for (const std::string* p : {&m.options.transformer, &m.options.text_encoder})
        if (!p->empty() && !fs::exists(*p))
            throw std::runtime_error(std::format("{}: {} does not exist", name, *p));
    if (!holds_h3(m.dir))
        throw std::runtime_error(std::format("{}: {} holds no MiniMax-H3 (model_index.json, text_encoder/, video_vae/, audio_vae/)", name, m.dir));
    if (fs::is_directory(lora_dir))
        m.lora_dir = lora_dir;
    m.references = references ? *references : names_ref2va(m.options.transformer.empty() ? m.dir : m.options.transformer);
    return m;
}

} // namespace

VideoModel load_video_model(const std::string& dir, const std::string& name) { return load_model(dir, name, ""); }

std::vector<VideoModel> video_models(const std::string& root)
{
    std::vector<VideoModel> out;
    std::vector<fs::path> dirs;
    std::error_code ec;
    for (const auto& e : fs::directory_iterator(root, ec))
        if (e.is_directory(ec) && (fs::exists(e.path() / "model_index.json") || fs::exists(e.path() / "mako.json") ||
                                   !single_file(e.path()).empty()))
            dirs.push_back(e.path());
    std::ranges::sort(dirs);
    std::string parts;
    for (const auto& d : dirs)
        if (holds_h3(d) && !names_ref2va(d.string())) {
            parts = d.string();
            break;
        }
    if (parts.empty())
        for (const auto& d : dirs)
            if (holds_h3(d)) {
                parts = d.string();
                break;
            }
    for (const auto& d : dirs) {
        try {
            out.push_back(load_model(d.string(), d.filename().string(), parts));
        } catch (const std::exception& e) {
            log::warn("video model {}: {}", d.filename().string(), e.what());
        }
    }
    return out;
}

std::vector<LoraChoice> lora_choices(const std::vector<video::Lora>& defaults, const std::string& dir)
{
    std::vector<LoraChoice> out;
    auto known = [&](const std::string& name) { return std::ranges::contains(out, name, &LoraChoice::name); };
    for (const auto& l : defaults) {
        const std::string name = fs::path(l.path).stem().string();
        if (!known(name))
            out.push_back({name, l.path, l.scale});
    }
    if (dir.empty())
        return out;
    std::vector<fs::path> files;
    std::error_code ec;
    for (const auto& e : fs::directory_iterator(dir, ec))
        if (e.path().extension() == ".safetensors" && e.is_regular_file(ec))
            files.push_back(e.path());
    std::ranges::sort(files);
    for (const auto& f : files)
        if (!known(f.stem().string()))
            out.push_back({f.stem().string(), f.string(), std::nullopt});
    return out;
}

LoraNotes lora_notes(const std::string& path)
{
    LoraNotes n;
    const auto sidecar = fs::path(path).replace_extension(".json");
    if (std::ifstream in(sidecar); in) {
        const std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        glz::generic v;
        if (glz::read_json(v, text) || !v.is_object()) {
            log::warn("{}: not a JSON object; its notes left out", sidecar.string());
        } else {
            if (v.contains("trigger") && v["trigger"].is_string())
                n.trigger = v["trigger"].get_string();
            if (v.contains("description") && v["description"].is_string())
                n.description = v["description"].get_string();
            if (v.contains("steps") && v["steps"].is_number() && v["steps"].get_number() > 0)
                n.steps = uint32_t(v["steps"].get_number());
        }
    }
    if (n.steps == 0) { // the turbo LoRAs say it in their file names: ..._4step_..., ..._8Step...
        const std::string l = lower(fs::path(path).stem().string());
        for (uint32_t k : {4u, 8u})
            if (l.contains(std::format("{}step", k)) || l.contains(std::format("{}-step", k)))
                n.steps = k;
    }
    if (!n.description.empty())
        return n;
    auto f = safetensors::File::open(path);
    if (!f)
        return n;
    const auto& m = f->metadata();
    if (auto it = m.find("modelspec.description"); it != m.end())
        n.description = it->second;
    if (n.trigger.empty())
        for (const char* k : {"trigger_word", "modelspec.trigger_phrase"})
            if (auto it = m.find(k); it != m.end() && !it->second.empty()) {
                n.trigger = it->second;
                break;
            }
    return n;
}

std::vector<video::Lora> resolve_loras(std::string_view json, const std::vector<LoraChoice>& choices)
{
    glz::generic v;
    if (glz::read_json(v, json) || !v.is_array())
        throw std::invalid_argument("an array of {\"name\", \"scale\"} objects");
    std::vector<video::Lora> out;
    std::vector<std::string> names;
    for (auto& e : v.get_array()) {
        if (!e.is_object() || !e.contains("name") || !e["name"].is_string())
            throw std::invalid_argument("each entry needs a \"name\"");
        std::string name = e["name"].get_string();
        if (name.ends_with(".safetensors"))
            name.resize(name.size() - std::string_view(".safetensors").size());
        const auto it = std::ranges::find(choices, name, &LoraChoice::name);
        if (it == choices.end())
            throw std::invalid_argument(std::format("no LoRA named \"{}\" (GET /v1/videos/loras lists them)", name));
        if (std::ranges::contains(names, name))
            throw std::invalid_argument(std::format("\"{}\" named twice", name));
        float scale = 1.0f;
        if (e.contains("scale") && !e["scale"].is_null()) {
            if (!e["scale"].is_number() || !std::isfinite(e["scale"].get_number()) ||
                std::abs(e["scale"].get_number()) > 10)
                throw std::invalid_argument(std::format("\"{}\": scale must be a number from -10 to 10", name));
            scale = float(e["scale"].get_number());
        }
        names.push_back(name);
        out.push_back({it->path, scale});
    }
    return out;
}

} // namespace mako::server
