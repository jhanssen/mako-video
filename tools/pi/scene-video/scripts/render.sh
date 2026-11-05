#!/bin/sh
# Makes a video with mako_video_serve's /v1/videos, waits for it, and prints the MP4's path.
#   render.sh [--model NAME] [--lora NAME[:SCALE]]... [--no-loras] [--steps N] [--seconds F] [--seed N]
#             [--first-frame FILE] [--last-frame FILE] [--ref-image FILE | --ref-video FILE | --ref-audio FILE]...
#             WIDTHxHEIGHT|auto "prompt"
# --model makes it with that video model (default: the server's); --lora (repeatable) with these LoRAs in place of the
# model's defaults; --no-loras with none; --steps N schedule points (0: the model's 50; a turbo LoRA wants 9 for 8
# steps, 5 for 4); --seconds F the length, 5 to 15 (default 5.2); --seed N the noise's seed (default: a new one);
# --first-frame / --last-frame FILE keyframes (fl2va; size "auto" takes the first frame's aspect ratio);
# --ref-image / --ref-video / --ref-audio FILE references (ref2va, a Ref2VA model; repeatable, read in the order given:
# the prompt names them "Picture 1", "Video 1", "Audio 1"). MAKO_VIDEO_URL (mako_video_serve; default
# http://127.0.0.1:8083/v1), MAKO_VIDEO_API_KEY (when the server wants one), MAKO_VIDEO_DIR (where to save a copy when
# the server is on another machine; default ~/Movies/mako), MAKO_VIDEO_OPEN=0 (not opened in the video player).
set -eu
usage() {
    echo "usage: render.sh [--model NAME] [--lora NAME[:SCALE]]... [--no-loras] [--steps N] [--seconds F] [--seed N]" >&2
    echo "                 [--first-frame FILE] [--last-frame FILE] [--ref-image|--ref-video|--ref-audio FILE]..." >&2
    echo "                 WIDTHxHEIGHT|auto \"prompt\"" >&2
    exit 2
}
model=
loras=
no_loras=
steps=
seconds=
seed=
first=
last=
refs=
while [ $# -gt 0 ]; do
    case $1 in
    --model) [ $# -ge 2 ] || usage; model=$2; shift 2 ;;
    --lora) [ $# -ge 2 ] || usage; loras="$loras$2
"; shift 2 ;;
    --no-loras) no_loras=1; shift ;;
    --steps) [ $# -ge 2 ] || usage; steps=$2; shift 2 ;;
    --seconds) [ $# -ge 2 ] || usage; seconds=$2; shift 2 ;;
    --seed) [ $# -ge 2 ] || usage; seed=$2; shift 2 ;;
    --first-frame) [ $# -ge 2 ] || usage; first=$2; [ -r "$first" ] || { echo "render.sh: cannot read $first" >&2; exit 2; }; shift 2 ;;
    --last-frame) [ $# -ge 2 ] || usage; last=$2; [ -r "$last" ] || { echo "render.sh: cannot read $last" >&2; exit 2; }; shift 2 ;;
    --ref-image | --ref-video | --ref-audio)
        [ $# -ge 2 ] || usage
        [ -r "$2" ] || { echo "render.sh: cannot read $2" >&2; exit 2; }
        refs="$refs${1#--ref-} $2
"
        shift 2 ;;
    --help | -h) usage ;;
    --*) echo "render.sh: unknown option $1" >&2; usage ;;
    *) break ;;
    esac
done
[ $# -eq 2 ] || usage
size=$1
prompt=$2
url=${MAKO_VIDEO_URL:-http://127.0.0.1:8083/v1}
auth=
[ -z "${MAKO_VIDEO_API_KEY:-}" ] || auth="Authorization: Bearer $MAKO_VIDEO_API_KEY"
body=$(PROMPT=$prompt SIZE=$size MODEL=$model LORAS=$loras NO_LORAS=$no_loras STEPS=$steps SECONDS_=$seconds SEED=$seed FIRST=$first LAST=$last REFS=$refs python3 -c '
import base64, json, os
e = os.environ
b = {"prompt": e["PROMPT"], "size": e["SIZE"]}
if e["MODEL"]: b["model"] = e["MODEL"]
if e["STEPS"]: b["steps"] = int(e["STEPS"])
if e["SECONDS_"]: b["seconds"] = float(e["SECONDS_"])
if e["SEED"]: b["seed"] = int(e["SEED"])
if e["NO_LORAS"]: b["loras"] = []
elif e["LORAS"].strip():
    b["loras"] = [{"name": s.split(":")[0], "scale": float(s.split(":")[1]) if ":" in s else 1.0} for s in e["LORAS"].split("\n") if s]
def data(path): return base64.b64encode(open(path, "rb").read()).decode()
if e["FIRST"]: b["input_reference"] = data(e["FIRST"])
if e["LAST"]: b["last_frame"] = data(e["LAST"])
refs = [l.split(" ", 1) for l in e["REFS"].split("\n") if l]
if refs: b["references"] = [{"type": k, "data": data(p)} for k, p in refs]
print(json.dumps(b))')
response=$(printf '%s' "$body" | curl -sS -X POST "$url/videos" -H 'Content-Type: application/json' ${auth:+-H "$auth"} --data-binary @-)
id=$(printf '%s' "$response" | python3 -c 'import json, sys
r = json.load(sys.stdin)
if r.get("error"): sys.exit("render.sh: " + r["error"]["message"])
print(r["id"])')
echo "$id queued" >&2
status=queued
stage=
while [ "$status" = queued ] || [ "$status" = in_progress ]; do
    sleep 5
    response=$(curl -sS "$url/videos/$id" ${auth:+-H "$auth"})
    set -- $(printf '%s' "$response" | python3 -c 'import json, sys
r = json.load(sys.stdin)
if r.get("error") and not r.get("status"): sys.exit("render.sh: " + r["error"]["message"])
print(r["status"], r["progress"], (r.get("stage") or "-").replace(" ", "_"))')
    status=$1
    if [ "${3:-}" != "$stage" ]; then
        stage=${3:-}
        echo "$id: $(echo "$stage" | tr _ " ") ($2%)" >&2
    fi
done
if [ "$status" != completed ]; then
    printf '%s' "$response" | python3 -c 'import json, sys
r = json.load(sys.stdin)
sys.exit("render.sh: " + r["status"] + ": " + ((r.get("error") or {}).get("message") or ""))'
fi
path=$(printf '%s' "$response" | python3 -c 'import json, sys; print(json.load(sys.stdin).get("path") or "")')
if [ -z "$path" ] || [ ! -r "$path" ]; then
    # Another machine's server: download the MP4.
    dir=${MAKO_VIDEO_DIR:-$HOME/Movies/mako}
    mkdir -p "$dir"
    path=$dir/$id.mp4
    curl -sS "$url/videos/$id/content" ${auth:+-H "$auth"} -o "$path"
fi
[ "${MAKO_VIDEO_OPEN:-1}" = 0 ] || open "$path" 2>/dev/null || true
echo "$path"
