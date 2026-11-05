#!/bin/sh
# Lists mako_video_serve's models: name, whether it takes references, its default LoRAs and steps, its notes.
set -eu
url=${MAKO_VIDEO_URL:-http://127.0.0.1:8083/v1}
auth=
[ -z "${MAKO_VIDEO_API_KEY:-}" ] || auth="Authorization: Bearer $MAKO_VIDEO_API_KEY"
curl -sS "$url/videos/models" ${auth:+-H "$auth"} | python3 -c '
import json, sys
r = json.load(sys.stdin)
if r.get("error"): sys.exit(r["error"]["message"])
for m in r["data"]:
    flags = [f for f, on in (("default", m.get("default")), ("loaded", m.get("loaded")), ("takes references", m.get("references"))) if on]
    loras = ", ".join("%s %g" % (l["name"], l["scale"]) for l in m.get("loras", []))
    line = "%s: steps %d" % (m["id"], m["steps"])
    if loras: line += ", LoRAs " + loras
    if flags: line += " (" + ", ".join(flags) + ")"
    print(line)
    for k in ("description", "prompting"):
        if m.get(k): print("  %s: %s" % (k, m[k]))'
