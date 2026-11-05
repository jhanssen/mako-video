#!/bin/sh
# Lists the LoRAs a request can name for a model (default: the server's default model).
#   loras.sh [MODEL]
set -eu
url=${MAKO_VIDEO_URL:-http://127.0.0.1:8083/v1}
auth=
[ -z "${MAKO_VIDEO_API_KEY:-}" ] || auth="Authorization: Bearer $MAKO_VIDEO_API_KEY"
curl -sS "$url/videos/loras${1:+?model=$1}" ${auth:+-H "$auth"} | python3 -c '
import json, sys
r = json.load(sys.stdin)
if r.get("error"): sys.exit(r["error"]["message"])
for l in r["data"]:
    extra = [f"{k} {l[k]}" for k in ("default_scale", "steps", "trigger") if l.get(k)]
    print(l["name"] + (" (" + ", ".join(extra) + ")" if extra else "") + (": " + l["description"] if l.get("description") else ""))'
