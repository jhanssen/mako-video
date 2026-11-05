---
name: scene-video
description: Make a short video with sound of the current scene, character, place or moment from this conversation with the local MiniMax-H3 video model (mako_video_serve), from text, from a first and last frame, or from reference pictures, clips and soundtracks
---

# Scene video

Turn what the conversation has established into one video prompt, make it with mako_video_serve, and show the result.
A video takes minutes (the model's notes say how many); say so before starting, and do not start another one until it
is done.

0. Pick the model (`scripts/models.sh` lists them: the plain one for text, keyframes; the Ref2VA one for references) and
   its LoRA (`scripts/loras.sh MODEL`: a turbo LoRA runs 8 steps in a quarter of the time; `--steps 9` goes with an
   8-step one, `--steps 5` with a 4-step one).
1. Write the prompt as "Writing the prompt" below says.
2. Pick a size: 1344x768 for a wide scene, 768x1344 for a tall one, 960x544 or 512x288 for a quick look (each side a
   multiple of 32); `auto` takes the first frame's aspect ratio when there is one, else 1344x768. Pick a length,
   `--seconds 5` to `15` (default 5.2).
3. Run, from this skill's directory:

   ```sh
   scripts/render.sh --model MODEL --lora LORA --steps 9 [--seconds F] WIDTHxHEIGHT "<the prompt>"
   ```

   With a picture to start from: `--first-frame FILE` (and `--last-frame FILE` for where it ends). With references
   (the Ref2VA model): `--ref-image FILE`, `--ref-video FILE` (its sound comes along), `--ref-audio FILE`, in the order
   the prompt names them as "Picture 1", "Video 1", "Audio 1". It prints progress to standard error and, when done,
   the MP4's path, and opens it in the video player. When the video server shares the Mac with the chat model's
   mako_serve (`--llm`), the chat model is unloaded while the video model is loaded; the next message loads it again
   (~10 s), and this conversation continues as before.
4. Reply with the path and the prompt used, in a line or two. If the user wants changes, revise the prompt and make it
   again; what the user says about a video describes that video, not words to add to the prompt.

## Writing the prompt

The model knows only what the prompt says and fills the rest with its most common choice. Its text encoder is a
language model: write English sentences, not keyword lists. Say each thing once, in one place. Build it in this order:

1. The subject and what it does, as one continuous action over the length of the clip (a 5 s clip holds one beat: a
   fox crossing a field, a hand pouring coffee; not a story).
2. The setting and the light.
3. The camera: still, a slow push in, a pan following the subject. One move.
4. The sound: what is heard, in a clause of its own ("soft footsteps in snow and distant birdsong"; "she says: 'we
   should go'" for speech). The model makes a matching stereo soundtrack; without a sound clause it invents one.
5. The look, in a few words (film, documentary, animation), if it matters.

With references, name them as the model sees them: "the red fox from Picture 1", "the room in Video 1", "the voice
in Audio 1". With a first frame, describe what moves from it, not what is already in it.

## Options

`--model`, `--lora NAME[:SCALE]` (repeatable; `--no-loras` for none), `--steps`, `--seconds`, `--seed N` (the same
seed and settings make the same video), `--first-frame`, `--last-frame`, `--ref-image`, `--ref-video`,
`--ref-audio`. `MAKO_VIDEO_URL` names the server (default `http://127.0.0.1:8083/v1`), `MAKO_VIDEO_API_KEY` its
key, `MAKO_VIDEO_OPEN=0` leaves the player closed.
