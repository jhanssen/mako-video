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
4. Reply with the path and a line on what the prompt asked for (not the whole prompt, which is long). If the user
   wants changes, revise the prompt and make it again; what the user says about a video describes that video, not
   words to add to the prompt.

## Writing the prompt

MiniMax trained H3 on a fixed prompt format (its "Context-IR"), and the model follows prompts in that format far
more closely than free text. Write every prompt in it. The full rules are MiniMax's prompt-writing skill,
`h3-prompt-writing` (installed beside this one when present: read its SKILL.md and `references/base-en.txt`, or
`references/ref-en.txt` for references); the shape is:

```text
integrated_multimodal_description: [Shot 1] Live-action, cinematic, a medium-wide shot frames a red fox crossing a
snowfield at sunrise... The camera tracks right with small amplitude at slow speed as it trots... [Shot 2] At
00:03.500, the shot cuts to a close-up of...

overall_soundscape: Soft, rhythmic crunching of paws in dry snow under a light wind, with distant birdsong.

non_diegetic_music: N/A
```

- `integrated_multimodal_description` is the timeline: the style and composition first (`Live-action, cinematic`,
  `2D-animated`, `3D CG`, `claymation`...), then subjects, actions, shot changes, speech and the sounds tied to a
  moment. `[Shot 1]` has no timestamp; later shots start `[Shot N] At MM:SS.mmm, the shot cuts to...`, times
  increasing and within the clip. A 5 s clip holds one or two shots. Camera moves are written as actions with
  MiniMax's vocabulary: push in / pull out, pan, truck, tilt, pedestal, arc shot, tracking shot, static shot, zoom,
  shake slightly / strongly, POV, roll; `with small/large amplitude`, `at slow/fast speed` when it matters.
- Speakers get stable IDs and their lines go in `<d>`: `The old fisherman with a low, gravelly voice (S1) says:
  <d>[English] We should go.</d>`; voice-over is `says in an off-screen voiceover` followed by `while his lips remain
  completely closed`. On-screen text goes in double quotes, verbatim.
- `overall_soundscape`: one to four sentences of ambience and physical sounds for the whole clip (no dialogue or
  music; `N/A` only for silence). `non_diegetic_music`: one to three sentences on instruments, tempo and dynamics
  (no mood words), `N/A` when there is none.
- Keyframes add an alignment line first, then a blank line. First frame: `For the target video, at 0.00 seconds into
  the target video, <Picture 1> (from [Shot 1]) is fully referenced.` and the description starts from what the
  picture shows and moves on. First and last: `How the reference pictures align with the target video — Picture 1
  (from Shot 1) aligns with the 0.00-second mark of the target video; Picture 2 (from Shot 1) aligns with the
  S.SS-second mark of the target video.` with the clip's real length (below), one shot, the path between the two.
- References (the Ref2VA model) use the six-section form instead: `subject_definitions:` (one line per `<Subject N>`,
  `<Picture N>`, `<Video N>`, `<Audio N>` saying what each is), `summary:` (one paragraph starting with the task
  type, e.g. `[reference generation]`), `retention_analysis:` (one line per label: `fully_preserved`,
  `partially_preserved`, `attribute_transfer`, `weak_reference`; audio `fully_copy`, `partially_copy`,
  `reference`), `detailed_description:` (the timeline, 350-500 words, the style in a sentence before `[Shot 1]`,
  labels cited where they appear), then `overall_soundscape:` and `non_diegetic_music:`. The labels are numbered per
  kind in the order the files are given to `render.sh`: the first `--ref-image` is `<Picture 1>`, the first
  `--ref-video` is `<Video 1>` and its sound `<Audio 1>` (a `--ref-audio` before it would be `<Audio 1>` instead).
- Lengths the clip can have (24 fps, 17 n + 5 frames), for the alignment lines and the shot times: 5.17 s (the
  default), 5.88, 6.58, 7.29, 8.00, 8.71, 9.42, 10.13, 10.83, 11.54, 12.25, 12.96, 13.67, 14.38, 15.08; `--seconds`
  rounds up to the next.

Keep every detail visible or audible, each thing said once; say what moves, not only what is there.

## Options

`--model`, `--lora NAME[:SCALE]` (repeatable; `--no-loras` for none), `--steps`, `--seconds`, `--seed N` (the same
seed and settings make the same video), `--first-frame`, `--last-frame`, `--ref-image`, `--ref-video`,
`--ref-audio`. `MAKO_VIDEO_URL` names the server (default `http://127.0.0.1:8083/v1`), `MAKO_VIDEO_API_KEY` its
key, `MAKO_VIDEO_OPEN=0` leaves the player closed.
