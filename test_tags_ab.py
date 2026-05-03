"""A/B inference test for paralinguistic tag tokens: Standard vs Turbo Chatterbox.

Generates a wav per (tag, model) pair using each model's bundled default voice,
plus a tokenization report showing how each model's tokenizer encoded every
prompt. Output layout lives under outputs/tag_ab/.
"""

import re
from pathlib import Path

import torch
import torchaudio as ta

from chatterbox.tts import ChatterboxTTS
from chatterbox.tts_turbo import ChatterboxTurboTTS


OUT = Path("outputs/tag_ab")

BASELINE_TEXT = "She walked into the room and greeted everyone with a polite smile before sitting down."

# Tag -> carrier sentence. Tag in square brackets is substituted verbatim into
# the text so the tokenizer sees the exact form. {TAG} placeholder is used for
# the shared-renamed group where each model gets a different spelling.
CARRIER_SENTENCES: dict[str, str] = {
    "_baseline": BASELINE_TEXT,

    # Turbo-native (emotion/style)
    "angry":        "[angry] I cannot believe you did that again, after everything I told you.",
    "fear":         "[fear] Did you hear that noise coming from down the hallway?",
    "surprised":    "[surprised] Oh my goodness, I had absolutely no idea you would be here today.",
    "whispering":   "She leaned closer and [whispering] told him the secret she had been keeping.",
    "dramatic":     "[dramatic] And that is when everything changed forever, in a single moment.",
    "narration":    "[narration] The old house stood at the edge of the forest, silent and waiting.",
    "crying":       "[crying] I just don't know what to do anymore, it all feels so hopeless.",
    "happy":        "[happy] This is the best news I have heard all year long, thank you so much.",
    "sarcastic":    "[sarcastic] Oh sure, that sounds like a completely brilliant plan, no notes at all.",
    "sigh":         "She paused and then [sigh] continued reading the letter he had sent her.",
    "cough":        "Excuse me for a moment [cough] the air in here is rather dusty.",
    "groan":        "He opened the package and [groan] realized it was the wrong item again.",
    "sniff":        "She wiped her eyes and [sniff] tried to compose herself before speaking.",
    "gasp":         "She opened the letter and [gasp] read the news she had been dreading.",
    "chuckle":      "That's actually pretty funny [chuckle] I wasn't expecting that punchline at all.",
    "laugh":        "He told the joke and everyone [laugh] filled the room with noise.",
    "clear throat": "Before the speech she [clear throat] stepped up to the microphone.",
    "shush":        "[shush] Keep your voice down, the baby is finally sleeping in the other room.",

    # Standard-native (sound-effect)
    "bark":         "The dog ran outside and [bark] chased the mailman down the street.",
    "howl":         "The wolves in the distance began to [howl] at the rising full moon.",
    "meow":         "The kitten looked up at me and [meow] demanded her evening dinner.",
    "sneeze":       "She stepped into the dusty attic and [sneeze] covered her nose immediately.",
    "snore":        "He fell asleep on the couch and began to [snore] loudly through the movie.",
    "chew":         "She took a bite of the apple and [chew] thoughtfully while reading the paper.",
    "sip":          "She raised the cup of tea and [sip] slowly before setting it back down.",
    "kiss":         "He leaned close and [kiss] her gently on the forehead before leaving.",
    "whistle":      "He walked down the sunny street and [whistle] a cheerful little tune.",
    "humming":      "She worked on the puzzle and [humming] a song from her childhood.",
    "giggle":       "The children heard the silly joke and [giggle] at the dinner table.",
    "guffaw":       "He read the absurd headline and [guffaw] nearly spilling his coffee everywhere.",
    "cry":          "The baby woke up in the night and began to [cry] in the dark.",
    "mumble":       "He stared at his feet and [mumble] something I could not quite understand.",

    # Shared / differently-named concepts — {TAG} gets model-specific form.
    "whisper":      "She leaned closer and {TAG} told him the secret she had been keeping.",
    "laugh_shared": "He told the joke and everyone {TAG} filled the room with noise.",
    "sigh_shared":  "She paused and then {TAG} continued reading the letter.",
    "cough_shared": "Excuse me for a moment {TAG} the air in here is rather dusty.",
    "sniff_shared": "She wiped her eyes and {TAG} tried to compose herself.",
    "gasp_shared":  "She opened the letter and {TAG} read the news she had been dreading.",
    "groan_shared": "He opened the package and {TAG} realized it was the wrong item again.",
}

TURBO_NATIVE = [
    "angry", "fear", "surprised", "whispering", "dramatic",
    "narration", "crying", "happy", "sarcastic", "sigh",
    "cough", "groan", "sniff", "gasp", "chuckle", "laugh",
    "clear throat", "shush",
]

STANDARD_NATIVE = [
    "bark", "howl", "meow", "sneeze", "snore",
    "chew", "sip", "kiss", "whistle", "humming",
    "giggle", "guffaw", "cry", "mumble",
]

# (concept_key_in_CARRIERS, turbo_tag, standard_tag)
SHARED_BUT_RENAMED = [
    ("whisper",      "whispering", "whisper"),
    ("laugh_shared", "laugh",      "laughter"),
    ("sigh_shared",  "sigh",       "sigh"),
    ("cough_shared", "cough",      "cough"),
    ("sniff_shared", "sniff",      "sniff"),
    ("gasp_shared",  "gasp",       "gasp"),
    ("groan_shared", "groan",      "groan"),
]


def slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


def encode_std(model: ChatterboxTTS, text: str) -> list[int]:
    return list(model.tokenizer.encode(text))


def encode_turbo(model: ChatterboxTurboTTS, text: str) -> list[int]:
    return model.tokenizer(text, return_tensors="pt").input_ids[0].tolist()


def run_one(model, encode_fn, text, out_path: Path, log, label: str) -> None:
    ids = encode_fn(model, text)
    log.write(f"{label}\n  text: {text!r}\n  ids ({len(ids)} tok): {ids}\n\n")
    log.flush()
    print(f"  -> {label}  ({len(ids)} tokens)  {out_path.name}")
    with torch.no_grad():
        wav = model.generate(text)
    ta.save(str(out_path), wav.cpu(), model.sr)


def main() -> None:
    for sub in ("baseline", "turbo_native", "standard_native", "shared_renamed"):
        (OUT / sub).mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading Standard ChatterboxTTS on {device} ...")
    std = ChatterboxTTS.from_pretrained(device=device)
    print(f"Loading Turbo ChatterboxTurboTTS on {device} ...")
    turbo = ChatterboxTurboTTS.from_pretrained(device=device)

    log_path = OUT / "tokenization_report.txt"
    with open(log_path, "w") as log:
        log.write(f"# Tag A/B tokenization report\n# std = ChatterboxTTS (704 BPE)\n# turbo = ChatterboxTurboTTS (GPT2 50257 + 19 tags)\n\n")

        # 1. Baseline (no tag)
        print("\n== Baseline ==")
        run_one(std, encode_std, CARRIER_SENTENCES["_baseline"],
                OUT / "baseline/standard_no_tag.wav", log, "std   _baseline")
        run_one(turbo, encode_turbo, CARRIER_SENTENCES["_baseline"],
                OUT / "baseline/turbo_no_tag.wav", log, "turbo _baseline")

        # 2. Turbo-native tags × both models
        print("\n== Turbo-native tags ==")
        for tag in TURBO_NATIVE:
            text = CARRIER_SENTENCES[tag]
            stem = slug(tag)
            run_one(turbo, encode_turbo, text,
                    OUT / f"turbo_native/{stem}__turbo.wav", log, f"turbo [{tag}]")
            run_one(std, encode_std, text,
                    OUT / f"turbo_native/{stem}__standard.wav", log, f"std   [{tag}]")

        # 3. Standard-native tags × both models
        print("\n== Standard-native tags ==")
        for tag in STANDARD_NATIVE:
            text = CARRIER_SENTENCES[tag]
            stem = slug(tag)
            run_one(turbo, encode_turbo, text,
                    OUT / f"standard_native/{stem}__turbo.wav", log, f"turbo [{tag}]")
            run_one(std, encode_std, text,
                    OUT / f"standard_native/{stem}__standard.wav", log, f"std   [{tag}]")

        # 4. Shared-but-renamed
        print("\n== Shared-but-renamed tags ==")
        for concept_key, turbo_tag, std_tag in SHARED_BUT_RENAMED:
            template = CARRIER_SENTENCES[concept_key]
            text_t = template.replace("{TAG}", f"[{turbo_tag}]")
            text_s = template.replace("{TAG}", f"[{std_tag}]")
            concept = concept_key.replace("_shared", "")
            run_one(turbo, encode_turbo, text_t,
                    OUT / f"shared_renamed/{concept}__turbo_{slug(turbo_tag)}.wav",
                    log, f"turbo [{turbo_tag}]")
            run_one(std, encode_std, text_s,
                    OUT / f"shared_renamed/{concept}__standard_{slug(std_tag)}.wav",
                    log, f"std   [{std_tag}]")

    write_readme(OUT)
    print(f"\nDone. Report: {log_path}")
    print(f"Audio:  {OUT}/")


def write_readme(out: Path) -> None:
    readme = out / "README.md"
    readme.write_text(
        "# Paralinguistic Tag A/B Test\n\n"
        "Generated by `test_tags_ab.py`. Each tag is run through both models using "
        "bundled default voices (no `audio_prompt_path`).\n\n"
        "## Folders\n"
        "- `baseline/` — same sentence, no tags, to calibrate each model's default voice.\n"
        "- `turbo_native/` — tags in Turbo's vocab (IDs 50257–50275). Cross-tested on Standard to hear char-split OOV behavior.\n"
        "- `standard_native/` — tags in Standard's vocab (IDs 604+). Cross-tested on Turbo.\n"
        "- `shared_renamed/` — same concept, different surface form per model (e.g. Turbo `[whispering]` vs Standard `[whisper]`).\n\n"
        "## Filename convention\n"
        "`<concept>__<model>[_tagform].wav` — pair up by concept prefix.\n\n"
        "See `tokenization_report.txt` for per-prompt token IDs.\n"
    )


if __name__ == "__main__":
    main()
