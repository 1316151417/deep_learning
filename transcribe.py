#!/usr/bin/env python3
"""Transcribe MP3 audio to English text + SRT subtitles using faster-whisper.

- Device: auto-detect CUDA > MPS > CPU. faster-whisper's CTranslate2 backend only
  supports 'cpu' and 'cuda' (NOT MPS), so MPS is reported but resolved to CPU on
  Apple Silicon. On a CUDA box it auto-selects cuda + float16.
- Language: English, task=transcribe (no translation).
- For each MP3 in the cwd it writes a .txt (plain concatenated text) and a .srt.
"""

import os
import re
import sys
import glob


# ============================ CONFIG ============================
# Pick the model here. Larger = more accurate but slower. All auto-download on
# first run from the HuggingFace Hub.
MODEL = "base"          # <— change me. Options:
#   "large-v3"         : best quality, slowest. ~2.5-4h for 3.5h audio on M4 CPU.
#   "distil-large-v3"  : near-large quality (English, nearly lossless), ~1.7x
#                        faster. RECOMMENDED for long English audio.
#   "turbo"            : OpenAI large-v3-turbo. Fast + high quality; great on CUDA.
#   "medium"           : good quality, ~2x faster than large.
#   "small"            : faster, noticeably lower accuracy. ~25-40 min.
#   "base" / "tiny"    : fastest, lowest accuracy (quick tests only).

BEAM_SIZE = 5               # decoding beam size; 1 = greedy/fastest, 5 = default quality
VAD_FILTER = True           # skip silence -> much faster on long audio
LANGUAGE = "en"             # source language; set None to auto-detect
# ================================================================


def slugify(name: str) -> str:
    """Turn a filename into a safe base name for output files."""
    name = os.path.splitext(os.path.basename(name))[0]
    return re.sub(r"[^\w\-. ]+", "_", name).strip() or "transcript"


def format_timestamp(seconds: float) -> str:
    """Seconds -> SRT timestamp HH:MM:SS,mmm."""
    if seconds < 0:
        seconds = 0.0
    ms = int(round((seconds - int(seconds)) * 1000))
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if ms == 1000:  # rounding rollover
        ms, s = 0, s + 1
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def detect_device():
    """Return (faster_whisper_device, compute_type, human_label).

    Order: CUDA > MPS > CPU. faster-whisper/CTranslate2 cannot use MPS, so MPS
    is reported but resolved to CPU with int8 quantization.
    """
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda", "float16", "CUDA GPU (compute_type=float16)"
    except Exception:
        pass

    mps = False
    try:
        import torch
        mps = torch.backends.mps.is_available()
    except Exception:
        pass

    label = "MPS detected but unsupported by faster-whisper -> CPU (int8)" if mps else "CPU (int8)"
    return "cpu", "int8", label


def transcribe_file(model, audio_path: str, txt_path: str, srt_path: str):
    print(f"\n=== Transcribing: {audio_path} ===", flush=True)
    segments, info = model.transcribe(
        audio_path,
        language=LANGUAGE,
        task="transcribe",
        beam_size=BEAM_SIZE,
        vad_filter=VAD_FILTER,
        vad_parameters={"min_silence_duration_ms": 500},
        condition_on_previous_text=False,
    )
    print(f"detected language={info.language} (p={info.language_probability:.2f}), "
          f"duration={info.duration:.1f}s", flush=True)

    lines, srt_chunks, idx = [], [], 0
    for seg in segments:
        text = (seg.text or "").strip()
        if not text:
            continue
        idx += 1
        lines.append(text)
        srt_chunks.append(
            f"{idx}\n"
            f"{format_timestamp(seg.start)} --> {format_timestamp(seg.end)}\n"
            f"{text}\n"
        )
        # Live progress (flush so it shows up in background logs)
        print(f"[{idx:04d}] {format_timestamp(seg.start)} -> {format_timestamp(seg.end)} | {text}",
              flush=True)

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(" ".join(lines).strip() + "\n")
    with open(srt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(srt_chunks))


def main():
    mp3s = sorted(set(glob.glob("*.mp3") + glob.glob("*.MP3")))
    if not mp3s:
        print("No MP3 files found in current directory.", flush=True)
        sys.exit(1)

    print(f"Found {len(mp3s)} MP3 file(s):", flush=True)
    for m in mp3s:
        print(f"  - {m}", flush=True)

    device, compute_type, label = detect_device()
    print(f"\nDevice: {label}  (faster-whisper device='{device}', compute_type='{compute_type}')",
          flush=True)

    from faster_whisper import WhisperModel
    print(f"Loading model '{MODEL}' (downloads on first run)...", flush=True)
    model = WhisperModel(MODEL, device=device, compute_type=compute_type)
    print("Model loaded.", flush=True)

    single = len(mp3s) == 1
    for m in mp3s:
        if single:
            txt_path, srt_path = "transcript.txt", "transcript.srt"
        else:
            base = slugify(m)
            txt_path, srt_path = f"{base}.txt", f"{base}.srt"
        try:
            transcribe_file(model, m, txt_path, srt_path)
            print(f"  -> wrote {txt_path} and {srt_path}", flush=True)
        except Exception as e:
            print(f"  !! FAILED on {m}: {e}", flush=True)

    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
