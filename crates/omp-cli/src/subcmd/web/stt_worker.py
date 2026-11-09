"""omt dictation worker (spawned by `omt web`, see stt.rs).

One JSON request per stdin line: {"id", "path", "language"?}; one JSON reply
per stdout line: {"id", "text", "language"} or {"id", "error"}. The model
stays loaded between requests; the worker exits after OMT_STT_IDLE seconds
without a request (the hub respawns it on demand) to free the GPU.
"""
import json
import os
import select
import sys

MODEL = os.environ.get("OMT_STT_MODEL", "large-v3-turbo")
IDLE = float(os.environ.get("OMT_STT_IDLE", "600"))


def load():
    from faster_whisper import WhisperModel

    errors = []
    for device, compute in (("cuda", "int8_float16"), ("cpu", "int8")):
        try:
            return WhisperModel(MODEL, device=device, compute_type=compute), device
        except Exception as e:  # no CUDA / out of VRAM: next option
            errors.append(f"{device}: {e}")
    raise RuntimeError("; ".join(errors))


def reply(obj):
    sys.stdout.write(json.dumps(obj, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def main():
    model = None
    while True:
        ready, _, _ = select.select([sys.stdin], [], [], IDLE)
        if not ready:
            return
        line = sys.stdin.readline()
        if not line:
            return
        req = json.loads(line)
        try:
            if model is None:
                model, device = load()
                print(f"stt: {MODEL} on {device}", file=sys.stderr, flush=True)
            segments, info = model.transcribe(
                req["path"],
                language=req.get("language") or None,
                beam_size=5,
                vad_filter=True,
            )
            text = " ".join(s.text.strip() for s in segments).strip()
            reply({"id": req.get("id"), "text": text, "language": info.language})
        except Exception as e:
            reply({"id": req.get("id"), "error": str(e)})


if __name__ == "__main__":
    main()
