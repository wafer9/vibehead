#!/usr/bin/env python3
"""Resample VividHead videos to 30 FPS and audio to 24 kHz.

Each input line must be a JSON object with ``key``, ``video`` and ``audio``
fields, as in ``data/vivi/train.list``.
"""

import argparse
import json
import logging
import os
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


DEFAULT_LIST = "data/vivi/train.list"
DEFAULT_VIDEO_DIR = "/nfs-speech-cfs/wangzhou/data/tts/VividHead/videos_30pfs"
DEFAULT_AUDIO_DIR = "/nfs-speech-cfs/wangzhou/data/tts/VividHead/audios_24k"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", default=DEFAULT_LIST, dest="list_path")
    parser.add_argument("--video-dir", default=DEFAULT_VIDEO_DIR)
    parser.add_argument("--audio-dir", default=DEFAULT_AUDIO_DIR)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    return parser.parse_args()


def load_records(list_path, limit=None):
    records = []
    with open(list_path, encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {list_path}:{line_number}: {exc}") from exc
            for field in ("key", "video", "audio"):
                if not record.get(field):
                    raise ValueError(f"missing {field!r} at {list_path}:{line_number}")
            records.append(record)
            if limit is not None and len(records) >= limit:
                break
    return records


def run_ffmpeg(ffmpeg, input_path, output_path, extra_args, overwrite):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{output_path.stem}.", suffix=output_path.suffix, dir=output_path.parent
    )
    os.close(fd)
    os.unlink(temporary)
    try:
        command = [ffmpeg, "-hide_banner", "-loglevel", "error"]
        command += ["-y" if overwrite else "-n", "-i", str(input_path)]
        command += extra_args + [str(temporary)]
        subprocess.run(command, check=True)
        os.replace(temporary, output_path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def process_record(record, args):
    key = str(record["key"])
    video_output = Path(args.video_dir) / f"{key}.mp4"
    audio_output = Path(args.audio_dir) / f"{key}.wav"
    completed = []

    if args.overwrite or not video_output.exists():
        run_ffmpeg(
            args.ffmpeg,
            record["video"],
            video_output,
            ["-map", "0:v:0", "-vf", "fps=30", "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p"],
            args.overwrite,
        )
        completed.append("video")
    if args.overwrite or not audio_output.exists():
        run_ffmpeg(
            args.ffmpeg,
            record["audio"],
            audio_output,
            ["-map", "0:a:0", "-vn", "-ar", "24000", "-c:a", "pcm_s16le"],
            args.overwrite,
        )
        completed.append("audio")
    return key, completed


def main():
    args = parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be at least 1")
    if not args.overwrite:
        logging.info("Existing output files will be skipped; use --overwrite to replace them.")
    records = load_records(args.list_path, args.limit)
    logging.info("Loaded %d records from %s", len(records), args.list_path)

    failures = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        jobs = {pool.submit(process_record, record, args): record for record in records}
        for index, job in enumerate(as_completed(jobs), 1):
            record = jobs[job]
            try:
                key, completed = job.result()
                logging.info("[%d/%d] %s: %s", index, len(records), key, ", ".join(completed) or "skipped")
            except Exception as exc:
                failures += 1
                logging.error("[%d/%d] %s failed: %s", index, len(records), record["key"], exc)

    if failures:
        raise SystemExit(f"{failures} record(s) failed")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
