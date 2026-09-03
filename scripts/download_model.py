#!/usr/bin/env python3
# Copyright 2026 Arctel.net
# SPDX-License-Identifier: Apache-2.0

"""Resumable downloader for Hugging Face models supporting断点续传 (breakpoint resume).

Automatically detects already downloaded files, resumes partially downloaded files
via HTTP Range requests or aria2c, and handles mirrors (HF_ENDPOINT) and authentication (HF_TOKEN).
"""

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request


def handle_signal(sig, frame):
    sys.stdout.write("\n\n[Interrupted] Download paused. Re-run command to resume.\n")
    sys.stdout.flush()
    sys.exit(130)


signal.signal(signal.SIGINT, handle_signal)
signal.signal(signal.SIGTERM, handle_signal)


def human_size(bytes_val: int | float | None) -> str:
    if bytes_val is None:
        return "unknown"
    val = float(bytes_val)
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if val < 1024.0:
            return f"{val:.2f} {unit}" if unit != "B" else f"{int(val)} B"
        val /= 1024.0
    return f"{val:.2f} PB"


def format_eta(seconds: float | None) -> str:
    if seconds is None or seconds < 0 or seconds > 360000:
        return "--:--"
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def fetch_repo_files(
    endpoint: str, model_id: str, revision: str = "main", token: str | None = None
) -> list[dict]:
    api_url = f"{endpoint.rstrip('/')}/api/models/{model_id}?blobs=true"
    headers = {"User-Agent": "Cyphr-Downloader/1.0"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    req = urllib.request.Request(api_url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        if e.code in (401, 403):
            sys.exit(
                f"[Error] Repository '{model_id}' requires authentication (HTTP {e.code}).\n"
                f"Please pass --token or set the HF_TOKEN environment variable."
            )
        elif e.code == 404:
            sys.exit(
                f"[Error] Model repository '{model_id}' not found on {endpoint} (HTTP 404).\n"
                f"Please verify the model ID or check your HF_ENDPOINT setting."
            )
        else:
            sys.exit(
                f"[Error] Failed to fetch repository info from {endpoint} (HTTP {e.code}): {e}"
            )
    except urllib.error.URLError as e:
        sys.exit(
            f"[Error] Network error reaching {endpoint}: {e.reason}\n"
            f"Hint: If connecting to Hugging Face fails, try setting:\n"
            f"      export HF_ENDPOINT=https://hf-mirror.com"
        )

    siblings = data.get("siblings", [])
    if not siblings:
        sys.exit(f"[Error] No files found in repository '{model_id}'.")

    files = []
    for s in siblings:
        rfilename = s.get("rfilename")
        if not rfilename:
            continue
        size = s.get("size")
        lfs = s.get("lfs")
        sha256 = None
        if lfs:
            sha256 = lfs.get("sha256")
            if size is None:
                size = lfs.get("size")

        download_url = f"{endpoint.rstrip('/')}/{model_id}/resolve/{revision}/{rfilename}"
        files.append(
            {
                "rfilename": rfilename,
                "size": size,
                "sha256": sha256,
                "url": download_url,
            }
        )
    return files


def download_file_python(
    url: str,
    target_path: str,
    expected_size: int | None = None,
    token: str | None = None,
    max_retries: int = 5,
) -> bool:
    part_path = target_path + ".part"
    os.makedirs(os.path.dirname(target_path), exist_ok=True)

    retries = 0
    filename = os.path.basename(target_path)
    chunk_size = 512 * 1024  # 512 KB chunks

    while retries <= max_retries:
        resume_size = 0
        if os.path.exists(part_path):
            resume_size = os.path.getsize(part_path)
            if expected_size is not None and resume_size > expected_size:
                print(
                    f"[Warning] Incomplete file '{filename}.part' exceeds expected size."
                    " Restarting."
                )
                try:
                    os.remove(part_path)
                except OSError:
                    pass
                resume_size = 0

        headers = {"User-Agent": "Cyphr-Downloader/1.0"}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        if resume_size > 0:
            headers["Range"] = f"bytes={resume_size}-"

        req = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                is_partial = resp.status == 206
                if resume_size > 0 and not is_partial:
                    # Server did not honor Range, reset to beginning
                    resume_size = 0
                    mode = "wb"
                else:
                    mode = "ab" if resume_size > 0 else "wb"

                total_size = expected_size
                if total_size is None:
                    content_length = resp.headers.get("Content-Length")
                    if content_length:
                        total_size = resume_size + int(content_length)

                current_bytes = resume_size
                t_start = time.time()
                t_last = t_start
                bytes_last = current_bytes
                speed = 0.0

                with open(part_path, mode) as f:
                    while True:
                        chunk = resp.read(chunk_size)
                        if not chunk:
                            break
                        f.write(chunk)
                        current_bytes += len(chunk)

                        now = time.time()
                        dt = now - t_last
                        if dt >= 0.5:
                            speed = (current_bytes - bytes_last) / dt
                            t_last = now
                            bytes_last = current_bytes

                            if total_size:
                                pct = min(100.0, current_bytes / total_size * 100.0)
                                rem_bytes = max(0, total_size - current_bytes)
                                eta_sec = rem_bytes / speed if speed > 0 else None
                                sys.stdout.write(
                                    f"\r\033[K  [{filename}] {pct:5.1f}% | "
                                    f"{human_size(current_bytes)} / {human_size(total_size)} | "
                                    f"{human_size(speed)}/s | ETA {format_eta(eta_sec)}"
                                )
                            else:
                                sys.stdout.write(
                                    f"\r\033[K  [{filename}] {human_size(current_bytes)} | "
                                    f"{human_size(speed)}/s"
                                )
                            sys.stdout.flush()

            # End of stream reached
            final_size = os.path.getsize(part_path)
            if expected_size is not None and final_size != expected_size:
                raise OSError(
                    f"File size mismatch: downloaded {final_size} bytes, "
                    f"expected {expected_size} bytes."
                )

            os.replace(part_path, target_path)
            pct_str = "100.0%" if expected_size else ""
            sys.stdout.write(
                f"\r\033[K✓ [{filename}] {pct_str} {human_size(final_size)} complete\n"
            )
            sys.stdout.flush()
            return True

        except (OSError, urllib.error.URLError, TimeoutError) as e:
            retries += 1
            if retries > max_retries:
                sys.stdout.write("\n")
                print(f"[Error] Failed to download {filename} after {max_retries} retries: {e}")
                return False
            wait_time = min(2**retries, 16)
            sys.stdout.write(
                f"\r\033[K! [{filename}] Network interrupted ({e}). "
                f"Resuming in {wait_time}s (attempt {retries}/{max_retries})...\n"
            )
            sys.stdout.flush()
            time.sleep(wait_time)


def download_with_aria2(
    files: list[dict],
    dest: str,
    token: str | None = None,
    concurrency: int = 4,
    connections: int = 8,
) -> bool:
    os.makedirs(dest, exist_ok=True)
    input_lines = []
    for f in files:
        target_path = os.path.join(dest, f["rfilename"])
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        input_lines.append(f["url"])
        input_lines.append(f"  dir={os.path.dirname(target_path)}")
        input_lines.append(f"  out={os.path.basename(target_path)}")
        input_lines.append("  header=User-Agent: Cyphr-Downloader/1.0")
        if token:
            input_lines.append(f"  header=Authorization: Bearer {token}")

    input_path = os.path.join(dest, ".aria2_inputs.txt")
    with open(input_path, "w", encoding="utf-8") as f_in:
        f_in.write("\n".join(input_lines) + "\n")

    cmd = [
        "aria2c",
        "-c",
        "-j",
        str(concurrency),
        "-x",
        str(connections),
        "-s",
        str(connections),
        "-k",
        "1M",
        "--file-allocation=none",
        "--summary-interval=1",
        "--console-log-level=warn",
        "--auto-file-renaming=false",
        "--allow-overwrite=true",
        "-i",
        input_path,
    ]
    try:
        ret = subprocess.run(cmd)
        return ret.returncode == 0
    finally:
        if os.path.exists(input_path):
            try:
                os.remove(input_path)
            except OSError:
                pass


def main():
    parser = argparse.ArgumentParser(description="Resumable model downloader for Hugging Face Hub")
    parser.add_argument("--model-id", default="Qwen/Qwen3-ASR-0.6B", help="Model repository ID")
    parser.add_argument("--dest", required=True, help="Destination directory")
    parser.add_argument("--revision", default="main", help="Git revision / branch / tag")
    parser.add_argument(
        "--endpoint",
        default=os.getenv("HF_ENDPOINT", "https://huggingface.co"),
        help="Hugging Face endpoint URL or mirror (e.g. https://hf-mirror.com)",
    )
    parser.add_argument(
        "--token",
        default=os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN"),
        help="Hugging Face API token",
    )
    parser.add_argument(
        "--tool",
        choices=["auto", "aria2c", "python"],
        default=os.getenv("DOWNLOAD_TOOL", "auto"),
        help="Downloader tool: 'auto' (prefers aria2c if installed), 'aria2c', or 'python'",
    )

    args = parser.parse_args()
    dest = os.path.abspath(args.dest)
    os.makedirs(dest, exist_ok=True)

    print(f"Model ID : {args.model_id}")
    print(f"Revision : {args.revision}")
    print(f"Endpoint : {args.endpoint}")
    print(f"Target   : {dest}")

    files = fetch_repo_files(args.endpoint, args.model_id, args.revision, args.token)
    total_repo_size = sum(f["size"] for f in files if f.get("size"))
    print(f"Files    : {len(files)} items ({human_size(total_repo_size)})\n")

    # Check already downloaded files
    needed_files = []
    already_done = 0
    for f in files:
        target_path = os.path.join(dest, f["rfilename"])
        expected = f.get("size")
        if os.path.exists(target_path):
            if expected is not None and os.path.getsize(target_path) == expected:
                already_done += 1
                print(f"✓ [Skip] {f['rfilename']} ({human_size(expected)}) already exists")
                continue
            elif expected is None and os.path.getsize(target_path) > 0:
                already_done += 1
                print(f"✓ [Skip] {f['rfilename']} already exists")
                continue
        needed_files.append(f)

    if not needed_files:
        print(f"\nAll {len(files)} files are already up to date in {dest}.")
        return

    remaining_size = sum(f["size"] for f in needed_files if f.get("size"))
    count_str = f"{len(needed_files)}/{len(files)} files ({human_size(remaining_size)})"
    print(f"\nDownloading remaining {count_str}...")
    print("Tip: You can press Ctrl+C anytime; download will resume without losing progress.\n")

    tool = args.tool
    has_aria2 = bool(shutil.which("aria2c"))
    if tool == "auto":
        tool = "aria2c" if has_aria2 else "python"

    if tool == "aria2c" and not has_aria2:
        print(
            "[Warning] aria2c requested but not found in PATH. Falling back to python downloader."
        )
        tool = "python"

    if tool == "aria2c":
        print(
            f"Using aria2c for multi-connection resumable download ({len(needed_files)} files)..."
        )
        success = download_with_aria2(needed_files, dest, args.token)
        if not success:
            print(
                "\n[Warning] aria2c reported errors. "
                "Verifying and falling back to python for remaining files..."
            )
            # Check which files still need downloading
            still_needed = []
            for f in needed_files:
                target_path = os.path.join(dest, f["rfilename"])
                expected = f.get("size")
                is_done = os.path.exists(target_path) and (
                    expected is None or os.path.getsize(target_path) == expected
                )
                if not is_done:
                    still_needed.append(f)
            if still_needed:
                for idx, f in enumerate(still_needed, 1):
                    target_path = os.path.join(dest, f["rfilename"])
                    print(f"[{idx}/{len(still_needed)}] Downloading {f['rfilename']}...")
                    ok = download_file_python(f["url"], target_path, f.get("size"), args.token)
                    if not ok:
                        sys.exit(f"\n[Error] Failed to download {f['rfilename']}.")
    else:
        print(f"Using Python resumable stream download ({len(needed_files)} files)...")
        for idx, f in enumerate(needed_files, 1):
            target_path = os.path.join(dest, f["rfilename"])
            ok = download_file_python(f["url"], target_path, f.get("size"), args.token)
            if not ok:
                sys.exit(f"\n[Error] Failed to download {f['rfilename']}.")

    # Final verification
    missing = []
    for f in files:
        target_path = os.path.join(dest, f["rfilename"])
        expected = f.get("size")
        if not os.path.exists(target_path):
            missing.append(f["rfilename"])
        elif expected is not None and os.path.getsize(target_path) != expected:
            missing.append(f"{f['rfilename']} (size mismatch)")

    if missing:
        missing_msg = "\n".join(f" - {m}" for m in missing)
        sys.exit(f"\n[Error] Verification failed for {len(missing)} files:\n{missing_msg}")

    print(f"\nSuccessfully downloaded and verified all {len(files)} files in {dest}.")


if __name__ == "__main__":
    main()
