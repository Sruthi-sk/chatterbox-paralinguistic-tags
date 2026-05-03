"""Extract N rows (wav + script_content) from the cached parquet shards."""
import argparse, csv, glob, io, pathlib, sys
import pyarrow.parquet as pq

CACHE_GLOB = (
    "/home/sruthi/.cache/huggingface/hub/"
    "datasets--InternalCan--stage1-processed-with-audio-aligned/"
    "snapshots/*/data/train-*-of-00104.parquet"
)


def iter_rows(shards):
    for shard in shards:
        pf = pq.ParquetFile(shard)
        for rg in range(pf.num_row_groups):
            tbl = pf.read_row_group(rg, columns=["script_content", "preprocessed_audio", "filename"])
            for script, audio, fname in zip(
                tbl["script_content"].to_pylist(),
                tbl["preprocessed_audio"].to_pylist(),
                tbl["filename"].to_pylist(),
            ):
                yield script, audio, fname


def maybe_decode_to_wav(raw: bytes) -> bytes:
    """If bytes are already RIFF/WAV, return as-is; otherwise decode + re-encode as WAV."""
    if raw[:4] == b"RIFF" and raw[8:12] == b"WAVE":
        return raw
    import soundfile as sf
    data, sr = sf.read(io.BytesIO(raw))
    buf = io.BytesIO()
    sf.write(buf, data, sr, format="WAV")
    return buf.getvalue()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", "--num", type=int, default=50)
    ap.add_argument("-o", "--out", default="./samples")
    args = ap.parse_args()

    out = pathlib.Path(args.out)
    (out / "wavs").mkdir(parents=True, exist_ok=True)

    shards = sorted(glob.glob(CACHE_GLOB))
    if not shards:
        sys.exit(f"No shards found at {CACHE_GLOB}")
    print(f"Using {len(shards)} shard(s); extracting {args.num} rows")

    meta_path = out / "metadata.csv"
    with open(meta_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["idx", "wav_path", "original_filename", "script_content"])

        n = 0
        for script, audio, orig_name in iter_rows(shards):
            if audio is None or audio.get("bytes") is None:
                continue
            wav_bytes = maybe_decode_to_wav(audio["bytes"])
            wav_name = f"{n:03d}.wav"
            (out / "wavs" / wav_name).write_bytes(wav_bytes)
            w.writerow([n, f"wavs/{wav_name}", orig_name, script])
            n += 1
            if n >= args.num:
                break

    print(f"Wrote {n} wav files to {out/'wavs'}")
    print(f"Wrote metadata to {meta_path}")


if __name__ == "__main__":
    main()