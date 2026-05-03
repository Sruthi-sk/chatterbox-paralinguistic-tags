import glob, pyarrow.parquet as pq

shard = sorted(glob.glob(
    "/home/sruthi/.cache/huggingface/hub/"
    "datasets--InternalCan--stage1-processed-with-audio-aligned/"
    "snapshots/*/data/train-00000-of-00104.parquet"
))[0]

# just the first row group, then first 5 rows — avoids loading all 449 MB
pf = pq.ParquetFile(shard)
print(pf.schema_arrow)
print(pf.num_row_groups, "row groups,", pf.metadata.num_rows, "rows total")

df = pf.read_row_group(0).slice(0, 5).to_pandas()
print(df)


# from datasets import load_dataset
# ds = load_dataset(
#     "parquet",
#     data_files=sorted(glob.glob(
#         "/home/sruthi/.cache/huggingface/hub/"
#         "datasets--InternalCan--stage1-processed-with-audio-aligned/"
#         "snapshots/*/data/train-0000[0-4]-of-00104.parquet"
#     )),
#     split="train",
#     streaming=True,   # doesn't materialize everything
# )
# print(next(iter(ds)))


# look at dtypes and sample content without loading everything into memory
# import glob, pyarrow.parquet as pq
# shard = sorted(glob.glob(
#     "/home/sruthi/.cache/huggingface/hub/"
#     "datasets--InternalCan--stage1-processed-with-audio-aligned/"
#     "snapshots/*/data/train-00000-of-00104.parquet"
# ))[0]
# pf = pq.ParquetFile(shard)
# print("=== SCHEMA ===")
# print(pf.schema_arrow)
# print("=== ROW 0 sample ===")
# row = pf.read_row_group(0).slice(0, 1).to_pydict()
# for k, v in row.items():
#     sample = v[0]
#     if isinstance(sample, dict):
#         shown = {kk: (type(vv).__name__ + f"(len={len(vv)})" if isinstance(vv, (bytes, str, list)) else repr(vv)[:60]) for kk, vv in sample.items()}
#         print(f"  {k}: dict -> {shown}")
#     elif isinstance(sample, (bytes, str)):
#         print(f"  {k}: {type(sample).__name__} len={len(sample)} preview={sample[:80]!r}")
#     else:
#         print(f"  {k}: {type(sample).__name__} {repr(sample)[:120]}")