"""Download the project datasets into data/raw/.

Run from the repo root, one dataset at a time or several together:

    python scripts/download_data.py --list
    python scripts/download_data.py brats2021
    python scripts/download_data.py rhuh mu africa
    python scripts/download_data.py verify

Downloads are resumable: files that are already complete are skipped.
Needs only the Python standard library, plus:
  - the `kaggle` CLI for brats2021 (pip install kaggle), with your Kaggle API token
    in ~/.kaggle/kaggle.json or in the KAGGLE_USERNAME / KAGGLE_KEY environment variables;
  - `ascli` (IBM aspera-cli) for the TCIA datasets, or a browser with IBM Aspera Connect.
Never put a Kaggle token (or any other credential) in this repository.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import os
import re
import shutil
import subprocess
import sys
import tarfile
import urllib.request
import zipfile
from pathlib import Path

TCIA = "https://faspex.cancerimagingarchive.net/aspera/faspex/public/package?context="
TCIA_FILES = "https://www.cancerimagingarchive.net/wp-content/uploads/"

DATASETS = {
    "brats2021": {
        "folder": "brats2021",
        "what": "BraTS 2021 Task 1: training cohort, 1,251 labelled patients (Kaggle mirror)",
        "size": "13.4 GB download, about 13 GB after extraction",
        "expect": "1,251 patient folders, 5 .nii.gz files each",
    },
    "rhuh": {
        "folder": "longitudinal_rhuh_gbm",
        "what": "RHUH-GBM: 40 patients x 3 timepoints, external test + change over time (TCIA)",
        "size": "2.9 GB",
        "expect": "40 patients, 720 NIfTI files",
        "package": TCIA + "eyJyZXNvdXJjZSI6InBhY2thZ2VzIiwidHlwZSI6ImV4dGVybmFsX2Rvd25sb2FkX3BhY2thZ2UiLCJpZCI6IjY4NCIsInBhc3Njb2RlIjoiNjFiMWU3OWZkYTU4MmRhNTBjZmU1NDZmYjkyZDQ0MzI2NjZjODZjNCIsInBhY2thZ2VfaWQiOiI2ODQiLCJlbWFpbCI6ImhlbHBAY2FuY2VyaW1hZ2luZ2FyY2hpdmUubmV0In0=",
        "page": "https://www.cancerimagingarchive.net/collection/rhuh-gbm/",
        "extras": ["clinical_data_TCIA_RHUH-GBM.csv"],
    },
    "mu": {
        "folder": "longitudinal_mu_glioma_post",
        "what": "MU-Glioma-Post: 203 patients, 1-6 post-treatment timepoints, change over time (TCIA)",
        "size": "11 GB",
        "expect": "203 patients, about 596 timepoints",
        "package": TCIA + "eyJyZXNvdXJjZSI6InBhY2thZ2VzIiwidHlwZSI6ImV4dGVybmFsX2Rvd25sb2FkX3BhY2thZ2UiLCJpZCI6IjEwMzAiLCJwYXNzY29kZSI6ImYyMGMxZDY1ODgwOWM1MWJhNWE3NGIwNTY0NGE4YWNmYzVlYTBmZjUiLCJwYWNrYWdlX2lkIjoiMTAzMCIsImVtYWlsIjoiaGVscEBjYW5jZXJpbWFnaW5nYXJjaGl2ZS5uZXQifQ==",
        "page": "https://www.cancerimagingarchive.net/collection/mu-glioma-post/",
        "extras": [
            "MU-Glioma-Post_ClinicalData-July2025.xlsx",
            "MU-Glioma-Post_Segmentation_Volumes.xlsx",
            "MR_Scanner_data.xlsx",
        ],
    },
    "africa": {
        "folder": "external_brats_africa",
        "what": "BraTS-Africa: 146 patients from Nigeria, external test (TCIA)",
        "size": "1.6 GB",
        "expect": "146 patients (95 glioma + 51 other brain tumors)",
        "package": TCIA + "eyJyZXNvdXJjZSI6InBhY2thZ2VzIiwidHlwZSI6ImV4dGVybmFsX2Rvd25sb2FkX3BhY2thZ2UiLCJpZCI6Ijk0OCIsInBhc3Njb2RlIjoiOTg2MzVlMGRmNzc3NWQ0NWJmZTQ2NjlhYzQwNjNmYjcxMjU0MzI1NyIsInBhY2thZ2VfaWQiOiI5NDgiLCJlbWFpbCI6ImhlbHBAY2FuY2VyaW1hZ2luZ2FyY2hpdmUubmV0In0=",
        "page": "https://www.cancerimagingarchive.net/collection/brats-africa/",
        "extras": ["BraTS-Africa_TCIA_datainfo_v2.xlsx"],
    },
    "utsw": {
        "folder": "external_utsw_glioma",
        "what": "UTSW-Glioma: 625 patients (362 expert-refined masks), external test + MGMT test (TCIA)",
        "size": "22.9 GB (two skull-stripping versions per patient; keep one after download)",
        "expect": "625 patients, 6,349 NIfTI files",
        "package": TCIA + "eyJyZXNvdXJjZSI6InBhY2thZ2VzIiwidHlwZSI6ImV4dGVybmFsX2Rvd25sb2FkX3BhY2thZ2UiLCJpZCI6IjEyNjUiLCJwYXNzY29kZSI6IjM0MzA0NTU2NTgxNDA0ZjYyZTU1ODcwNWI4NjYyZWFjYjlkZGZiNjUiLCJwYWNrYWdlX2lkIjoiMTI2NSIsImVtYWlsIjoiaGVscEBjYW5jZXJpbWFnaW5nYXJjaGl2ZS5uZXQifQ==",
        "page": "https://www.cancerimagingarchive.net/collection/utsw-glioma/",
        "extras": ["UTSW_Glioma_Metadata-2-1.tsv"],
    },
    "ds007045": {
        "folder": "external_ds007045",
        "what": "OpenNeuro ds007045: 363 GBM patients with MGMT, secondary external test + optional MGMT module",
        "size": "about 9.4 GB for one preprocessed version (the full dataset is 48 GB)",
        "expect": "363 patients: 4 scans each + 363 tumor masks",
    },
}

# ds007045 ships three preprocessed versions of the same scans; we download one.
DS007045_TIERS = {
    "brain": "derivatives/processing/",  # MNI152, skull-stripped (closest to BraTS)
    "n4": "derivatives/processing_MNI152_skull-stripped_N4/",  # + bias-field correction
    "zscore": "derivatives/processing_MNI152_skull-stripped_N4_z-scoring/",  # + z-score
}
OPENNEURO_S3 = "https://s3.amazonaws.com/openneuro.org"


def fetch(url: str, dest: Path) -> None:
    """Download url to dest unless dest already exists with the same size."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=120) as response:
        size = int(response.headers.get("Content-Length", -1))
        if dest.exists() and dest.stat().st_size == size:
            return
        partial = dest.with_name(dest.name + ".part")
        with open(partial, "wb") as out:
            shutil.copyfileobj(response, out, length=1 << 20)
    partial.replace(dest)


def extract_archives(folder: Path, keep: bool) -> None:
    """Unpack every .zip, then every .tar, found in folder; delete them unless keep."""
    for archive in sorted(folder.glob("*.zip")):
        print(f"  unzipping {archive.name}")
        with zipfile.ZipFile(archive) as z:
            z.extractall(folder)
        if not keep:
            archive.unlink()
    # the big archive first, so the single-patient archives are unpacked on top of it
    for archive in sorted(folder.glob("*.tar"), key=lambda p: -p.stat().st_size):
        print(f"  extracting {archive.name}")
        with tarfile.open(archive) as t:
            t.extractall(folder, filter="data")
        if not keep:
            archive.unlink()


def get_brats2021(root: Path, keep: bool, extract: bool) -> None:
    folder = root / DATASETS["brats2021"]["folder"]
    folder.mkdir(parents=True, exist_ok=True)
    if shutil.which("kaggle") is None:
        sys.exit("The kaggle CLI is missing: run `pip install kaggle` and set up your API token.")
    for name in ["BraTS2021_Training_Data.tar", "BraTS2021_00495.tar", "BraTS2021_00621.tar"]:
        print(f"  downloading {name}")
        subprocess.run(
            ["kaggle", "datasets", "download", "dschettler8845/brats-2021-task1",
             "-f", name, "-p", str(folder)],
            check=True,
        )
    if extract:
        extract_archives(folder, keep)


def get_tcia(key: str, root: Path) -> None:
    info = DATASETS[key]
    folder = root / info["folder"]
    folder.mkdir(parents=True, exist_ok=True)
    for name in info["extras"]:
        print(f"  downloading {name}")
        fetch(TCIA_FILES + name, folder / name)
    if shutil.which("ascli") is None:
        print(
            f"  ascli is not installed, so download the images in the browser instead:\n"
            f"    1. Install IBM Aspera Connect: https://www.ibm.com/products/aspera/downloads\n"
            f"    2. Open {info['page']} and click Download ({info['size']}) under Data Access.\n"
            f"    3. Save the package into {folder.resolve()}"
        )
        return
    subprocess.run(
        ["ascli", "faspex5", "packages", "receive", f"--url={info['package']}",
         f"--to-folder={folder}"],
        check=True,
    )


def get_ds007045(root: Path, tier: str) -> None:
    folder = root / DATASETS["ds007045"]["folder"]
    wanted = (DS007045_TIERS[tier], "derivatives/segmentation/")
    keys, token = [], None
    while True:  # list the public S3 bucket, 1,000 keys per page
        url = f"{OPENNEURO_S3}?list-type=2&prefix=ds007045/"
        if token:
            url += "&continuation-token=" + urllib.request.quote(token)
        page = urllib.request.urlopen(url, timeout=120).read().decode()
        keys += re.findall(r"<Key>(.*?)</Key>", page)
        match = re.search(r"<NextContinuationToken>(.*?)</NextContinuationToken>", page)
        if not match:
            break
        token = match.group(1)
    chosen = [
        k for k in keys
        if k.count("/") == 1 and not k.endswith("/")  # top-level files: participants.tsv, README...
        or any(k.startswith("ds007045/" + w) for w in wanted)
    ]
    print(f"  {len(chosen)} files ({tier} version + tumor masks + participant table)")
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        jobs = [pool.submit(fetch, f"{OPENNEURO_S3}/{k}", folder / k.split("/", 1)[1]) for k in chosen]
        for done, job in enumerate(concurrent.futures.as_completed(jobs), 1):
            job.result()
            if done % 200 == 0 or done == len(jobs):
                print(f"  {done}/{len(jobs)} files")


def verify(root: Path) -> None:
    """Print what is on disk next to what each dataset should contain."""
    for key, info in DATASETS.items():
        folder = root / info["folder"]
        if not folder.exists():
            print(f"{key:10} not downloaded")
            continue
        files = [p for p in folder.rglob("*") if p.is_file()]
        nifti = sum(p.name.endswith((".nii", ".nii.gz")) for p in files)
        archives = [p.name for p in files if p.suffix in (".zip", ".tar", ".part")]
        size = sum(p.stat().st_size for p in files) / 1e9
        print(f"{key:10} {nifti:6d} NIfTI files, {size:6.1f} GB   expected: {info['expect']}")
        if archives:
            print(f"{'':10} unfinished or unextracted archives: {', '.join(archives[:5])}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("datasets", nargs="*", help=f"any of: {', '.join(DATASETS)}, verify")
    parser.add_argument("--list", action="store_true", help="show the datasets and their sizes")
    parser.add_argument("--data-root", type=Path, default=Path(os.environ.get("DATA_RAW", "data/raw")),
                        help="where to put the data (default: data/raw, or $DATA_RAW)")
    parser.add_argument("--tier", choices=DS007045_TIERS, default="brain",
                        help="which ds007045 preprocessed version to download (default: brain)")
    parser.add_argument("--keep-archives", action="store_true", help="keep .zip/.tar files after extracting")
    parser.add_argument("--no-extract", action="store_true",
                        help="leave brats2021 as archives (best for Google Drive: one big file copies fast)")
    args = parser.parse_args()

    if args.list or not args.datasets:
        for key, info in DATASETS.items():
            print(f"{key:10} {info['size']}\n{'':10} {info['what']}")
        return
    for key in args.datasets:
        print(f"== {key}")
        if key == "verify":
            verify(args.data_root)
        elif key == "brats2021":
            get_brats2021(args.data_root, args.keep_archives, not args.no_extract)
        elif key == "ds007045":
            get_ds007045(args.data_root, args.tier)
        elif key in DATASETS:
            get_tcia(key, args.data_root)
        else:
            sys.exit(f"Unknown dataset '{key}'. Use --list to see the options.")


if __name__ == "__main__":
    main()
