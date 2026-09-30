# Getting the Data

Where the data lives, what we download, and the commands. Owner: Alaa (M1-T01) and Hagar (M1-T03).

## Where the data lives

**Google Drive (the university account with unlimited storage) is the team's master copy.** One folder, `BrainMRI_Data/raw/`, is shared with all six members. The data is downloaded once and everyone uses the same files. Shared files count against the owner's storage, not the teammates'.

| Work | Where | Why |
| --- | --- | --- |
| Master copy of all datasets | **Google Drive** `BrainMRI_Data/raw/<cohort>/` | Everyone can reach it, and it survives Colab sessions |
| Real training (3D U-Net, nnU-Net) | **Google Colab** (Kaggle Notebooks as backup when the Colab GPU runs out) | Free 16 GB GPU. The laptop GPU (4 GB) is too small, since nnU-Net needs at least 10 GB. |
| Checking, cleaning, EDA, volumes, radiomics, change over time, the app | **Local PC or Colab** | CPU work; 16 GB RAM is enough |
| Quick tests (data loader, tiny 2-epoch run), predicting on a few cases | **Local GPU** (RTX 3050 Ti, 4 GB) | Small patches (e.g. 96³), batch size 1, mixed precision |

**Keep big datasets on Drive as archives, not as thousands of loose files.** Reading many small files through Drive in Colab is very slow. At the start of a Colab session, copy the archive to the Colab disk and extract it there (a few minutes for BraTS 2021).

Check the real quota first: at drive.google.com, open **Settings**, then **Storage**. Many universities now set a per-person limit.

## What we download

| Dataset | Use | Size | Needed by |
| --- | --- | --- | --- |
| BraTS 2021 Task 1 (Kaggle) | Training, validation, locked test | 13.4 GB | Now (M1) |
| RHUH-GBM (TCIA) | Change over time + external test | 2.9 GB | M3 (10 Oct) |
| MU-Glioma-Post (TCIA) | Change over time + post-treatment test | 11 GB | M3 (10 Oct) |
| BraTS-Africa (TCIA) | External test | 1.6 GB | M5 final tests (21 Oct) |
| UTSW-Glioma (TCIA) | External test (362 refined masks) + MGMT test | 22.9 GB | M5 final tests (21 Oct) |
| ds007045 (OpenNeuro), one preprocessed version | MGMT module (optional) + secondary external test | 9.4 GB (full dataset 48 GB; the other versions are copies of the same scans) | M3/M5, only if MGMT goes ahead |
| ~~BraTS 2024 (Kaggle mirror)~~ | Not downloaded | 159 GB | See below |

**Why not the BraTS 2024 Kaggle mirror:** it holds only 700 of the roughly 1,621 labelled cases, and it wrongly says CC0: the real license is CC BY-NC. MU-Glioma-Post already covers post-treatment scans and change over time. MU-Glioma-Post is also probably one of BraTS 2024's sources, so training on BraTS 2024 would stop MU-Glioma-Post from counting as an external test. If the team wants it anyway, use the official Synapse copy: complete, 43 GB compressed, needs a free Synapse account.

Total: about 61 GB.

## Step 1: put BraTS 2021 on Drive (Colab, once)

Ready-made notebook: `notebooks/00_download_brats2021_to_drive.ipynb` (upload it to Colab and choose **Run all**). The cells below are the short version.

Use Colab for this download, because Kaggle to Colab to Drive is fast and doesn't use your home internet.

First, add your Kaggle credentials as Colab secrets:
1. On kaggle.com, open **Settings**, then **API**, then **Create New Token**.
2. Open the downloaded `kaggle.json` and copy the username and key.
3. In Colab, click the key icon on the left and add two secrets: `KAGGLE_USERNAME` and `KAGGLE_KEY`.

Never put the token in the repo or in a notebook.

```python
from google.colab import userdata, drive
import os
drive.mount("/content/drive")
os.environ["KAGGLE_USERNAME"] = userdata.get("KAGGLE_USERNAME")
os.environ["KAGGLE_KEY"] = userdata.get("KAGGLE_KEY")
DRIVE = "/content/drive/MyDrive/BrainMRI_Data/raw"
```

```bash
!pip -q install kaggle
!for f in BraTS2021_Training_Data.tar BraTS2021_00495.tar BraTS2021_00621.tar; do kaggle datasets download dschettler8845/brats-2021-task1 -f $f -p /content/dl; done
!mkdir -p "{DRIVE}/brats2021" && cp /content/dl/* "{DRIVE}/brats2021/" && ls -lh "{DRIVE}/brats2021"
```

Then share `BrainMRI_Data` with the team (**Viewer** is enough). Each member opens the shared folder and chooses **Add shortcut to Drive**, so the folder appears under `MyDrive` in their Colab.

## Step 2: the other datasets

Ready-made Colab notebook for the four TCIA datasets: `notebooks/01_download_tcia_to_drive.ipynb`. It saves each dataset as one `.tar` on Drive, the same way as BraTS 2021. The options below are the alternatives.

Run `scripts/download_data.py` either on the local PC or in Colab. `--data-root` decides where the files go.

- **TCIA sets (RHUH, MU, Africa, UTSW):**
  - **Browser:** easiest on the local PC. Install **IBM Aspera Connect** (https://www.ibm.com/products/aspera/downloads) and download each package. The script prints the page links. Then upload the files to Drive with **Google Drive for desktop**, or drag them into drive.google.com.
  - **Command line:** install `ascli` (Ruby ≥ 3.1, then `gem install aspera-cli`, then `ascli config ascp install`) and the script downloads the packages by itself. TCIA has a Colab notebook for installing it: https://github.com/kirbyju/TCIA_Notebooks/blob/main/TCIA_Aspera_CLI_Downloads.ipynb. TCIA warns that `ascli` is noticeably slower on free Colab.
- **ds007045:** straight from OpenNeuro's public server. It works anywhere.

```bash
python scripts/download_data.py --list
python scripts/download_data.py rhuh mu africa utsw --data-root "G:/My Drive/BrainMRI_Data/raw"
python scripts/download_data.py ds007045 --data-root "G:/My Drive/BrainMRI_Data/raw"
python scripts/download_data.py verify --data-root "G:/My Drive/BrainMRI_Data/raw"
```

`G:/My Drive/...` is where Google Drive for desktop usually appears on Windows; check the letter on your PC. In Colab, use `/content/drive/MyDrive/BrainMRI_Data/raw`. For a local copy instead, leave out `--data-root` and files go to `data/raw/`, which git ignores.

## Step 3: start of every Colab training session

```python
from google.colab import drive
drive.mount("/content/drive")
DRIVE = "/content/drive/MyDrive/BrainMRI_Data/raw"
```

```bash
!mkdir -p /content/data/brats2021 && cp "{DRIVE}/brats2021/"* /content/data/brats2021/
!cd /content/data/brats2021 && (unzip -q '*.zip' 2>/dev/null; rm -f *.zip) && tar -xf BraTS2021_Training_Data.tar && tar -xf BraTS2021_00495.tar && tar -xf BraTS2021_00621.tar && rm *.tar && ls | wc -l
```

The big archive is extracted first, so the two single-patient archives go on top of it.

- **Train from `/content/data/`**, the fast local disk.
- **Save checkpoints and results to Drive** (e.g. `MyDrive/BrainMRI_Data/runs/<experiment>`), because free sessions end after a few hours. nnU-Net can resume from its latest checkpoint.
- **Use the patient split files from the repo**, so every member trains on exactly the same train/validation/test groups. The locked test patients are never used in training.

**Kaggle as backup:** in a Kaggle notebook, click **Add Input**, search `brats-2021-task1`, and add it. Under **Settings → Accelerator**, choose a GPU. Then run:

```python
import glob, tarfile
tar_path = glob.glob("/kaggle/input/**/BraTS2021_Training_Data.tar", recursive=True)[0]
tarfile.open(tar_path).extractall("/tmp/brats2021", filter="data")   # not /kaggle/working (about 20 GB, saved as output)
```
