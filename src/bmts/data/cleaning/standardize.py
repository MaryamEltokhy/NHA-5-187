# Standardization pipeline: RAS orientation, pad to 240x240x155, brain z-score, label remap (BraTS2021 4->3).
import os, shutil, tarfile
import numpy as np, pandas as pd, nibabel as nib
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from tqdm.auto import tqdm

BASE = Path('/content/drive/MyDrive/BrainMRI_Data')
DATA = Path('/content/data')          # where the tar files are extracted
OUT = Path('/content/processed')      # where standardized images are written
TARGET = np.array([240, 240, 155])
KEEP = ['t1', 't1ce', 't2', 'flair', 'seg']
LABEL_MAP = {'brats2021': {4: 3}, 'utsw': {4: 3}}   # africa, rhuh and mu keep their labels
TARS = {
    'brats2021': BASE / 'raw/brats2021/BraTS2021_Training_Data.tar',
    'africa': BASE / 'raw/external_brats_africa/external_brats_africa.tar',
    'rhuh': BASE / 'raw/longitudinal_rhuh_gbm/longitudinal_rhuh_gbm.tar',
    'mu': BASE / 'raw/longitudinal_mu_glioma_post/longitudinal_mu_glioma_post.tar',
    'utsw': BASE / 'raw/external_utsw_glioma/external_utsw_glioma.tar',
}


def is_junk(name):
    base = os.path.basename(name)
    return base == '.DS_Store' or base.startswith('._') or '__MACOSX' in name


def ensure_extracted(name):
    marker = DATA / f'.done_{name}'
    target = DATA / 'brats2021' if name == 'brats2021' else DATA
    if marker.exists():
        return
    if name == 'brats2021' and len(list(target.glob('BraTS2021_*'))) >= 1251:
        return
    size = TARS[name].stat().st_size
    if shutil.disk_usage('/content').free < size * 1.05:
        raise RuntimeError(f'not enough disk to extract {name}')
    target.mkdir(parents=True, exist_ok=True)
    with tarfile.open(TARS[name], 'r|', bufsize=16 * 1024 * 1024) as tar, \
         tqdm(total=size, unit='B', unit_scale=True, desc=name) as bar:
        for member in tar:
            if not is_junk(member.name):
                tar.extract(member, target, filter='data')
            bar.update(member.size)
    marker.touch()


def list_jobs(datasets):
    rows = []
    for ds in datasets:
        if ds == 'brats2021':
            for sub in sorted((DATA / 'brats2021').glob('BraTS2021_*')):
                if sub.is_dir():
                    for k in KEEP:
                        rows.append((ds, sub.name, '0', k, str(sub / f'{sub.name}_{k}.nii.gz')))
        else:
            man = pd.read_csv(BASE / f'manifests/{ds}_manifest.csv')
            for r in man[man['scan'].isin(KEEP)].itertuples():
                rows.append((ds, r.subject_id, str(r.timepoint_raw), r.scan, str(DATA / r.extracted_path)))
    jobs = pd.DataFrame(rows, columns=['dataset', 'subject', 'timepoint', 'scan', 'path'])
    jobs['out_path'] = [str(OUT / d / s / t / f'{k}.nii.gz') for d, s, t, k in
                        zip(jobs['dataset'], jobs['subject'], jobs['timepoint'], jobs['scan'])]
    return jobs


def standardize_file(row):
    if os.path.exists(row.out_path):
        return 'skipped', ''
    try:
        img = nib.as_closest_canonical(nib.load(row.path))
        data = np.asarray(img.dataobj, dtype=np.float32)
        if row.scan == 'seg':
            data = np.rint(data)
            for old, new in LABEL_MAP.get(row.dataset, {}).items():
                data[data == old] = new
            data = data.astype(np.uint8)
        else:
            brain = data != np.median(data)
            values = data[brain]
            data = np.where(brain, (data - values.mean()) / (values.std() + 1e-8), 0).astype(np.float32)
        shape = np.array(data.shape)
        if (shape > TARGET).any():
            return 'failed', f'shape {tuple(shape)} bigger than target'
        before = (TARGET - shape) // 2
        data = np.pad(data, list(zip(before, TARGET - shape - before)))
        affine = img.affine.copy()
        affine[:3, 3] -= affine[:3, :3] @ before
        Path(row.out_path).parent.mkdir(parents=True, exist_ok=True)
        nib.save(nib.Nifti1Image(data, affine), row.out_path)
        return 'done', ''
    except Exception as e:
        return 'failed', f'{type(e).__name__}: {e}'


def run(datasets, subjects=None, workers=4):
    jobs = list_jobs(datasets)
    if subjects is not None:
        jobs = jobs[jobs['subject'].isin(subjects)].copy()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        res = list(tqdm(ex.map(standardize_file, jobs.itertuples()), total=len(jobs)))
    jobs['status'] = [r[0] for r in res]
    jobs['error'] = [r[1] for r in res]
    print(jobs['status'].value_counts().to_string())
    return jobs


def regenerate(splits=('train', 'val'), datasets=('brats2021', 'mu', 'rhuh')):
    # test and external stay locked: ask for them only for the final evaluation
    idx = pd.read_csv(BASE / 'qc/unified_subject_index_v4.csv')
    idx = idx[idx['split'].isin(splits) & idx['dataset_source'].isin(datasets)]
    for ds in datasets:
        ensure_extracted(ds)
    return run(list(datasets), subjects=set(idx['original_subject_id']))
