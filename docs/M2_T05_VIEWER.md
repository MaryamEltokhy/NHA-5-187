# M2-T05 — Brain MRI Viewer

## What this task does

A small browser page that lets the user choose a cleaned MRI case, choose an MRI modality, move through axial slices with a slider, and display the existing `seg.nii.gz` tumor mask on top of the MRI in color.

**Important:** M2-T05 does **not** run the 3D U-Net. The task depends on M1-T04's cleaned NIfTI files, so there is no GPU requirement and no Cloud/Claude Pro requirement.

## Input layout

The viewer expects the cleaned M1-T04 layout documented in `docs/DATA_LOADER.md`:

```text
<processed_root>/
└── <dataset>/
    └── <subject>/
        └── <timepoint>/
            ├── t1.nii.gz
            ├── t1ce.nii.gz
            ├── t2.nii.gz
            ├── flair.nii.gz
            └── seg.nii.gz
```

For the team's Colab layout this is normally:

```text
/content/processed/<dataset>/<subject>/<timepoint>/...
```

## Local run (Windows/macOS/Linux)

From the repository root:

```bash
python -m pip install -r environment/requirements-viewer.txt
streamlit run apps/m2_t05_viewer.py
```

Then open the local URL printed by Streamlit (normally `http://localhost:8501`).

## Colab run

The app itself is CPU-only and does not need a GPU. A team member can install the two viewer libraries and run Streamlit in a Colab runtime, but the simplest workflow for this task is to run it on a normal laptop/desktop that has access to the cleaned files.

## What counts as done

- A case is selectable.
- The page shows an MRI modality slice.
- The axial-slice slider changes the displayed slice.
- `seg.nii.gz` is overlaid in discrete colors.
- The app refuses incomplete/mismatched cases with a readable error.

## Color meaning

| Canonical label | Meaning | Viewer color |
|---|---|---|
| 1 | NCR/NET | red |
| 2 | edema | green |
| 3 | enhancing tumor | blue |
| 4 | resection cavity | yellow |

The labels come from the project's canonical label convention in `src/bmts/common/constants.py` and `docs/DATA_LOADER.md`.

## Suggested evidence for the task board

Take one screenshot showing:
1. a real cleaned patient selected,
2. the FLAIR modality,
3. the slice slider,
4. a visible colored mask overlay.

Attach that screenshot to M2-T05 and mark the task Done when the team confirms the expected files are being used.
