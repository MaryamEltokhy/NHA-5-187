"""Project-wide constants: MRI modalities and the canonical label scheme.

Canonical labels (docs/PROJECT_STRUCTURE.md §3.2): 0 background, 1 NCR/NET, 2 ED, 3 ET, 4 RC (post-treatment only).
The cleaning step (M1-T04) writes every mask in this scheme, e.g. BraTS 2021's ET label 4 becomes 3.
"""

MODALITIES = ("t1", "t1ce", "t2", "flair")  # channel order of every model input
SEG = "seg"

BACKGROUND, NCR, ED, ET, RC = 0, 1, 2, 3, 4

# Evaluation regions, in the channel order of the model output. RC is never part of a tumor region.
REGIONS = {
    "TC": (NCR, ET),       # tumor core
    "WT": (NCR, ED, ET),   # whole tumor
    "ET": (ET,),           # enhancing tumor
}
