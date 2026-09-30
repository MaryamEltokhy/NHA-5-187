# Dataset Research — Brain MRI Tumor Segmentation, Longitudinal Analysis, Radiomics and MGMT

**Project:** AI-Powered Brain MRI Tumor Analysis, Segmentation, and Quantitative Assessment System (DEPI, ML track, 2026)
**Purpose of this document:** replace the proposal's two Kaggle-linked datasets with a verified, official-source dataset strategy covering (A) 3D segmentation training, (B) external generalization testing, (C) longitudinal same-patient analysis, (D) radiomics, and (E) the optional MGMT module, with explicit overlap/leakage analysis.
**Research date:** 16 September 2026. Every card below was produced from the official landing page and/or dataset paper and then independently fact-checked by a second pass that re-fetched the sources; corrections from that pass are already merged. Items that could not be confirmed on an official page are marked **[verify on download]** and are assigned to task `M1-T01` in `PROJECT_PLAN_NOTION.md`.
**Rule applied:** official hosts (TCIA, Synapse, OpenNeuro, Springer Nature figshare, institutional portals) are the authoritative sources; Kaggle/HuggingFace re-uploads are listed only as convenience mirrors.

---

## 1. Executive summary — what changed versus the proposal

| Proposal assumption | Verified finding | Consequence |
| --- | --- | --- |
| BraTS 2021 Task 1 obtained from Kaggle | The authoritative release is TCIA *RSNA-ASNR-MICCAI-BraTS-2021* (DOI 10.7937/jc8x-9874, **CC BY 4.0**, open, both tasks + an ID crosswalk to TCIA source collections). The identical voxels are also on Synapse as BraTS 2023 Adult Glioma (CC BY-NC, click-through, ET relabelled 3). The Kaggle Task 1 upload is a user mirror | Use TCIA (or Synapse) as the source; Kaggle only as fallback. Use the crosswalk for provenance-aware splitting and overlap checks |
| BraTS 2024 Adult Glioma Post-Treatment from Kaggle | Official host is Synapse (BraTS 2024 → continued as BraTS-Lighthouse 2025 Task 1), **CC BY-NC 4.0**, registration + click-through. Training set ≈ 1,621 labelled post-treatment timepoints from ≈ 731 subjects (initial release 1,350), label scheme **1 NETC, 2 SNFH, 3 ET, 4 RC**, subject IDs with ordinal timepoint suffixes (`-100`, `-101`, …), no dates. The Kaggle mirror is partial (700 cases) and mislabels the license | Keep it, but as a post-treatment/supplementary cohort with label harmonization; it is itself a large ordinal longitudinal source |
| "Compatible independent data for generalization" unspecified | Most public glioma sets **are inside BraTS 2021** (UPenn-GBM 447 subjects, UCSF-PDGM 298, TCGA-GBM/LGG 243, CPTAC-GBM 39, IvyGAP 34; BraTS 2018–2020 and MSD Task01 are subsets). Confirmed independent, open sets exist: **UTSW-Glioma** (2026, 625 pre-op, CC BY 4.0), **BraTS-Africa** (TCIA, CC BY 4.0), **RHUH-GBM** (Spain, CC BY 4.0 NIfTI), **OpenNeuro ds007045** (363 GBM, 8 European sites, CC0), **MU-Glioma-Post** (post-treatment, CC BY 4.0) | External generalization cohorts selected from the independent list only |
| A longitudinal dataset "may be needed" | Several usable options: **MU-Glioma-Post** (203 patients, ≈ 596 timepoints, refined masks at every timepoint, days-from-diagnosis), **RHUH-GBM** (40 × 3 timepoints with expert masks, intervals preserved), BraTS 2024 same-subject pairs (ordinal only), LUMIERE (91 patients, 638 timepoints, automatic masks, weeks, RANO ratings) | Longitudinal module is feasible with real data; no synthetic fallback expected to be needed |
| Optional MGMT "where labels are available" | Labels are widely available (BraTS 2021 Task 2 via crosswalk 695; UCSF-PDGM 416; ds007045 363; UTSW 281; MU-Glioma-Post ≈ 163; LUMIERE 80), **but** the published evidence shows the MRI→MGMT signal is weak (Kaggle 2021 winner AUC ≈ 0.62; independent re-evaluations at chance; radiomics meta-analysis external AUC ≈ 0.65). RHUH-GBM has **no** MGMT labels (only IDH) | Keep MGMT optional, pre-register H0 (AUC = 0.5), report CIs and permutation null; negative results acceptable |
| — | Several relevant collections became **controlled-access** (NIH policy, dbGaP) in 2025: TCGA-GBM/LGG raw DICOM, IvyGAP, CPTAC-GBM radiology, Burdenko-GBM-Progression, GLIS-RT, QIN GBM Treatment Response, RHUH raw DICOM, BraTS-Africa unprocessed images | Only open NIfTI derivatives are used |

---

## 2. Datasets already in the proposal

### 2.1 BraTS 2021 Task 1 (segmentation) and Task 2 (MGMT) — RSNA-ASNR-MICCAI BraTS 2021

| Field | Value |
| --- | --- |
| Dataset name | RSNA-ASNR-MICCAI BraTS 2021 — Task 1 (Brain Tumor Segmentation), Task 2 (Radiogenomic Classification: MGMT) |
| Official dataset source | The Cancer Imaging Archive (TCIA), analysis-result collection *RSNA-ASNR-MICCAI-BraTS-2021* (DOI 10.7937/jc8x-9874). Identical Task 1 voxels re-released on Synapse as **BraTS 2023 Adult Glioma (GLI)** |
| Official URL | https://www.cancerimagingarchive.net/analysis-result/rsna-asnr-miccai-brats-2021/ (the legacy `wiki.cancerimagingarchive.net/display/DOI/RSNA-ASNR-MICCAI-BraTS-2021` page now returns 404) · Synapse: https://www.synapse.org/Synapse:syn51156910/wiki/622351 (files syn64952532 → syn51514105) · Organizer page: https://www.med.upenn.edu/cbica/brats2021/ · Task 2 Kaggle competition: https://www.kaggle.com/c/rsna-miccai-brain-tumor-radiogenomic-classification |
| Convenience mirrors (not authoritative) | Kaggle `dschettler8845/brats-2021-task1` (user re-upload of the Task 1 training tar); HuggingFace re-uploads of the 2023 GLI zips |
| Hosting institution | TCIA (NCI/UAMS); challenge organized by RSNA, ASNR and MICCAI with UPenn CBICA |
| Associated challenge | BraTS 2021 (MICCAI 2021); data reused unchanged in BraTS 2022 and BraTS 2023 GLI; pre-treatment half of BraTS-Lighthouse 2025 Task 1 (re-preprocessed) |
| Related paper | Baid et al., "The RSNA-ASNR-MICCAI BraTS 2021 Benchmark on Brain Tumor Segmentation and Radiogenomic Classification", arXiv:2107.02314 (2021); Bakas et al., Sci Data 2017 (TCGA segmentations); Menze et al., IEEE TMI 2015 |
| Number of subjects | 2,040 in the challenge (1,251 training + 219 validation + 570 sequestered test). Publicly downloadable: TCIA package lists **1,480** subjects (1,470 Task 1 cases + 9–10 Task 2-only cases); Synapse: 1,470 (1,251 with labels + 219 without) |
| Number of scans | One pre-operative study per subject × 4 modalities; 1,251 segmentation masks |
| MRI modalities | T1, T1ce (T1Gd), T2, FLAIR |
| Segmentation labels | 0 background, **1 NCR/NET, 2 ED, 4 ET** (label 3 unused); regions ET = {4}, TC = {1,4}, WT = {1,2,4}. Synapse 2023 copy: ET = 3 |
| Clinical labels | None in Task 1 (age/survival were BraTS 2020 Task 2 only) |
| Molecular labels / MGMT | Task 2: binary MGMT (0/1) for 585 Kaggle training subjects (307 methylated / 278 unmethylated; 3 flagged as corrupt → 582). The TCIA crosswalk `BraTS2021_MappingToTCIA.xlsx` publishes MGMT for **695** cases in total; 594 Task 1 training cases (with masks) and 92 validation cases carry a label. Assays heterogeneous across sites |
| Longitudinal availability | None (single pre-operative timepoint) |
| Subject identifiers | `BraTS2021_XXXXX` (Task 1) / `BraTS21ID` (Task 2) — same numbering; crosswalk maps 767 cases to TCIA source IDs (UPENN-GBM 447, UCSF-PDGM 298–299, TCGA-GBM 135, TCGA-LGG 108, CPTAC-GBM 39, IvyGAP 34, ACRIN-FMISO 4); 413 rows are "new, not previously in TCIA" |
| Acquisition dates | Not in NIfTI; crosswalk gives (shifted) study dates for TCIA-derived cases only |
| File format | NIfTI (.nii.gz) 240×240×155; Task 2 DICOM |
| Voxel dimensions | 1 mm isotropic, co-registered to SRI24, skull-stripped |
| Annotation quality | Expert-approved manual refinement of STAPLE-fused automatic masks (nnU-Net, DeepScan, DeepMedic), 1–4 raters, neuroradiologist approval |
| Training suitability | **Excellent** — the primary development cohort |
| External validation suitability | Not applicable (it is the training set) |
| Longitudinal suitability | None |
| Radiomics suitability | Good (standard space, consistent preprocessing; no scanner metadata per case; intensities not bias-corrected) |
| Access / registration | TCIA: open, no registration; IBM Aspera Connect plugin for the faspex package. Synapse: free account + self-sign click-through. Kaggle Task 2: account + rules |
| License | TCIA package: **CC BY 4.0**; Synapse 2023 copy: CC BY-NC 4.0 + mandatory attribution ("obtained as part of the BraTS Challenge project through Synapse ID syn51156910"); original source DICOM (TCGA/CPTAC/IvyGAP/ACRIN): NIH Controlled Data Access Policy |
| Research-use restrictions | Citation of the BraTS papers required; no redistribution of the Synapse copy |
| Download procedure | TCIA page → Data Access → "Challenge data both tasks (142 GB)" via Aspera (+ crosswalk xlsx). Task 1 only: Synapse `synapse get -r syn51514105` (TrainingData.zip, ValidationData.zip, `BraTS2023_2017_GLI_Mapping.xlsx`) |
| Estimated storage | Task 1 NIfTI ≈ 12–13 GB compressed (≈ 110–220 GB if fully unpacked to float32); TCIA package incl. Task 2 DICOM 142 GB |
| Known limitations | No scanner/site metadata per case; label 4 convention; unlabelled validation cases; hidden test never released |
| Class imbalance | ET is the smallest class; some cases have no ET; background dominates (> 99 % of voxels) |
| Institutional bias | North American / Western European tertiary centres; 1.5 T and 3 T mix |
| Scanner/protocol diversity | Multi-institutional (≥ 13 contributors, USA/Canada/Germany/Switzerland/Hungary/India) |
| Potential subject overlap | Identical to BraTS 2022/2023 GLI; superset of BraTS 2017–2020; near-complete overlap with MSD Task01; contains UPenn-GBM (447), UCSF-PDGM (298–299), TCGA-GBM/LGG (243), CPTAC-GBM (39), IvyGAP (34), ACRIN (4); 38 BraTS 2024 post-treatment subjects share the ID space (pre→post pairs) |
| Leakage risks | Any of the above cannot be an external test set; use the crosswalk to exclude mapped IDs from external cohorts; the 1,470 vs 1,480 discrepancy must be resolved during indexing |

### 2.2 BraTS 2024 Adult Glioma Post-Treatment (BraTS-GLI 2024; BraTS-Lighthouse 2025 Task 1 post-treatment half)

| Field | Value |
| --- | --- |
| Dataset name | BraTS 2024 Adult Glioma Post-Treatment segmentation dataset |
| Official dataset source | Synapse (Sage Bionetworks) — BraTS 2024 project syn53708249; data folder syn59059776 (Training, AdditionalTraining, Validation zips, demographics xlsx); 2025 continuation syn64153130 |
| Official URL | https://www.synapse.org/Synapse:syn53708249/wiki/627500 · 2025 task page https://www.synapse.org/Synapse:syn64153130/wiki/631053 · data access https://www.synapse.org/Synapse:syn64153130/wiki/631048 |
| Convenience mirrors (not authoritative) | Kaggle `i212385nomanarif/2024-brats-glioma` (partial: 700 cases, uncompressed, license mislabelled CC0); HuggingFace `Aff77/BraTS-2024-Complete` (unofficial Dec-2024 copy) |
| Hosting institution | Sage Bionetworks on behalf of the BraTS consortium (UCSD/Scripps, Indiana University, UCSF, Duke, Missouri, Heidelberg, Michigan) |
| Associated challenge | BraTS 2024 (MICCAI 2024) — Adult Glioma Post-Treatment; BraTS-Lighthouse 2025 Task 1 |
| Related paper | de Verdier et al., "The 2024 Brain Tumor Segmentation (BraTS) Challenge: Glioma Segmentation on Post-treatment MRI", arXiv:2405.18368 |
| Number of subjects | Reported as cases: ≈ 2,200 total (70/10/20 split). The organizers' metadata file in syn59059776 (`BraTS-PTG supplementary demographic information and metadata.xlsx`, 1,809 rows) lists **1,350 Train + 271 Train-additional = 1,621 labelled training timepoints** and **188 validation timepoints**, ≈ 818 unique subject IDs (≈ 731 training subjects + 87 validation subjects) |
| Number of scans | ≈ 1,621 × 4 modalities + masks (+ 188 × 4 unlabelled) |
| MRI modalities | T1, T1ce, T2, FLAIR |
| Segmentation labels | **1 NETC** (non-enhancing tumor core), **2 SNFH** (surrounding non-enhancing FLAIR hyperintensity), **3 ET**, **4 RC** (resection cavity); WT = 1+2+3, TC = 1+3, ET = 3 (RC scored separately in the official `metrics_GLI.py`) |
| Clinical / molecular labels / MGMT | Metadata xlsx with per-case **Site** and **Site Subject ID** plus demographics (use for site-stratified analysis and to attempt ID linkage with MU-Glioma-Post / UCSD-PTGBM **[verify on download]**); no MGMT/IDH labels documented |
| Longitudinal availability | **Yes, ordinal**: IDs `BraTS-GLI-XXXXX-TTT`; TTT = 100, 101, 102… successive post-treatment scans of the same subject (000/001 = pre-op baselines in the 2025 pre-treatment half). ≈ 76 % of training subjects have ≥ 2 timepoints (suffix distribution 100: 728, 101: 560, 102: 153, 103: 88, 104: 52, …) |
| Subject identifiers | 5-digit subject ID; IDs ≤ 01790 share the BraTS 2021/2023 numbering (38 subjects / 78 cases = follow-ups of BraTS 2021 subjects); IDs ≥ 02005 new |
| Acquisition dates | None; no intervals; ordering by suffix only |
| File format | NIfTI (.nii.gz) |
| Voxel dimensions | 1 mm isotropic, skull-stripped (HD-BET), affine-registered to a linear symmetric MNI atlas (some contributing sites processed with FeTS/SRI24 in-house pipelines) |
| Annotation quality | Hybrid: automatic pre-segmentation, manual refinement by 1–4 raters, neuroradiologist approval |
| Training suitability | Good for post-treatment segmentation; label semantics differ from BraTS 2021 (SNFH includes treatment change; RC new) |
| External validation suitability | Post-treatment evaluation of a BraTS-2021-trained model is valid for subjects not in the shared-ID set; not independent of MU-Glioma-Post / UCSD-PTGBM / UCSF-ALPTDG |
| Longitudinal suitability | Good for ordering-only comparisons (no rate of change) |
| Radiomics suitability | Moderate (post-treatment heterogeneity; no scanner metadata) |
| Access / registration | Synapse account → BraTS 2024 Data Access page (wiki 627759, still live) → "Access BraTS 2024 Data" (syn64952546) → accept post-challenge terms (self-sign click-through). The BraTS-Lighthouse 2025 route is currently hidden; the 2025 post-challenge set is announced for TCIA hosting but not yet listed there |
| License | **CC BY-NC 4.0** + mandatory attribution ("obtained as part of the BraTS Challenge project through Synapse ID: syn53708249") and citation of the flagship and task papers |
| Research-use restrictions | Non-commercial; no redistribution; cite arXiv:2405.18368 |
| Download procedure | Synapse web or `synapse get -r syn59059776` after access approval (TrainingData.zip, AdditionalTrainingData.zip, ValidationData.zip, metadata xlsx) |
| Estimated storage | ≈ 43 GB compressed training + ≈ 5.4 GB validation (≈ 230 GB if unpacked float32) |
| Known limitations | No dates/intervals; no molecular labels; label-semantics shift vs pre-treatment; subject-level grouping required for any split |
| Class imbalance | ET small and often absent post-treatment; RC variable |
| Institutional bias | Seven US/German academic centres |
| Scanner/protocol diversity | Multi-institutional, heterogeneous |
| Potential subject overlap | UCSD-PTGBM (≈ 350 cases), UCSF-LPTDG/ALPTDG (≈ 600), University of Missouri (≈ 400, i.e., MU-Glioma-Post), 38 BraTS 2021 subjects |
| Leakage risks | Training on it invalidates MU-Glioma-Post/UCSD-PTGBM/ALPTDG as external sets; within-dataset leakage if timepoints of one subject cross splits |

---

## 3. Newly researched datasets

### 3.1 UTSW-Glioma (University of Texas Southwestern) — **selected: External A + MGMT external**

| Field | Value |
| --- | --- |
| Official source / URL | TCIA collection *UTSW-Glioma*, DOI 10.7937/dfae-1b86 — https://www.cancerimagingarchive.net/collection/utsw-glioma/ |
| Hosting institution | TCIA; data from UT Southwestern Medical Center (Dallas) |
| Associated challenge | None |
| Related paper | Reddy et al., Sci Data 13, 934 (2026) https://www.nature.com/articles/s41597-026-07274-4 (precursor: CVPRW 2024) |
| Subjects / scans | 625 adult glioma patients; one pre-operative session each; 6,349 NIfTI files (two skull-stripped variants per case) |
| Modalities | T1, T1Gd, T2, T2-FLAIR (all 625) |
| Segmentation labels | ET / NCR / ED per FeTS-BraTS protocol; **362 expert-refined** (multi-stage QC, 4-neuroradiologist consensus for hard cases), **263 unrefined** automatic FeTS masks. Integer values not printed; expected FeTS scheme 1 NCR, 2 ED, 4 ET **[verify on download]** |
| Clinical / molecular / MGMT | Age, sex, WHO grade, IDH, 1p/19q; **MGMT for 281/625** (114 methylated / 167 unmethylated; qMSP, 10 % cut-off) |
| Longitudinal | None (post-surgical cases excluded) |
| Subject IDs / dates | TSV subject IDs; no scan dates (age at imaging only) |
| Format / voxel | NIfTI, SRI24, 1 mm isotropic (matrix not stated; FeTS standard 240×240×155 **[verify]**) |
| Annotation quality | High for the 362 refined; no inter-rater statistics published |
| Suitability | Training: possible but reserved; **External validation: excellent** (independent institution, same preprocessing space); Longitudinal: no; Radiomics: good (consistent pipeline; intensities un-normalized); MGMT: good external test set |
| Access / license | Open, no registration; Aspera plugin; **CC BY 4.0** (images and TSV); TCIA citation required |
| Download | TCIA page → Aspera package (22.9 GB) + metadata TSV |
| Limitations | 263 auto masks unusable as ground truth; initial masks for all cases came from a FeTS model trained on BraTS-era data (annotation-style correlation) |
| Class imbalance / bias / diversity | Single institution; mixed grades; Siemens/Philips/GE scanners 1.5 T and 3 T |
| Overlap / leakage | No documented overlap with BraTS (UT Southwestern is not a BraTS contributor; BraTS 2021 lists UT MD Anderson and UT Health San Antonio only). Use only the 362 refined masks for headline metrics |

### 3.2 BraTS-Africa / BraTS-SSA (Sub-Saharan Africa adult glioma) — **selected: External B**

| Field | Value |
| --- | --- |
| Official source / URL | TCIA collection *BraTS-Africa* — https://www.cancerimagingarchive.net/collection/brats-africa/ (processed NIfTI + segmentations, CC BY 4.0); challenge splits on Synapse (BraTS-SSA 2025 folder syn65550519 / syn64377310, registration required; 2023/2024 Synapse folders archived) |
| Hosting institution | TCIA; Nigerian centres (Crestview Radiology Lagos, LASUTH, LUTH, LILY Hospital Benin, NSIA-Kano, MEDHUB Umuahia); SPARK Academy / BraTS organizers |
| Associated challenge | BraTS 2023/2024/2025 SSA task (60 train / 15–35 val / 20–30 test) |
| Related paper | Adewole et al., "The Brain Tumor Segmentation (BraTS) Challenge 2023: Glioma Segmentation in Sub-Saharan Africa", arXiv:2305.19369; organizers' Neuro-Oncology Advances 2026 paper |
| Subjects / scans | TCIA: **146 subjects** (95 presumed adult diffuse glioma + 51 other CNS neoplasms), one pre-op study each; 730 processed series |
| Modalities | T1, T1ce, T2, FLAIR |
| Segmentation labels | BraTS 2023 scheme 1 NCR, 2 ED, 3 ET; semi-automatic (nnU-Net pre-annotation) refined by residents, corrected by two board-certified radiologists, approved by an expert BraTS neuroradiologist |
| Clinical / molecular / MGMT | Radiologic diagnosis category, scanner metadata; no molecular labels |
| Longitudinal | None |
| Format / voxel | NIfTI, SRI24, 1 mm isotropic |
| Suitability | External validation: **best available domain-shift set** (all 1.5 T, low-resource protocols, different population); Training: 60-case SSA training split could be used for adaptation experiments only; Radiomics: limited (lower SNR/CNR, artifacts); MGMT: none |
| Access / license | TCIA processed package: open, Aspera, **CC BY 4.0**; unprocessed images: NIH controlled access (not downloadable); Synapse 2025 data: CC BY-NC + registration |
| Storage | 1.6 GB processed (3.7 GB collection) |
| Limitations | No pathology confirmation ("presumed" glioma); missing sequences common in the source population; 51 non-glioma cases must be reported separately |
| Overlap / leakage | Independent of BraTS 2021 sources; SSA training cases are used in BraTS 2023–2025 challenges, so models pretrained on those challenge sets must not be evaluated here |

### 3.3 RHUH-GBM (Río Hortega University Hospital Glioblastoma, Spain) — **selected: External C + Longitudinal B**

| Field | Value |
| --- | --- |
| Official source / URL | TCIA collection *RHUH-GBM*, DOI 10.7937/4545-c905 — https://www.cancerimagingarchive.net/collection/rhuh-gbm/ |
| Hosting institution | TCIA; Río Hortega University Hospital, Valladolid, Spain |
| Related paper | Cepeda et al., "The Río Hortega University Hospital Glioblastoma dataset: a comprehensive collection of preoperative, early postoperative and recurrence MRI scans (RHUH-GBM)", Data in Brief 2023 |
| Subjects / scans | 40 patients × **3 timepoints** (pre-operative, early post-operative ≤ 72 h, recurrence/progression) = 120 studies; NIfTI package 720 files |
| Modalities | T1, T1ce, T2, FLAIR, ADC |
| Segmentation labels | Necrosis / peritumoral signal alteration (edema **plus** non-enhancing tumor) / enhancing tumor, DeepMedic pre-segmentation manually corrected by two expert neurosurgeons at **every timepoint**; documented as 1/2/3 but the authors' code implies 1/2/4 **[verify on download]**; no resection-cavity label |
| Clinical / molecular / MGMT | Age, sex, extent of resection, PFS, OS, **IDH** (4 mutant / 36 wild-type); **no MGMT** (the proposal-era assumption that RHUH has MGMT is false) |
| Longitudinal | Yes: three timepoints; DICOM dates shifted but intervals preserved (pre→post median 6 d; post→recurrence median 196.5 d); CSV gives days imaging→surgery, PFS, OS |
| Format / voxel | NIfTI, SRI24, 1 mm isotropic, SynthStrip skull-stripping, **already z-score normalized** (T1/T1ce/T2/FLAIR) |
| Suitability | External validation: good (independent Spanish centre; expert labels) with the TC caveat (non-enhancing tumor merged into "peritumoral" → TC under-defined; report WT/ET primarily); Longitudinal: **excellent** (expert masks at all timepoints, intervals); Radiomics: limited (pre-normalized intensities cannot be undone without controlled DICOM); MGMT: none |
| Access / license | NIfTI + segmentations + clinical CSV: open, Aspera, **CC BY 4.0**; raw DICOM: NIH controlled access |
| Storage | 2.9 GB (NIfTI) |
| Limitations | Small (n = 40); GBM only; post-op cavity unlabelled |
| Overlap / leakage | No BraTS contribution (no Spanish institution in BraTS 2021/2024 contributor lists); probable overlap only with the same group's in-house cohorts |

### 3.4 MU-Glioma-Post (University of Missouri post-treatment glioma) — **selected: Longitudinal A + post-treatment external**

| Field | Value |
| --- | --- |
| Official source / URL | TCIA collection *MU-Glioma-Post*, DOI 10.7937/7K9K-3C83 — https://www.cancerimagingarchive.net/collection/mu-glioma-post/ |
| Hosting institution | TCIA; University of Missouri (Columbia) |
| Related paper | Nature Sci Data (2025) dataset descriptor (Missouri group, FeTS/FL-PoST pipeline); related FeTS 2.0 abstract (Neuro-Oncology 2025) |
| Subjects / scans | **203 patients**; post-treatment timepoints per patient 1–6; total ≈ **596** timepoints (594/596/617 quoted across official sources; 2,978 NIfTI series ≈ 596 × 5) |
| Modalities | T1, T1ce, T2, FLAIR (+ segmentation) per timepoint |
| Segmentation labels | **1 NETC, 2 SNFH, 3 ET, 4 RC** (BraTS 2024 scheme); nnU-Net/FeTS automatic masks refined under three-tier neuroradiologist review (Dice auto-vs-final WT 0.99 → labels retain model characteristics) |
| Clinical / molecular / MGMT | Clinical spreadsheet: diagnosis, grade, treatment, progression events (imaging/clinical), resections, survival; **MGMT ≈ 163/203** (66 methylated / 97 unmethylated); IDH |
| Longitudinal | **Yes, core feature**: `PatientID_XXXX/Timepoint_X/`; **days from diagnosis to each timepoint** in the clinical sheet (mean FU1–FU6 ≈ 66/144/231/290/356/458 d); no pre-operative scan **[verify whether clinical dates are absolute or offsets]** |
| Format / voxel | NIfTI, SRI24, 1 mm isotropic, N4-corrected, BrainMaGe skull-stripping; intensities not normalized |
| Suitability | Longitudinal: **excellent** (GT at every timepoint, intervals); External validation (post-treatment): good for models not trained on BraTS 2024; Training: possible for post-treatment fine-tuning (then loses external status); Radiomics: good; MGMT: usable but post-treatment imaging |
| Access / license | Open, no DUA; Aspera package (11 GB) + XLSX; **CC BY 4.0** |
| Limitations | Model-influenced labels; timepoint count inconsistencies across sources; whether the composite WT/TC definitions exclude RC is inferred (BraTS 2024 convention) |
| Overlap / leakage | **Highly likely a source of ≈ 400 BraTS 2024 post-treatment cases** (not explicitly stated; IDs not linkable) → never evaluate a BraTS-2024-trained model here; clean for BraTS-2021-only models |

### 3.5 OpenNeuro ds007045 — Real-world multi-institutional glioblastoma MRI with MGMT (2026) — **selected: External D (secondary) + MGMT development**

| Field | Value |
| --- | --- |
| Official source / URL | OpenNeuro dataset ds007045 — https://openneuro.org/datasets/ds007045 (v2.0.1, 2026-09-04; paper cites v1.0.1) |
| Hosting institution | OpenNeuro (Stanford); contributed by 8 European hospitals: Bologna, Karlsruhe, Novosibirsk, Turin, Tarragona, Foggia, Messina, Bari |
| Related paper | Filimonova, Leone, Carbone et al., "Glioblastoma MRI Dataset with Standardized Preprocessing, Expert-Validated Segmentation, and MGMT Profiling", Sci Data 13, 1213 (2026) |
| Subjects / scans | **363** GBM patients (v2.0.1; 337 in v1.0.1), one pre-operative session each; raw + 3 derivative tiers |
| Modalities | T1, T1ce, T2, FLAIR |
| Segmentation labels | 1 NCR, 2 ED (non-enhancing/edematous), 3 ET (BraTS 2023 scheme; verified by decoding masks); CNN pre-segmentation (nnU-Net/SegResNet pretrained on BraTS-METS 2024) with single-pass expert review — ≈ 71 % unchanged |
| Clinical / molecular / MGMT | Age, sex, site, scanner; **MGMT for 363/363** (166 methylated / 197 unmethylated; MSP, local labs) |
| Longitudinal | None |
| Format / voxel | BIDS; raw native + derivatives in FSL MNI152 1 mm (182×218×182), skull-stripped, N4 + z-scored variant ("radiomics-ready"); defaced raw |
| Suitability | External validation: good multi-site independence but **label-quality caveat** (CNN-derived masks) → secondary; MGMT: **best development cohort** (complete labels, multi-site); Radiomics: good; Longitudinal: none |
| Access / license | Fully open, anonymous; **CC0**; citation requested |
| Storage | ≈ 19 GB |
| Limitations | Linear MNI registration (not SRI24); one empty derivative folder; edema boundaries least reliable; site imbalance (Bologna 171, Karlsruhe 110) |
| Overlap / leakage | No BraTS/TCGA/UPenn/UCSF contributors; inferred independent |

### 3.6 UCSF-PDGM (UCSF Preoperative Diffuse Glioma MRI) — evaluated; **MGMT labels only** (not an external segmentation set)

| Field | Value |
| --- | --- |
| Official source / URL | TCIA collection *UCSF-PDGM* — https://www.cancerimagingarchive.net/collection/ucsf-pdgm/ (v5) |
| Related paper | Calabrese et al., Radiology: AI 2022 |
| Subjects / scans | 495 patients / 501 studies (6 short-interval follow-ups renamed `_FU###d`); 12,030 files; T1, T1c, T2, FLAIR + SWI, DWI/ADC, ASL, DTI derivatives |
| Segmentation labels | ET / NCR / ED; automated BraTS-ensemble masks corrected by annotators and approved by neuroradiologists; integer scheme likely 1/2/4 **[verify]** |
| Molecular / MGMT | IDH, 1p/19q, grade, **MGMT status for 416 determinate cases** (302 positive / 114 negative) + quantitative index |
| Longitudinal | Minimal (6 pairs) |
| Access / license | Open; Aspera (142 GB); **CC BY 4.0** |
| Overlap / leakage | **298 (–299) studies are BraTS 2021 training/validation cases** (official crosswalk; TCIA hosts the corresponding package); further UCSF cases probably in the hidden BraTS test set. Also 62 patients overlap UCSF-ALPTDG. → Not an external segmentation test; MGMT labels usable with tagging |

### 3.7 UPenn-GBM — evaluated; **not selected**

| Field | Value |
| --- | --- |
| Official source / URL | TCIA collection *UPENN-GBM* — https://www.cancerimagingarchive.net/collection/upenn-gbm/ ; Bakas et al., Sci Data 2022 |
| Subjects / scans | 630 patients; 611 baseline + 60 follow-up (`_11` / `_21`; 41 follow-ups have their baseline in the collection); T1, T1Gd, T2, FLAIR (+ DTI, DSC derivatives) |
| Segmentation labels | ET/NCR/ED; automatic STAPLE (DeepMedic + DeepSCAN + nnU-Net) for all baselines, **232 manually refined** |
| Molecular / MGMT | MGMT for 317 baseline (140/177) per paper (CSV: 262 baseline rows labelled); IDH; survival; precomputed CaPTk radiomics |
| Access / license | Open; NBIA/Aspera; **CC BY 4.0** |
| Overlap / leakage | **447 baseline subjects are BraTS 2021 cases** (403 train / 44 val) — including 148 of the refined masks and 249 of the MGMT-labelled subjects; half of the follow-up subjects have baselines in BraTS training → cannot serve as external or clean longitudinal test |

### 3.8 LUMIERE (Longitudinal Glioblastoma MRI with Expert RANO Evaluation) — evaluated; **optional Longitudinal D**

| Field | Value |
| --- | --- |
| Official source / URL | Springer Nature figshare collection 5904905 — https://springernature.figshare.com/collections/The_LUMIERE_Dataset_Longitudinal_Glioblastoma_MRI_with_Expert_RANO_Evaluation/5904905 ; Suter et al., Sci Data 9:768 (2022) |
| Subjects / scans | 91 GBM patients (Inselspital Bern); **638 timepoints** (2–21 per patient, mean 7); 599 complete 4-sequence studies; 616 timepoints with expert RANO rating |
| Segmentation labels | **Automatic only** (DeepBraTumIA: 1 necrosis, 2 contrast-enhancing, 3 edema; HD-GLIO-AUTO: 1 non-enhancing, 2 enhancing); outliers intentionally retained |
| Molecular / MGMT | MGMT qualitative for 80/91 (37 methylated / 43 not); quantitative % for 64; IDH; survival |
| Longitudinal / dates | Week index since pre-operative MRI (`week-NNN`, intra-week suffix); no day-level intervals |
| Access / license | Open figshare download; figshare metadata says CC0 but the authors state **non-commercial use** in paper, README and GitHub → treat as non-commercial |
| Storage | 32.6 GB imaging (+ optional 54 GB feature maps) |
| Use | Only public set with response ratings; training data of BraTS 2025 Task 11 (BraTPRO). Useful to compare the project's trend summary against RANO classes; not usable as segmentation ground truth |
| Overlap / leakage | Possible unmappable overlap with historic BraTS 2013 Bern cases inside BraTS 2021 → caveat; no overlap with BraTS 2024 |

### 3.9 UCSD-PTGBM (UC San Diego post-treatment glioblastoma) — evaluated; **documented alternate to MU-Glioma-Post**

| Field | Value |
| --- | --- |
| Official source / URL | TCIA collection *UCSD-PTGBM*, DOI 10.7937/fwv2-dt74 — https://www.cancerimagingarchive.net/collection/ucsd-ptgbm/ ; Gagnon et al., Sci Data 13:183 (2026) |
| Subjects / scans | 178 patients; 243 post-treatment timepoints (main 136/184 + "BraTS-GLI 2024 Test Data" 42/59); T1, T1ce, T2, FLAIR (+ DWI/ADC, perfusion-derived maps) |
| Segmentation labels | ET / NETC / SNFH / RC (BraTS 2024 scheme; integers **[verify]**), plus "cellular tumor" masks; automated pre-segmentation with radiologist correction |
| Molecular / MGMT | MGMT for 149 timepoints (76/73); IDH; OS/PFS; 51 timepoints of treatment-related change only (pseudoprogression / radiation necrosis labels) |
| Longitudinal | Multiple timepoints for a subset (≈ 1.37 per subject); no dates documented on the landing page |
| Access / license | Open; Aspera (45 GB); **CC BY 4.0** |
| Overlap / leakage | Direct overlap with BraTS 2024 (UCSD contributed ≈ 350 cases) and BraTS 2025 test data → valid only for BraTS-2021-only models; kept as an alternate longitudinal/post-treatment cohort |

### 3.10 BraTS 2023 Adult Glioma (GLI) — evaluated; **not an additional dataset**

Synapse syn51156910 (files syn64952532 → syn51514105). Patient- and scan-identical to BraTS 2021 (1,251 training + 219 validation), renamed `BraTS-GLI-XXXXX-000`, ET relabelled 3; `BraTS2023_2017_GLI_Mapping.xlsx` maps to TCGA/IvyGAP/CPTAC. License CC BY-NC 4.0 (click-through). Role: alternative download route for the primary cohort only. Never an external set. The BraTS-Lighthouse 2025 pre-treatment half was re-preprocessed and "is no longer identical" — also not external.

### 3.11 Erasmus Glioma Database (EGD) — evaluated; **not selected**

XNAT Health-RI https://xnat.health-ri.nl/data/projects/egd ; van der Voort et al., Data in Brief 2021. 774 pre-operative adult diffuse gliomas (Erasmus MC 2008–2018), T1/T1GD/T2/FLAIR in MNI 1 mm; **whole-tumor mask only** (374 manual, 400 automatic from a CNN trained partly on BraTS 2019); IDH, 1p/19q, grade; **no MGMT** released; custom CC BY-NC-SA-derived license forbidding redistribution; access by e-mail request + signed DUA (turnaround undocumented). Independent of BraTS. Reason not selected: WT-only labels (no TC/ET), controlled access; could be a WT-only tertiary external check if access is granted quickly.

### 3.12 CFB-GBM v2.0 (Centre François Baclesse, France) — evaluated; **not selected**

TCIA *CFB-GBM* (DOI 10.7937/v9pn-2f72, v4 2026-09-11, CC BY 4.0, open). 264 newly diagnosed GBM (Stupp protocol), up to 3 timepoints (t0 pre-chemoradiation/post-surgery, t1 ≈ 4 months, t2 ≈ 6 months; 120 patients with all three), relative timing in weeks, RT dose/CT included. **Single binary GTV mask** (enhancing + necrosis merged; no edema), t1/t2 GTVs generated by a BraTS-2021-pretrained nnU-Net (label circularity), native T1 rare, T2 sparse, 208 GB. Independent of BraTS. Reason not selected: label scheme incompatible with WT/TC/ET and size; documented as a future longitudinal option (TC-like only).

### 3.13 UCSF-ALPTDG (UCSF Adult Longitudinal Post-Treatment Diffuse Glioma) — evaluated; **not selected**

https://imagingdatasets.ucsf.edu/dataset/2 (DOI 10.58078/C21592). 298 patients × exactly 2 consecutive post-treatment timepoints (596 scans), co-registered pairs, four-class labels (ET, SNFH, NETC/NCR, RC), relative days (surgery→scan 1, scan 1→scan 2), MGMT for a subset. Custom UCSF **non-commercial DUA** (account + click-through; terms retrieved from the sibling UCSF-BMSR DUA: research only, no transfer outside the user organization). Near-complete overlap with BraTS 2024 post-treatment (UCSF-LPTDG); 62 patients overlap UCSF-PDGM. Reason not selected: access friction and overlap; the same use case is covered by MU-Glioma-Post.

### 3.14 TCGA-GBM / TCGA-LGG, BraTS-TCGA-GBM/LGG, CPTAC-GBM, IvyGAP — evaluated; **not selected**

TCIA. Raw DICOM (longitudinal for many TCGA/IvyGAP patients; shifted dates preserving intervals) is now **NIH controlled access** (dbGaP phs004225; TCIA stopped hosting July 2025). The open NIfTI analysis results (BraTS-TCGA-GBM 102 training subjects, BraTS-TCGA-LGG 65; CC BY 3.0) are pre-operative and are **inside BraTS 2021**. MGMT exists via cBioPortal (gbm_tcga_pub2013: 350 samples; lgggbm_tcga_pub: 932). Reason not selected: leakage + controlled access.

### 3.15 BraTS 2018 / 2019 / 2020 and MSD Task01_BrainTumour — evaluated; **excluded (leakage)**

BraTS 2018/2019/2020 training sets (285/335/369) are nested subsets of BraTS 2021 training (CBICA IPP registration; citation-based terms). MSD Task01 (484 labelled + 266 unlabelled; CC BY-SA 4.0; open) is a renamed copy of BraTS 2016/2017 cases with scrambled IDs — unmappable, largely inside BraTS 2021. None can be an external test set.

### 3.16 Burdenko-GBM-Progression, GLIS-RT, QIN GBM Treatment Response — evaluated; **excluded (access + labels)**

TCIA; all three now under the NIH Controlled Data Access Policy (dbGaP phs004225) because images contain faces; RT-structure contours (GTV/CTV/PTV) or no masks; Burdenko has 180 patients with planning MRI + 1–8 follow-ups and MGMT for ≈ 92, GLIS-RT 230 single-timepoint, QIN 54 pre/post. Not usable within the project timeline.

### 3.17 ReMIND (Brigham and Women's) — evaluated; **not selected**

TCIA, CC BY 4.0, open. 114 surgical patients (88 adult diffuse gliomas), pre-operative + intra-operative MRI and 3D ultrasound; manual **whole-tumor** masks only; MGMT for 83/88 (47 methylated / 29 unmethylated / 7 partial); placeholder dates. Independent of BraTS. Reason: WT-only labels and intra-operative focus.

### 3.18 Identified in the sweep but not evaluated in depth (future options)

| Dataset | Source | Note |
| --- | --- | --- |
| IMAGO — Amsterdam IMAging and Clinical GliOma dataset (2026) | https://www.nature.com/articles/s41597-026-07424-8 | First release 500 adult diffuse glioma patients with pre-operative 4-modality MRI and post-operative MRI (≤ 14 days) for 381; MNI space; nnU-Net-derived segmentations; access mechanism not verified (abstract only). Potential future external/longitudinal (pre→post) cohort — evaluate access in M1 if time permits |
| MOTUM — multi-center multi-origin brain tumor MRI (2024) | https://doi.gin.g-node.org/10.12751/g-node.tvzqc5/ | 67 patients (29 high-grade gliomas + 38 metastases), 3 Chinese hospitals, 2D 5-mm acquisitions, CC BY 4.0; independent of BraTS; small and low-resolution — optional robustness check only |
| BraTS-Reg (2022) | https://bratsreg.github.io/ | 250 pre-op/follow-up pairs with expert landmarks from BraTS-affiliated institutions and TCIA collections → overlaps BraTS 2021; useful only as a registration-method benchmark (not a test set) |
| BraTS 2025 Task 11 BraTPRO | https://www.synapse.org/Synapse:syn64153130/wiki/631459 | RANO-class prediction; training data = LUMIERE; hidden 306-patient test — informs the trend-vs-RANO comparison, no new public data |
| One further sweep candidate | — | Its research card did not complete during the research run; the sweep had already summarized it among IMAGO / MOTUM / BraTS-Reg above, and it is not needed for the strategy |
| CGGA MRI_268 | — | ≈ 191 joinable MGMT labels, no FLAIR, no masks, license undocumented — not usable |
| BraTS 2026 | https://www.synapse.org/brats2026 | No adult glioma pre/post-treatment or SSA task in 2026 (METS, PEDs, GoAT, Inpainting, Path only) — no new glioma data expected from the 2026 cycle |

---

## 4. Dataset comparison

| Dataset | Subjects (timepoints) | Modalities | Labels (scheme) | Manual/refined GT | MGMT | Longitudinal | Space | License | Access | BraTS-2021 overlap |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BraTS 2021 T1 (TCIA) | 1,251 labelled (+219) | 4 | 1/2/4 | Yes | via crosswalk 695 | No | SRI24 | CC BY 4.0 | Open | — (is the training set) |
| BraTS 2024 PT (Synapse) | ≈ 731 (≈ 1,621) | 4 | 1 NETC/2 SNFH/3 ET/4 RC | Yes | No | Ordinal | MNI-lin | CC BY-NC | Registration | 38 subjects shared-ID |
| UTSW-Glioma | 625 (625) | 4 | FeTS 1/2/4 [verify] | 362 refined | 281 | No | SRI24 | CC BY 4.0 | Open | None documented |
| BraTS-Africa (TCIA) | 146 (146) | 4 | 1/2/3 | Yes (semi-auto refined) | No | No | SRI24 | CC BY 4.0 | Open | None |
| RHUH-GBM | 40 (120) | 4 + ADC | 1/2/3 or 1/2/4 [verify] | Yes, all timepoints | **No** | 3 TPs, intervals | SRI24, z-scored | CC BY 4.0 (NIfTI) | Open | None |
| MU-Glioma-Post | 203 (≈ 596) | 4 | 1/2/3/4 (2024) | Refined | ≈ 163 | 1–6 TPs, days | SRI24 | CC BY 4.0 | Open | None (overlaps BraTS 2024) |
| ds007045 | 363 (363) | 4 | 1/2/3 | Semi-auto reviewed | 363 | No | MNI152 | CC0 | Open | None (inferred) |
| UCSF-PDGM | 495 (501) | 4 + adv. | 1/2/4 [verify] | Yes | 416 | 6 pairs | Patient FLAIR space, 1 mm (not SRI24) | CC BY 4.0 | Open | 298 studies |
| UPenn-GBM | 630 (671) | 4 + adv. | 1/2/4 | 232 refined | 317 | 60 FU | SRI24 | CC BY 4.0 | Open | 447 subjects |
| LUMIERE | 91 (638) | 4 | auto (2 tools) | No | 80 | 2–21 TPs, weeks | MNI/native | CC0 meta / non-commercial stated | Open | Possible (Bern 2013) |
| UCSD-PTGBM | 178 (243) | 4 + DWI | 2024 scheme [verify] | Refined | 149 TPs | Subset | Not documented [verify] | CC BY 4.0 | Open | None (overlaps BraTS 2024/2025) |
| EGD | 774 (774) | 4 | WT only | 374 manual | No | No | MNI | Custom NC-SA | Request + DUA | None |
| CFB-GBM | 264 (554) | T1Gd/FLAIR (+) | GTV only | Partial | No | 3 TPs, weeks | t0-registered | CC BY 4.0 | Open | None (inferred) |
| UCSF-ALPTDG | 298 (596) | 4 | 2024 scheme | Yes | Subset | 2 TPs, days | SRI24 | UCSF DUA (NC) | Account + DUA | Via BraTS 2024/PDGM |
| TCGA/CPTAC/IvyGAP (raw) | 262/199/…/39 | DICOM | RT/none | — | cBioPortal | Yes | native | Controlled | dbGaP | Inside BraTS |
| Burdenko / GLIS-RT / QIN | 180/230/54 | DICOM | RTSTRUCT | — | Burdenko ≈ 92 | Burdenko yes | native | Controlled | dbGaP | None |
| ReMIND | 114 (228) | 4 + iMRI/iUS | WT manual | Yes (WT) | 83 | Intra-op only | native | CC BY 4.0 | Open | None |

## 5. Suitability matrix

| Dataset | A. Segmentation training | B. External generalization | C. Longitudinal | D. Radiomics | E. MGMT |
| --- | --- | --- | --- | --- | --- |
| BraTS 2021 | ★★★ primary | — | — | ★★ | ★★ (crosswalk labels) |
| BraTS 2024 PT | ★★ supplementary (ablation) | ★ post-treatment (non-independent of MU/UCSD/ALPTDG) | ★★ ordinal only | ★ | — |
| UTSW-Glioma | ★ (reserved) | ★★★ (refined subset) | — | ★★ | ★★★ external test |
| BraTS-Africa | ★ (adaptation only) | ★★★ domain shift | — | ★ | — |
| RHUH-GBM | — | ★★ (WT/ET; TC caveat) | ★★★ (expert GT, intervals) | ★ | — |
| MU-Glioma-Post | ★ (post-treatment fine-tune, optional) | ★★ post-treatment | ★★★ | ★★ | ★★ (post-treatment) |
| ds007045 | ★ | ★★ (label caveat) | — | ★★★ (radiomics-ready tier) | ★★★ development |
| UCSF-PDGM | — (overlap) | — | — | ★★ | ★★ (tag BraTS-mapped) |
| UPenn-GBM | — (overlap) | — | — | ★ | ★ (overlap) |
| LUMIERE | — | — | ★★ (RANO only) | ★ | ★ |
| UCSD-PTGBM | — | ★ alternate | ★★ alternate | ★ | ★ |
| EGD | — | ★ WT only (if access) | — | ★ | — |
| CFB-GBM | — | — | ★ TC-like | — | — |
| Others (ALPTDG, TCGA raw, Burdenko, GLIS-RT, QIN, ReMIND, MSD, BraTS 2018–2020) | — | — | — | — | — |

## 6. Access and licensing summary

| Dataset | Registration | Agreement | License | Redistribution | Demo display permitted |
| --- | --- | --- | --- | --- | --- |
| BraTS 2021 (TCIA) | No (Aspera plugin) | TCIA usage policy + citation | CC BY 4.0 | No | Yes, with attribution |
| BraTS 2021 via Synapse 2023 GLI | Synapse account | Click-through | CC BY-NC 4.0 | No | Yes (non-commercial), attribution |
| BraTS 2024 PT | Synapse account (+ 2025 form) | Click-through / challenge rules | CC BY-NC 4.0 | No | Yes (non-commercial), attribution |
| UTSW-Glioma | No | TCIA policy + citation | CC BY 4.0 | No | Yes |
| BraTS-Africa (TCIA) | No | TCIA policy + citation | CC BY 4.0 | No | Yes |
| RHUH-GBM NIfTI | No | TCIA policy + citation | CC BY 4.0 | No | Yes |
| MU-Glioma-Post | No | TCIA policy + citation | CC BY 4.0 | No | Yes |
| ds007045 | No | Citation requested | CC0 | Allowed | Yes |
| UCSF-PDGM | No | TCIA policy + citation | CC BY 4.0 | No | Yes |
| LUMIERE | No | Authors' non-commercial statement | CC0 (metadata) / NC (authors) | No | Yes (non-commercial) |
| UCSD-PTGBM | No | TCIA policy + citation | CC BY 4.0 | No | Yes |
| EGD | E-mail + account | Signed DUA | Custom NC-SA, no redistribution | No | Restricted |
| UCSF-ALPTDG | Account | Click-through DUA | UCSF NC | No | Restricted |
| Controlled TCIA sets | dbGaP DAR | NIH policy | — | No | No |

Project rule: all selected cohorts are usable for non-commercial academic research; each member accepts terms individually; no dataset files enter the repository; demo cases are drawn from CC BY 4.0 / CC0 cohorts.

## 7. Label harmonization rules (canonical: 1 NCR, 2 ED/SNFH, 3 ET, 4 RC; WT = {1,2,3}, TC = {1,3}, ET = {3})

| Source scheme | Datasets | Remap | Derivable regions | Notes |
| --- | --- | --- | --- | --- |
| BraTS 2017–2021: 1 NCR/NET, 2 ED, 4 ET | BraTS 2021 (TCIA), UPenn-GBM, UCSF-PDGM [verify], UTSW [verify], BraTS 2018–2020 | 4 → 3 | WT/TC/ET | Reference |
| BraTS 2023: 1 NCR, 2 ED, 3 ET | Synapse BraTS 2023 GLI, BraTS-Africa, ds007045 | identity | WT/TC/ET | — |
| BraTS 2024 post-treatment: 1 NETC, 2 SNFH, 3 ET, 4 RC | BraTS 2024 PT, MU-Glioma-Post, UCSD-PTGBM [verify], UCSF-ALPTDG | identity | WT/TC/ET (RC excluded by default) | SNFH includes treatment-related change → ED semantics differ; report explicitly |
| RHUH: necrosis / peritumoral (edema + non-enhancing tumor) / enhancing (1/2/3 documented, 1/2/4 in code) | RHUH-GBM | 1→1, 2→2, (3\|4)→3 | WT, ET reliable; TC under-defined | No cavity label |
| LUMIERE DeepBraTumIA: 1 necrosis, 2 CE, 3 edema | LUMIERE | 1→1, 2→3, 3→2 | WT/TC/ET (automatic) | HD-GLIO: 1 non-enh→2, 2 CE→3 (no NCR) |
| Binary WT | EGD, ReMIND | 1→WT | WT only | — |
| GTV (ET + NCR) | CFB-GBM | 1→TC-like | TC only | — |
| MSD Task01: 0/1/2/3 (edema, non-enhancing, enhancing) | MSD | not used | — | leakage |

BraTS evaluation note: from BraTS 2023 the official metrics are lesion-wise Dice/HD95 in addition to voxel-wise; the project reports voxel-wise Dice/IoU/Sensitivity/Precision/HD95 per region (proposal) and may add lesion-wise Dice as a stretch item.

## 8. Overlap and leakage analysis

### 8.1 Overlap matrix (✔ = documented overlap, ? = possible/unmappable, ✘ = none documented/expected)

| | BraTS 2021 | BraTS 2024 PT | UCSF-PDGM | UPenn-GBM | TCGA/CPTAC/IvyGAP | MSD/BraTS18–20 | MU-Glioma-Post | UCSD-PTGBM | UCSF-ALPTDG | LUMIERE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BraTS 2021 | — | ✔ 38 subj (pre→post) | ✔ 298 | ✔ 447 | ✔ 243+39+34 | ✔ subsets | ✘ | ✘ | ? (via PDGM ≤ 62) | ? (Bern 2013) |
| BraTS 2024 PT | ✔ | — | ? (UCSF pre-op) | ✘ | ✘ | ✘ | ✔ (≈ 400 cases likely) | ✔ (≈ 350) | ✔ (≈ 600) | ✘ |
| UTSW-Glioma | ✘ | ✘ | ✘ | ✘ | ✘ | ✘ | ✘ | ✘ | ✘ | ✘ |
| BraTS-Africa | ✘ | ✘ | ✘ | ✘ | ✘ | ✘ | ✘ | ✘ | ✘ | ✘ |
| RHUH-GBM | ✘ | ✘ | ✘ | ✘ | ✘ | ✘ | ✘ | ✘ | ✘ | ✘ |
| ds007045 | ✘ (inferred) | ✘ | ✘ | ✘ | ✘ | ✘ | ✘ | ✘ | ✘ | ✘ |

### 8.2 Leakage rules adopted

1. **Patient-level roles** — all timepoints of a canonical subject share its roles (`DEV_TRAIN`, `DEV_VAL`, `DEV_TEST`, `SUPP`, `EXT_TEST`, `LONG`, `MOL`, `DEMO`). Several evaluation-only roles may coexist (e.g. `EXT_TEST` + `LONG` for MU-Glioma-Post), but a training or tuning role never coexists with a test role, and `DEV_TEST` subjects are excluded from MGMT development. Enforced by a CI test.
2. **Crosswalk exclusion** — BraTS 2021 `BraTS2021_MappingToTCIA.xlsx` marks every BraTS case's TCIA origin; any external/molecular cohort row whose ID is in the crosswalk is tagged `overlap_flag = brats2021` and excluded from external evaluation (UCSF-PDGM 298, UPenn-GBM 447, TCGA 243, CPTAC 39, IvyGAP 34).
3. **BraTS 2024 shared IDs** — subject IDs ≤ 01790 are treated as follow-ups of BraTS 2021 subjects; used only as deliberate pre→post pairs, never as independent post-treatment test cases.
4. **Training contamination rule** — any experiment that trains on BraTS 2024 PT (supplementary ablation) or fine-tunes on MU-Glioma-Post may not be evaluated on MU-Glioma-Post, UCSD-PTGBM or UCSF-ALPTDG; the main model trains on BraTS 2021 only.
5. **Image-hash duplicate scan** across all cohorts (FLAIR + T1ce perceptual hashes after resampling) to catch unmappable duplicates (MSD, LUMIERE/Bern, "new" BraTS institutions).
6. **Label-provenance reporting** — CNN-derived masks (UTSW unrefined, ds007045 uncorrected ≈ 71 %, LUMIERE, EGD auto, CFB t1/t2) are never used as headline ground truth; expert-refined subsets are reported separately.
7. **Pretrained-weight rule** — no weights pretrained on BraTS 2023/2024/2025 or FeTS federated models are used, so BraTS-Africa, MU-Glioma-Post and UCSD-PTGBM remain valid tests.
8. **Single-evaluation events** for every lockbox cohort under the frozen protocol (Milestone 5).

## 9. Recommended dataset strategy

| Cohort role | Dataset(s) | Milestones | Data role | Why | Combination risks | Harmonization required |
| --- | --- | --- | --- | --- | --- | --- |
| **Primary Development Cohort** | BraTS 2021 Task 1 (TCIA CC BY 4.0; Synapse 2023 GLI as identical alternative) — patient-level ≈ 70/15/15 stratified by crosswalk source, ET presence, WT-volume tertile | M1–M5 | Training (`DEV_TRAIN`) + validation (`DEV_VAL`) | Largest expert-labelled 4-modality glioma set; standard space; proposal's primary dataset | Hidden test cases of the same institutions exist in other collections (UCSF/UPenn) → those collections cannot be external | 4 → 3 ET remap; z-score; crop |
| **Held-Out Internal Test** | BraTS 2021 `DEV_TEST` (≈ 185 cases) in the lockbox | M5 | Internal test, once | In-distribution reference | Contamination if opened early → protocol-hash guard | Same as above |
| **Supplementary / Post-Treatment Cohort** | BraTS 2024 Adult Glioma Post-Treatment (Synapse, CC BY-NC) | M1 harmonization; M2 post-treatment error analysis + training ablation; M3 ordinal pairs; M5 post-treatment evaluation + ablation | Secondary analysis; supplementary training only in a labelled ablation | Proposal dataset; RC labels; post-treatment abnormalities; large same-subject ordinal pairs | Using it for training invalidates MU-Glioma-Post/UCSD/ALPTDG as external; SNFH vs ED semantics; shared-ID subjects | 2024 scheme → canonical (identity), RC policy; MNI-lin vs SRI24 geometry check |
| **External Generalization A** | UTSW-Glioma, 362 refined masks (TCIA CC BY 4.0) | M2 preprocess; M5 evaluate | External test + MGMT external test | Independent US institution, 2026, same space, MGMT | None known; verify label integers | 4 → 3 remap [verify]; choose FeTS skull-strip variant |
| **External Generalization B** | BraTS-Africa (TCIA CC BY 4.0), 95 glioma (+ 51 other CNS reported separately) | M2 preprocess; M5 evaluate | External test (domain shift) | Different continent, 1.5 T, low-resource protocols | Presumed (not pathology-confirmed) gliomas; small | Identity labels; expect lower SNR |
| **External Generalization C** | RHUH-GBM pre-operative timepoint (40) | M2 preprocess; M5 evaluate | External test (WT/ET; TC caveat) | Independent Spanish centre; expert masks | Pre-normalized intensities; TC definition | Label integer check; skip z-score |
| **External Generalization D (secondary)** | OpenNeuro ds007045 (363) | M2 preprocess; M5 evaluate (secondary table) | External test with label caveat | 8-site European independence | CNN-derived masks; MNI152 space | Crop/pad; identity labels |
| **Post-treatment External / Longitudinal A** | MU-Glioma-Post (203 / ≈ 596) | M1 feasibility; M2 preprocess; M3 development pairs; M5 reserved pairs + post-treatment test | Longitudinal (interval-aware) + post-treatment external | Refined masks at every timepoint; days from diagnosis; CC BY 4.0 | Overlaps BraTS 2024 → only for BraTS-2021-only models; model-influenced labels | Identity labels; RC policy |
| **Longitudinal B** | RHUH-GBM (40 × 3) | M3 development (25 subjects) + M5 validation (15 reserved) | Longitudinal (pre→post→recurrence, intervals) | Expert masks; verification logic can use dates | Small n | As above |
| **Longitudinal C (scale, ordinal)** | BraTS 2024 PT same-subject pairs | M3, M4 | Ordering-only path; change-map QA | Large N | No intervals | — |
| **Longitudinal D (optional)** | LUMIERE (91 / 638) | M3/M5 optional | Trend vs RANO comparison | Only public RANO ratings | Automatic masks; non-commercial statement; possible Bern/BraTS 2013 overlap | DeepBraTumIA remap |
| **Molecular / MGMT (optional)** | Development: ds007045 (363) + UCSF-PDGM (416, tag BraTS-mapped) + BraTS 2021 Task 2/crosswalk (695, de-duplicated); external test: UTSW (281) | M1 feasibility; M3 module; M5 final | Optional secondary prediction; exploratory | Labels exist; independent external test possible | Weak signal (literature); assay heterogeneity; UCSF/UPenn/BraTS overlap in development set (dedupe by crosswalk) | Radiomics on canonical masks; site as covariate |
| **Demo** | 2–3 cases from `DEV_VAL`, MU-Glioma-Post or RHUH-GBM (never lockboxes) | M4–M5 | Demonstration | CC BY 4.0 display permitted | — | — |

## 10. Longitudinal dataset strategy

- **Verification logic inputs available per cohort:** MU-Glioma-Post — subject ID + timepoint index + days from diagnosis; RHUH-GBM — subject ID + timepoint index + shifted dates/CSV intervals; BraTS 2024 — subject ID + ordinal suffix only (interval unknown → *flagged* path); LUMIERE — patient + week index (+ intra-week order).
- **Registration:** all selected cohorts are already skull-stripped and atlas-registered per timepoint (SRI24 / MNI); intra-subject rigid (optionally affine) registration of follow-up to baseline is still performed and QC'd (NCC/MI, brain-mask Dice) because timepoints were registered independently.
- **Development vs validation pairs:** development on ≈ 70 % of MU-Glioma-Post subjects and 25 RHUH subjects; ≈ 30 % of MU-Glioma-Post subjects and 15 RHUH subjects are reserved in the lockbox for M5 (chosen by seed at M1).
- **Ground-truth-based validation:** predicted-mask change vs GT-mask change (absolute/percent per region), trend agreement (decreased/stable/increased with documented thresholds), rate-of-change error where intervals exist (MU-Glioma-Post, RHUH).
- **Expected yield:** ≥ 100 interval-aware GT pairs (MU-Glioma-Post consecutive pairs ≈ 390 timepoint transitions across 203 patients; RHUH 80 transitions) — comfortably above the M1 go threshold; synthetic-pair fallback unlikely to be needed.
- **Post-treatment semantics:** the module measures imaging change of canonical regions (RC excluded, SNFH treated as ED with an explicit caveat); it does not output RANO or clinical response classes. LUMIERE RANO ratings, if used, serve only for a descriptive agreement analysis.
- **Alternates:** UCSD-PTGBM (open, CC BY 4.0) if MU-Glioma-Post proves problematic; IMAGO (2026) for pre→post pairs if access is confirmed.

## 11. MGMT feasibility

- **Labels:** available in BraTS 2021 (695 via crosswalk; 594 training cases with masks; heterogeneous assays), UCSF-PDGM (416; 298 overlap BraTS), UPenn-GBM (317; 447 overlap BraTS), ds007045 (363; complete; 8 sites; MSP), UTSW (281; qMSP), MU-Glioma-Post (≈ 163; post-treatment), UCSD-PTGBM (149 timepoints), LUMIERE (80), ReMIND (83), Burdenko (≈ 92, controlled). **Not** in RHUH-GBM, BraTS-Africa, BraTS 2024 PT, EGD, CFB-GBM.
- **Evidence on predictability:** BraTS 2021 Task 2 winner AUC ≈ 0.62 (within the top 0.13 % of random submissions; large public/private shake-up); Saeed et al. 2023 (Med Image Anal) — AUC saturates ≈ 0.62, no tumour-linked signal; Kim et al. 2022 (Cancers) — 60 % of 420 models at chance, Kaggle winner on external SNUH cohort AUROC 0.56; Robinet et al. 2023 — "unable to determine MGMT from MRI"; radiomics meta-analysis (Doniselli et al. 2024, Eur Radiol) — pooled AUC 0.78 but I² = 94 %, externally validated grade-IV studies AUC 0.65 (0.57–0.73); Calabrese et al. 2022 — MGMT only "fair" internally.
- **Decision:** MGMT remains **optional/stretch**. If attempted: pre-register H0 (AUC = 0.5); develop on ds007045 (+ UCSF-PDGM/BraTS Task 2 de-duplicated by crosswalk, with site as a covariate) using patient-level stratified CV × ≥ 3 seeds; compare PyRadiomics + Random Forest / XGBoost vs deep features from the segmentation model vs hybrid; report bootstrap 95 % CIs, label-permutation null distribution, calibration; single external evaluation on UTSW; treat any AUC > 0.70 as a leakage alarm; present negative results honestly. IDH status (external AUC 0.92–0.98 in recent literature) is the feasible radiogenomic showcase for future work.

## 12. Storage and download plan

| Cohort | Compressed download | Notes |
| --- | --- | --- |
| BraTS 2021 Task 1 | ≈ 12–13 GB (TCIA both-tasks package 142 GB incl. Task 2 DICOM) | Prefer Task-1-only download; Synapse 2023 GLI ≈ 13.4 GB identical |
| BraTS 2024 PT | ≈ 43 GB (+ 5.4 GB validation) | Synapse zips |
| UTSW-Glioma | 22.9 GB | keep one skull-strip variant (≈ 11 GB) |
| BraTS-Africa | 1.6 GB | processed package |
| RHUH-GBM | 2.9 GB | NIfTI package |
| MU-Glioma-Post | ≈ 11 GB | + XLSX |
| ds007045 | ≈ 19 GB | raw + derivatives (use z-scored derivative tier for radiomics) |
| UCSF-PDGM (optional) | 142 GB (clinical CSV alone < 1 MB) | download images only if the MGMT module proceeds |
| LUMIERE (optional) | 32.6 GB | non-commercial |
| **Core total** | **≈ 115 GB** raw | processed copies ≈ 1.5× → plan ≥ 300 GB local + Azure Blob DVC remote |

## 13. Final recommendation

1. **Primary development cohort:** BraTS 2021 Task 1 from TCIA (CC BY 4.0), patient-level split with a lockbox; crosswalk-aware stratification.
2. **Supplementary/post-treatment cohort:** BraTS 2024 Adult Glioma Post-Treatment from Synapse — harmonized to the canonical label scheme; used for post-treatment analysis, ordinal longitudinal pairs and a supplementary-training ablation only.
3. **External generalization:** UTSW-Glioma (refined subset), BraTS-Africa, RHUH-GBM (pre-op), ds007045 (secondary), MU-Glioma-Post (post-treatment) — all open, independent of BraTS 2021, evaluated once each under the frozen protocol and reported separately.
4. **Longitudinal:** MU-Glioma-Post (primary, interval-aware, GT at every timepoint), RHUH-GBM (pre→post→recurrence, expert GT), BraTS 2024 pairs (ordinal path), LUMIERE optional for RANO comparison.
5. **MGMT:** optional, pre-registered negative-result-tolerant design on ds007045/UCSF-PDGM/BraTS Task 2 with UTSW as external test.
6. **Not used:** UPenn-GBM, UCSF-PDGM (as segmentation test), TCGA/CPTAC/IvyGAP raw, BraTS 2018–2020, MSD, BraTS 2023 GLI (identical), EGD, CFB-GBM, UCSF-ALPTDG, Burdenko/GLIS-RT/QIN, ReMIND.

## 14. Items to verify on download (assigned to M1-T01 / M1-T08)

- Integer label values in UTSW-Glioma, RHUH-GBM, UCSF-PDGM, UCSD-PTGBM masks (`np.unique`).
- BraTS 2021 TCIA package: 1,480 vs 1,470 subjects; whether Task 1 can be downloaded without Task 2 DICOM.
- BraTS 2024 PT: final training case/subject counts after download; demographics xlsx contents.
- MU-Glioma-Post: whether clinical dates are absolute or day offsets; exact timepoint total (594/596/617).
- ds007045: use v2.0.1; confirm empty N4-only derivative folder; site codes.
- LUMIERE: license interpretation (non-commercial) recorded in the governance register if used.

## 15. Sources consulted (official)

TCIA: RSNA-ASNR-MICCAI-BraTS-2021 analysis result (DOI 10.7937/jc8x-9874); UTSW-Glioma (10.7937/dfae-1b86); BraTS-Africa; RHUH-GBM (10.7937/4545-c905); MU-Glioma-Post (10.7937/7K9K-3C83); UCSD-PTGBM (10.7937/fwv2-dt74); UCSF-PDGM; UPENN-GBM; CFB-GBM (10.7937/v9pn-2f72); ReMIND (10.7937/3RAG-D070); TCGA-GBM/LGG; BraTS-TCGA-GBM/LGG; CPTAC-GBM; IvyGAP; Burdenko-GBM-Progression; GLIS-RT; QIN GBM Treatment Response; TCIA Data Usage Policies (NIH Controlled Data Access, dbGaP phs004225). Synapse (REST wiki): BraTS 2021 (syn25829067), BraTS 2023 GLI (syn51156910 / wikis 622351, 627000), BraTS 2024 GLI post-treatment (syn53708249 / 627500, 627759), BraTS-Lighthouse 2025 (syn64153130 / 631053, 631048, 631064, 631251, 631459), BraTS 2026 (syn74274097). OpenNeuro ds007045 (GraphQL metadata, participants.tsv, masks). Springer Nature figshare LUMIERE collection 5904905 (API). Health-RI XNAT EGD project. UCSF ci2 imagingdatasets.ucsf.edu/dataset/2. Papers: Baid et al. arXiv:2107.02314; de Verdier et al. arXiv:2405.18368; Adewole et al. arXiv:2305.19369; Reddy et al. Sci Data 2026; Filimonova et al. Sci Data 2026; Gagnon et al. Sci Data 2026; Suter et al. Sci Data 2022; Cepeda et al. Data in Brief 2023; Calabrese et al. Radiology: AI 2022; Bakas et al. Sci Data 2017 and 2022; van der Voort et al. Data in Brief 2021; Juvekar et al. Sci Data 2024; MSD (Antonelli et al. 2022); MGMT literature: Saeed et al. Med Image Anal 2023; Kim et al. Cancers 2022; Robinet et al. Cancers 2023; Emchinov 2022; Faghani et al. J Digit Imaging 2023; Doniselli et al. Eur Radiol 2024; Calabrese et al. Neuro-Oncol Adv 2022; Byeon et al. npj Digit Med 2025.
