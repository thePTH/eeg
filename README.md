# EEG Dataset Setup

This repository expects EEG data to be provided in **BIDS-like format** inside the `data/` directory.

Raw EEG recordings are **not included** in this repository and must be added manually before running the pipeline.

---

# Required dataset structure

Place your dataset inside:

```
data/
├── participants.tsv
├── sub-001/
│   └── eeg/
│       ├── sub-001_task-eyesclosed_eeg.set
│       ├── sub-001_task-eyesclosed_eeg.json
│       └── sub-001_task-eyesclosed_channels.tsv
├── sub-002/
│   └── eeg/
│       └── ...
└── sub-XXX/
```

Only the following elements are required:

* subject folders: `sub-XXX`
* EEG recordings inside `sub-XXX/eeg/`
* the file `participants.tsv`

Other BIDS metadata files (e.g. `dataset_description.json`) are **optional** for this project.

---

# ⚠️ Most important file: `participants.tsv`

The pipeline relies heavily on this file.

It must be located at:

```
data/participants.tsv
```

and must follow exactly this structure:

```
participant_id	Gender	Age	Group	MMSE
sub-001	F	57	A	16
sub-002	F	78	A	22
sub-003	M	70	A	14
sub-004	F	67	A	20
...
```

## Column description

| Column         | Description                                           |
| -------------- | ----------------------------------------------------- |
| participant_id | Subject identifier (must match folder name `sub-XXX`) |
| Gender         | `M` or `F`                                            |
| Age            | Age in years                                          |
| Group          | Subject group label                                   |
| MMSE           | Mini-Mental State Examination score                   |

⚠️ The `participant_id` values **must exactly match** subject folder names.

Example:

```
data/sub-001/
```

must correspond to:

```
participant_id = sub-001
```

---

# EEG file requirements

Each subject must contain:

```
sub-XXX/eeg/sub-XXX_task-eyesclosed_eeg.set
```

The pipeline currently expects:

* EEGLAB `.set` format
* task name: `eyesclosed`
* one recording per subject

Example:

```
data/sub-012/eeg/sub-012_task-eyesclosed_eeg.set
```

---

# Minimal checklist before running the pipeline

Make sure:

* `data/` exists at repository root
* `participants.tsv` is present
* subject folders follow `sub-XXX`
* each subject contains an EEG file
* filenames match `sub-XXX_task-eyesclosed_eeg.set`
