# Data and artifact dictionary

CSV files retain their original names so scripts and archived checkpoints remain traceable.

| Field | Meaning |
| --- | --- |
| `timestamp` | Recorded sample time; excluded from model features |
| `motor1_position` … `motor6_position` | Six motor position channels; source acquisition units retained |
| `motor1_temperature` … `motor6_temperature` | Six motor temperature channels; motor 6 is the prediction target (°C) |
| `motor1_voltage` … `motor6_voltage` | Six motor voltage channels; source acquisition units retained |
| `label` | Synthetic-fault label: 0 healthy, 1 fault; excluded from predictors |

`source_all_data_b1.csv`: 104,000-row nominal source sequence. `target_all_data_b1.csv`: 15,600-row nominal target sequence. Source partitions are 70/15/15%; target partitions are 80/10/10%. The provided split CSVs are used by archived evaluation scripts. NaN rows are dropped by the original loaders before windowing.

The filename encodes acquisition/fault-injection variants; `b1` is the supported archived detector track. This package includes the eleven CSV inputs required by its training/evaluation commands. Additional acquisition variants, raw monitoring logs and secondary robot-platform experiments are retained only in the curated research workspace.

Checkpoints: PyTorch state dictionaries. Scalers: joblib-serialized scikit-learn MinMaxScaler objects. Five target seed families contain target-only and layer-freezing variants at eight training ratios. Use package-supplied scalers with their matching source checkpoint.

The two SHA-256 manifests cover immutable source/reference files and artifacts. Generated outputs are excluded, so evaluation and figure rendering do not invalidate verification. Primary checkpoint inputs retain their names under `legacy/codes/data/`; descriptive cleaned CSVs and DoE designs are separately located in `data/` and supplied through the artifact bundle.
