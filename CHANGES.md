# Changelog

## 4D (3D + time) support & vessel-wall losses

This repository extends nnU-Net v2 with an end-to-end **4D pipeline** (three spatial
dimensions + one time dimension)
> **Note:** This fork depends on 4D-enabled forks of `acvl_utils`,
> `batchgeneratorsv2` and `dynamic_network_architectures` (see `requirements.txt`).

### Added

**4D experiment planning**
- New `ExperimentPlanner4D` (`experiment_planning/experiment_planners/experiement_planner_4D.py`)
  producing a `4d_fullres` configuration.
  - Time is always the first axis; `suppress_transpose=True` is forced.
  - `determine_transpose()` extended to 4 axes.
  - Spatial plan is derived from the 3 spatial axes, then the time axis is prepended.

**4D preprocessing**
- New `TimeResampledPreprocessor`
  (`preprocessing/preprocessors/time_resample_preprocessor.py`) which resamples the
  time axis to a fixed number of steps, with presets
  `TimeResampledPreprocessor_16` and `TimeResampledPreprocessor_32`.

**4D training**
- Data augmentation pipeline is now time-aware: intensity, low-res, gamma and
  one-hot morphology transforms are wrapped in `TransformEachTimestep` for 4D
  input (`TransformNothing` for ≤3D), so each timestep is augmented consistently.
- `configure_rotation_dummyDA_mirroring_and_initial_patch_size()` handles `dim == 4`
  (mirror axes `(1, 2, 3)`, i.e. spatial only — no mirroring in time).
- `get_patch_size()` supports 4-element patch sizes by rotating the spatial part only.
- New loss `Time_DC_and_CE_loss` (Dice + CE variant for 4D targets).

**4D inference**
- Sliding-window slicer generation extended to 4D volumes (4 nested step loops).
- `predict_sliding_window_return_logits` now accepts 4D and 5D input
  (`(c, t, x, y(, z))`).

**Other**
- New Aorta trainer variant `nnUNetTrainer_500epochs_lre3` (500 epochs, `lr = 1e-3`).
- Pinned `requirements.txt` (Torch 2.7.0, NumPy 2.2.5, blosc2 3.5.1, plus the 4D forks).
- Convenience entry-point scripts `run_preprocessing` and `run_training`.
- VS Code launch configuration for debugging the 4D planner.

### Changed

- **Default configurations now include `4d_fullres`**, in both
  `preprocess()` and the `plan_and_preprocess` CLI (`-c`), with a default of
  4 processes.
- **Image I/O**
  - `NibabelIO` now reads 4D/5D NIfTI files: images are squeezed and fully
    transposed, and spacing is assembled as `[time] + spatial[::-1]`. Raises if
    more than one extra (time) dimension is present. The strict `ndim == 3`
    assertion was removed.
  - `NibabelIO.write_seg` squeezes and uses a generic transpose instead of `(2, 1, 0)`.
  - `SimpleITKIO`: placeholder (commented out, untested) branch for 6D NIfTI with a
    time dimension.
- **Resampling** (`default_resampling.py`): `resample_data_or_seg` and
  `resample_data_or_seg_to_spacing` accept 4D and 5D data; `do_separate_z` is still
  restricted to 4D and now asserts this explicitly.
- **Cropping**: `create_nonzero_mask` accepts `(C, X, Y, Z, T)`.
- **Fingerprint extraction**: foreground-intensity collection no longer asserts 4D
  input (supports `(c, x, y, z, t)`)
- **Blosc2 dataset**: chunk/block shapes are cast to plain Python `int` tuples before
  `blosc2.asarray`, fixing `numpy.int64`-related failures.

### Breaking changes / migration notes

- `4d_fullres` is part of the default configuration list — datasets without it are
  skipped, but process-count tuples passed via `-np` may need an extra entry.
- `NibabelIO` no longer restricts input to 3D images, and the returned transpose /
  spacing order changed for non-3D inputs. Plans and preprocessed data generated
  with older versions may be incompatible.
