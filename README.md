# PyRATS

**Spectral ageing and energetics of extended radio sources, from a set of multi-frequency radio images to publication-ready maps, SEDs and physical properties.**

PyRATS takes cropped radio continuum images of one source at two or more frequencies, puts them on a common pixel grid and common restoring beam, and derives:

- the integrated radio SED, with a full thermal ⊕ zero-level ⊕ flux-scale error budget;
- the **injection index**, from a six-model spectral-ageing ensemble (JP, KP, JP-Tribble, KP-Tribble, CI, CI-off) with explicit identifiability checks and AICc/BIC model averaging;
- **equipartition and minimum-energy magnetic fields**, energy densities, total energy and minimum pressure, with Monte-Carlo uncertainties;
- resolved **spectral-index, curvature and error maps**, and optionally resolved **spectral-age, break-frequency, B_eq and pressure maps**.

It drives three established codes from one command line, on the same pixels and the same error model:

| engine | role |
|---|---|
| [synchrofit](https://github.com/synchrofit/synchrofit) | the six-model spectral-ageing ensemble |
| pysynch (M. Hardcastle) | equipartition / minimum energy, aged electron spectra |
| [BRATS](https://github.com/JeremyHarwood/BRATS) | resolved spectral ageing, as an independent cross-check |

The name comes from **Py**Synch + B**RATS**.

---

## Contents

- [Pipeline](#pipeline)
- [Installation](#installation)
- [Quick start](#quick-start)
- [Input requirements](#input-requirements)
- [Key options](#key-options)
- [Outputs](#outputs)
- [Defaults you should know about](#defaults-you-should-know-about)
- [Conventions](#conventions)
- [Reproducibility](#reproducibility)
- [Citing](#citing)
- [License](#license)

---

## Pipeline

```
input FITS maps
  -> robust noise + calibration-systematic characterisation
  -> astrometric reference selection
  -> WCS regridding onto one common pixel grid
  -> convolution to one common restoring beam
  -> hysteresis source segmentation
  -> integrated SED (non-detections kept, never clipped)
  -> 3-D geometry and source volume (several independent estimators)
  -> injection index from the model ensemble
  -> equipartition and minimum-energy fields, u_e, u_B, u_min, E, p
  -> self-consistent alpha_inj <-> B <-> aged-electron-spectrum iteration
  -> resolved spectral-index / curvature / error maps
  -> optional resolved ageing maps (PyRATS and/or BRATS)
  -> FITS + PNG + JSON + CSV + human-readable reports
```

Every stage reports what it assumed. When a result is degenerate or underconstrained — a model that is not identifiable with the number of bands available, a fitted parameter pinned at its prior, a fallback to a simpler electron spectrum — the reports say so next to the number rather than silently substituting.

---

## Installation

PyRATS is a single script, `PyRATS.py`. Its dependencies are installed separately.

### 1. Python environment

With conda (recommended, because pysynch needs GSL and a C compiler):

```bash
conda env create -f environment.yml
```

```bash
conda activate pyrats
```

This installs the core stack and synchrofit. Alternatively, with pip into an existing Python 3.11 environment:

```bash
pip install -r requirements.txt
```

### 2. pysynch (needed for equipartition, minimum energy and resolved ageing)

pysynch compiles a C extension against [GSL](https://www.gnu.org/software/gsl/). With the conda environment above active, clone pysynch and install it from the clone:

```bash
pip install /path/to/pysynch
```

Without pysynch, PyRATS still produces the regridded maps, SEDs and spectral-index maps, and says which products it skipped.

### 3. BRATS (optional; only for `--engine brats` or `--engine both`)

Build [BRATS](https://github.com/JeremyHarwood/BRATS) following its own documentation, and either put `brats` on your `PATH` or pass `--brats-path`. PyRATS can also run BRATS inside a container (`--brats-container docker|podman|apptainer|singularity`).

### Check the installation

```bash
python PyRATS.py --version
```

### Tested environment

PyRATS 2.1.0 was validated on macOS with:

| package | version |
|---|---|
| Python | 3.11.15 |
| numpy | 2.4.6 |
| scipy | 1.16.3 |
| astropy | 8.0.1 |
| matplotlib | 3.11.1 |
| radio-beam | 0.3.9 |
| reproject | 0.21.0 |
| synchrofit | 1.0.0 |
| pysynch | 0.2 (distribution name `synchrotron`) |
| BRATS | 2.6.3.3 |

Other versions will probably work but have not been tested.

---

## Quick start

Label each image with `LABEL=path`. The labels are used in every table, plot and file name.

```bash
python PyRATS.py \
  --image LOFAR_144MHz=lofar_144.fits \
  --image GMRT_650MHz=gmrt_650.fits \
  --image MeerKAT_1284MHz=meerkat_1284.fits \
  --image VLA_3GHz=vla_3000.fits \
  --reference-label MeerKAT_1284MHz \
  --redshift 0.15 \
  --output-dir my_source
```

A fuller run, with both ageing engines, resolved ageing maps, per-band flux-scale errors and BRATS maps:

```bash
python PyRATS.py \
  --image GMRT_650MHz=gmrt_650.fits \
  --image MeerKAT_1284MHz=meerkat_1284.fits \
  --image VLA_3GHz=vla_3000.fits \
  --cal-frac GMRT_650MHz=0.10 --cal-frac MeerKAT_1284MHz=0.05 --cal-frac VLA_3GHz=0.05 \
  --redshift 0.15 \
  --reference-label MeerKAT_1284MHz --contour-label MeerKAT_1284MHz \
  --engine both --resolved-ageing \
  --ensemble-models JP_Tribble CI \
  --brats-signaltonoise 10 \
  --output-dir my_source_full
```

Run with no `--image` to be prompted for the inputs interactively. `python PyRATS.py --help` lists every option.

---

## Input requirements

- **At least two** FITS images of the same source, ideally cropped to the source and its surroundings.
- `BUNIT = 'Jy/beam'`.
- A restoring beam: `BMAJ`, `BMIN`, `BPA`. This is mandatory; there is no default beam.
- A frequency: either a `FREQ` WCS axis, or one of `RESTFRQ`, `RESTFREQ`, `FREQ`, `OBSFREQ`, `CRVAL3`, `CRVAL4`. PyRATS uses the header value, never the file name — check the header of any image whose name suggests a round frequency.
- `--redshift` is **required**. There is deliberately no default: distances, luminosities, B_CMB and every spectral age depend on it.
- `--reference-label` is **required**: name the map whose absolute astrometry you trust; every other map is aligned to it. PyRATS will not guess, and deliberately does not use beam size as a proxy for astrometric quality. Alternatively, pass `--astrometry-rms LABEL=ARCSEC` for maps whose astrometry you have measured independently.

Noise levels and flux-scale uncertainties are measured automatically. Override them per band with `--rms LABEL=JYBEAM` and `--cal-frac LABEL=FRAC`; `--show-error-provenance` prints how every value was obtained.

---

## Key options

A selection; see `--help` for the rest.

**Physics**

| option | default | meaning |
|---|---|---|
| `--redshift` | *required* | source redshift |
| `--reference-label` | *required* | astrometric reference map, by label |
| `--kappa` | `0` | proton : electron energy ratio κ; ζ = 1 + κ |
| `--gamma-min`, `--gamma-max` | `10`, `1e7` | Lorentz-factor limits of the electron spectrum |
| `--alpha-inj` | fitted | fix the injection index instead of fitting it (signed, e.g. `-0.5`) |
| `--inject-range S_MIN S_MAX` | `2.01 3.00` | prior on the injection energy index s |
| `--b-field-fraction F` | `0.4` | report a budget at F × B_eq |
| `--analysis-at-sub-equipartition` | off | actually *use* F × B_eq for the ageing maps and BRATS |
| `--filling-factor` | `1.0` | volume filling factor |

**Models and engines**

| option | default | meaning |
|---|---|---|
| `--engine` | `synchrofit` | `synchrofit`, `brats`, or `both` (cross-compared) |
| `--ensemble-models` | all six | subset of `JP KP JP_Tribble KP_Tribble CI CI-OFF` |
| `--brats-models` | `JP KP JP_Tribble CI CI-OFF` | models BRATS fits |
| `--resolved-ageing` | off | adaptive binning + per-region ageing fits |
| `--brats-signaltonoise` | `1.0` | BRATS region S/N target — see [below](#defaults-you-should-know-about) |
| `--brats-max-myears` | `50` | upper edge of the BRATS age grid |

**Segmentation and contours**

| option | default | meaning |
|---|---|---|
| `--detection-map` | best peak S/N band | band used to seed the source mask |
| `--source-sigma`, `--source-grow-sigma` | `5`, `3` | hysteresis seed / growth thresholds |
| `--contour-label` | reference map | band whose contours are drawn on every map |
| `--contour-sigma` | `5` | anchor of the contour ladder, in σ |
| `--contour-factor` | `2` | ratio between successive levels |
| `--contour-drop-lowest` | `2` | levels omitted from the bottom of the ladder |
| `--contour-levels-sigma` | — | explicit ladder, e.g. `20 40 80 160 320 640` |

**Run control**: `--output-dir` (default `PyRATS_Products`), `--seed`, `--thorough` (larger Monte-Carlo counts and posterior grids), `--keep-stale`.

---

## Outputs

Everything is written to `--output-dir`.

| file | contents |
|---|---|
| `PyRATS_report.txt` | run summary, **including the exact command line** |
| `source_properties.txt` | every headline physical quantity with its provenance and caveats |
| `PyRATS_results.json` | every intermediate quantity, machine-readable |
| `integrated_SED.csv` | per-band flux densities and error components |
| `model_ranking.csv` | ensemble ranking, weights, injection indices, break frequencies |
| `radio_SED*.png`, `model_comparison.png`, `injection_index_scan.png` | SED and model figures |
| `luminosity_vs_rest_frequency.png`, `volume_emissivity_vs_rest_frequency.png`, `energy_budget.png` | energetics figures |
| `spectral_index*.fits/.png`, `spectral_curvature*.fits/.png` | resolved spectral index, curvature, statistical / systematic / total errors, χ², band count |
| `resolved_*.fits/.png`, `resolved_regions.csv` | resolved age, break frequency, B_eq, pressure (with `--resolved-ageing`) |
| `source_mask.fits`, `depth_map.fits`, `source_segmentation_QA.png` | segmentation and line-of-sight depth |
| `common_beam_FITS/`, `*_commonbeam.png` | the regridded, common-beam input maps |
| `brats_maps/`, `BRATS_summary.txt`, `brats_workspace/` | BRATS maps (painted onto the PyRATS grid), native BRATS outputs and PGPLOT figures, and the exact BRATS command files |

The BRATS command files in `brats_workspace/commands/` are plain text and can be pasted into an interactive BRATS session to check or continue a run by hand.

---

## Defaults you should know about

These choices affect the numbers you quote. Check them against your source before publishing.

- **κ = 0 (electron–positron) by default.** B_eq scales roughly as (1 + κ)^(2/7), so `--kappa 100` (the classical proton-dominated assumption) raises it by about 3.7×. κ is an assumption; the data cannot constrain it.
- **`--b-field-fraction` only reports.** By default the ageing maps and BRATS use the full B_eq; the F × B_eq block is a what-if budget. Add `--analysis-at-sub-equipartition` to run the analysis at F × B_eq. `source_properties.txt` states which field was used.
- **Ages are conditional on B.** Below B_CMB/√3 inverse-Compton losses dominate and a weaker field gives a *shorter* age. The reports flag when the field is on that side.
- **γ_max may be lowered automatically.** An aged electron spectrum underflows far above its break, so PyRATS reduces γ_max until pysynch can normalise it, and records the value used as `gamma_max_used`. If even that fails it falls back to an unbroken power law and marks the result `** FALLBACK **` in `source_properties.txt`. Treat fallback energetics as provisional.
- **Few bands limit what can be fitted.** AICc needs n > k + 1 data points, so with four bands and three free parameters PyRATS ranks models on BIC and says so. Models that are not identifiable are reported as such, not averaged in.
- **`--brats-signaltonoise` defaults to 1.** At that value BRATS builds single-pixel regions and its automatic search-area scaling does not engage; resolved BRATS ages can then be unusable. A value around 10 was used in validation.
- **Only the 20σ-and-above contours are drawn by default** (`--contour-drop-lowest 2`). The ladder is still anchored at 5σ, and the mask used to confine contours to the source still follows the 5σ isophote. Use `--contour-drop-lowest 0` to draw the 5σ and 10σ levels too.

---

## Conventions

```
S_nu  ∝ nu^alpha                 alpha is NEGATIVE for optically thin synchrotron
alpha_positive = -alpha          both are reported everywhere
N(E)  ∝ E^(-s)                   s is the injection ENERGY index
alpha_inj_positive = (s - 1)/2
s = 1 - 2 * alpha_signed
```

`--alpha-inj` takes the **signed** value: `--alpha-inj -0.5` means α_inj = 0.5 in the positive convention. BRATS is handed the positive convention internally.

Units: energy densities in J m⁻³, magnetic fields in µG in reports (Tesla internally, 1 µG = 10⁻¹⁰ T), volumes in m³, ages in Myr.

---

## Reproducibility

- `PyRATS_results.json` records the **exact command line**, working directory, Python and package versions, and every configuration value; `PyRATS_report.txt` prints the command line at the top. Re-running that command in the same environment reproduces the run.
- Monte-Carlo steps use a fixed default seed (`--seed`).
- Defaults can change between versions. When reproducing someone else's results, pass every option explicitly rather than relying on defaults.

---

## Citing

If you use PyRATS, please cite it using `CITATION.cff` (GitHub shows a "Cite this repository" button). Please **also cite the codes PyRATS drives**, since the physics is theirs:

- **BRATS** — Harwood et al. (2013, 2015)
- **synchrofit** — Turner et al. (2018)
- **pysynch** — Hardcastle et al. (1998)

Further references for the methods implemented:

| reference | used for |
|---|---|
| Burbidge (1959) | minimum energy |
| Pacholczyk (1970) | synchrotron formalism |
| Jaffe & Perola (1973) | JP ageing model |
| Kardashev (1962) | KP ageing model |
| Tribble (1993) | inhomogeneous-field ageing |
| Myers & Spangler (1985) | CI / CI-off |
| Beck & Krause (2005) | revised equipartition |
| Croston et al. (2005) | proton content (ζ) in radio sources |

---

## License

PyRATS is free software, released under the **GNU General Public License v3.0 or later** — see [`LICENSE`](LICENSE).

PyRATS does not include synchrofit, pysynch or BRATS. They are installed separately and remain under their own licences.

### Renamed from PySynch_BRATS

Earlier versions were called `PySynch_BRATS.py`. Output files have been renamed to match: `PySynch_results.json` → `PyRATS_results.json`, `PySynch_report.txt` → `PyRATS_report.txt`, default output directory `PySynch_Products` → `PyRATS_Products`, and the FITS provenance keyword `PYSYNCH` → `PYRATS`.
