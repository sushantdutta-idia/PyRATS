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

PyRATS — spectral ageing and energetics of radio sources (PySynch + BRATS)
Copyright (C) 2026  Sushant Dutta

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details. You should have received a copy of the GNU General Public License along with this program.  If not, see <https://www.gnu.org/licenses/>.

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

## This Script

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

===============================================================================
WHAT THIS PIPELINE PRODUCES
===============================================================================

From a set of cropped radio continuum FITS images at different frequencies:

  input FITS maps
    -> automatic robust noise + calibration-systematic characterisation
    -> astrometric reference selection
    -> WCS regridding onto one common pixel grid (true footprint intersection)
    -> convolution to one common restoring beam
    -> whole-source hysteresis segmentation
    -> integrated SED (with proper non-detection handling)
    -> 3-D geometry and SOURCE VOLUME (four independent estimators)
    -> INJECTION INDEX from a six-model spectral-ageing ensemble
         (JP, KP, JP-Tribble, KP-Tribble, CI, CI-OFF)
       with explicit identifiability gating and AICc/BIC model averaging
    -> EQUIPARTITION and TRUE MINIMUM-ENERGY magnetic fields
    -> u_e, u_B, u_min, total energy, minimum pressure (+ full Monte Carlo)
    -> self-consistent alpha_inj <-> B <-> aged-electron-spectrum iteration
    -> resolved SPECTRAL-INDEX, CURVATURE and ERROR maps
       (statistical and calibration systematics reported separately)
    -> optional resolved SPECTRAL-AGE / BREAK-FREQUENCY / B_eq / pressure maps
    -> FITS + publication-ready PNG + JSON + CSV + human-readable report

===============================================================================
SCIENTIFIC RATIONALE
===============================================================================

(1) SYNCHROFIT API ADAPTER  [correctness-critical]
    Current synchrofit master exposes

        spectral_fitter(…)  -> (params, discretized_parameters,
                                  marginal_distributions, probability_vector,
                                  mesh_parameters, normalisation_vector)
        spectral_fitter_(…) -> params                       (8-tuple)
        spectral_model(fit_type, mesh_parameters,
                       normalisation_vector, frequency,
                       probability_vector, …)
        spectral_model_(params, frequency, mc_length=…, …)

    This version introspects the installed API at import time and normalises BOTH
    conventions onto a single internal record.  It also exploits the new API’s
    `probability_vector` / `mesh_parameters` to build *marginal posteriors*
    for the injection index and break frequency, which is far more defensible
    than a grid-peak +/- value when the spectrum is sparsely sampled.

(2) INJECTION INDEX  [the central scientific quantity here]
    A low-frequency power-law slope is an OBSERVED spectral index.  It equals
    the injection index only if the emitting plasma is unaged at those
    frequencies, which cannot be assumed.
    This version:
      * always fits the full six-model ensemble when synchrofit is available;
      * gates every model on formal identifiability (n_points > k + 1),
        because with 3-4 bands AICc is undefined for the 3-parameter models
        and CI-OFF (k = 4) is unconstrained;
      * additionally runs a RESTRICTED ensemble in which s is held fixed and
        only (normalisation, break) are free (k = 2), which IS identifiable
        with 4 bands, and reports it as the recommended product at low band
        count;
      * performs a parametric bootstrap over the flux uncertainties so the
        quoted error is not purely the grid spacing;
      * for the Tribble variants, marginalises over the magnetic-field
        uncertainty instead of conditioning on one B value;
      * combines models by AICc weights and reports the between-model scatter
        explicitly as a *model systematic*, separate from the statistical error;
      * enforces the physical consistency requirement alpha_inj <= alpha_obs
        (an aged spectrum can only be steeper than the injected one).

(3) EQUIPARTITION vs MINIMUM ENERGY  [conceptual correction]
    PySynch’s `normalize(method=‘equipartition’)` solves u_B = zeta * u_e and
    then sets total_energy_density = zeta*u_e + u_B = 2 u_B.
    This version reports both solutions, both energy budgets, the minimum pressure
    p_min = u_min/3, and verifies the (4/3)^(2/7) ratio as an automatic
    numerical QA check on the whole normalisation chain.

    NOTE ON zeta: in PySynch, `zeta` is the ratio u_B/u_particles enforced at
    equipartition, i.e. zeta = 1 + kappa where kappa is the ratio of the
    energy density in non-radiating particles (protons) to that in electrons.
    zeta = 1 therefore means an electron-positron plasma with NO proton
    contribution.  This version exposes —kappa as the physically transparent control
    and documents the mapping in every output file.

(4) SELF-CONSISTENT ALPHA_INJ <-> B <-> AGED SPECTRUM
    B_eq depends on alpha_inj; the Tribble fits depend on B; the aged electron
    spectrum depends on the age, which depends on B. This version iterates to a fixed point (tolerance
    on ln B), and — importantly — uses pysynch’s `spectrum=‘aged’` (a JP
    aged electron spectrum implemented in C) or `spectrum=‘broken’` for the
    equipartition normalisation once a break has been measured, instead of
    assuming an unbroken power law.  This matters whenever the normalisation
    band lies at or above the break, where a power-law assumption
    systematically over-estimates the number of radiating electrons.

(5) VOLUME
    A single global sphere/ellipsoid/cylinder is a poor description of a WAT,
    a tail, an X-shaped source or a double.  This version adds a per-column,
    per-connected-segment integration,

        V = sum over columns, over segments  pi * (d_seg/2)^2 * dx

    performed along both principal axes, which handles concave and
    disconnected morphologies correctly.  All four estimators (sphere,
    ellipsoid, cylinder, integrated) are reported; the integrated one is the
    default and the spread between them enters the error budget as a
    geometry systematic.  A filling factor phi is an explicit parameter
    (B_eq scales as phi^(-2/7)).  The geometry error is no longer a hardcoded
    15%: it is measured by re-deriving the volume at several segmentation
    thresholds and under both beam-deconvolution conventions.

(6) SPECTRAL-INDEX MAPS
    This version vectorises the fit (orders of magnitude faster) and reports
    sigma_alpha_stat, sigma_alpha_sys and sigma_alpha_tot as separate maps,
    plus chi2/dof, band-count and — where >= 4 bands exist — a spectral
    CURVATURE map, which is the resolved diagnostic that actually
    distinguishes ageing from a pure power law.

(7) RESOLVED AGEING MAPS
    This version adaptively bins the source (quadtree to a target signal-to-noise, floored
    at one beam), fixes the injection index at the ensemble value, and fits
    only (normalisation, break) per region — which IS identifiable.  For
    speed the model spectra are precomputed ONCE as pysynch emissivity
    templates on an age grid (JP, via setage) or break grid (CI, via
    setbreak); each region then costs one analytic normalisation and a chi2.

(8) TWO INDEPENDENT FITTING ENGINES: synchrofit AND BRATS
    —engine {synchrofit, brats, both}

    BRATS (Harwood et al. 2013, 2015; ascl:1806.025) is driven exactly the way
    its own Python wrapper drives it — a text file of commands piped to the
    program’s stdin — so results are identical to typing them by hand. The
    pipeline stages the common-beam maps into a BRATS working directory,
    writes the DS9 region files, generates the command script, runs it, and
    reads the exported tables and FITS maps back.

    —engine both runs the two backends over the same maps, with the same
    per-map RMS and flux-scale errors, and cross-compares them. They share no
    source code, so agreement is evidence about the data and disagreement is a
    model-implementation systematic worth quoting.

    —brats-interactive instead prepares the workspace and opens BRATS in a
    NEW terminal window, for when you want to drive its PGPLOT session by
    hand. If no terminal can be opened, the workspace and a README are still
    written and the exact command to run is reported.

    —open-brats is the other way round, and is usually what you want: the
    pipeline runs to completion first — every fit, map, figure and table
    written — and only then opens BRATS in a new window, sitting in the
    prepared workspace with this run’s injection index and magnetic field
    already in place. Because it is the last thing that happens, nothing
    about the science products depends on it.

    Where a scripted BRATS run has just finished, that session is pointed at
    the .brats file its `fullexport` wrote, so a single `fullimport` restores
    every region, fit, age and error rather than recomputing them — a
    resolved fit can be hours of work and is otherwise lost when the scripted
    process exits. —open-brats also works with —engine synchrofit, where no
    BRATS stage ran at all: the workspace is built at that point instead.

    Interactive BRATS draws through PGPLOT, which needs an X server. For a
    containerised BRATS the display is forwarded automatically (the host X
    socket on Linux, host.docker.internal on macOS); if no X server can be
    found this is reported up front, because the failure mode is otherwise
    baffling — BRATS starts, takes commands, and then does nothing the
    moment you ask it to plot.

    BRATS can be driven natively or inside a container
    (—brats-container docker|podman|apptainer|singularity). The container
    path is not a convenience: BRATS depends on PGPLOT and FUNTOOLS, which
    are packaged on Linux but have no maintained macOS build, so on a Mac it
    is usually the only way to run BRATS at all. A Dockerfile is written into
    every workspace, so the workspace is a complete, self-contained record —
    a collaborator who receives it can rebuild the exact environment the
    numbers came from. The stdin-piping contract is identical either way.

(9) READING BRATS’ EXPORTS CORRECTLY  [correctness-critical]
    BRATS’ exportdata files DO NOT all have the same column layout, and the
    layout is not implied by the menu label. Read from the writer in main.c:

        ages / chi-squared / normalisation   1 column
        age errors                           2 columns, “+plus -minus”
        spectral index                       2 columns, alpha, sigma_alpha
        injection index                      2 columns, inject, SUM chi^2
        injection index by region            2 columns, region_id, value
        region array                         3 columns, x, y, region_id

    Reading column 0 of every file — the obvious thing to do, and what this script
    does — is therefore right for only about half of them. For the injection
    index it returns the trial GRID instead of the chi-square curve, so
    minimising it returns the first grid node whatever the data say: the
    reported “BRATS injection index” was just `mininject`, independent of the
    observations. For the per-region injection exports it returned region ID
    numbers. For the region array it returned x pixel coordinates. Each of
    those is a wrong number that looks entirely plausible in a summary table.
    This script encodes the layout explicitly, per token, and reads the column that
    holds the measurement.

    The injection index is then finished properly rather than left as a node
    number: the chi-square minimum is refined with a parabola through the
    surrounding nodes, and the Delta chi-square = 1 interval is taken AFTER
    dividing the summed chi-square by the beam area, because regions inside
    one beam are not independent measurements (BRATS Cookbook). Skipping that
    division gives an error bar roughly sqrt(A_beam) too small. A minimum
    that lands on the edge of the search range is reported as a LIMIT, not a
    measurement.

(10) BRATS RESULTS BECOME SCIENCE PRODUCTS
    BRATS reports resolved quantities per REGION, plus a region-array export
    giving the region ID of every pixel. This script uses that to paint every
    exported quantity back onto the pipeline’s own pixel grid, so the
    spectral-age, age-error, chi-square and spectral-index maps come out as
    FITS with the real WCS attached and as publication figures. This is done
    in addition to BRATS’ own FITS writer because that writer emits a minimal
    header (nothing can be overlaid on it without re-attaching the WCS by
    hand), `exportasfits` silently produces PNGs on older builds, and the
    error export carries BOTH wings whereas the FITS map holds only one.

    The BRATS source itself notes that its x/y mapping “has become crossed
    over somewhere”, so the axis order is not taken on trust: both
    orientations are tried and scored against the pipeline’s own source mask,
    and a low on-source fraction is reported as a warning rather than being
    quietly plotted.

    These land in brats_maps/, deliberately not beside the synchrofit
    products: two files called spectral_index.fits from different codes would
    be an easy and serious mistake, and comparing the two engines is the
    whole point of running both.

(11) CONTOUR OVERLAYS
    A spectral-index or spectral-age map is a derived quantity with no
    morphology of its own. Drawn alone there is no way to tell which
    structure a given value belongs to — hotspot, lobe, or tail. Every
    derived map is therefore drawn with total-intensity contours from a
    reference band over it, on the conventional radio ladder (3 sigma, then
    doubling), with the restoring beam marked. The reference defaults to the
    astrometric reference map, which is the one the pixel grid is actually
    registered against; —contour-label, —contour-sigma, —contour-factor
    and —contour-max-levels control it, and —no-contours turns it off.

    Model coverage differs between the engines and is enforced rather than
    papered over: BRATS fits JP, KP, Tribble (JP), CI and CI-off, but it
    CANNOT fit KP-Tribble — fitkptribble is commented out in its source and
    the export menu has no Tribble (KP) entries. Ask for KP-Tribble and the
    pipeline says so and points you at synchrofit, which does implement it.

===============================================================================
CONVENTIONS  (stated explicitly because sign conventions cause real errors)
===============================================================================

    S_nu  proportional to  nu^(alpha)          alpha is SIGNED and NEGATIVE
                                               for optically thin synchrotron
    alpha_positive = -alpha                    both are reported everywhere
    N(E) dE proportional to E^(-s) dE          s is the injection ENERGY index
    alpha_inj_positive = (s - 1)/2
    s = 1 - 2*alpha_signed                     ( = pysynch’s `injection` = p )

    Energy densities are J m^-3, magnetic fields Tesla internally and
    microGauss in reports (1 uG = 1e-10 T), volumes m^3, ages Myr.

===============================================================================
EXTERNAL DEPENDENCIES
===============================================================================

    required : numpy, scipy, astropy, matplotlib, radio_beam, reproject
    optional : mhardcastle/pysynch  (+ GSL)   — equipartition / minimum energy
                                                 and the JP/broken emissivity
                                                 templates used for resolved
                                                 ageing maps
    optional : synchrofit/synchrofit          — the six-model ageing ensemble

    The pipeline degrades gracefully: without pysynch you still get maps,
    SEDs and spectral indices; without synchrofit you still get equipartition
    and the pysynch-native JP analysis.  Nothing is ever silently substituted
    — an unavailable model is reported as unavailable, never replaced by a
    power law behind your back.

===============================================================================
KEY REFERENCES
===============================================================================
    Burbidge (1959)                    minimum energy
    Pacholczyk (1970)                  synchrotron formalism, F(x)
    Jaffe & Perola (1973)              JP ageing model
    Kardashev (1962); Pacholczyk       KP ageing model
    Tribble (1993)                     inhomogeneous-field ageing
    Myers & Spangler (1985)            CI / CI-off
    Beck & Krause (2005)               revised equipartition
    Hardcastle et al. (1998, 2002)     pysynch formalism
    Harwood et al. (2013, 2015)        resolved spectral ageing (BRATS)
    Turner et al. (2018a,b)            synchrofit; Tribble implementation
    Croston et al. (2005)              zeta / proton content in FR sources


---

## Installation

PyRATS is a single Python script, `PyRATS.py`. Its dependencies are installed separately.

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

### 2. PySynch (needed for equipartition, minimum energy and resolved ageing)

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

PyRATS is validated on Linux/macOS with:

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

Label each image with `LABEL=path`. The labels are used in every table, plot and file name. An example to run this script is given below

```bash
python PyRATS.py \
  --image image_1.fits \
  --image image_2.fits \
  --image image_3.fits \
  --image image_4.fits \
  --reference-label image_2 \
  --redshift xx \
  --output-dir directory_name
```

A fuller run example, with both ageing engines, resolved ageing maps, per-band flux-scale errors and BRATS maps:

```bash
python PyRATS.py \
  --image image_1.fits \
  --image image_2.fits \
  --image image_3.fits \
  --cal-frac image_1=0.10 --cal-frac image_2=0.05 --cal-frac image_3=0.05 \
  --redshift xx \
  --reference-label image_2 --contour-label image_2 \
  --engine both --resolved-ageing \
  --ensemble-models JP_Tribble CI \
  --brats-signaltonoise 10 \
  --output-dir directory_name
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
- **SynchroFit** — Turner et al. (2018)
- **PySynch** — Hardcastle et al. (1998)

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

PyRATS does not include SynchroFit, PySynch or BRATS. They are installed separately and remain under their own licences.
