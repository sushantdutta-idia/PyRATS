#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PySynch_BRATS.py

Author : Sushant Dutta


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

        spectral_fitter(...)  -> (params, discretized_parameters,
                                  marginal_distributions, probability_vector,
                                  mesh_parameters, normalisation_vector)
        spectral_fitter_(...) -> params                       (8-tuple)
        spectral_model(fit_type, mesh_parameters,
                       normalisation_vector, frequency,
                       probability_vector, ...)
        spectral_model_(params, frequency, mc_length=..., ...)

    v1 called `spectral_fitter` and then unpacked the result as if it were the
    8-tuple `params`.  On any recent synchrofit this raises inside the parse
    block and every ageing fit silently degrades to status="parse_failed".
    v2 introspects the installed API at import time and normalises BOTH
    conventions onto a single internal record.  It also exploits the new API's
    `probability_vector` / `mesh_parameters` to build *marginal posteriors*
    for the injection index and break frequency, which is far more defensible
    than a grid-peak +/- value when the spectrum is sparsely sampled.

(2) INJECTION INDEX  [the central scientific quantity here]
    A low-frequency power-law slope is an OBSERVED spectral index.  It equals
    the injection index only if the emitting plasma is unaged at those
    frequencies, which cannot be assumed.  v1 wired only the single
    user-selected model into main() and never called its own ensemble routine.
    v2:
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
    pysynch's `normalize(method='equipartition')` solves u_B = zeta * u_e and
    then sets total_energy_density = zeta*u_e + u_B = 2 u_B.  v1 reported that
    number as "u_min".  It is the EQUIPARTITION total energy density, not the
    minimum-energy one.  The classical minimum-energy condition is
    u_B = (3/4) u_particles, giving

        B_eq / B_me = (4/3)^(2/7) ~= 1.0851

    pysynch already ships `method='minimum_energy'`, which v1 never used.
    v2 reports both solutions, both energy budgets, the minimum pressure
    p_min = u_min/3, and verifies the (4/3)^(2/7) ratio as an automatic
    numerical QA check on the whole normalisation chain.

    NOTE ON zeta: in pysynch, `zeta` is the ratio u_B/u_particles enforced at
    equipartition, i.e. zeta = 1 + kappa where kappa is the ratio of the
    energy density in non-radiating particles (protons) to that in electrons.
    zeta = 1 therefore means an electron-positron plasma with NO proton
    contribution.  v2 exposes --kappa as the physically transparent control
    and documents the mapping in every output file.

(4) SELF-CONSISTENT ALPHA_INJ <-> B <-> AGED SPECTRUM
    B_eq depends on alpha_inj; the Tribble fits depend on B; the aged electron
    spectrum depends on the age, which depends on B.  v1 broke this circle by
    taking a single provisional pass.  v2 iterates to a fixed point (tolerance
    on ln B), and -- importantly -- uses pysynch's `spectrum='aged'` (a JP
    aged electron spectrum implemented in C) or `spectrum='broken'` for the
    equipartition normalisation once a break has been measured, instead of
    assuming an unbroken power law.  This matters whenever the normalisation
    band lies at or above the break, where a power-law assumption
    systematically over-estimates the number of radiating electrons.

(5) VOLUME
    A single global sphere/ellipsoid/cylinder is a poor description of a WAT,
    a tail, an X-shaped source or a double.  v2 adds a per-column,
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
    v1 looped in pure Python over every pixel and folded the calibration
    systematic into the per-pixel weights.  That conflates two very different
    errors: image noise is independent between pixels, whereas a flux-scale
    error is COHERENT across the entire map and does not average down.  v2
    vectorises the fit (orders of magnitude faster) and reports
    sigma_alpha_stat, sigma_alpha_sys and sigma_alpha_tot as separate maps,
    plus chi2/dof, band-count and -- where >= 4 bands exist -- a spectral
    CURVATURE map, which is the resolved diagnostic that actually
    distinguishes ageing from a pure power law.

(7) RESOLVED AGEING MAPS
    Per-pixel JP/KP/CI fitting is not identifiable with 3-4 bands.  v2
    adaptively bins the source (quadtree to a target signal-to-noise, floored
    at one beam), fixes the injection index at the ensemble value, and fits
    only (normalisation, break) per region -- which IS identifiable.  For
    speed the model spectra are precomputed ONCE as pysynch emissivity
    templates on an age grid (JP, via setage) or break grid (CI, via
    setbreak); each region then costs one analytic normalisation and a chi2.

(8) TWO INDEPENDENT FITTING ENGINES: synchrofit AND BRATS
    --engine {synchrofit, brats, both}

    BRATS (Harwood et al. 2013, 2015; ascl:1806.025) is driven exactly the way
    its own Python wrapper drives it -- a text file of commands piped to the
    program's stdin -- so results are identical to typing them by hand. The
    pipeline stages the common-beam maps into a BRATS working directory,
    writes the DS9 region files, generates the command script, runs it, and
    reads the exported tables and FITS maps back.

    --engine both runs the two backends over the same maps, with the same
    per-map RMS and flux-scale errors, and cross-compares them. They share no
    source code, so agreement is evidence about the data and disagreement is a
    model-implementation systematic worth quoting.

    --brats-interactive instead prepares the workspace and opens BRATS in a
    NEW terminal window, for when you want to drive its PGPLOT session by
    hand. If no terminal can be opened, the workspace and a README are still
    written and the exact command to run is reported.

    --open-brats is the other way round, and is usually what you want: the
    pipeline runs to completion first -- every fit, map, figure and table
    written -- and only then opens BRATS in a new window, sitting in the
    prepared workspace with this run's injection index and magnetic field
    already in place. Because it is the last thing that happens, nothing
    about the science products depends on it.

    Where a scripted BRATS run has just finished, that session is pointed at
    the .brats file its `fullexport` wrote, so a single `fullimport` restores
    every region, fit, age and error rather than recomputing them -- a
    resolved fit can be hours of work and is otherwise lost when the scripted
    process exits. --open-brats also works with --engine synchrofit, where no
    BRATS stage ran at all: the workspace is built at that point instead.

    Interactive BRATS draws through PGPLOT, which needs an X server. For a
    containerised BRATS the display is forwarded automatically (the host X
    socket on Linux, host.docker.internal on macOS); if no X server can be
    found this is reported up front, because the failure mode is otherwise
    baffling -- BRATS starts, takes commands, and then does nothing the
    moment you ask it to plot.

    BRATS can be driven natively or inside a container
    (--brats-container docker|podman|apptainer|singularity). The container
    path is not a convenience: BRATS depends on PGPLOT and FUNTOOLS, which
    are packaged on Linux but have no maintained macOS build, so on a Mac it
    is usually the only way to run BRATS at all. A Dockerfile is written into
    every workspace, so the workspace is a complete, self-contained record --
    a collaborator who receives it can rebuild the exact environment the
    numbers came from. The stdin-piping contract is identical either way.

(9) READING BRATS' EXPORTS CORRECTLY  [correctness-critical]
    BRATS' exportdata files DO NOT all have the same column layout, and the
    layout is not implied by the menu label. Read from the writer in main.c:

        ages / chi-squared / normalisation   1 column
        age errors                           2 columns, "+plus -minus"
        spectral index                       2 columns, alpha, sigma_alpha
        injection index                      2 columns, inject, SUM chi^2
        injection index by region            2 columns, region_id, value
        region array                         3 columns, x, y, region_id

    Reading column 0 of every file -- the obvious thing to do, and what v2.0
    did -- is therefore right for only about half of them. For the injection
    index it returns the trial GRID instead of the chi-square curve, so
    minimising it returns the first grid node whatever the data say: the
    reported "BRATS injection index" was just `mininject`, independent of the
    observations. For the per-region injection exports it returned region ID
    numbers. For the region array it returned x pixel coordinates. Each of
    those is a wrong number that looks entirely plausible in a summary table.
    v2.1 encodes the layout explicitly, per token, and reads the column that
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
    giving the region ID of every pixel. v2.1 uses that to paint every
    exported quantity back onto the pipeline's own pixel grid, so the
    spectral-age, age-error, chi-square and spectral-index maps come out as
    FITS with the real WCS attached and as publication figures. This is done
    in addition to BRATS' own FITS writer because that writer emits a minimal
    header (nothing can be overlaid on it without re-attaching the WCS by
    hand), `exportasfits` silently produces PNGs on older builds, and the
    error export carries BOTH wings whereas the FITS map holds only one.

    The BRATS source itself notes that its x/y mapping "has become crossed
    over somewhere", so the axis order is not taken on trust: both
    orientations are tried and scored against the pipeline's own source mask,
    and a low on-source fraction is reported as a warning rather than being
    quietly plotted.

    These land in brats_maps/, deliberately not beside the synchrofit
    products: two files called spectral_index.fits from different codes would
    be an easy and serious mistake, and comparing the two engines is the
    whole point of running both.

(11) CONTOUR OVERLAYS
    A spectral-index or spectral-age map is a derived quantity with no
    morphology of its own. Drawn alone there is no way to tell which
    structure a given value belongs to -- hotspot, lobe, or tail. Every
    derived map is therefore drawn with total-intensity contours from a
    reference band over it, on the conventional radio ladder (3 sigma, then
    doubling), with the restoring beam marked. The reference defaults to the
    astrometric reference map, which is the one the pixel grid is actually
    registered against; --contour-label, --contour-sigma, --contour-factor
    and --contour-max-levels control it, and --no-contours turns it off.

    Model coverage differs between the engines and is enforced rather than
    papered over: BRATS fits JP, KP, Tribble (JP), CI and CI-off, but it
    CANNOT fit KP-Tribble -- fitkptribble is commented out in its source and
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
    s = 1 - 2*alpha_signed                     ( = pysynch's `injection` = p )

    Energy densities are J m^-3, magnetic fields Tesla internally and
    microGauss in reports (1 uG = 1e-10 T), volumes m^3, ages Myr.

===============================================================================
EXTERNAL DEPENDENCIES
===============================================================================

    required : numpy, scipy, astropy, matplotlib, radio_beam, reproject
    optional : mhardcastle/pysynch  (+ GSL)   -- equipartition / minimum energy
                                                 and the JP/broken emissivity
                                                 templates used for resolved
                                                 ageing maps
    optional : synchrofit/synchrofit          -- the six-model ageing ensemble

    The pipeline degrades gracefully: without pysynch you still get maps,
    SEDs and spectral indices; without synchrofit you still get equipartition
    and the pysynch-native JP analysis.  Nothing is ever silently substituted
    -- an unavailable model is reported as unavailable, never replaced by a
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
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import re
import shutil
import subprocess
import sys
import textwrap
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u
from astropy.cosmology import FlatLambdaCDM
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
from radio_beam import Beam, Beams
from reproject import reproject_interp
from scipy.ndimage import binary_dilation, binary_fill_holes, label
#: `label` is shadowed by a loop variable in several functions below, so
#: the connected-component code uses this unambiguous alias instead.
from scipy.ndimage import label as connected_components

__version__ = "2.1.0"


# ===========================================================================
# PART 1.  OPTIONAL BACKENDS -- introspected, never assumed
# ===========================================================================

class SynchrofitAPI:
    """
    Version-agnostic wrapper around synchrofit.

    synchrofit has shipped two incompatible public conventions:

      "modern"  spectral_fitter(...)  -> (params8, discretized_parameters,
                                          marginal_distributions,
                                          probability_vector, mesh_parameters,
                                          normalisation_vector)
                spectral_model(fit_type, mesh_parameters,
                               normalisation_vector, frequency,
                               probability_vector, b_field, redshift, pcrit)

      "legacy"  spectral_fitter(...)  -> params8
                spectral_model(params8, frequency, mc_length=..., err_width=...)

    In the modern package the legacy entry points survive as
    `spectral_fitter_` and `spectral_model_`.  We prefer the legacy-style
    entry points when present (they are simpler and single-process), but we
    keep the modern ones so that the probability grid can be harvested for
    marginal posteriors.

    Every method here returns plain Python/NumPy objects; nothing synchrofit
    specific leaks into the science layer.
    """

    #: params8 field order, fixed by synchrofit itself
    PARAM_FIELDS = (
        "fit_type",
        "log10_break_frequency_Hz",
        "log10_break_frequency_err",
        "injection_index_s",
        "injection_index_s_err",
        "remnant_fraction",
        "remnant_fraction_err",
        "normalisation",
    )

    def __init__(self) -> None:
        self.available = False
        self.import_error: Optional[BaseException] = None
        self.flavour = "none"
        self._fitter = None            # returns params8 (possibly wrapped)
        self._model_legacy = None      # spectral_model_(params8, freq, ...)
        self._model_modern = None      # spectral_model(fit_type, mesh, ...)
        self._ages = None
        self._module = None
        self.version = "unknown"
        self._load()

    # -- import -------------------------------------------------------------
    def _load(self) -> None:
        module = None
        for import_path in ("sf.synchrofit", "synchrofit.synchrofit", "synchrofit", "sf"):
            try:
                module = __import__(import_path, fromlist=["*"])
                break
            except Exception as exc:                      # noqa: BLE001
                self.import_error = exc
                module = None
        if module is None:
            return

        self._module = module
        self.version = str(getattr(module, "__version__", "unknown"))

        fitter_legacy = getattr(module, "spectral_fitter_", None)
        fitter_modern = getattr(module, "spectral_fitter", None)
        model_legacy = getattr(module, "spectral_model_", None)
        model_modern = getattr(module, "spectral_model", None)
        self._ages = getattr(module, "spectral_ages", None)

        if callable(fitter_legacy):
            # Modern package: the underscore variants are the legacy ones.
            self._fitter = fitter_legacy
            self.flavour = "modern"
        elif callable(fitter_modern):
            # Old packages return params8 directly; new ones return a
            # 6-tuple. The shape is resolved at call time by
            # `_normalise_fitter_return`, which is safer than guessing from a
            # version string that many installs do not even set.
            self._fitter = fitter_modern
            self.flavour = "legacy_or_modern_single_entry"
        else:
            self.import_error = RuntimeError(
                "synchrofit imported but exposes neither spectral_fitter nor "
                "spectral_fitter_"
            )
            return

        self._model_legacy = model_legacy if callable(model_legacy) else None
        self._model_modern = model_modern if callable(model_modern) else None
        # Old packages: spectral_model IS the legacy signature.
        if self._model_legacy is None and self._model_modern is not None:
            self._model_legacy = self._model_modern
            self._model_modern = None

        self.available = True

    # -- helpers ------------------------------------------------------------
    #: fields some synchrofit versions append after the eight standard ones
    PARAM_EXTRA_FIELDS = ("max_probability",)

    @staticmethod
    def _looks_like_params8(obj: Any) -> bool:
        """
        Is this synchrofit's parameter tuple?

        The eight standard fields are followed, in some versions, by extra
        diagnostics -- current `spectral_fitter_` appends `max_probability`,
        making it a NINE-tuple. Insisting on exactly eight rejects that
        outright, and because the rejection happens inside the fit call every
        model in the ensemble comes back `fit_failed`: no injection index, no
        SED model curve, no emissivity curve, and nothing to say why beyond a
        signature complaint. Anything with at least the eight known fields
        and a fit-type string in front is accepted, and the surplus is kept
        rather than discarded.
        """
        return (
            isinstance(obj, (tuple, list))
            and len(obj) >= 8
            and isinstance(obj[0], str)
        )

    def _normalise_fitter_return(self, raw: Any) -> Dict[str, Any]:
        """
        Reduce any synchrofit fitter return value to

            {"params8": tuple, "grid": {...} or None}

        `grid` carries mesh_parameters / probability_vector /
        normalisation_vector when the modern API supplied them, so that
        marginal posteriors can be computed downstream.
        """
        if self._looks_like_params8(raw):
            return {"params8": tuple(raw), "grid": None}

        if isinstance(raw, (tuple, list)) and len(raw) >= 1:
            head = raw[0]
            if self._looks_like_params8(head):
                grid = None
                if len(raw) >= 6:
                    grid = {
                        "discretized_parameters": raw[1],
                        "marginal_distributions": raw[2],
                        "probability_vector": np.asarray(raw[3], dtype=float),
                        "mesh_parameters": np.asarray(raw[4], dtype=float),
                        "normalisation_vector": np.asarray(raw[5], dtype=float),
                    }
                return {"params8": tuple(head), "grid": grid}
            # (params8, debug_output)
            if len(raw) == 2 and self._looks_like_params8(raw[0]):
                return {"params8": tuple(raw[0]), "grid": None}

        raise RuntimeError(
            "Unrecognised synchrofit spectral_fitter return signature "
            f"(type={type(raw).__name__}, len="
            f"{len(raw) if hasattr(raw, '__len__') else 'n/a'}). "
            "Please report the installed synchrofit version."
        )

    # -- public -------------------------------------------------------------
    def fit(
        self,
        frequency_hz: np.ndarray,
        flux: np.ndarray,
        dflux: np.ndarray,
        fit_type: str,
        n_breaks: int,
        break_range: Sequence[float],
        n_injects: int,
        inject_range: Sequence[float],
        n_remnants: int,
        remnant_range: Sequence[float],
        n_iterations: int,
        b_field_t: Optional[float] = None,
        redshift: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Run one synchrofit fit and return a normalised record."""
        if not self.available:
            raise RuntimeError("synchrofit is not available.")

        kwargs: Dict[str, Any] = dict(
            frequency=np.asarray(frequency_hz, dtype=float),
            luminosity=np.asarray(flux, dtype=float),
            dluminosity=np.asarray(dflux, dtype=float),
            fit_type=str(fit_type),
            n_breaks=int(n_breaks),
            break_range=[float(break_range[0]), float(break_range[1])],
            n_injects=int(n_injects),
            inject_range=[float(inject_range[0]), float(inject_range[1])],
            n_remnants=int(n_remnants),
            remnant_range=[float(remnant_range[0]), float(remnant_range[1])],
            n_iterations=int(n_iterations),
            write_model=False,
        )
        if str(fit_type).upper() in {"TJP", "TKP", "TCI"}:
            if b_field_t is None or not np.isfinite(b_field_t) or b_field_t <= 0:
                raise ValueError(
                    f"{fit_type} requires a positive magnetic field strength."
                )
            if redshift is None or not np.isfinite(redshift) or redshift <= 0:
                raise ValueError(f"{fit_type} requires a positive redshift.")
            kwargs["b_field"] = float(b_field_t)
            kwargs["redshift"] = float(redshift)

        with _suppressed_stdout():
            raw = self._fitter(**kwargs)

        record = self._normalise_fitter_return(raw)
        params_full = record["params8"]
        params8 = tuple(params_full[:len(self.PARAM_FIELDS)])
        out = {
            name: params8[i] for i, name in enumerate(self.PARAM_FIELDS)
        }
        # Keep whatever this version appended after the standard eight.
        extras = tuple(params_full[len(self.PARAM_FIELDS):])
        for i, value in enumerate(extras):
            name = (
                self.PARAM_EXTRA_FIELDS[i]
                if i < len(self.PARAM_EXTRA_FIELDS) else f"extra_{i}"
            )
            try:
                out[name] = float(value)
            except (TypeError, ValueError):
                out[name] = value
        out["fit_type"] = str(out["fit_type"])
        for key in self.PARAM_FIELDS:
            if key == "fit_type":
                continue
            out[key] = float(out[key])
        out["_params8"] = params8
        out["_grid"] = record["grid"]
        return out

    def model(
        self,
        record: Dict[str, Any],
        frequency_hz: np.ndarray,
        b_field_t: Optional[float] = None,
        redshift: Optional[float] = None,
        mc_length: int = 200,
        err_width: int = 1,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Evaluate the fitted model on `frequency_hz`.

        Returns (model_flux, model_flux_err); either may be None if the
        installed synchrofit could not evaluate the model.
        """
        freq = np.asarray(frequency_hz, dtype=float)
        tribble = str(record.get("fit_type", "")).upper() in {"TJP", "TKP", "TCI"}

        if self._model_legacy is not None and record.get("_params8") is not None:
            kwargs: Dict[str, Any] = {
                "mc_length": int(mc_length),
                "err_width": int(err_width),
                "write_model": False,
            }
            if tribble:
                kwargs["b_field"] = (
                    float(b_field_t) if b_field_t is not None else None
                )
                kwargs["redshift"] = (
                    float(redshift) if redshift is not None else None
                )
            try:
                with _suppressed_stdout():
                    out = self._model_legacy(record["_params8"], freq, **kwargs)
                model_flux = np.asarray(out[0], dtype=float)
                model_err = (
                    np.asarray(out[1], dtype=float) if len(out) > 1 else None
                )
                if model_flux.shape == freq.shape:
                    return model_flux, model_err
            except Exception:                                  # noqa: BLE001
                pass

        grid = record.get("_grid")
        if self._model_modern is not None and grid is not None:
            try:
                with _suppressed_stdout():
                    arr = self._model_modern(
                        record["fit_type"],
                        grid["mesh_parameters"],
                        grid["normalisation_vector"],
                        freq,
                        grid["probability_vector"],
                        float(b_field_t) if (tribble and b_field_t) else None,
                        float(redshift) if (tribble and redshift) else None,
                    )
                arr = np.atleast_2d(np.asarray(arr, dtype=float))
                # rows are realisations drawn from the probability grid
                model_flux = np.nanmedian(arr, axis=0)
                model_err = np.nanstd(arr, axis=0)
                if model_flux.shape == freq.shape:
                    return model_flux, model_err
            except Exception:                                  # noqa: BLE001
                pass

        return None, None

    def ages(
        self,
        fit_type: str,
        break_frequency_hz: float,
        remnant_fraction: float,
        b_field_t: float,
        redshift: float,
    ) -> Optional[Dict[str, float]]:
        """
        Total / active / inactive spectral age in Myr.

        synchrofit's spectral_ages() validates its inputs with strict
        isinstance(..., float) checks, so every argument is coerced to a
        genuine Python float here (np.float32 would be rejected).
        """
        if self._ages is None:
            return None
        try:
            if not np.isfinite(break_frequency_hz) or break_frequency_hz <= 0:
                return None
            if not np.isfinite(b_field_t) or b_field_t <= 0:
                return None
            if not np.isfinite(redshift) or redshift <= 0:
                return None
            with _suppressed_stdout():
                tau, t_on, t_off = self._ages(
                    (
                        str(fit_type),
                        float(break_frequency_hz),
                        float(max(remnant_fraction, 0.0)),
                    ),
                    float(b_field_t),
                    float(redshift),
                )
            return {
                "tau_Myr": float(tau),
                "t_on_Myr": float(t_on),
                "t_off_Myr": float(t_off),
            }
        except Exception:                                      # noqa: BLE001
            return None


class _suppressed_stdout:
    """
    Context manager that muffles synchrofit's very chatty coloured stdout.

    synchrofit prints a block of ANSI-coloured text for every single fit.  In
    an ensemble of six models x an adaptive-binning map that is thousands of
    lines of noise.  stderr is deliberately left alone so genuine warnings
    still reach the user.
    """

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled and not bool(
            os.environ.get("PYSYNCH_VERBOSE_BACKENDS")
        )
        self._saved = None
        self._devnull = None

    def __enter__(self):
        if self.enabled:
            self._saved = sys.stdout
            self._devnull = open(os.devnull, "w")
            sys.stdout = self._devnull
        return self

    def __exit__(self, *exc):
        if self.enabled:
            sys.stdout = self._saved
            if self._devnull is not None:
                self._devnull.close()
        return False


SYNCHROFIT = SynchrofitAPI()
HAVE_SYNCHROFIT = SYNCHROFIT.available

try:
    from synchro import SynchSource
    import synch as _synch_c
    HAVE_PYSYNCH = True
    PYSYNCH_IMPORT_ERROR: Optional[BaseException] = None
except Exception as _exc:                                      # noqa: BLE001
    SynchSource = None                                          # type: ignore
    _synch_c = None                                             # type: ignore
    HAVE_PYSYNCH = False
    PYSYNCH_IMPORT_ERROR = _exc


def require_pysynch() -> None:
    if not HAVE_PYSYNCH:
        raise RuntimeError(
            "mhardcastle/pysynch (module `synchro`, C extension `synch`) could "
            "not be imported, but an equipartition / minimum-energy or "
            "resolved-ageing product was requested.\n"
            f"Import error: {PYSYNCH_IMPORT_ERROR}\n"
            "Install pysynch and its GSL dependency, or rerun with "
            "--no-equipartition."
        )


def require_synchrofit() -> None:
    if not HAVE_SYNCHROFIT:
        raise RuntimeError(
            "synchrofit was requested but is not importable.\n"
            "  python -m pip install git+https://github.com/synchrofit/synchrofit.git\n"
            f"Import error: {SYNCHROFIT.import_error}"
        )


# ===========================================================================
# PART 2.  PHYSICAL CONSTANTS AND SCIENTIFIC DEFAULTS
# ===========================================================================

# CODATA 2018, SI
C_LIGHT = 2.99792458e8            # m s^-1
E_CHARGE = 1.602176634e-19        # C
M_ELECTRON = 9.1093837015e-31     # kg
MU_0 = 1.25663706212e-6           # N A^-2
SIGMA_T = 6.6524587321e-29        # m^2
K_BOLTZ = 1.380649e-23            # J K^-1
H_PLANCK = 6.62607015e-34         # J s
STEFAN_A = 7.565723e-16           # J m^-3 K^-4  (radiation constant)
T_CMB0 = 2.72548                  # K  (Fixsen 2009)
JANSKY = 1.0e-26                  # W m^-2 Hz^-1
MYR = 3.1556952e13                # s
TESLA_PER_MICROGAUSS = 1.0e-10

#: nu_c = SYNCH_NU_C * B[T] * gamma^2   for pitch angle 90 deg
#: from nu_c = (3/2) gamma^2 e B sin(theta) / (2 pi m_e)
SYNCH_NU_C = 3.0 * E_CHARGE / (4.0 * math.pi * M_ELECTRON)   # ~4.1993e10

DEFAULT_REDSHIFT = 0.146914
DEFAULT_GAMMA_MIN = 10.0
DEFAULT_GAMMA_MAX = 1.0e7
DEFAULT_KAPPA = 0.0               # proton:electron energy ratio -> zeta = 1+kappa
DEFAULT_FILLING_FACTOR = 1.0
ADOPTED_FIELD_FRACTION = 0.4      # only used for the legacy "B_adopted" report

DEFAULT_COSMOLOGY = FlatLambdaCDM(H0=70.0, Om0=0.3, Tcmb0=T_CMB0)

DEFAULT_SOURCE_SIGMA = 5.0
DEFAULT_GROW_SIGMA = 3.0
DEFAULT_MIN_SNR_MAP = 3.0
DEFAULT_MIN_FIT_BANDS = 3

DEFAULT_ALPHA_BOOTSTRAP = 2000
DEFAULT_BEQ_MC = 1000             # fast mode; --thorough raises this
DEFAULT_THOROUGH_BEQ_MC = 8000
DEFAULT_ALPHA_ERR_FLOOR = 0.02
DEFAULT_CAL_FRAC = 0.05

DEFAULT_BMIN_T = 1.0e-13
DEFAULT_BMAX_T = 1.0e-6
DEFAULT_SEED = 20260825

#: Physically motivated prior on the injection energy index.
#: s in [2.0, 3.0]  <->  alpha_inj_positive in [0.5, 1.0].
#: v1 used [1.5, 3.2], whose lower edge (alpha_inj = 0.25) is flatter than any
#: observed hotspot and lets the fitter buy curvature with an unphysical s.
DEFAULT_INJECT_RANGE = (2.01, 3.00)
DEFAULT_BREAK_RANGE = (7.0, 13.0)   # log10 nu_break / Hz

SPECTRAL_AGE_MODELS: Tuple[str, ...] = (
    "JP", "KP", "JP_Tribble", "KP_Tribble", "CI", "CI-OFF",
)

MODEL_MENU: Dict[str, Optional[str]] = {
    "Power-law": None,
    "JP": "JP",
    "JP-Tribble": "JP_Tribble",
    "KP": "KP",
    "KP-Tribble": "KP_Tribble",
    "CI": "CI",
    "CI-OFF": "CI-OFF",
    "Ensemble": "ENSEMBLE",       # AICc-weighted average of all six
}


def b_cmb_tesla(redshift: float, synchrofit_compatible: bool = False) -> float:
    """
    Equivalent magnetic field of the CMB radiation energy density,

        B_CMB = sqrt(2 mu_0 u_CMB),   u_CMB = a T^4,  T = T_0 (1+z)

    With T_0 = 2.72548 K this gives B_CMB(0) = 3.2379 uG, i.e. 0.32379 nT.

    synchrofit hardcodes 0.318 nT, which corresponds to T_0 ~ 2.70 K.  The
    difference propagates into spectral ages at the ~3% level for B << B_CMB.
    Set `synchrofit_compatible=True` to reproduce synchrofit's number exactly
    when you need bit-comparable ages.
    """
    if synchrofit_compatible:
        return 0.318e-9 * (1.0 + float(redshift)) ** 2
    t = T_CMB0 * (1.0 + float(redshift))
    u_cmb = STEFAN_A * t ** 4
    return math.sqrt(2.0 * MU_0 * u_cmb)


def alpha_signed_to_s(alpha_signed: float) -> float:
    """S_nu ~ nu^alpha  ->  N(E) ~ E^-s.   s = 1 - 2*alpha_signed."""
    return 1.0 - 2.0 * float(alpha_signed)


def s_to_alpha_signed(s: float) -> float:
    """N(E) ~ E^-s  ->  S_nu ~ nu^alpha.   alpha_signed = (1 - s)/2."""
    return 0.5 * (1.0 - float(s))


# Backwards-compatible aliases used by the v1 API surface.
alpha_to_p = alpha_signed_to_s
_radio_alpha_from_s = s_to_alpha_signed


# ===========================================================================
# PART 3.  FITS INGESTION, NOISE AND CALIBRATION CHARACTERISATION
# ===========================================================================

@dataclass
class RadioMap:
    label: str
    path: Path
    data: np.ndarray
    header: fits.Header
    wcs: WCS
    beam: Beam
    freq_hz: float
    rms_jybeam: float
    rms_method: str = "unknown"
    cal_frac: float = DEFAULT_CAL_FRAC
    cal_method: str = "unknown"
    #: fraction of the map area used by the noise estimator; diagnostic only
    rms_area_fraction: float = float("nan")


def frequency_from_header(header: fits.Header) -> float:
    """
    Recover the continuum reference frequency in Hz.

    Priority: an explicit FREQ world-coordinate axis, then the usual rest
    frequency keywords.  Any candidate outside 1 MHz - 1 THz is rejected as a
    mis-parsed keyword rather than trusted.
    """
    for axis in range(1, 7):
        ctype = str(header.get(f"CTYPE{axis}", "")).upper()
        if "FREQ" not in ctype:
            continue
        try:
            value = float(header.get(f"CRVAL{axis}"))
        except (TypeError, ValueError):
            continue
        if np.isfinite(value) and 1.0e6 <= value <= 1.0e12:
            return value

    for key in ("RESTFRQ", "RESTFREQ", "FREQ", "OBSFREQ", "CRVAL3", "CRVAL4"):
        try:
            value = float(header[key])
        except (KeyError, TypeError, ValueError):
            continue
        if np.isfinite(value) and 1.0e6 <= value <= 1.0e12:
            return value

    raise RuntimeError(
        "No physical continuum frequency could be recovered from the FITS "
        "header (looked for a FREQ WCS axis and RESTFRQ/RESTFREQ/FREQ/OBSFREQ)."
    )


def beam_from_header(header: fits.Header) -> Beam:
    try:
        beam = Beam.from_fits_header(header)
    except Exception as exc:                                   # noqa: BLE001
        raise RuntimeError(
            "Missing or invalid BMAJ/BMIN/BPA. A restoring beam is mandatory: "
            "without it, Jy/beam cannot be converted to Jy and no flux density "
            "is defined."
        ) from exc
    if beam.major.to_value(u.arcsec) <= 0 or beam.minor.to_value(u.arcsec) <= 0:
        raise RuntimeError(f"Non-physical restoring beam in header: {beam}")
    return beam


def pixel_scales_arcsec(wcs: WCS) -> Tuple[float, float]:
    """Celestial pixel scales in arcsec/pixel, unit-aware."""
    scales = np.asarray(
        proj_plane_pixel_scales(wcs.celestial), dtype=float
    ).ravel()
    if scales.size < 2 or not np.all(np.isfinite(scales[:2])):
        raise RuntimeError(
            f"Could not determine celestial pixel scales from WCS: {scales!r}"
        )

    sx, sy = abs(float(scales[0])), abs(float(scales[1]))
    cunit1 = str(wcs.celestial.wcs.cunit[0] or "deg")
    cunit2 = str(wcs.celestial.wcs.cunit[1] or "deg")

    def to_arcsec(value: float, unit_name: str) -> float:
        name = unit_name.strip().lower()
        if name in {"deg", "degree", "degrees", ""}:
            return value * 3600.0
        if name in {"arcsec", "asec", "arcsecond", "arcseconds", "sec"}:
            return value
        if name in {"arcmin", "amin", "arcminute", "arcminutes"}:
            return value * 60.0
        if name in {"rad", "radian", "radians"}:
            return math.degrees(value) * 3600.0
        return value * 3600.0

    return to_arcsec(sx, cunit1), to_arcsec(sy, cunit2)


def pixel_area_arcsec2(wcs: WCS) -> float:
    sx, sy = pixel_scales_arcsec(wcs)
    return sx * sy


def beam_area_arcsec2(beam: Beam) -> float:
    """Gaussian beam solid angle: pi/(4 ln2) * BMAJ * BMIN."""
    return (
        (math.pi / (4.0 * math.log(2.0)))
        * beam.major.to_value(u.arcsec)
        * beam.minor.to_value(u.arcsec)
    )


def _robust_mad_sigma(values: np.ndarray, min_n: int = 20) -> float:
    """Gaussian-equivalent sigma from the median absolute deviation."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size < min_n:
        return float("nan")
    med = np.nanmedian(v)
    return float(1.482602218505602 * np.nanmedian(np.abs(v - med)))


def _iterative_sigma_clip(
    values: np.ndarray, nsig: float = 3.0, iterations: int = 8, min_n: int = 20
) -> np.ndarray:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < min_n:
        return x
    for _ in range(iterations):
        med = np.nanmedian(x)
        sig = _robust_mad_sigma(x, min_n=min_n)
        if not np.isfinite(sig) or sig <= 0:
            break
        keep = np.abs(x - med) <= nsig * sig
        if np.count_nonzero(keep) == x.size:
            break
        x = x[keep]
        if x.size < min_n:
            break
    return x


# retained for API compatibility with v1
sigma_clip_values = _iterative_sigma_clip


def _tile_rms_statistics(
    data: np.ndarray, tile_size: int, min_finite_fraction: float = 0.70
) -> List[Dict[str, float]]:
    """Robust local noise statistics on a regular tiling of the image."""
    ny, nx = data.shape
    out: List[Dict[str, float]] = []
    for y0 in range(0, ny, tile_size):
        y1 = min(ny, y0 + tile_size)
        for x0 in range(0, nx, tile_size):
            x1 = min(nx, x0 + tile_size)
            sub = np.asarray(data[y0:y1, x0:x1], dtype=float)
            finite = sub[np.isfinite(sub)]
            if finite.size < 30 or finite.size / sub.size < min_finite_fraction:
                continue
            clipped = _iterative_sigma_clip(finite, nsig=3.0, iterations=6)
            if clipped.size < 30:
                continue
            median = float(np.nanmedian(clipped))
            mad_sigma = _robust_mad_sigma(clipped)
            if not np.isfinite(mad_sigma) or mad_sigma <= 0:
                continue
            rms = float(np.sqrt(np.mean((clipped - median) ** 2)))
            if not np.isfinite(rms) or rms <= 0:
                continue
            out.append({
                "x0": float(x0), "x1": float(x1),
                "y0": float(y0), "y1": float(y1),
                "n": float(clipped.size), "median": median,
                "mad_sigma": mad_sigma, "rms": rms,
                "area": float((y1 - y0) * (x1 - x0)),
            })
    return out


def _select_clean_noise_tiles(
    tiles: List[Dict[str, float]], robust_sigma: float
) -> List[Dict[str, float]]:
    """
    Reject tiles contaminated by emission, sidelobes or calibration residuals.

    Selecting simply the lowest-noise tiles biases the RMS low, so the cut is
    made jointly on tile median (must sit at the map background) and tile
    sigma (a two-sided interval; the lower guard prevents picking accidentally
    over-clean regions such as a masked-out corner).
    """
    if len(tiles) < 5:
        return tiles

    meds = np.asarray([t["median"] for t in tiles], dtype=float)
    sigs = np.asarray([t["mad_sigma"] for t in tiles], dtype=float)

    med0 = float(np.nanmedian(meds))
    med_scale = _robust_mad_sigma(meds, min_n=5)
    if not np.isfinite(med_scale) or med_scale <= 0:
        med_scale = max(abs(robust_sigma) * 0.25, np.finfo(float).eps)

    med_keep = np.abs(meds - med0) <= max(3.0 * med_scale, 1.0 * robust_sigma)

    ref = sigs[med_keep] if np.any(med_keep) else sigs
    sig_mid = float(np.nanmedian(ref))
    sig_scale = _robust_mad_sigma(ref, min_n=5)
    if not np.isfinite(sig_scale) or sig_scale <= 0:
        sig_scale = max(0.25 * sig_mid, np.finfo(float).eps)

    sig_keep = (
        (sigs >= max(0.5 * sig_mid, sig_mid - 3.0 * sig_scale))
        & (sigs <= sig_mid + 3.0 * sig_scale)
    )

    keep = med_keep & sig_keep
    selected = [t for t, ok in zip(tiles, keep) if ok]

    if len(selected) < max(4, int(0.20 * len(tiles))):
        keep = (
            np.abs(meds - med0) <= max(5.0 * med_scale, 2.0 * robust_sigma)
        ) & sig_keep
        selected = [t for t, ok in zip(tiles, keep) if ok]

    return selected


def estimate_rms_jybeam(
    data: np.ndarray,
    header: fits.Header,
    trust_header: bool = True,
) -> Tuple[float, str, float]:
    """
    Robust image noise in Jy/beam.

    Returns (rms, provenance_string, area_fraction_used).

    Strategy, in order:
      1. an explicit and trustworthy noise keyword written by the reduction;
      2. a spatially resolved tile estimate with contaminated tiles rejected,
         cross-checked against a negative-side estimator (real emission is
         one-sided positive, so the negative half of the pixel distribution is
         a nearly emission-free sample);
      3. a global robust consensus if the tiling degenerates.

    This is a NOISE estimate.  It is deliberately not an estimate of the flux
    scale (calibration) uncertainty, which is handled separately and treated
    as a coherent systematic rather than as noise.
    """
    if trust_header:
        for key in ("RMS", "NOISE", "MAPRMS", "RMSNOISE", "RMS_JYBEAM",
                    "SIGMA", "SIGMA_MAD", "MAD_RMS"):
            if key not in header:
                continue
            try:
                value = float(header[key])
            except (TypeError, ValueError):
                continue
            if np.isfinite(value) and value > 0:
                return value, f"FITS header keyword {key}", 1.0

    values = np.asarray(data, dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size < 200:
        raise RuntimeError("Too few finite pixels for a robust noise estimate.")

    global_med = float(np.nanmedian(finite))
    global_mad = _robust_mad_sigma(finite)
    if not np.isfinite(global_mad) or global_mad <= 0:
        raise RuntimeError("Global MAD could not provide an initial RMS.")

    negative = _iterative_sigma_clip(
        finite[finite <= global_med], nsig=3.0, iterations=8
    )
    negative_sigma = _robust_mad_sigma(negative)
    if not (np.isfinite(negative_sigma) and negative_sigma > 0):
        negative_sigma = float("nan")

    try:
        beam = Beam.from_fits_header(header)
        scales = np.asarray(
            proj_plane_pixel_scales(WCS(header).celestial), dtype=float
        ) * 3600.0
        pix = float(np.nanmean(np.abs(scales)))
        beam_pix = max(
            beam.major.to_value(u.arcsec), beam.minor.to_value(u.arcsec)
        ) / max(pix, 1e-12)
    except Exception:                                          # noqa: BLE001
        beam_pix = 8.0

    ny, nx = values.shape
    nominal = int(max(16, min(256, round(max(4.0 * beam_pix, min(ny, nx) / 10)))))
    tile_size = max(8, min(nominal, max(8, min(ny, nx) // 4)))

    tiles = _tile_rms_statistics(values, tile_size=tile_size)
    clean = _select_clean_noise_tiles(tiles, global_mad)

    if clean:
        tile_mad = np.asarray([t["mad_sigma"] for t in clean], dtype=float)
        tile_rms = np.asarray([t["rms"] for t in clean], dtype=float)
        spatial_sigma = float(np.nanmedian(tile_mad))
        rms_over_mad = float(np.nanmedian(tile_rms) / max(spatial_sigma, 1e-30))

        estimates = [spatial_sigma]
        if np.isfinite(negative_sigma):
            estimates.append(negative_sigma)
        estimates = [e for e in estimates if np.isfinite(e) and e > 0]
        rms_final = float(np.median(estimates))

        scatter = float(np.nanstd(tile_mad, ddof=1)) if tile_mad.size > 1 else 0.0
        frac_unc = float(np.clip(
            scatter / max(spatial_sigma, 1e-30) / math.sqrt(max(tile_mad.size, 1)),
            0.02, 0.50,
        ))
        area_fraction = float(
            sum(t["area"] for t in clean) / max(float(ny * nx), 1.0)
        )
        method = (
            f"spatial robust consensus (tiles={len(clean)}/{len(tiles)}, "
            f"tile={tile_size}px, area_used={area_fraction:.2f}, "
            f"negative_side={'yes' if np.isfinite(negative_sigma) else 'no'}, "
            f"RMS/MAD={rms_over_mad:.3f}, "
            f"estimator_frac_unc={frac_unc:.3f})"
        )
        return rms_final, method, area_fraction

    clipped_sigma = _robust_mad_sigma(
        _iterative_sigma_clip(finite, nsig=3.0, iterations=8)
    )
    candidates = [
        x for x in (negative_sigma, clipped_sigma, global_mad)
        if np.isfinite(x) and x > 0
    ]
    if not candidates:
        raise RuntimeError("All robust RMS estimators failed.")
    return (
        float(np.median(candidates)),
        f"robust global consensus ({len(candidates)} estimators)",
        1.0,
    )


#: Conservative flux-scale priors.  These are SYSTEMATIC priors on the
#: absolute flux scale, not image-derived measurements, and they are recorded
#: verbatim in the provenance so a referee can see exactly what was assumed.
CALIBRATION_PRIORS: Tuple[Tuple[Tuple[str, ...], float, str], ...] = (
    (("UGMRT", "GMRT"),            0.10, "uGMRT/GMRT 10%"),
    (("MEERKAT",),                 0.05, "MeerKAT 5%"),
    (("LOFAR", "LOTSS"),           0.10, "LOFAR 10%"),
    (("ASKAP", "RACS"),            0.05, "ASKAP 5%"),
    (("ATCA",),                    0.05, "ATCA 5%"),
    (("VLA", "EVLA", "JVLA"),      0.05, "VLA 5%"),
    (("SKA",),                     0.05, "SKA 5%"),
    (("WSRT", "APERTIF"),          0.08, "WSRT/Apertif 8%"),
    (("MWA",),                     0.10, "MWA 10%"),
    (("VLSS", "TGSS"),             0.15, "VLSS/TGSS 15%"),
    (("NVSS", "FIRST"),            0.03, "NVSS/FIRST 3%"),
)


def estimate_calibration_fraction(header: fits.Header) -> Tuple[float, str]:
    """
    Fractional absolute flux-scale uncertainty.

    This is a multiplicative systematic and is NOT statistically identifiable
    from a single science image, so it is taken from reduction metadata when
    present and otherwise from a documented instrument prior.  It is
    deliberately never inferred from the map RMS.
    """
    for key in ("CALFRAC", "CALERR", "FLUXCAL", "CALFRACT", "AMPERR", "FLUXERR"):
        if key in header:
            try:
                v = float(header[key])
            except (TypeError, ValueError):
                continue
            if np.isfinite(v) and 0 <= v < 1:
                return v, f"FITS header keyword {key}"

    text = " ".join(
        str(header.get(k, "")).upper()
        for k in ("TELESCOP", "INSTRUME", "ORIGIN", "OBSERVAT", "SURVEY")
    )
    for keys, value, label in CALIBRATION_PRIORS:
        if any(k in text for k in keys):
            return value, f"automatic instrument prior: {label}"

    return (
        DEFAULT_CAL_FRAC,
        f"generic prior: {DEFAULT_CAL_FRAC:.0%} (instrument not identified)",
    )


def _sanitize_beam(beam: Beam, label: str) -> Beam:
    """
    Force a radio_beam.Beam into angular major/minor/PA.

    Some headers encode BMAJ/BMIN in a way that yields steradian-valued
    attributes; radio_beam.Beams then tries to combine those as angles and
    raises an sr <-> arcsec conversion error deep inside common_beam().
    Failing loudly here gives a far more useful message.
    """
    try:
        major, minor, pa = beam.major, beam.minor, beam.pa
        if not major.unit.is_equivalent(u.arcsec):
            raise RuntimeError(f"BMAJ has non-angular unit {major.unit!s}")
        if not minor.unit.is_equivalent(u.arcsec):
            raise RuntimeError(f"BMIN has non-angular unit {minor.unit!s}")
        if not pa.unit.is_equivalent(u.deg):
            raise RuntimeError(f"BPA has non-angular unit {pa.unit!s}")
        major = major.to(u.arcsec)
        minor = minor.to(u.arcsec)
        pa = pa.to(u.deg)
        if major <= 0 * u.arcsec or minor <= 0 * u.arcsec:
            raise RuntimeError(f"non-positive restoring beam {major}, {minor}")
        if minor > major:
            major, minor = minor, major
            pa = pa + 90.0 * u.deg
        return Beam(major=major, minor=minor, pa=pa)
    except Exception as exc:                                   # noqa: BLE001
        raise RuntimeError(f"{label}: could not sanitise restoring beam: {exc}") from exc


def load_radio_map(
    label_: str,
    path: Path,
    rms_jybeam: Optional[float] = None,
    cal_frac: Optional[float] = None,
    trust_header_rms: bool = True,
) -> RadioMap:
    with fits.open(path, memmap=False) as hdul:
        hdu = hdul[0]
        if hdu.data is None:
            for candidate in hdul[1:]:
                if getattr(candidate, "data", None) is not None:
                    hdu = candidate
                    break
        data = np.squeeze(np.asarray(hdu.data, dtype=float))
        header = hdu.header.copy()

    if data.ndim != 2:
        raise RuntimeError(
            f"{path}: expected a 2-D continuum image after squeezing; got "
            f"shape {data.shape}."
        )
    if not np.any(np.isfinite(data)):
        raise RuntimeError(f"{path}: image contains no finite pixels.")

    bunit = str(header.get("BUNIT", "")).upper().replace(" ", "")
    if "JY/BEAM" not in bunit and "JYBEAM" not in bunit:
        raise RuntimeError(
            f"{path}: expected BUNIT='Jy/beam'; got {header.get('BUNIT')!r}. "
            "Surface-brightness units are required so that integrated flux "
            "densities are well defined."
        )

    wcs = WCS(header).celestial
    if not wcs.has_celestial:
        raise RuntimeError(f"{path}: no usable celestial WCS.")

    beam = _sanitize_beam(beam_from_header(header), label_)
    freq = frequency_from_header(header)

    area_fraction = 1.0
    if rms_jybeam is None:
        rms_jybeam, rms_method, area_fraction = estimate_rms_jybeam(
            data, header, trust_header=trust_header_rms
        )
    else:
        rms_method = "user supplied"
    if not np.isfinite(rms_jybeam) or rms_jybeam <= 0:
        raise RuntimeError(f"{path}: invalid RMS={rms_jybeam}.")

    if cal_frac is None:
        cal_frac, cal_method = estimate_calibration_fraction(header)
    else:
        cal_method = "user supplied"
    if not np.isfinite(cal_frac) or not 0 <= cal_frac < 1:
        raise ValueError(f"{label_}: invalid calibration fraction={cal_frac}.")

    return RadioMap(
        label=label_, path=Path(path).resolve(), data=data, header=header,
        wcs=wcs, beam=beam, freq_hz=float(freq),
        rms_jybeam=float(rms_jybeam), rms_method=str(rms_method),
        cal_frac=float(cal_frac), cal_method=str(cal_method),
        rms_area_fraction=float(area_fraction),
    )


# ===========================================================================
# PART 4.  ASTROMETRIC REFERENCE, COMMON GRID, COMMON BEAM
# ===========================================================================

def choose_reference_map(
    maps: Sequence[RadioMap],
    explicit_label: Optional[str],
    astrometry_rms: Dict[str, float],
) -> RadioMap:
    """
    Pick the astrometric reference.

    Regridding does NOT repair absolute astrometry: it propagates whatever
    reference frame it is handed.  Choosing the reference by beam size (a
    tempting proxy) is wrong, because angular resolution and absolute
    positional accuracy are independent properties.  We therefore require the
    user to either name the reference or supply measured astrometric RMS
    values, rather than guessing.
    """
    if explicit_label:
        for m in maps:
            if m.label == explicit_label:
                return m
        raise RuntimeError(
            f"--reference-label {explicit_label!r} matches no input map. "
            f"Available: {', '.join(m.label for m in maps)}"
        )

    supplied = [
        m for m in maps
        if m.label in astrometry_rms
        and np.isfinite(astrometry_rms[m.label])
        and astrometry_rms[m.label] >= 0
    ]
    if supplied:
        return min(supplied, key=lambda m: astrometry_rms[m.label])

    raise RuntimeError(
        "No defensible astrometric reference was specified. Use "
        "--reference-label LABEL, or supply --astrometry-rms LABEL=ARCSEC for "
        "the maps whose absolute astrometry you have independently validated. "
        "Beam size is deliberately NOT used as an astrometric-quality proxy."
    )


def build_common_target(
    reference: RadioMap,
    maps: Sequence[RadioMap],
    reference_pixels: Optional[Tuple[int, int]] = None,
) -> Tuple[WCS, Tuple[int, int]]:
    """Common grid = the true sky-footprint intersection of every input map."""
    if reference_pixels is not None:
        nx, ny = reference_pixels
        return reference.wcs.celestial.deepcopy(), (ny, nx)

    ref_shape = reference.data.shape
    common = np.ones(ref_shape, dtype=bool)
    for m in maps:
        _, footprint = reproject_interp(
            (np.ones_like(m.data, dtype=float), m.wcs.celestial),
            reference.wcs.celestial,
            shape_out=ref_shape,
            order="nearest-neighbor",
            return_footprint=True,
        )
        common &= np.isfinite(footprint) & (footprint >= 0.99)

    yy, xx = np.where(common)
    if xx.size < 256:
        raise RuntimeError(
            "The true common sky footprint of the supplied maps is smaller "
            f"than 256 pixels ({xx.size}). Check that the cutouts overlap."
        )
    x0, x1 = int(xx.min()), int(xx.max()) + 1
    y0, y1 = int(yy.min()), int(yy.max()) + 1

    target = reference.wcs.celestial.deepcopy()
    target.wcs.crpix[0] -= x0
    target.wcs.crpix[1] -= y0
    return target, (y1 - y0, x1 - x0)


def _beam_deconvolution_is_valid(target: Beam, native: Beam) -> bool:
    try:
        target.deconvolve(native)
        return True
    except Exception:                                          # noqa: BLE001
        return False


def make_common_beam(maps: Sequence[RadioMap], verbose: bool = True) -> Beam:
    """
    Smallest beam every native beam can be convolved up to.

    radio_beam's optimised common beam is tried first, then VALIDATED against
    every native beam.  If any deconvolution fails (a real and common
    numerical failure mode of the quadratic solution near-degenerate beams),
    we fall back to a circular beam slightly larger than the largest native
    axis, which is guaranteed broader than every input ellipse and removes
    all position-angle dependence.
    """
    if not maps:
        raise RuntimeError("No maps supplied for common-beam determination.")

    clean = [_sanitize_beam(m.beam, m.label) for m in maps]

    try:
        candidate = _sanitize_beam(Beams(clean).common_beam(), "optimised common beam")
        if all(_beam_deconvolution_is_valid(candidate, b) for b in clean):
            return candidate
    except Exception:                                          # noqa: BLE001
        pass

    majors = np.asarray([b.major.to_value(u.arcsec) for b in clean], dtype=float)
    minors = np.asarray([b.minor.to_value(u.arcsec) for b in clean], dtype=float)
    max_fwhm = float(np.nanmax(np.maximum(majors, minors)))
    if not np.isfinite(max_fwhm) or max_fwhm <= 0:
        raise RuntimeError("Could not determine an enclosing common beam.")

    fallback = None
    for factor in (1.05, 1.10, 1.20, 1.35, 1.50, 2.00):
        trial = Beam(
            major=(factor * max_fwhm) * u.arcsec,
            minor=(factor * max_fwhm) * u.arcsec,
            pa=0.0 * u.deg,
        )
        if all(_beam_deconvolution_is_valid(trial, b) for b in clean):
            fallback = trial
            break

    if fallback is None:
        raise RuntimeError(
            "Could not construct a common restoring beam deconvolvable from "
            "every native beam. Native beams (arcsec): "
            + "; ".join(
                f"{m.label}: {b.major.to_value(u.arcsec):.3f}x"
                f"{b.minor.to_value(u.arcsec):.3f}@"
                f"{b.pa.to_value(u.deg):.1f}"
                for m, b in zip(maps, clean)
            )
        )

    if verbose:
        print(
            "\nNOTE: the optimised radio_beam common beam could not be "
            "deconvolved from every input beam. Using a conservative circular "
            f"common beam of {fallback.major.to_value(u.arcsec):.4f} arcsec."
        )
    return fallback


def normalized_nan_convolution(
    image: np.ndarray, kernel: np.ndarray, min_weight: float = 0.05
) -> np.ndarray:
    """
    NaN-aware convolution.

    Blanked pixels are treated as missing rather than as zeros; the result is
    renormalised by the convolved weight map so that flux is not artificially
    suppressed near image edges and blanked regions.  Pixels whose effective
    weight falls below `min_weight` are blanked, because there the result is
    dominated by extrapolation.
    """
    from scipy.signal import fftconvolve

    finite = np.isfinite(image)
    values = np.where(finite, image, 0.0)
    weights = finite.astype(float)

    conv_values = fftconvolve(values, kernel, mode="same")
    conv_weights = fftconvolve(weights, kernel, mode="same")

    out = np.full_like(image, np.nan, dtype=float)
    good = conv_weights > min_weight
    out[good] = conv_values[good] / conv_weights[good]
    return out


def regrid_and_common_beam(
    radio_map: RadioMap,
    target_wcs: WCS,
    target_shape: Tuple[int, int],
    common_beam: Beam,
) -> Tuple[np.ndarray, np.ndarray, fits.Header]:
    """
    Regrid, then convolve to the common beam, in that order.

    Order matters: convolving first and regridding second resamples an already
    smoothed image and mixes interpolation error into the beam, whereas
    regridding first keeps the convolution kernel exactly matched to the final
    pixel grid.

    Units: the data start in Jy per NATIVE beam.  Convolution conserves
    surface brightness per unit solid angle, so the conversion to Jy per
    COMMON beam is a multiplication by (Omega_common / Omega_native).
    """
    data, footprint = reproject_interp(
        (radio_map.data, radio_map.wcs.celestial),
        target_wcs,
        shape_out=target_shape,
        order="bilinear",
        return_footprint=True,
    )
    data = np.asarray(data, dtype=float)
    footprint = np.asarray(footprint, dtype=float)
    data[footprint < 0.99] = np.nan

    native_beam = _sanitize_beam(radio_map.beam, radio_map.label)
    target_beam = _sanitize_beam(common_beam, "common beam")
    try:
        kernel_beam = target_beam.deconvolve(native_beam)
    except Exception as exc:                                   # noqa: BLE001
        raise RuntimeError(
            f"{radio_map.label}: could not build the common-beam kernel. "
            f"native={native_beam.major.to_value(u.arcsec):.4f}x"
            f"{native_beam.minor.to_value(u.arcsec):.4f}@"
            f"{native_beam.pa.to_value(u.deg):.2f}; "
            f"target={target_beam.major.to_value(u.arcsec):.4f}x"
            f"{target_beam.minor.to_value(u.arcsec):.4f}@"
            f"{target_beam.pa.to_value(u.deg):.2f}"
        ) from exc

    pixscale_arcsec = float(np.mean(pixel_scales_arcsec(target_wcs)))
    if not np.isfinite(pixscale_arcsec) or pixscale_arcsec <= 0:
        raise RuntimeError(
            f"{radio_map.label}: invalid target pixel scale {pixscale_arcsec!r}."
        )
    pixscale = pixscale_arcsec * u.arcsec

    # radio_beam has moved the pixel-scale argument between positional and
    # keyword form across releases; try both before failing.
    try:
        kernel_obj = kernel_beam.as_kernel(pixscale)
    except TypeError:
        try:
            kernel_obj = kernel_beam.as_kernel(pixscale=pixscale)
        except TypeError as exc:
            raise RuntimeError(
                f"{radio_map.label}: incompatible radio_beam Beam.as_kernel() API."
            ) from exc

    kernel = np.asarray(kernel_obj.array, dtype=float)
    kernel_sum = float(np.sum(kernel))
    if not np.isfinite(kernel_sum) or kernel_sum <= 0:
        raise RuntimeError(f"{radio_map.label}: invalid convolution kernel.")
    kernel /= kernel_sum

    smoothed = normalized_nan_convolution(data, kernel)
    smoothed *= float((common_beam.sr / native_beam.sr).decompose().value)

    header = target_wcs.to_header(relax=True)
    header["NAXIS"] = 2
    header["NAXIS1"] = target_shape[1]
    header["NAXIS2"] = target_shape[0]
    header["BUNIT"] = "Jy/beam"
    header["BMAJ"] = common_beam.major.to_value(u.deg)
    header["BMIN"] = common_beam.minor.to_value(u.deg)
    header["BPA"] = common_beam.pa.to_value(u.deg)
    header["RESTFRQ"] = radio_map.freq_hz
    header["OBJECT"] = (
        str(radio_map.header.get("OBJECT", "")).strip() or "RADIO_SOURCE"
    )
    for key in ("TELESCOP", "INSTRUME", "DATE-OBS", "OBSERVAT"):
        if key in radio_map.header:
            header[key] = radio_map.header[key]
    header["HISTORY"] = (
        f"PySynch v{__version__}: regridded then convolved to common beam"
    )
    return smoothed, footprint, header


def write_fits_map(path: Path, data: np.ndarray, header: fits.Header) -> None:
    arr = np.asarray(data)
    if arr.ndim != 2:
        raise ValueError(f"Only 2-D maps can be written; got {arr.shape}.")
    arr = arr.astype(np.float32) if arr.dtype.kind == "f" else arr

    h = header.copy()
    h["NAXIS"] = 2
    h["NAXIS1"] = arr.shape[1]
    h["NAXIS2"] = arr.shape[0]
    for key in list(h.keys()):
        ukey = str(key).upper()
        if ukey in {"NAXIS3", "NAXIS4"} or re.fullmatch(r"NAXIS[5-9][0-9]*", ukey):
            try:
                del h[key]
            except Exception:                                  # noqa: BLE001
                pass
    fits.PrimaryHDU(arr, h).writeto(path, overwrite=True, output_verify="fix")


def product_header(
    common_header: fits.Header, bunit: str, beam: Optional[Beam] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> fits.Header:
    h = common_header.copy()
    h["NAXIS"] = 2
    if beam is not None:
        h["BMAJ"] = beam.major.to_value(u.deg)
        h["BMIN"] = beam.minor.to_value(u.deg)
        h["BPA"] = beam.pa.to_value(u.deg)
    h["BUNIT"] = bunit
    h["PYSYNCH"] = (__version__, "PySynch pipeline version")
    if extra:
        for key, value in extra.items():
            try:
                h[key[:8].upper()] = value
            except Exception:                                  # noqa: BLE001
                pass
    for key in list(h.keys()):
        ukey = str(key).upper()
        if ukey in {"NAXIS3", "NAXIS4"} or re.fullmatch(r"NAXIS[5-9][0-9]*", ukey):
            try:
                del h[key]
            except Exception:                                  # noqa: BLE001
                pass
    return h


def save_fits(path: Path, data: np.ndarray, header: fits.Header) -> None:
    write_fits_map(path, data, header)


# ===========================================================================
# PART 5.  SEGMENTATION
# ===========================================================================

def beam_area_pixels(common_beam: Beam, wcs: WCS) -> float:
    return beam_area_arcsec2(common_beam) / pixel_area_arcsec2(wcs)


def hysteresis_source_mask(
    snr_map: np.ndarray,
    high_sigma: float = 5.0,
    low_sigma: float = 3.0,
    fill_holes: bool = True,
    single_target: bool = True,
) -> np.ndarray:
    """
    Statistically secure seeds at `high_sigma`, grown through connected
    emission down to `low_sigma`.

    A plain N-sigma cut either truncates diffuse emission (high N) or picks up
    noise islands (low N).  Hysteresis keeps the false-positive rate set by
    the seed threshold while recovering the low-surface-brightness extent that
    dominates the volume and hence the equipartition field.

    `single_target=True` (the default, and correct for this pipeline: every
    input is a small cutout built around ONE source) additionally keeps only
    the connected component closest to the image centre. Without it, on any
    field crowded enough that a neighbour also clears `high_sigma` somewhere
    within itself -- not a rare situation once the detection map is chosen
    for depth rather than frequency -- every such neighbour is kept too, and
    integrated photometry silently includes their flux. Measured on
    J021926-051535: switching the detection map to the deepest band (MeerKAT,
    peak/rms ~ 700) grew the mask from 188 to 489 pixels across THREE
    disconnected components, of which only one -- 241 px, centred within a
    pixel of the known target -- was the source; the other two were
    independent field sources 15-30 pixels away that each happened to clear
    5 sigma on their own.
    """
    if low_sigma >= high_sigma:
        raise ValueError("low_sigma must be strictly smaller than high_sigma.")

    high = np.isfinite(snr_map) & (snr_map >= high_sigma)
    low = np.isfinite(snr_map) & (snr_map >= low_sigma)
    if not np.any(high):
        return np.zeros_like(snr_map, dtype=bool)

    labels, ncomp = label(low, structure=np.ones((3, 3), dtype=int))
    keep = np.unique(labels[high])
    keep = keep[keep > 0]

    if single_target and keep.size > 1:
        cy, cx = (snr_map.shape[0] - 1) / 2.0, (snr_map.shape[1] - 1) / 2.0
        best_label, best_d2 = None, np.inf
        for lb in keep:
            yy, xx = np.where(labels == lb)
            d2 = (yy.mean() - cy) ** 2 + (xx.mean() - cx) ** 2
            if d2 < best_d2:
                best_d2, best_label = d2, lb
        keep = np.array([best_label])

    mask = np.isin(labels, keep)

    if fill_holes:
        # Interior holes in a detected lobe are noise dips, not real cavities
        # at this significance; leaving them out would bias the volume low.
        mask = binary_fill_holes(mask)
    return mask


def build_detection_map(common_maps: Sequence[RadioMap]) -> Tuple[np.ndarray, float]:
    """
    Detection image = the lowest-frequency common-beam map.

    For optically thin synchrotron emission the low-frequency map traces the
    oldest, most extended plasma and therefore gives the most complete
    morphology; using a high-frequency map would systematically truncate the
    aged outer regions that matter most for the volume.
    """
    ref = min(common_maps, key=lambda m: m.freq_hz)
    return ref.data / ref.rms_jybeam, float(ref.rms_jybeam)


def all_band_coverage(common_maps: Sequence[RadioMap]) -> np.ndarray:
    """Pixels with finite data in EVERY band (regridding footprint intersection)."""
    cover = np.ones_like(common_maps[0].data, dtype=bool)
    for m in common_maps:
        cover &= np.isfinite(m.data)
    return cover


# ===========================================================================
# PART 6.  GEOMETRY AND SOURCE VOLUME
# ===========================================================================
#
# The volume is the single largest lever on the equipartition field:
#
#       B_eq  proportional to  V^(-2/7)
#
# so a factor-2 error in V moves B_eq by only ~18%, but it moves the total
# energy  E = u V  almost linearly.  It is therefore worth estimating V
# several independent ways and carrying the spread as a systematic rather
# than committing to one idealised solid.
# ---------------------------------------------------------------------------

def pixel_direction_to_sky_pa(wcs: WCS, dx: float, dy: float) -> float:
    """
    Position angle (deg, North through East) of a pixel-frame direction.

    Uses the WCS linear transformation so that image rotation, flips and
    non-square pixels are all handled, rather than assuming the image is
    north-up/east-left.
    """
    cd = np.asarray(wcs.celestial.pixel_scale_matrix, dtype=float)
    east = cd[0, 0] * dx + cd[0, 1] * dy
    north = cd[1, 0] * dx + cd[1, 1] * dy
    return float(np.degrees(np.arctan2(east, north)) % 180.0)


def beam_extent_arcsec(beam: Beam, sky_pa_deg: float) -> float:
    """
    FWHM of an elliptical Gaussian beam measured along a given sky position
    angle:  theta(psi) = a b / sqrt((b cos d)^2 + (a sin d)^2),  d = psi - BPA.
    """
    a = float(beam.major.to_value(u.arcsec))
    b = float(beam.minor.to_value(u.arcsec))
    d = math.radians(float(sky_pa_deg) - float(beam.pa.to_value(u.deg)))
    denom = math.hypot(b * math.cos(d), a * math.sin(d))
    if denom <= 0:
        return a
    return a * b / denom


def deconvolve_extent(observed: float, beam_extent: float, mode: str) -> float:
    """
    Remove the beam contribution from a measured angular extent.

    Two conventions are in common use and they disagree by tens of per cent
    for marginally resolved structure:

      'quadrature'  sqrt(d^2 - theta_b^2)   -- exact for a Gaussian component
                                               convolved with a Gaussian beam
      'linear'      d - theta_b             -- appropriate for a sharp-edged
                                               (top-hat) source, whose
                                               isophotal extent grows roughly
                                               linearly with beam size

    A radio lobe is neither, so v2 evaluates both and treats the difference as
    a genuine geometric systematic instead of silently picking one.
    """
    d = float(observed)
    t = float(beam_extent)
    if mode == "quadrature":
        return math.sqrt(max(d * d - t * t, 0.0))
    if mode == "linear":
        return max(d - t, 0.0)
    raise ValueError(f"Unknown deconvolution mode {mode!r}")


def principal_axes(mask: np.ndarray, wcs: WCS) -> Dict[str, Any]:
    """Second-moment principal axes of the mask, in arcsec and sky PA."""
    yy, xx = np.where(mask)
    if xx.size < 20:
        raise RuntimeError(
            f"Only {xx.size} pixels in the source mask; geometry is unreliable. "
            "Lower --source-sigma or check the input images."
        )

    sx, sy = pixel_scales_arcsec(wcs)
    x0, y0 = float(np.mean(xx)), float(np.mean(yy))
    coords = np.column_stack(((xx - x0) * sx, (yy - y0) * sy))
    cov = np.cov(coords.T)
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[::-1]
    vals = vals[order]
    vecs = vecs[:, order]

    major_vec, minor_vec = vecs[:, 0], vecs[:, 1]
    pix_diag = math.sqrt(sx * sy)
    major_obs = float(np.ptp(coords @ major_vec) + pix_diag)
    minor_obs = float(np.ptp(coords @ minor_vec) + pix_diag)

    # eigenvector components are in arcsec-scaled space; convert back to pixels
    major_pa = pixel_direction_to_sky_pa(wcs, major_vec[0] / sx, major_vec[1] / sy)
    minor_pa = pixel_direction_to_sky_pa(wcs, minor_vec[0] / sx, minor_vec[1] / sy)

    return {
        "centroid_pixel": (x0, y0),
        "major_vec_pixel": (float(major_vec[0] / sx), float(major_vec[1] / sy)),
        "minor_vec_pixel": (float(minor_vec[0] / sx), float(minor_vec[1] / sy)),
        "major_obs_arcsec": major_obs,
        "minor_obs_arcsec": minor_obs,
        "major_pa_deg": major_pa,
        "minor_pa_deg": minor_pa,
        "axis_ratio": major_obs / max(minor_obs, 1e-12),
        "area_arcsec2": float(np.count_nonzero(mask) * sx * sy),
        "n_pixels": int(np.count_nonzero(mask)),
        "rms_major_arcsec": float(math.sqrt(max(vals[0], 0.0))),
        "rms_minor_arcsec": float(math.sqrt(max(vals[1], 0.0))),
    }


def integrated_volume(
    mask: np.ndarray,
    wcs: WCS,
    beam: Beam,
    along: str = "major",
    deconv: str = "quadrature",
    axes: Optional[Dict[str, Any]] = None,
) -> Dict[str, float]:
    """
    Volume by integrating a circular cross-section along a principal axis.

        V = sum over pixels  A_pixel * (pi/4) * d_dec^2 / d_obs

    which is algebraically identical to marching along the chosen axis and
    summing  pi (d_dec/2)^2 dx  over every connected transverse segment.

    Working per SEGMENT rather than per column is what makes this correct for
    doubles, X-shapes, ring-like remnants and wide-angle tails, where a single
    column can cut two physically separate lobes: a naive column width would
    fill in the gap between them.  The ray-trace implementation
    (`local_width_map`) gets this for free, because a ray that leaves the mask
    stops.

    The assumption -- unavoidable without polarisation or X-ray depth
    information -- is that the line-of-sight depth of each structure equals
    its measured transverse width.  That is the standard choice in the
    literature and is restated in every output file.

    `along='major'` integrates along the major axis and measures widths across
    the minor axis (the usual choice for a jet, tail or lobe pair);
    `along='minor'` does the reverse and is retained as an independent
    realisation for the error budget.
    """
    axes = axes or principal_axes(mask, wcs)
    if along == "major":
        direction = axes["minor_vec_pixel"]
        width_pa = axes["minor_pa_deg"]
    elif along == "minor":
        direction = axes["major_vec_pixel"]
        width_pa = axes["major_pa_deg"]
    else:
        raise ValueError("`along` must be 'major' or 'minor'.")

    widths = local_width_map(
        mask, wcs, beam, direction_pixel=direction,
        direction_sky_pa_deg=width_pa, deconv=deconv,
    )
    depth = widths["depth_arcsec"]
    finite = np.isfinite(depth)
    volume = float(np.sum(depth[finite]) * pixel_area_arcsec2(wcs))

    obs = widths["width_obs_arcsec"]
    obs_finite = obs[np.isfinite(obs)]
    theta_beam = float(widths["beam_extent_arcsec"])
    return {
        "volume_arcsec3": volume,
        "along": along,
        "deconvolution": deconv,
        "beam_extent_arcsec": theta_beam,
        "n_pixels": int(np.count_nonzero(finite)),
        "median_segment_width_arcsec": (
            float(np.median(obs_finite)) if obs_finite.size else float("nan")
        ),
        "max_segment_width_arcsec": (
            float(np.max(obs_finite)) if obs_finite.size else float("nan")
        ),
        "unresolved_fraction": (
            float(np.mean(obs_finite < theta_beam)) if obs_finite.size else 1.0
        ),
    }


def classify_geometry(
    axes: Dict[str, Any], beam: Beam, wcs: WCS, deconv: str = "quadrature"
) -> Dict[str, Any]:
    """
    Assign one of pysynch's idealised solids, for comparison with the
    literature and as a cross-check on the integrated volume.
    """
    sx, sy = pixel_scales_arcsec(wcs)
    pix = math.sqrt(sx * sy)

    major_obs = axes["major_obs_arcsec"]
    minor_obs = axes["minor_obs_arcsec"]
    theta_major = beam_extent_arcsec(beam, axes["major_pa_deg"])
    theta_minor = beam_extent_arcsec(beam, axes["minor_pa_deg"])

    major_dec = deconvolve_extent(major_obs, theta_major, deconv)
    minor_dec = deconvolve_extent(minor_obs, theta_minor, deconv)

    major_resolved = major_obs >= 1.5 * theta_major
    minor_resolved = minor_obs >= 1.5 * theta_minor
    axis_ratio = major_obs / max(minor_obs, 1e-12)

    common = {
        "major_obs_arcsec": major_obs,
        "minor_obs_arcsec": minor_obs,
        "major_dec_arcsec": major_dec,
        "minor_dec_arcsec": minor_dec,
        "beam_extent_major_arcsec": theta_major,
        "beam_extent_minor_arcsec": theta_minor,
        "major_resolved": bool(major_resolved),
        "minor_resolved": bool(minor_resolved),
        "axis_ratio": float(axis_ratio),
        "area_arcsec2": axes["area_arcsec2"],
        "major_pa_deg": axes["major_pa_deg"],
    }

    area = axes["area_arcsec2"]
    r_area_obs = math.sqrt(area / math.pi)

    if major_dec <= 0 or minor_dec <= 0 or axis_ratio < 1.25:
        r_dec = max(
            deconvolve_extent(2.0 * r_area_obs, 0.5 * (theta_major + theta_minor), deconv) / 2.0,
            pix / 2.0,
        )
        return {
            **common,
            "geometry": "sphere",
            "radius_arcsec": float(r_dec),
            "radius_obs_arcsec": float(r_area_obs),
            "volume_arcsec3": float(4.0 * math.pi * r_dec ** 3 / 3.0),
            "resolved": bool(major_resolved and minor_resolved),
        }

    if axis_ratio >= 2.2 and minor_resolved:
        length = major_dec
        radius = max(minor_dec / 2.0, pix / 2.0)
        return {
            **common,
            "geometry": "cylinder",
            "length_arcsec": float(length),
            "radius_arcsec": float(radius),
            "volume_arcsec3": float(math.pi * radius ** 2 * length),
            "resolved": bool(major_resolved and minor_resolved),
        }

    return {
        **common,
        "geometry": "ellipsoid",
        "major_arcsec": float(major_dec),
        "minor_arcsec": float(minor_dec),
        # pysynch prolate spheroid: V = pi * minor^2 * major / 6 with FULL axes
        "volume_arcsec3": float(math.pi * minor_dec ** 2 * major_dec / 6.0),
        "resolved": bool(major_resolved and minor_resolved),
        "note": (
            "Elongated but the minor axis is not well enough resolved to "
            "justify a cylinder."
        ),
    }


def arcsec3_to_m3(volume_arcsec3: float, redshift: float,
                  cosmology=DEFAULT_COSMOLOGY) -> float:
    """Convert a projected-angular volume to proper metres^3 at the source."""
    kpc_per_arcsec = float(
        (1.0 / cosmology.arcsec_per_kpc_proper(redshift)).value
    )
    m_per_arcsec = kpc_per_arcsec * 1000.0 * 3.0856775814913673e16
    return float(volume_arcsec3) * m_per_arcsec ** 3


def build_geometry(
    mask: np.ndarray,
    common_beam: Beam,
    wcs: WCS,
    redshift: float,
    threshold_masks: Optional[Sequence[np.ndarray]] = None,
    filling_factor: float = DEFAULT_FILLING_FACTOR,
    cosmology=DEFAULT_COSMOLOGY,
) -> Dict[str, Any]:
    """
    Full geometry description with FOUR independent volume estimates and a
    data-driven geometric uncertainty.

    The uncertainty is built from three measurable contributions rather than a
    hardcoded percentage:

      * deconvolution convention   (quadrature vs linear)
      * integration axis           (along major vs along minor)
      * segmentation threshold     (volumes recomputed on `threshold_masks`)

    These are combined as the standard deviation of ln V over all realisations,
    which is the quantity the Monte Carlo actually needs.
    """
    axes = principal_axes(mask, wcs)
    kpc_per_arcsec = float((1.0 / cosmology.arcsec_per_kpc_proper(redshift)).value)

    realisations: Dict[str, float] = {}
    diagnostics: Dict[str, Any] = {}

    # Which integration axes are physically defensible?
    #
    # Integrating ALONG the major axis (widths measured across the minor axis)
    # assumes the line-of-sight depth equals the transverse width -- the
    # standard choice.  Integrating along the MINOR axis instead assumes the
    # depth equals the full major-axis length, which for a jet or a tail means
    # claiming a 120 kpc line-of-sight depth for a 20 kpc wide structure.  For
    # an elongated source that is not an alternative hypothesis, it is simply
    # wrong, and including it would inflate the quoted geometric error by
    # nearly an order of magnitude for no physical reason.  It is therefore
    # kept as a realisation only when the source is close to round, where the
    # choice of axis genuinely is ambiguous.
    axis_ratio = float(axes["axis_ratio"])
    along_options = ["major"]
    if axis_ratio < 1.5:
        along_options.append("minor")

    for along in along_options:
        for deconv in ("quadrature", "linear"):
            res = integrated_volume(
                mask, wcs, common_beam, along=along, deconv=deconv, axes=axes
            )
            key = f"integrated_{along}_{deconv}"
            realisations[key] = res["volume_arcsec3"]
            diagnostics[key] = res

    solids: Dict[str, Dict[str, Any]] = {}
    for deconv in ("quadrature", "linear"):
        solid = classify_geometry(axes, common_beam, wcs, deconv=deconv)
        solids[deconv] = solid
        realisations[f"solid_{solid['geometry']}_{deconv}"] = solid["volume_arcsec3"]

    threshold_volumes: List[float] = []
    if threshold_masks:
        for i, alt in enumerate(threshold_masks):
            if alt is None or not np.any(alt):
                continue
            try:
                alt_axes = principal_axes(alt, wcs)
                res = integrated_volume(
                    alt, wcs, common_beam, along="major",
                    deconv="quadrature", axes=alt_axes,
                )
                threshold_volumes.append(res["volume_arcsec3"])
                realisations[f"threshold_variant_{i}"] = res["volume_arcsec3"]
            except Exception:                                  # noqa: BLE001
                continue

    #: Adopted volume: the integrated estimator along the major axis with
    #: quadrature deconvolution.  This is the estimator that makes the fewest
    #: assumptions about the 3-D shape while still respecting the beam.
    adopted_arcsec3 = realisations["integrated_major_quadrature"]

    values = np.asarray(
        [v for v in realisations.values() if np.isfinite(v) and v > 0], dtype=float
    )
    if values.size >= 2:
        ln_values = np.log(values)
        ln_sigma = float(np.std(ln_values, ddof=1))
    else:
        ln_sigma = 0.30                      # ~35% if nothing else is measurable
    # A floor: no volume of an irregular radio source is known to better than
    # ~20%, and pretending otherwise produces spuriously tight B_eq errors.
    ln_sigma = float(max(ln_sigma, 0.20))
    frac_error = float(math.expm1(ln_sigma))

    volume_m3 = arcsec3_to_m3(adopted_arcsec3, redshift, cosmology) * float(filling_factor)

    geometry = {
        "adopted_volume_arcsec3": float(adopted_arcsec3),
        "adopted_volume_m3": float(volume_m3),
        "adopted_volume_kpc3": float(adopted_arcsec3 * kpc_per_arcsec ** 3
                                     * filling_factor),
        "filling_factor": float(filling_factor),
        "volume_estimator": "integrated_major_quadrature",
        "volume_realisations_arcsec3": {k: float(v) for k, v in realisations.items()},
        "volume_ln_sigma": ln_sigma,
        "volume_fractional_error": frac_error,
        "n_threshold_variants": len(threshold_volumes),
        "integration_axes_used": along_options,
        "integration_axis_note": (
            "Only the major axis is integrated for elongated sources "
            "(axis ratio >= 1.5): integrating along the minor axis would "
            "assume a line-of-sight depth equal to the full source length, "
            "which is not a physically plausible alternative for a jet, tail "
            "or lobe pair."
        ),
        "kpc_per_arcsec": kpc_per_arcsec,
        "projected_area_arcsec2": float(axes["area_arcsec2"]),
        "projected_area_kpc2": float(axes["area_arcsec2"] * kpc_per_arcsec ** 2),
        "largest_angular_size_arcsec": float(axes["major_obs_arcsec"]),
        "largest_linear_size_kpc": float(axes["major_obs_arcsec"] * kpc_per_arcsec),
        "minor_axis_arcsec": float(axes["minor_obs_arcsec"]),
        "minor_axis_kpc": float(axes["minor_obs_arcsec"] * kpc_per_arcsec),
        "axis_ratio": float(axes["axis_ratio"]),
        "major_pa_deg": float(axes["major_pa_deg"]),
        "n_pixels": int(axes["n_pixels"]),
        "beams_per_source": float(
            axes["n_pixels"] / max(beam_area_pixels(common_beam, wcs), 1e-12)
        ),
        "solid_model": solids["quadrature"],
        "solid_model_linear": solids["linear"],
        "integration_diagnostics": diagnostics,
        "assumption_note": (
            "The line-of-sight depth of each transverse segment is assumed "
            "equal to its measured (beam-deconvolved) plane-of-sky width. "
            "This is a modelling assumption, not a measurement, and it is the "
            "dominant systematic on the total energy content."
        ),
        # pysynch-compatible geometry keys, retained so the classic
        # sphere/ellipsoid/cylinder pathway remains available
        **{
            k: v for k, v in solids["quadrature"].items()
            if k.endswith("_arcsec") or k in {"geometry", "resolved"}
        },
    }
    geometry["geometry_error_fraction"] = frac_error
    geometry["geometry_confidence"] = (
        "high" if frac_error < 0.30 else "medium" if frac_error < 0.60 else "low"
    )
    return geometry


def geometry_physical_summary(geometry: Dict[str, Any], redshift: float) -> Dict[str, Any]:
    """Compact, human-facing physical summary of the adopted geometry."""
    return {
        "geometry_model": geometry.get("geometry"),
        "volume_estimator": geometry.get("volume_estimator"),
        "kpc_per_arcsec": geometry.get("kpc_per_arcsec"),
        "volume_m3": geometry.get("adopted_volume_m3"),
        "volume_kpc3": geometry.get("adopted_volume_kpc3"),
        "volume_fractional_error": geometry.get("volume_fractional_error"),
        "filling_factor": geometry.get("filling_factor"),
        "largest_linear_size_kpc": geometry.get("largest_linear_size_kpc"),
        "minor_axis_kpc": geometry.get("minor_axis_kpc"),
        "projected_area_kpc2": geometry.get("projected_area_kpc2"),
        "axis_ratio": geometry.get("axis_ratio"),
        "beams_per_source": geometry.get("beams_per_source"),
        "assumption_note": geometry.get("assumption_note"),
    }


# ===========================================================================
# PART 7.  INTEGRATED PHOTOMETRY
# ===========================================================================

def measure_background_offset(
    radio_map: RadioMap,
    source_mask: np.ndarray,
    inner_growth: int = 6,
    outer_growth: int = 18,
) -> Dict[str, float]:
    """
    Residual zero-level offset in an annulus around the source.

    For a source covering N beams, an unrecognised background offset b
    contributes b*N to the integrated flux -- a term that grows linearly with
    source size while the thermal term grows only as sqrt(N).  For the large,
    low-surface-brightness structures this pipeline targets, the zero level is
    frequently the DOMINANT flux uncertainty, so it is measured rather than
    assumed to be zero.
    """
    structure = np.ones((3, 3), dtype=bool)
    inner = binary_dilation(source_mask, structure=structure, iterations=int(inner_growth))
    outer = binary_dilation(source_mask, structure=structure, iterations=int(outer_growth))
    annulus = outer & (~inner) & np.isfinite(radio_map.data)

    values = radio_map.data[annulus]
    values = values[np.isfinite(values)]
    if values.size < 100:
        return {
            "background_jybeam": 0.0,
            "background_err_jybeam": float(radio_map.rms_jybeam),
            "n_pixels": int(values.size),
            "method": "insufficient annulus pixels; offset assumed zero",
        }

    clipped = _iterative_sigma_clip(values, nsig=3.0, iterations=6)
    if clipped.size < 50:
        clipped = values
    level = float(np.nanmedian(clipped))

    beam_pix = beam_area_pixels(
        _sanitize_beam(radio_map.beam, radio_map.label), radio_map.wcs
    )
    n_indep = max(clipped.size / max(beam_pix, 1.0), 1.0)
    level_err = float(radio_map.rms_jybeam / math.sqrt(n_indep))

    return {
        "background_jybeam": level,
        "background_err_jybeam": level_err,
        "n_pixels": int(clipped.size),
        "n_independent_beams": float(n_indep),
        "method": f"sigma-clipped median of a {inner_growth}-{outer_growth} pixel annulus",
    }


def integrated_flux_for_mask(
    radio_map: RadioMap,
    mask: np.ndarray,
    background: Optional[Dict[str, float]] = None,
    subtract_background: bool = True,
) -> Dict[str, float]:
    """
    Integrate a mask and build a defensible error budget.

        S = sum(I_i) * Omega_pix / Omega_beam

        sigma^2 = N_beam * rms^2            thermal / deconvolution noise
                + (N_beam * sigma_b)^2      residual zero-level (COHERENT)
                + (f_cal * S)^2             absolute flux scale (COHERENT)

    A negative integrated flux at a high frequency is a legitimate outcome of
    a non-detection and is returned as-is; it is never clipped to a positive
    floor, which would manufacture a spurious spectral turnover.  The caller
    decides via `flux_is_detection` whether the point may enter a fit.
    """
    beam = _sanitize_beam(radio_map.beam, radio_map.label)
    good = mask & np.isfinite(radio_map.data)
    npix = int(np.count_nonzero(good))
    if npix < 5:
        raise RuntimeError(
            f"{radio_map.label}: only {npix} finite pixels inside the source "
            "mask; no flux density can be measured."
        )

    pix_area = pixel_area_arcsec2(radio_map.wcs)
    beam_area = beam_area_arcsec2(beam)
    pix_per_beam = beam_area / pix_area
    nbeam = npix / pix_per_beam

    values = radio_map.data[good]
    bkg = background or {"background_jybeam": 0.0,
                         "background_err_jybeam": 0.0,
                         "method": "none"}
    offset = float(bkg["background_jybeam"]) if subtract_background else 0.0
    offset_err = float(bkg.get("background_err_jybeam", 0.0))

    flux = float(np.nansum(values - offset)) * pix_area / beam_area

    sigma_thermal = radio_map.rms_jybeam * math.sqrt(max(nbeam, 1.0))
    sigma_zero = nbeam * offset_err
    sigma_cal = radio_map.cal_frac * abs(flux)
    sigma_total = math.sqrt(sigma_thermal ** 2 + sigma_zero ** 2 + sigma_cal ** 2)

    return {
        "flux_Jy": float(flux),
        "flux_err_Jy": float(sigma_total),
        "flux_err_stat_Jy": float(math.hypot(sigma_thermal, sigma_zero)),
        "flux_err_thermal_Jy": float(sigma_thermal),
        "flux_err_zerolevel_Jy": float(sigma_zero),
        "flux_err_cal_Jy": float(sigma_cal),
        "nbeam": float(nbeam),
        "npix": int(npix),
        "background_subtracted_Jy_beam": float(offset),
        "peak_Jy_beam": float(np.nanmax(values)) if values.size else float("nan"),
        "peak_snr": float(np.nanmax(values) / radio_map.rms_jybeam)
        if values.size else float("nan"),
    }


def flux_is_detection(flux_jy: float, flux_err_jy: float, min_snr: float = 3.0) -> bool:
    """A flux point is usable only if it is a significant positive detection."""
    if not np.isfinite(flux_jy) or not np.isfinite(flux_err_jy) or flux_err_jy <= 0:
        return False
    return bool(flux_jy > 0.0 and flux_jy / flux_err_jy >= min_snr)


def kcorrection_ln_sigma(redshift: float, alpha_err: Optional[float]) -> float:
    """
    Fractional uncertainty the K-correction imposes on a rest-frame quantity.

    L_nu = 4 pi D_L^2 S_nu / (1+z)^(1+alpha), so d(ln L)/d(alpha) = -ln(1+z)
    and the spectral-index error enters as ln(1+z) * sigma_alpha.

    This is NOT a small correction at high redshift and it is easy to forget.
    For J021926-051535 at z = 1.47 with sigma_alpha = 0.32 it comes to 28.5
    per cent, against flux errors of 5-10 per cent -- so it dominates the
    error budget of every luminosity and emissivity point on the plot.

    It is a CORRELATED systematic, not per-point scatter: one alpha is used
    for every band, so it slides the whole curve up or down together rather
    than moving points relative to one another. It must therefore be quoted
    separately from the error bars, never folded into them, or a reader will
    take it for independent noise and under-weight a real trend.
    """
    if alpha_err is None or not np.isfinite(alpha_err) or alpha_err <= 0:
        return 0.0
    return float(math.log1p(float(redshift)) * float(alpha_err))

def rest_lnu_from_flux(
    flux_jy: np.ndarray, redshift: float, alpha: float = -0.7,
    cosmology=DEFAULT_COSMOLOGY, k_correct: bool = True,
) -> np.ndarray:
    """
    Monochromatic rest-frame luminosity density.

        L_nu = 4 pi D_L^2 S_nu / (1+z)^(1+alpha)

    The (1+z)^(1+alpha) factor is the standard radio K-correction.  v1 used
    (1+z)^1, i.e. implicitly alpha = 0, which for a typical alpha = -0.7 source
    at z = 0.15 mis-scales the luminosity by ~10%.  Set k_correct=False to
    reproduce the old behaviour.
    """
    dl_m = float(cosmology.luminosity_distance(redshift).to_value(u.m))
    s_si = np.asarray(flux_jy, dtype=float) * JANSKY
    exponent = (1.0 + float(alpha)) if k_correct else 1.0
    return 4.0 * math.pi * dl_m ** 2 * s_si / (1.0 + redshift) ** exponent


# ===========================================================================
# PART 8.  POWER-LAW AND CURVATURE FITTING
# ===========================================================================

def weighted_logpoly(
    nu_hz: np.ndarray,
    flux_jy: np.ndarray,
    err_jy: np.ndarray,
    degree: int = 1,
    pivot_hz: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Weighted least squares of  ln S  against  ln(nu/nu_pivot).

    degree=1 gives the spectral index; degree=2 additionally gives the
    curvature  c = d^2 ln S / d(ln nu)^2, which is the model-independent
    signature of spectral ageing (c < 0) or of a self-absorbed / injected
    component (c > 0).

    Pivoting at the weighted mean log-frequency decorrelates the normalisation
    from the slope, so the reported slope error is not inflated by the choice
    of reference frequency.
    """
    nu = np.asarray(nu_hz, dtype=float)
    f = np.asarray(flux_jy, dtype=float)
    e = np.asarray(err_jy, dtype=float)

    good = (
        np.isfinite(nu) & (nu > 0) & np.isfinite(f) & (f > 0)
        & np.isfinite(e) & (e > 0)
    )
    n = int(np.count_nonzero(good))
    if n < degree + 1:
        raise RuntimeError(
            f"Need at least {degree + 1} positive SED points for a "
            f"degree-{degree} log-log fit; have {n}."
        )

    x_raw = np.log(nu[good])
    y = np.log(f[good])
    # sigma of ln S: exact first-order propagation, adequate for S/N >~ 3
    sy = e[good] / f[good]
    w = 1.0 / np.maximum(sy ** 2, 1e-300)

    if pivot_hz is None:
        pivot = float(np.exp(np.sum(w * x_raw) / np.sum(w)))
    else:
        pivot = float(pivot_hz)
    x = x_raw - math.log(pivot)

    design = np.vander(x, degree + 1, increasing=True)   # [1, x, x^2, ...]
    wd = design * w[:, None]
    ata = design.T @ wd
    atb = wd.T @ y

    try:
        cov = np.linalg.inv(ata)
    except np.linalg.LinAlgError as exc:
        raise RuntimeError(
            "Frequency coverage is degenerate for this polynomial degree."
        ) from exc
    coeffs = cov @ atb

    model = design @ coeffs
    residual = y - model
    chi2 = float(np.sum(w * residual ** 2))
    dof = n - (degree + 1)
    errs = np.sqrt(np.maximum(np.diag(cov), 0.0))

    out: Dict[str, Any] = {
        "n_points": n,
        "degree": int(degree),
        "pivot_Hz": pivot,
        "coefficients": coeffs.tolist(),
        "coefficient_errors": errs.tolist(),
        "covariance": cov.tolist(),
        "chi2": chi2,
        "dof": int(dof),
        "reduced_chi2": float(chi2 / dof) if dof > 0 else float("nan"),
        "ln_normalisation": float(coeffs[0]),
        "flux_at_pivot_Jy": float(math.exp(coeffs[0])),
        "alpha": float(coeffs[1]),
        "alpha_err": float(errs[1]),
        "alpha_positive": float(-coeffs[1]),
        "residuals_sigma": (residual * np.sqrt(w)).tolist(),
    }
    if degree >= 2:
        # ln S = a0 + a1 x + a2 x^2  ->  d2 lnS / dx2 = 2 a2
        out["curvature"] = float(2.0 * coeffs[2])
        out["curvature_err"] = float(2.0 * errs[2])
        out["curvature_significance"] = float(
            abs(2.0 * coeffs[2]) / max(2.0 * errs[2], 1e-30)
        )
    return out


def weighted_powerlaw(
    nu_hz: np.ndarray, flux_jy: np.ndarray, err_jy: np.ndarray
) -> Tuple[float, float, float, int]:
    """Backwards-compatible thin wrapper: (alpha, alpha_err, chi2_red, n)."""
    res = weighted_logpoly(nu_hz, flux_jy, err_jy, degree=1)
    return (
        res["alpha"], res["alpha_err"],
        res["reduced_chi2"] if np.isfinite(res["reduced_chi2"]) else 0.0,
        res["n_points"],
    )


def bootstrap_alpha(
    nu_hz: np.ndarray, flux_jy: np.ndarray, err_jy: np.ndarray,
    n: int, seed: int, degree: int = 1,
) -> Dict[str, float]:
    """
    Parametric bootstrap of the log-log slope.

    Resampling the fluxes rather than the log-fluxes is important: at modest
    S/N the log transform is noticeably non-Gaussian, and a purely analytic
    log-space error underestimates the true uncertainty and is biased.
    """
    rng = np.random.default_rng(int(seed))
    nu = np.asarray(nu_hz, dtype=float)
    f = np.asarray(flux_jy, dtype=float)
    e = np.asarray(err_jy, dtype=float)

    values: List[float] = []
    curvature: List[float] = []
    for _ in range(int(n)):
        draw = rng.normal(f, e)
        try:
            res = weighted_logpoly(nu, draw, e, degree=degree)
        except Exception:                                      # noqa: BLE001
            continue
        if np.isfinite(res["alpha"]):
            values.append(res["alpha"])
            if degree >= 2 and np.isfinite(res.get("curvature", np.nan)):
                curvature.append(res["curvature"])

    if len(values) < max(50, int(0.05 * n)):
        raise RuntimeError(
            f"Only {len(values)}/{n} bootstrap realisations converged; the SED "
            "is too noisy for a bootstrap slope."
        )

    arr = np.asarray(values, dtype=float)
    out = {
        "alpha_median": float(np.median(arr)),
        "alpha_std": float(np.std(arr, ddof=1)),
        "alpha_p16": float(np.percentile(arr, 16)),
        "alpha_p84": float(np.percentile(arr, 84)),
        "n_success": int(arr.size),
        "n_requested": int(n),
    }
    if curvature:
        carr = np.asarray(curvature, dtype=float)
        out.update({
            "curvature_median": float(np.median(carr)),
            "curvature_std": float(np.std(carr, ddof=1)),
        })
    return out


def two_point_indices(
    nu_hz: Sequence[float], flux_jy: Sequence[float], err_jy: Sequence[float],
    labels: Optional[Sequence[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Spectral index between each adjacent pair of bands.

    A monotonic steepening of the two-point indices with frequency is the
    cleanest model-free evidence of ageing and is what justifies fitting an
    ageing model at all.  Reported so the reader can check that conclusion
    independently of any model.
    """
    nu = np.asarray(nu_hz, dtype=float)
    f = np.asarray(flux_jy, dtype=float)
    e = np.asarray(err_jy, dtype=float)
    order = np.argsort(nu)
    nu, f, e = nu[order], f[order], e[order]
    names = (
        [str(labels[i]) for i in order] if labels is not None
        else [f"band{i}" for i in range(len(nu))]
    )

    out: List[Dict[str, Any]] = []
    for i in range(len(nu) - 1):
        if not (f[i] > 0 and f[i + 1] > 0):
            continue
        ln_ratio = math.log(nu[i + 1] / nu[i])
        if ln_ratio == 0:
            continue
        alpha = math.log(f[i + 1] / f[i]) / ln_ratio
        rel = math.hypot(e[i] / f[i], e[i + 1] / f[i + 1])
        out.append({
            "band_low": names[i],
            "band_high": names[i + 1],
            "nu_low_GHz": float(nu[i] / 1e9),
            "nu_high_GHz": float(nu[i + 1] / 1e9),
            "alpha": float(alpha),
            "alpha_err": float(rel / abs(ln_ratio)),
            "alpha_positive": float(-alpha),
        })
    return out


# ===========================================================================
# PART 9.  SPECTRAL-AGEING MODEL DEFINITIONS
# ===========================================================================
#
# Six physically distinct models, all fitted through synchrofit:
#
#   JP          Jaffe & Perola (1973). Single injection burst; pitch angles
#               isotropised much faster than the loss time, so every electron
#               ages at the same rate. Exponential cut-off above the break.
#   KP          Kardashev-Pacholczyk. Same burst, but pitch angles are FROZEN,
#               so low-pitch-angle electrons survive and the spectrum has a
#               power-law-like tail. KP therefore always gives OLDER ages than
#               JP for the same data.
#   JP/KP       Tribble (1993) variants: the field is inhomogeneous and
#   -Tribble    Gaussian-random rather than uniform, so the observed spectrum
#               is an average over a distribution of local field strengths.
#               These REQUIRE an external B, which is exactly the quantity the
#               equipartition calculation provides -- hence the self-consistent
#               loop in Part 13.
#   CI          Continuous injection (Myers & Spangler 1985): the source is
#               still being fed. Break steepens by exactly delta-alpha = 0.5.
#   CI-OFF      Continuous injection followed by a quiescent phase; the extra
#               free parameter is the remnant fraction t_off/t_total.
#
# Free-parameter counts (normalisation always counts):
#   JP, KP, TJP, TKP, CI : 3   (normalisation, break frequency, s)
#   CI-OFF               : 4   (+ remnant fraction)
# With s held fixed each count drops by one.
# ---------------------------------------------------------------------------

def model_spec(name: str) -> Dict[str, Any]:
    table = {
        "JP":         ("JP",  False, (0.0, 0.0), 3),
        "KP":         ("KP",  False, (0.0, 0.0), 3),
        "JP_Tribble": ("TJP", True,  (0.0, 0.0), 3),
        "KP_Tribble": ("TKP", True,  (0.0, 0.0), 3),
        "CI":         ("CI",  False, (0.0, 0.0), 3),
        "CI-OFF":     ("CI",  False, (0.0, 1.0), 4),
    }
    if name not in table:
        raise ValueError(
            f"Unknown spectral-ageing model {name!r}. "
            f"Choose from {', '.join(table)}."
        )
    fit_type, tribble, remnant_range, k_free = table[name]
    return {
        "name": name,
        "fit_type": fit_type,
        "tribble": tribble,
        "remnant_range": remnant_range,
        "k_free": k_free,
        "fits_remnant": name == "CI-OFF",
    }


def _model_free_parameters(name: str, s_fixed: bool = False) -> int:
    k = model_spec(name)["k_free"]
    return k - 1 if s_fixed else k


def information_criteria(chi2: float, n: int, k: int) -> Dict[str, float]:
    """
    AIC / AICc / BIC from a chi-square, assuming Gaussian errors.

    AICc is the small-sample correction and is UNDEFINED for n <= k + 1.
    Returning NaN there (rather than a large finite number) is deliberate: it
    forces the caller to treat the model as unranked instead of quietly
    preferring whichever unconstrained model happened to fit best.
    """
    out = {"chi2": float(chi2), "n_points": int(n), "k_free": int(k)}
    dof = n - k
    out["dof"] = int(dof)
    out["reduced_chi2"] = float(chi2 / dof) if dof > 0 else float("nan")
    if not np.isfinite(chi2):
        out.update(AIC=float("nan"), AICc=float("nan"), BIC=float("nan"))
        return out
    aic = chi2 + 2.0 * k
    out["AIC"] = float(aic)
    out["AICc"] = (
        float(aic + (2.0 * k * (k + 1)) / (n - k - 1)) if n > k + 1 else float("nan")
    )
    out["BIC"] = float(chi2 + k * math.log(max(n, 1)))
    return out


def _grid_marginals(
    grid: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """
    Marginal posteriors for (log nu_break, s, remnant) from synchrofit's
    probability grid.

    synchrofit's `mesh_parameters` has columns (log10 break, injection index,
    remnant ratio) and `probability_vector` the corresponding un-normalised
    posterior weight.  Marginalising this grid is strictly more informative
    than the grid-peak +/- half-spacing that the point API reports, and it is
    the only way to see the s <-> nu_break degeneracy that dominates when the
    spectrum has few points.

    Caveat, stated in the output: with n_iterations > 1 synchrofit zooms the
    grid onto the peak, so the marginal is LOCAL.  Use --posterior-grid to
    force a single wide-grid pass whose marginals span the full prior.
    """
    if not grid:
        return None
    try:
        mesh = np.asarray(grid["mesh_parameters"], dtype=float)
        prob = np.asarray(grid["probability_vector"], dtype=float).ravel()
    except Exception:                                          # noqa: BLE001
        return None
    if mesh.ndim != 2 or mesh.shape[0] != prob.size or mesh.shape[1] < 2:
        return None

    finite = np.isfinite(prob) & (prob >= 0)
    if not np.any(finite):
        return None
    mesh, prob = mesh[finite], prob[finite]
    total = float(np.sum(prob))
    if not np.isfinite(total) or total <= 0:
        return None
    weights = prob / total

    def summarise(column: np.ndarray) -> Dict[str, float]:
        order = np.argsort(column)
        x = column[order]
        w = weights[order]
        cdf = np.cumsum(w)
        cdf /= cdf[-1]
        def q(p: float) -> float:
            return float(np.interp(p, cdf, x))
        mean = float(np.sum(w * x))
        var = float(np.sum(w * (x - mean) ** 2))
        return {
            "mean": mean,
            "std": float(math.sqrt(max(var, 0.0))),
            "median": q(0.5),
            "p16": q(0.16),
            "p84": q(0.84),
            "p2.5": q(0.025),
            "p97.5": q(0.975),
            "mode": float(x[int(np.argmax(w))]),
        }

    out: Dict[str, Any] = {
        "log10_break_frequency_Hz": summarise(mesh[:, 0]),
        "injection_index_s": summarise(mesh[:, 1]),
        "n_grid_points": int(mesh.shape[0]),
    }
    if mesh.shape[1] >= 3 and np.ptp(mesh[:, 2]) > 0:
        out["remnant_fraction"] = summarise(mesh[:, 2])

    # correlation between s and the break -- the degeneracy that matters
    s_col, b_col = mesh[:, 1], mesh[:, 0]
    ms, mb = float(np.sum(weights * s_col)), float(np.sum(weights * b_col))
    cov = float(np.sum(weights * (s_col - ms) * (b_col - mb)))
    vs = float(np.sum(weights * (s_col - ms) ** 2))
    vb = float(np.sum(weights * (b_col - mb) ** 2))
    if vs > 0 and vb > 0:
        out["corr_s_logbreak"] = float(cov / math.sqrt(vs * vb))

    alpha_stats = out["injection_index_s"]
    out["alpha_inj_signed"] = {
        key: s_to_alpha_signed(value)
        for key, value in alpha_stats.items()
        if key in {"mean", "median", "mode"}
    }
    out["alpha_inj_signed"]["std"] = 0.5 * alpha_stats["std"]
    # note: s -> alpha is order-reversing, so percentiles swap
    out["alpha_inj_signed"]["p16"] = s_to_alpha_signed(alpha_stats["p84"])
    out["alpha_inj_signed"]["p84"] = s_to_alpha_signed(alpha_stats["p16"])
    return out


def fit_spectral_age_model(
    name: str,
    frequency_hz: np.ndarray,
    flux_jy: np.ndarray,
    err_jy: np.ndarray,
    bfield_t: Optional[float],
    redshift: float,
    n_breaks: int = 31,
    n_injects: int = 31,
    n_remnants: int = 21,
    iterations: int = 3,
    inject_range: Sequence[float] = DEFAULT_INJECT_RANGE,
    break_range: Sequence[float] = DEFAULT_BREAK_RANGE,
    fixed_s: Optional[float] = None,
    harvest_posterior: bool = False,
) -> Dict[str, Any]:
    """
    Fit ONE spectral-ageing model and return a fully self-describing record.

    `fixed_s` pins the injection index, reducing the free-parameter count by
    one.  With three or four bands that is usually the only way to obtain a
    formally identifiable break frequency, and it is the mode used for the
    resolved maps.
    """
    spec = model_spec(name)
    s_fixed = fixed_s is not None
    k = _model_free_parameters(name, s_fixed=s_fixed)

    good = (
        np.isfinite(frequency_hz) & (np.asarray(frequency_hz) > 0)
        & np.isfinite(flux_jy) & (np.asarray(flux_jy) > 0)
        & np.isfinite(err_jy) & (np.asarray(err_jy) > 0)
    )
    nu = np.asarray(frequency_hz, dtype=float)[good]
    fl = np.asarray(flux_jy, dtype=float)[good]
    er = np.asarray(err_jy, dtype=float)[good]
    n = int(nu.size)

    identifiable = bool(n > k + 1)
    record: Dict[str, Any] = {
        "model": name,
        "fit_type": spec["fit_type"],
        "tribble": spec["tribble"],
        # The field this model was FITTED at. A Tribble spectrum is a
        # function of B, so its fitted normalisation is only valid at the
        # field used to obtain it. Re-evaluating the curve at a different
        # field -- the final self-consistent value, say -- shifts it off the
        # data it was fitted to, which looks like a bad fit rather than an
        # inconsistent evaluation.
        "b_field_T_used": (
            float(bfield_t) if (spec["tribble"] and bfield_t) else None
        ),
        "n_points": n,
        "k_free": k,
        "s_fixed": bool(s_fixed),
        "s_fixed_value": float(fixed_s) if s_fixed else None,
        "identifiable": identifiable,
        "underconstrained": not identifiable,
        "fit_success": False,
        "scientifically_usable": False,
        "used_for_model_consensus": False,
        "status": "not_attempted",
        "b_field_input_T": float(bfield_t) if bfield_t else None,
        "redshift": float(redshift),
    }

    if n < 2:
        record["status"] = "insufficient_data"
        return record
    if not HAVE_SYNCHROFIT:
        record["status"] = "synchrofit_unavailable"
        record["error"] = str(SYNCHROFIT.import_error)
        return record
    if not identifiable:
        # We deliberately still attempt the fit so the user can SEE the
        # best-fit parameters, but everything downstream is gated on
        # `identifiable` so an unconstrained model can never win a comparison.
        record["status"] = "underconstrained_fit_attempted_but_not_ranked"
    if spec["tribble"] and (
        bfield_t is None or not np.isfinite(bfield_t) or bfield_t <= 0
    ):
        record["status"] = "missing_bfield_for_tribble"
        return record

    if s_fixed:
        eff_inject_range = (float(fixed_s), float(fixed_s))
        eff_n_injects = 1
    else:
        eff_inject_range = (float(inject_range[0]), float(inject_range[1]))
        eff_n_injects = int(n_injects)

    try:
        fitted = SYNCHROFIT.fit(
            frequency_hz=nu, flux=fl, dflux=er,
            fit_type=spec["fit_type"],
            n_breaks=int(n_breaks),
            break_range=break_range,
            n_injects=eff_n_injects,
            inject_range=eff_inject_range,
            n_remnants=(int(n_remnants) if spec["fits_remnant"] else 1),
            remnant_range=spec["remnant_range"],
            n_iterations=(1 if harvest_posterior else int(iterations)),
            b_field_t=bfield_t if spec["tribble"] else None,
            redshift=redshift if spec["tribble"] else None,
        )
    except Exception as exc:                                   # noqa: BLE001
        record["status"] = "fit_failed"
        record["error"] = f"{type(exc).__name__}: {exc}"
        return record

    s_value = float(fitted["injection_index_s"])
    s_err = float(fitted["injection_index_s_err"])
    logb = float(fitted["log10_break_frequency_Hz"])
    logb_err = float(fitted["log10_break_frequency_err"])
    remnant = float(fitted["remnant_fraction"])
    remnant_err = float(fitted["remnant_fraction_err"])

    record.update({
        "synchrofit_fit_type": fitted["fit_type"],
        "injection_index_s": s_value,
        "injection_index_s_err": s_err,
        "alpha_inj_radio": s_to_alpha_signed(s_value),
        "alpha_inj_radio_err": 0.5 * abs(s_err),
        "alpha_inj_positive": -s_to_alpha_signed(s_value),
        "log10_break_frequency_Hz": logb,
        "log10_break_frequency_err": logb_err,
        "break_frequency_Hz": float(10.0 ** logb),
        "break_frequency_err_Hz": float(
            10.0 ** logb * math.log(10.0) * abs(logb_err)
        ),
        "remnant_fraction": remnant,
        "remnant_fraction_err": remnant_err,
        "normalisation": float(fitted["normalisation"]),
        "_params8": fitted.get("_params8"),
        "_grid": fitted.get("_grid"),
    })

    posterior = _grid_marginals(fitted.get("_grid"))
    if posterior is not None:
        posterior["grid_is_local"] = not harvest_posterior
        posterior["note"] = (
            "Marginals derived from synchrofit's own posterior grid."
            + ("" if harvest_posterior else
               " n_iterations>1 zooms the grid onto the peak, so these "
               "intervals are LOCAL; rerun with --posterior-grid for "
               "prior-spanning marginals.")
        )
        record["posterior"] = posterior

    # --- goodness of fit -------------------------------------------------
    chi2 = float("nan")
    model_flux, model_err = SYNCHROFIT.model(
        fitted, nu,
        b_field_t=bfield_t if spec["tribble"] else None,
        redshift=redshift if spec["tribble"] else None,
        # mc_length=1, NOT 200. The Monte-Carlo pass builds `model_err` and
        # nothing else -- chi2 below uses `model_flux` alone, and the flux it
        # returns is BIT-IDENTICAL either way (measured: max|dflux/flux| =
        # 0.000e+00 for JP, KP, CI and TJP). So the samples cannot move any
        # number this pipeline reports.
        #
        # They are not cheap: 93x slower for the standard models, and ~103 s
        # per call for the Tribble ones, which across two Tribble models and
        # up to 40 self-consistent iterations came to ~2.3 hours per run.
        #
        # `model_err` itself is not worth paying for. It is written to
        # `model_flux_err_Jy` and never read anywhere else in this file, and
        # the values are unphysical on their face -- of order 1e5 Jy for a
        # 3 mJy source, i.e. seven orders of magnitude too large. Quoting
        # them would be worse than omitting them.
        mc_length=1, err_width=1,
    )
    if model_flux is not None and model_flux.shape == fl.shape:
        finite_model = np.isfinite(model_flux) & (model_flux > 0)
        if np.all(finite_model):
            chi2 = float(np.sum(((fl - model_flux) / er) ** 2))
            record["model_flux_Jy"] = model_flux.tolist()
            # Deliberately not stored: see the mc_length note above. The
            # only way to get a defensible error band on the model curve is
            # to propagate the fitted parameter covariance, which the
            # posterior marginals in record["posterior"] already do.
            record["model_flux_err_Jy"] = None
            record["model_flux_err_note"] = (
                "Not computed. synchrofit's Monte-Carlo error array was "
                "unphysical (~1e5 Jy for a mJy source) and unused; use "
                "record['posterior'] for parameter uncertainties."
            )
            record["normalised_residuals"] = (
                ((fl - model_flux) / er).tolist()
            )
        else:
            record["model_evaluation_error"] = "model returned non-positive flux"
    else:
        record["model_evaluation_error"] = (
            "installed synchrofit could not evaluate the fitted model"
        )

    ic = information_criteria(chi2, n, k)
    record.update(ic)

    # --- spectral age ----------------------------------------------------
    age_field = bfield_t
    if age_field is not None and np.isfinite(age_field) and age_field > 0:
        ages = SYNCHROFIT.ages(
            fit_type=spec["fit_type"],
            break_frequency_hz=10.0 ** logb,
            remnant_fraction=max(remnant, 0.0),
            b_field_t=float(age_field),
            redshift=float(redshift),
        )
        if ages is not None:
            # Propagate the break-frequency error: tau ~ nu_b^(-1/2)
            ages["tau_err_Myr"] = float(
                0.5 * ages["tau_Myr"] * math.log(10.0) * abs(logb_err)
            )
            ages["b_field_used_T"] = float(age_field)
            ages["b_field_used_uG"] = float(age_field / TESLA_PER_MICROGAUSS)
            ages["note"] = (
                "Age depends on the assumed field as tau ~ B^0.5/(B^2+B_CMB^2); "
                "it is maximal at B = B_CMB/sqrt(3) and this dependence usually "
                "dominates over the statistical break-frequency error."
            )
            record["spectral_age"] = ages

    # --- quality gates ---------------------------------------------------
    redchi = record.get("reduced_chi2", float("nan"))
    quality = "identified" if identifiable else "underconstrained"
    if np.isfinite(redchi):
        if redchi <= 2.0:
            quality += "_acceptable_redchi2"
        elif redchi <= 5.0:
            quality += "_marginal_redchi2"
        else:
            quality += "_poor_redchi2"
    else:
        quality += "_no_gof"

    record["fit_quality"] = quality
    record["fit_success"] = True
    record["status"] = "success" if identifiable else "success_but_underconstrained"
    record["used_for_model_consensus"] = bool(
        identifiable and np.isfinite(record.get("AICc", np.nan))
    )
    record["scientifically_usable"] = bool(
        record["used_for_model_consensus"]
        and np.isfinite(redchi) and redchi <= 5.0
    )
    return record


# ===========================================================================
# PART 10.  OBSERVED SPECTRUM, MODEL ENSEMBLE, ROBUST INJECTION INDEX
# ===========================================================================

def characterise_observed_spectrum(
    nu_hz: np.ndarray,
    flux_jy: np.ndarray,
    err_jy: np.ndarray,
    detected: np.ndarray,
    labels: Optional[Sequence[str]] = None,
    bootstrap_n: int = DEFAULT_ALPHA_BOOTSTRAP,
    seed: int = DEFAULT_SEED,
) -> Dict[str, Any]:
    """
    Model-independent description of the integrated spectrum.

    IMPORTANT: everything here is an OBSERVED spectral index.  For an aged
    synchrotron source the observed index is always steeper than or equal to
    the injection index, so these numbers are UPPER LIMITS on the steepness of
    the injected spectrum, never a substitute for it.  v1 fed the
    lowest-three-band slope straight into the equipartition calculation as
    `alpha_inj`; v2 keeps the two concepts strictly separate and labels the
    provenance in every output.
    """
    nu = np.asarray(nu_hz, dtype=float)
    f = np.asarray(flux_jy, dtype=float)
    e = np.asarray(err_jy, dtype=float)
    det = np.asarray(detected, dtype=bool)

    out: Dict[str, Any] = {
        "frequency_GHz": (nu / 1e9).tolist(),
        "flux_Jy": f.tolist(),
        "flux_err_Jy": e.tolist(),
        "detected": det.tolist(),
        "n_bands": int(nu.size),
        "n_detections": int(np.count_nonzero(det)),
        "interpretation": (
            "These are OBSERVED spectral indices of the integrated source. "
            "They bound, but do not measure, the injection index."
        ),
    }
    if labels is not None:
        out["labels"] = [str(x) for x in labels]

    nd, fd, ed = nu[det], f[det], e[det]
    order = np.argsort(nd)
    nd, fd, ed = nd[order], fd[order], ed[order]

    out["two_point_indices"] = two_point_indices(
        nd, fd, ed,
        labels=[str(labels[i]) for i in np.where(det)[0][order]] if labels else None,
    )
    if len(out["two_point_indices"]) >= 2:
        seq = [x["alpha"] for x in out["two_point_indices"]]
        out["monotonic_steepening"] = bool(
            all(seq[i + 1] <= seq[i] + 1e-12 for i in range(len(seq) - 1))
        )
        out["total_steepening_delta_alpha"] = float(seq[0] - seq[-1])

    if nd.size >= 2:
        linear = weighted_logpoly(nd, fd, ed, degree=1)
        out["powerlaw_all_bands"] = linear
        out["alpha_all"] = linear["alpha"]
        out["alpha_all_err"] = linear["alpha_err"]
        out["chi2_red_all"] = linear["reduced_chi2"]
    else:
        out["alpha_all"] = float("nan")
        out["alpha_all_err"] = float("nan")
        out["chi2_red_all"] = float("nan")

    if nd.size >= 3:
        low = weighted_logpoly(nd[:3], fd[:3], ed[:3], degree=1)
        out["powerlaw_low_frequency"] = low
        out["alpha_low"] = low["alpha"]
        out["alpha_low_err"] = low["alpha_err"]
        out["chi2_red_low"] = low["reduced_chi2"]
        try:
            out["bootstrap_low"] = bootstrap_alpha(
                nd[:3], fd[:3], ed[:3], n=bootstrap_n, seed=seed
            )
        except Exception as exc:                               # noqa: BLE001
            out["bootstrap_low_error"] = str(exc)
    else:
        out["alpha_low"] = float("nan")
        out["alpha_low_err"] = float("nan")
        out["chi2_red_low"] = float("nan")

    if nd.size >= 3:
        try:
            quad = weighted_logpoly(nd, fd, ed, degree=2)
            out["logpoly_curvature"] = quad
            out["curvature"] = quad["curvature"]
            out["curvature_err"] = quad["curvature_err"]
            out["curvature_significance"] = quad["curvature_significance"]
            out["curvature_interpretation"] = (
                "significant negative curvature: spectrum steepens, consistent "
                "with radiative ageing"
                if quad["curvature"] < 0 and quad["curvature_significance"] >= 2
                else "significant positive curvature: spectrum flattens "
                     "(absorption, or a second injected component)"
                if quad["curvature"] > 0 and quad["curvature_significance"] >= 2
                else "no significant curvature detected at this sampling"
            )
        except Exception as exc:                               # noqa: BLE001
            out["curvature_error"] = str(exc)

    if nd.size >= 2:
        try:
            out["bootstrap_all"] = bootstrap_alpha(
                nd, fd, ed, n=bootstrap_n, seed=seed + 1,
                degree=2 if nd.size >= 4 else 1,
            )
        except Exception as exc:                               # noqa: BLE001
            out["bootstrap_all_error"] = str(exc)

    # Steepest defensible upper bound on alpha_inj: an aged spectrum can only
    # be steeper than injection, so the FLATTEST observed index is the
    # tightest data-driven bound.
    candidates = [
        v for v in (out.get("alpha_low"), out.get("alpha_all"))
        if v is not None and np.isfinite(v)
    ]
    tp = [x["alpha"] for x in out.get("two_point_indices", [])]
    candidates.extend([v for v in tp if np.isfinite(v)])
    if candidates:
        out["alpha_inj_upper_bound_signed"] = float(max(candidates))
        out["alpha_inj_upper_bound_positive"] = float(-max(candidates))
        out["alpha_inj_bound_note"] = (
            "alpha_inj must be FLATTER than or equal to the flattest observed "
            "index; this is the tightest model-free constraint available."
        )
    return out


def _refit_with_bfield(
    name: str, nu: np.ndarray, fl: np.ndarray, er: np.ndarray,
    b_t: float, redshift: float, cfg: Dict[str, Any],
    fixed_s: Optional[float] = None,
) -> Dict[str, Any]:
    return fit_spectral_age_model(
        name, nu, fl, er, b_t, redshift,
        n_breaks=cfg["n_breaks"], n_injects=cfg["n_injects"],
        n_remnants=cfg["n_remnants"], iterations=cfg["iterations"],
        inject_range=cfg["inject_range"], break_range=cfg["break_range"],
        fixed_s=fixed_s, harvest_posterior=cfg.get("harvest_posterior", False),
    )


def marginalise_tribble_over_bfield(
    name: str,
    nu: np.ndarray, fl: np.ndarray, er: np.ndarray,
    b_t: float, b_err_t: float, redshift: float,
    cfg: Dict[str, Any], n_samples: int = 3,
) -> Dict[str, Any]:
    """
    Tribble fits are CONDITIONAL on an assumed magnetic field.

    Quoting a Tribble injection index with only its grid error understates the
    uncertainty, because B itself is known only to tens of per cent from the
    equipartition calculation.  Here the fit is repeated across the B
    uncertainty (log-spaced, so the sampling respects the multiplicative
    nature of the B error) and the resulting spread in s is added in
    quadrature as a conditioning systematic.
    """
    base = _refit_with_bfield(name, nu, fl, er, b_t, redshift, cfg)
    if not base.get("fit_success") or b_err_t is None or b_err_t <= 0:
        return base

    ln_sigma = min(abs(b_err_t / max(b_t, 1e-30)), 0.9)
    offsets = np.linspace(-1.0, 1.0, max(int(n_samples), 3))
    s_values, break_values = [], []
    for off in offsets:
        b_trial = float(b_t * math.exp(off * ln_sigma))
        if not np.isfinite(b_trial) or b_trial <= 0:
            continue
        trial = _refit_with_bfield(name, nu, fl, er, b_trial, redshift, cfg)
        if trial.get("fit_success"):
            s_values.append(trial["injection_index_s"])
            break_values.append(trial["log10_break_frequency_Hz"])

    if len(s_values) >= 3:
        s_spread = float(np.std(np.asarray(s_values), ddof=1))
        b_spread = float(np.std(np.asarray(break_values), ddof=1))
        base["b_field_conditioning"] = {
            "b_field_T": float(b_t),
            "b_field_err_T": float(b_err_t),
            "n_samples": len(s_values),
            "s_spread_from_B": s_spread,
            "log10_break_spread_from_B": b_spread,
            "note": (
                "Systematic from conditioning the Tribble fit on an uncertain "
                "magnetic field; added in quadrature to the statistical error."
            ),
        }
        base["injection_index_s_err"] = float(
            math.hypot(base["injection_index_s_err"], s_spread)
        )
        base["alpha_inj_radio_err"] = 0.5 * base["injection_index_s_err"]
        base["log10_break_frequency_err"] = float(
            math.hypot(base["log10_break_frequency_err"], b_spread)
        )
    return base


def scan_injection_index(
    name: str,
    nu: np.ndarray, fl: np.ndarray, er: np.ndarray,
    b_t: Optional[float], redshift: float, cfg: Dict[str, Any],
    n_scan: int = 9,
) -> Optional[Dict[str, Any]]:
    """
    Profile likelihood in the injection index.

    With three or four bands the free-s fit is not identifiable, but the
    problem is not that s is unconstrained -- it is that s and the break
    frequency are DEGENERATE.  Scanning s on a grid, refitting the break at
    each node, and profiling chi2 exposes that degeneracy honestly and yields
    a defensible confidence interval (Delta chi2 = 1 for one parameter of
    interest) instead of a spuriously precise point value.
    """
    lo, hi = float(cfg["inject_range"][0]), float(cfg["inject_range"][1])
    grid = np.linspace(lo, hi, max(int(n_scan), 3))

    chi2_values: List[float] = []
    nodes: List[Dict[str, Any]] = []
    for s_val in grid:
        try:
            rec = _refit_with_bfield(
                name, nu, fl, er, b_t, redshift, cfg, fixed_s=float(s_val)
            )
        except Exception:                                      # noqa: BLE001
            continue
        chi2 = rec.get("chi2", float("nan"))
        if not np.isfinite(chi2):
            continue
        chi2_values.append(float(chi2))
        nodes.append({
            "s": float(s_val),
            "alpha_inj_signed": s_to_alpha_signed(float(s_val)),
            "chi2": float(chi2),
            "log10_break_frequency_Hz": rec.get("log10_break_frequency_Hz"),
            "break_frequency_Hz": rec.get("break_frequency_Hz"),
            "reduced_chi2": rec.get("reduced_chi2"),
        })

    if len(chi2_values) < 3:
        return None

    chi2_arr = np.asarray(chi2_values, dtype=float)
    s_arr = np.asarray([node["s"] for node in nodes], dtype=float)
    imin = int(np.argmin(chi2_arr))
    chi2_min = float(chi2_arr[imin])
    delta = chi2_arr - chi2_min

    def crossing(direction: int) -> float:
        """Linear interpolation of the Delta chi2 = 1 crossing."""
        idx = range(imin, len(s_arr) - 1) if direction > 0 else range(imin, 0, -1)
        for i in idx:
            j = i + 1 if direction > 0 else i - 1
            if delta[j] >= 1.0 > delta[i]:
                t = (1.0 - delta[i]) / max(delta[j] - delta[i], 1e-12)
                return float(s_arr[i] + t * (s_arr[j] - s_arr[i]))
        return float(s_arr[-1] if direction > 0 else s_arr[0])

    s_lo, s_hi = crossing(-1), crossing(+1)
    bounded = bool(s_lo > s_arr[0] + 1e-9 and s_hi < s_arr[-1] - 1e-9)

    return {
        "model": name,
        "nodes": nodes,
        "s_best": float(s_arr[imin]),
        "s_lo_1sigma": float(min(s_lo, s_hi)),
        "s_hi_1sigma": float(max(s_lo, s_hi)),
        "s_err_symmetric": float(0.5 * abs(s_hi - s_lo)),
        "alpha_inj_signed_best": s_to_alpha_signed(float(s_arr[imin])),
        "alpha_inj_signed_lo": s_to_alpha_signed(float(max(s_lo, s_hi))),
        "alpha_inj_signed_hi": s_to_alpha_signed(float(min(s_lo, s_hi))),
        "chi2_min": chi2_min,
        "interval_bounded_by_data": bounded,
        "prior_range_s": [lo, hi],
        "note": (
            "Delta chi2 = 1 profile interval with the break frequency refitted "
            "at every node."
            + ("" if bounded else
               " The interval reaches the edge of the assumed prior on s, i.e. "
               "the data alone do NOT bound the injection index; the quoted "
               "range is prior-limited.")
        ),
    }


def fit_spectral_ageing_ensemble(
    frequency_hz: np.ndarray,
    flux_jy: np.ndarray,
    err_jy: np.ndarray,
    bfield_t: Optional[float],
    redshift: float,
    cfg: Dict[str, Any],
    bfield_err_t: Optional[float] = None,
    models: Optional[Sequence[str]] = None,
    observed: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Fit the full six-model ensemble and build a defensible consensus.

    The consensus is an AICc-weighted average over models that are FORMALLY
    IDENTIFIABLE at the available number of frequency points.  Two error terms
    are reported separately and must not be conflated:

      within-model  : the weighted mean of the individual statistical errors
      between-model : the weighted scatter of the central values

    The second is a MODEL SYSTEMATIC.  For a source with four bands it is
    typically the larger of the two, and reporting only the first is the most
    common way that published injection indices end up quoted three times more
    precisely than the data support.
    """
    nu = np.asarray(frequency_hz, dtype=float)
    fl = np.asarray(flux_jy, dtype=float)
    er = np.asarray(err_jy, dtype=float)
    n_points = int(np.count_nonzero(np.isfinite(fl) & (fl > 0)))

    results: Dict[str, Any] = {
        "n_points": n_points,
        "b_field_input_T": float(bfield_t) if bfield_t else None,
        "b_field_input_uG": (
            float(bfield_t / TESLA_PER_MICROGAUSS) if bfield_t else None
        ),
        "inject_range_prior_s": list(cfg["inject_range"]),
        "break_range_prior_log10Hz": list(cfg["break_range"]),
        "synchrofit_available": bool(HAVE_SYNCHROFIT),
        "synchrofit_flavour": SYNCHROFIT.flavour,
    }

    # The ensemble list is configurable because the two Tribble variants cost
    # far more than the other four and do not scale down with the parameter
    # grid: their expense is the per-grid-point integration over the field
    # distribution (synchrofit uses nalpha=32, nfields=128), so shrinking
    # n_breaks/n_injects barely touches them. On this source one TJP fit took
    # ~9 minutes against ~1 minute for JP, KP, CI and CI-OFF combined, which
    # made them ~80 per cent of the whole run.
    #
    # Dropping them is a real loss of model coverage, not a free saving, so it
    # is an explicit choice via --ensemble-models rather than a default.
    if models is None:
        models = tuple(cfg.get("ensemble_models") or SPECTRAL_AGE_MODELS)
    if not HAVE_SYNCHROFIT:
        results["status"] = "synchrofit_unavailable"
        results["models"] = {}
        results["robust_injection_index"] = {
            "method": "unavailable",
            "reason": f"synchrofit import failed: {SYNCHROFIT.import_error}",
        }
        return results

    # ---- stage A: free injection index ---------------------------------
    fits: Dict[str, Any] = {}
    for name in models:
        spec = model_spec(name)
        if spec["tribble"] and bfield_t and bfield_err_t and cfg.get("thorough"):
            fits[name] = marginalise_tribble_over_bfield(
                name, nu, fl, er, float(bfield_t), float(bfield_err_t),
                redshift, cfg,
            )
        else:
            fits[name] = _refit_with_bfield(
                name, nu, fl, er, bfield_t, redshift, cfg
            )
    results["models"] = fits

    identifiable = [
        r for r in fits.values()
        if r.get("fit_success") and r.get("used_for_model_consensus")
    ]

    # ---- stage B: AICc model averaging ----------------------------------
    if identifiable:
        aicc = np.asarray([float(r["AICc"]) for r in identifiable])
        delta = aicc - float(np.nanmin(aicc))
        weights = np.exp(-0.5 * delta)
        weights = weights / np.sum(weights)

        alphas = np.asarray([float(r["alpha_inj_radio"]) for r in identifiable])
        alpha_errs = np.asarray(
            [float(r["alpha_inj_radio_err"]) for r in identifiable]
        )
        mu = float(np.sum(weights * alphas))
        within = float(np.sum(weights * alpha_errs ** 2))
        between = float(np.sum(weights * (alphas - mu) ** 2))

        breaks = np.asarray(
            [float(r["log10_break_frequency_Hz"]) for r in identifiable]
        )
        break_mu = float(np.sum(weights * breaks))
        break_between = float(np.sum(weights * (breaks - break_mu) ** 2))

        consensus = {
            "method": "AICc_model_average_free_s",
            "alpha_inj_radio": mu,
            "alpha_inj_positive": -mu,
            "alpha_inj_stat_err": float(math.sqrt(max(within, 0.0))),
            "alpha_inj_model_systematic": float(math.sqrt(max(between, 0.0))),
            "alpha_inj_radio_err": float(math.sqrt(max(within + between, 0.0))),
            "injection_index_s": alpha_signed_to_s(mu),
            "injection_index_s_err": float(
                2.0 * math.sqrt(max(within + between, 0.0))
            ),
            "log10_break_frequency_Hz": break_mu,
            "log10_break_frequency_err": float(math.sqrt(max(break_between, 0.0))),
            "n_models": len(identifiable),
            "models": [r["model"] for r in identifiable],
            "delta_AICc": {
                r["model"]: float(d) for r, d in zip(identifiable, delta)
            },
            "akaike_weights": {
                r["model"]: float(w) for r, w in zip(identifiable, weights)
            },
            "best_model": identifiable[int(np.argmin(delta))]["model"],
        }
    else:
        consensus = {
            "method": "no_identifiable_model",
            "alpha_inj_radio": None,
            "n_models": 0,
            "models": [],
            "reason": (
                f"With {n_points} frequency points no model satisfies "
                "n > k + 1, so AICc is undefined and no free-injection-index "
                "fit is statistically identifiable."
            ),
        }
    results["free_s_consensus"] = consensus

    # ---- stage C: injection-index profile scan --------------------------
    scans: Dict[str, Any] = {}
    scan_models = [m for m in cfg.get("scan_models", ("JP", "CI")) if m in models]
    if cfg.get("scan_injection", True):
        for name in scan_models:
            spec = model_spec(name)
            if spec["tribble"] and not bfield_t:
                continue
            scan = scan_injection_index(
                name, nu, fl, er,
                bfield_t if spec["tribble"] else bfield_t,
                redshift, cfg, n_scan=int(cfg.get("n_scan", 9)),
            )
            if scan is not None:
                scans[name] = scan
    results["injection_index_scans"] = scans

    # ---- stage D: assemble the recommended injection index ---------------
    results["robust_injection_index"] = _assemble_injection_index(
        consensus, scans, observed, n_points, cfg
    )

    # ---- stage E: restricted ensemble at the adopted s -------------------
    adopted = results["robust_injection_index"].get("injection_index_s")
    if adopted is not None and np.isfinite(adopted):
        restricted: Dict[str, Any] = {}
        for name in models:
            spec = model_spec(name)
            if spec["tribble"] and not bfield_t:
                continue
            restricted[name] = _refit_with_bfield(
                name, nu, fl, er, bfield_t, redshift, cfg,
                fixed_s=float(adopted),
            )
        results["restricted_models_fixed_s"] = restricted

        usable = [
            r for r in restricted.values()
            if r.get("fit_success") and np.isfinite(r.get("AICc", np.nan))
        ]
        if usable:
            aicc = np.asarray([float(r["AICc"]) for r in usable])
            delta = aicc - float(np.nanmin(aicc))
            weights = np.exp(-0.5 * delta) / np.sum(np.exp(-0.5 * delta))
            breaks = np.asarray(
                [float(r["log10_break_frequency_Hz"]) for r in usable]
            )
            bmu = float(np.sum(weights * breaks))
            results["restricted_consensus"] = {
                "method": "AICc_model_average_fixed_s",
                "fixed_s": float(adopted),
                "log10_break_frequency_Hz": bmu,
                "log10_break_frequency_stat_err": float(math.sqrt(np.sum(
                    weights * np.asarray(
                        [float(r["log10_break_frequency_err"]) for r in usable]
                    ) ** 2
                ))),
                "log10_break_frequency_model_systematic": float(
                    math.sqrt(np.sum(weights * (breaks - bmu) ** 2))
                ),
                "break_frequency_Hz": float(10.0 ** bmu),
                "models": [r["model"] for r in usable],
                "akaike_weights": {
                    r["model"]: float(w) for r, w in zip(usable, weights)
                },
                "best_model": usable[int(np.argmin(delta))]["model"],
                "delta_AICc": {
                    r["model"]: float(d) for r, d in zip(usable, delta)
                },
                "note": (
                    "With s held at the adopted injection index only the "
                    "normalisation and the break frequency are free (k = 2, or "
                    "3 for CI-OFF), so these fits ARE identifiable with four "
                    "bands. This is the recommended break-frequency and "
                    "spectral-age product at low band count."
                ),
            }
    return results


def _assemble_injection_index(
    consensus: Dict[str, Any],
    scans: Dict[str, Any],
    observed: Optional[Dict[str, Any]],
    n_points: int,
    cfg: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Decide the number that goes into the equipartition calculation, and be
    explicit about how much of its error is statistical and how much is model
    or prior dependence.

    Precedence:
      1. an AICc-identifiable free-s model average, when one exists;
      2. otherwise the profile-likelihood scan, which is honest about the
         s <-> nu_break degeneracy;
      3. otherwise the flattest observed index, as a bound-driven fallback,
         explicitly flagged as NOT a measured injection index.
    """
    out: Dict[str, Any] = {"n_points": int(n_points)}

    bound = None
    if observed:
        bound = observed.get("alpha_inj_upper_bound_signed")

    if consensus.get("alpha_inj_radio") is not None:
        alpha = float(consensus["alpha_inj_radio"])
        out.update({
            "alpha_inj_radio": alpha,
            "alpha_inj_positive": -alpha,
            "alpha_inj_stat_err": float(consensus["alpha_inj_stat_err"]),
            "alpha_inj_model_systematic": float(
                consensus["alpha_inj_model_systematic"]
            ),
            "alpha_inj_radio_err": float(consensus["alpha_inj_radio_err"]),
            "injection_index_s": alpha_signed_to_s(alpha),
            "method": "AICc_model_average_over_identifiable_models",
            "models_used": consensus["models"],
            "confidence": "model-identified",
        })
    elif scans:
        # Combine the available profile scans; the spread between models is a
        # model systematic exactly as in the AICc branch.
        best = [v for v in scans.values() if np.isfinite(v["s_best"])]
        s_vals = np.asarray([v["s_best"] for v in best], dtype=float)
        s_errs = np.asarray([v["s_err_symmetric"] for v in best], dtype=float)
        s_mu = float(np.mean(s_vals))
        stat = float(np.sqrt(np.mean(np.maximum(s_errs, 0.0) ** 2)))
        sysm = float(np.std(s_vals, ddof=1)) if s_vals.size > 1 else 0.0
        alpha = s_to_alpha_signed(s_mu)
        bounded = all(v["interval_bounded_by_data"] for v in best)
        out.update({
            "alpha_inj_radio": alpha,
            "alpha_inj_positive": -alpha,
            "alpha_inj_stat_err": 0.5 * stat,
            "alpha_inj_model_systematic": 0.5 * sysm,
            "alpha_inj_radio_err": 0.5 * float(math.hypot(stat, sysm)),
            "injection_index_s": s_mu,
            "method": "profile_likelihood_scan_over_fixed_s",
            "models_used": [v["model"] for v in best],
            "interval_bounded_by_data": bounded,
            "confidence": "degeneracy-limited" if bounded else "prior-limited",
            "warning": (
                "No model is formally identifiable at this number of frequency "
                "points, so the injection index comes from a profile scan. "
                + ("The Delta chi2 = 1 interval is set by the data."
                   if bounded else
                   "The interval runs to the edge of the assumed prior on s: "
                   "the data do NOT constrain the injection index and the "
                   "quoted value is prior-dominated. Additional low-frequency "
                   "coverage is required for a measurement.")
            ),
        })
    elif bound is not None and np.isfinite(bound):
        out.update({
            "alpha_inj_radio": float(bound),
            "alpha_inj_positive": float(-bound),
            "alpha_inj_stat_err": float(
                observed.get("alpha_low_err", DEFAULT_ALPHA_ERR_FLOOR)
                if observed else DEFAULT_ALPHA_ERR_FLOOR
            ),
            "alpha_inj_model_systematic": 0.10,
            "alpha_inj_radio_err": float(math.hypot(
                observed.get("alpha_low_err", DEFAULT_ALPHA_ERR_FLOOR)
                if observed else DEFAULT_ALPHA_ERR_FLOOR,
                0.10,
            )),
            "injection_index_s": alpha_signed_to_s(float(bound)),
            "method": "observed_flattest_index_used_as_injection_proxy",
            "confidence": "not-measured",
            "warning": (
                "NO ageing model could be fitted. The flattest observed index "
                "is used as a PROXY for the injection index. It is an upper "
                "bound on the steepness of the injected spectrum, not a "
                "measurement of it, and a 0.10 model systematic has been added "
                "to reflect that."
            ),
        })
    else:
        out.update({
            "alpha_inj_radio": None,
            "method": "undetermined",
            "confidence": "none",
        })
        return out

    # --- physical consistency check --------------------------------------
    if bound is not None and np.isfinite(bound) and out["alpha_inj_radio"] is not None:
        # alpha is SIGNED and negative, so "steeper" means MORE negative.
        # An aged spectrum can only be steeper than the injected one, hence
        # the requirement is  alpha_inj >= alpha_flattest_observed.
        # A violation is therefore a NEGATIVE difference, and the tolerance
        # must be one-sided in that direction: alpha_inj is allowed to be
        # flatter than the bound by any amount (that is the normal, expected
        # situation for an aged source), but not steeper by more than 2 sigma.
        violation = float(out["alpha_inj_radio"]) - float(bound)
        tolerance = 2.0 * max(out.get("alpha_inj_radio_err", 0.05), 1e-3)
        out["consistency_with_observed_bound"] = {
            "flattest_observed_alpha_signed": float(bound),
            "alpha_inj_minus_bound": violation,
            "tolerance_2sigma": float(tolerance),
            "consistent": bool(violation >= -tolerance),
            "requirement": (
                "For an aged synchrotron source the observed spectrum can only "
                "be steeper than the injected one, so alpha_inj must not be "
                "steeper than the flattest observed index."
            ),
        }
        if not out["consistency_with_observed_bound"]["consistent"]:
            out["warning"] = (
                (out.get("warning", "") + " ").strip()
                + " PHYSICAL INCONSISTENCY: the fitted injection index is "
                  "steeper than the flattest observed spectral index. This "
                  "usually indicates an unmodelled flux-scale offset between "
                  "bands, resolved-out flux at high frequency, or a "
                  "contaminating flat-spectrum core inside the mask."
            ).strip()

    out["alpha_convention"] = "S_nu ~ nu^alpha with alpha negative"
    return out


# ===========================================================================
# PART 11.  pysynch ENGINE
# ===========================================================================

def validate_pysynch_cosmology(cosmology=DEFAULT_COSMOLOGY) -> None:
    if not hasattr(cosmology, "arcsec_per_kpc_proper"):
        raise RuntimeError(
            "The configured cosmology does not provide arcsec_per_kpc_proper(), "
            "which pysynch requires."
        )
    test = cosmology.arcsec_per_kpc_proper(0.1)
    if not np.isfinite(test.value) or test.value <= 0:
        raise RuntimeError("Cosmology returned an invalid arcsec_per_kpc_proper(0.1).")


def equivalent_sphere_radius_m(volume_m3: float) -> float:
    """Radius of the sphere with the given volume."""
    if not np.isfinite(volume_m3) or volume_m3 <= 0:
        raise ValueError(f"Volume must be finite and positive; got {volume_m3!r}.")
    return float((3.0 * volume_m3 / (4.0 * math.pi)) ** (1.0 / 3.0))


def make_synch_source(
    volume_m3: float,
    alpha_inj: float,
    redshift: float,
    gmin: float,
    gmax: float,
    spectrum: str = "powerlaw",
    spectrum_kwargs: Optional[Dict[str, Any]] = None,
    cosmology=DEFAULT_COSMOLOGY,
):
    """
    Build a pysynch SynchSource with an EXACT prescribed volume.

    pysynch derives the volume from one of three idealised solids.  Because
    the volume enters only as a scalar divisor of the emissivity, and because
    we have a better volume estimate than any of those solids provides
    (Part 6), we instantiate the equivalent sphere -- radius (3V/4pi)^(1/3),
    supplied in METRES so no angular conversion is involved -- which
    reproduces the requested volume identically while leaving every other part
    of pysynch's physics untouched.

    `spectrum`:
        'powerlaw'  unbroken; the classical assumption
        'broken'    requires gbreak (Lorentz factor) and dpow (extra index
                    steepening above the break; 1.0 for continuous injection)
        'aged'      requires age (seconds) and ageb (Tesla); pysynch's C
                    implementation of the JP aged electron spectrum, with
                    cmbage=True adding the CMB field in quadrature
    """
    require_pysynch()
    kwargs: Dict[str, Any] = dict(
        type="sphere",
        rsph=equivalent_sphere_radius_m(volume_m3),
        gmin=float(gmin),
        gmax=float(gmax),
        z=float(redshift),
        injection=float(alpha_signed_to_s(alpha_inj)),
        spectrum=str(spectrum),
        cosmology=cosmology,
        verbose=False,
    )
    if spectrum_kwargs:
        kwargs.update(spectrum_kwargs)

    if spectrum == "broken":
        for key in ("gbreak", "dpow"):
            if key not in kwargs:
                raise ValueError(f"spectrum='broken' requires {key!r}.")
    elif spectrum == "aged":
        for key in ("age", "ageb"):
            if key not in kwargs:
                raise ValueError(f"spectrum='aged' requires {key!r}.")

    source = SynchSource(**kwargs)
    # Guard against any future change in pysynch's volume bookkeeping.
    if not np.isclose(source.volume, volume_m3, rtol=1e-6):
        source.volume = float(volume_m3)
    return source


def _pysynch_budget(
    source: Any, method: str, spectrum: str, zeta: float,
    lo: float, hi: float,
) -> Optional[Dict[str, Any]]:
    """
    The energy budget of a normalised pysynch source, or None if unusable.

    Every successful exit from `pysynch_normalise` goes through here, so the
    callers see ONE schema. The gamma_max recovery path used to build its own
    dict with different key names (u_total_J_m3 instead of
    total_energy_density_J_m3, and so on), and the first full run that
    actually needed the recovery died in equipartition_analysis with a
    KeyError. It had also skipped the finiteness and bracket-edge checks.

    Returns None when the energies are non-finite or the field sits on the
    edge of the search bracket; the caller decides whether to widen the
    bracket or try another gamma_max.
    """
    b = float(source.B)
    u_e = float(source.electron_energy_density)
    u_b = float(source.bfield_energy_density)
    u_tot = float(source.total_energy_density)
    if not all(np.isfinite(v) and v > 0 for v in (b, u_e, u_b, u_tot)):
        return None
    if not (lo * 1.001 < b < hi * 0.999):
        return None
    volume = float(source.volume)
    return {
        "method": str(method),
        "spectrum": str(spectrum),
        "B_T": b,
        "B_uG": b / TESLA_PER_MICROGAUSS,
        "volume_m3": volume,
        "zeta": float(zeta),
        "electron_energy_density_J_m3": u_e,
        "particle_energy_density_J_m3": float(zeta) * u_e,
        "magnetic_energy_density_J_m3": u_b,
        "total_energy_density_J_m3": u_tot,
        "total_energy_J": u_tot * volume,
        "pressure_Pa": u_tot / 3.0,
        "electron_number_density_m3": float(
            _synch_c.intne(source.synchnorm)
        ) if _synch_c is not None else float("nan"),
        "bracket_lo_T": lo,
        "bracket_hi_T": hi,
    }


def pysynch_normalise(
    volume_m3: float,
    alpha_inj: float,
    redshift: float,
    norm_freq_hz: float,
    norm_flux_jy: float,
    zeta: float,
    gmin: float,
    gmax: float,
    method: str,
    bmin_t: float = DEFAULT_BMIN_T,
    bmax_t: float = DEFAULT_BMAX_T,
    spectrum: str = "powerlaw",
    spectrum_kwargs: Optional[Dict[str, Any]] = None,
    cosmology=DEFAULT_COSMOLOGY,
    max_bracket_expansions: int = 6,
) -> Dict[str, float]:
    """
    Run one pysynch normalisation ('equipartition', 'minimum_energy' or
    'fixed') and return the full energy budget.

    The magnetic-field bracket is widened automatically if the solution lands
    on a boundary, because a solution pinned at the search edge is not a
    solution -- v1 returned one silently in some configurations.
    """
    require_pysynch()
    if norm_flux_jy <= 0:
        raise ValueError("The normalisation flux density must be positive.")

    lo, hi = float(bmin_t), float(bmax_t)
    last_error: Optional[BaseException] = None

    for _ in range(int(max_bracket_expansions)):
        source = make_synch_source(
            volume_m3, alpha_inj, redshift, gmin, gmax,
            spectrum=spectrum, spectrum_kwargs=spectrum_kwargs,
            cosmology=cosmology,
        )
        try:
            source.normalize(
                float(norm_freq_hz), float(norm_flux_jy),
                method=str(method), brange=(lo, hi), zeta=float(zeta),
            )
        except RuntimeError as exc:                            # noqa: PERF203
            last_error = exc
            # "Not bracketing a root/minimum" -> widen and retry
            lo /= 10.0
            hi *= 10.0
            continue
        except (ZeroDivisionError, FloatingPointError) as exc:
            # pysynch divides by the integrated electron number density
            # (synchro.py: `tno = synch.intne(1)`; `eln = 1.0/tno`). That
            # integral comes back as zero when almost the whole [gmin, gmax]
            # grid sits ABOVE the ageing break, where an aged/broken spectrum
            # has already underflowed to zero in floating point -- the
            # surviving electrons occupy too small a fraction of the grid to
            # register.
            #
            # The cause is therefore a gamma_max that is too LARGE relative
            # to the break, not too small. Measured on this source
            # (gamma_break = 5937): gamma_max = 1e6 (168x the break)
            # normalises fine and gives B_eq = 1.426 uG, stable to 0.7 per
            # cent all the way out to 3e6; 1e7 and 1e8 both fail outright.
            # The old message here advised widening gamma_max, which is
            # exactly backwards and walks the caller further into the wall.
            #
            # So retry downward. Electrons more than ~100x above the break
            # have radiated away, so shrinking gamma_max towards the break
            # discards nothing physical -- and the answer is only accepted
            # while it stays stable, which is checked by the caller.
            recovered = None
            g_hi = float(gmax)
            for _ in range(6):
                g_hi /= 10.0
                if g_hi <= float(gmin) * 10.0:
                    break
                try:
                    src2 = make_synch_source(
                        volume_m3, alpha_inj, redshift, gmin, g_hi,
                        spectrum=spectrum, spectrum_kwargs=spectrum_kwargs,
                        cosmology=cosmology,
                    )
                    src2.normalize(
                        float(norm_freq_hz), float(norm_flux_jy),
                        method=str(method), brange=(lo, hi), zeta=float(zeta),
                    )
                    budget = _pysynch_budget(
                        src2, method, spectrum, zeta, lo, hi,
                    )
                except Exception:                              # noqa: BLE001
                    continue
                if budget is None:
                    continue
                recovered = (budget, g_hi)
                break

            if recovered is None:
                raise RuntimeError(
                    "pysynch could not normalise the source: the electron "
                    "spectrum integrates to zero over "
                    f"gamma = {gmin:g} to {gmax:g}, and retrying with "
                    "successively smaller gamma_max did not recover it. "
                    f"With spectrum='{spectrum}' this means essentially the "
                    "whole Lorentz-factor grid lies above the ageing break. "
                    "LOWER --gamma-max towards the break (not higher), or "
                    "check that the fitted break and age are physical."
                ) from exc

            out, gmax_used = recovered
            out["gamma_max_used"] = float(gmax_used)
            out["gamma_max_requested"] = float(gmax)
            out["gamma_max_note"] = (
                "gamma_max was reduced from the requested value because the "
                "aged/broken electron spectrum underflowed to zero over the "
                "requested grid. Electrons far above the ageing break have "
                "radiated away, so this discards no physical population."
            )
            return out

        b = float(source.B)
        u_vals = (b, float(source.electron_energy_density),
                  float(source.bfield_energy_density),
                  float(source.total_energy_density))
        if not all(np.isfinite(v) and v > 0 for v in u_vals):
            raise RuntimeError(
                "pysynch returned non-finite or non-positive energy densities."
            )

        budget = _pysynch_budget(source, method, spectrum, zeta, lo, hi)
        if budget is not None:
            return budget

        if b <= lo * 1.001:
            lo /= 10.0
        else:
            hi *= 10.0

    raise RuntimeError(
        "The pysynch solution stayed at the edge of the magnetic-field search "
        f"bracket after {max_bracket_expansions} expansions "
        f"(final bracket {lo:.3e}-{hi:.3e} T"
        + (f"; last error: {last_error}" if last_error else "")
        + "). Check the normalisation flux, the volume and gamma_min/gamma_max."
    )


# Retained for API compatibility with v1 call sites.
def pysynch_beq(
    geometry: Dict[str, Any], alpha_inj: float, redshift: float,
    norm_freq_hz: float, norm_flux_jy: float, zeta: float,
    gmin: float, gmax: float, bmin_t: float = DEFAULT_BMIN_T,
    bmax_t: float = DEFAULT_BMAX_T,
) -> Dict[str, float]:
    volume = float(
        geometry.get("adopted_volume_m3")
        or arcsec3_to_m3(geometry["volume_arcsec3"], redshift)
    )
    result = pysynch_normalise(
        volume, alpha_inj, redshift, norm_freq_hz, norm_flux_jy, zeta,
        gmin, gmax, method="equipartition", bmin_t=bmin_t, bmax_t=bmax_t,
    )
    result["Beq_uG"] = result["B_uG"]
    return result


def calibrate_energy_scaling(
    volume_m3: float,
    alpha_inj: float,
    redshift: float,
    norm_freq_hz: float,
    norm_flux_jy: float,
    gmin: float,
    gmax: float,
    spectrum: str = "powerlaw",
    spectrum_kwargs: Optional[Dict[str, Any]] = None,
    cosmology=DEFAULT_COSMOLOGY,
    b_probe: Sequence[float] = (3.0e-10, 1.0e-9, 3.0e-9),
) -> Dict[str, float]:
    """
    Measure the local power law  u_e(B) = A B^(-n)  by direct evaluation.

    Why this exists.  For a power-law electron distribution between FIXED
    Lorentz-factor limits, the electron energy density required to produce a
    given emissivity scales as B^(-(s+1)/2), so u_e(B) is very nearly a pure
    power law and both field solutions have closed forms:

        equipartition   B_eq = (2 mu_0 zeta A)^(1/(2+n))
        minimum energy  B_me = (  mu_0 n zeta A)^(1/(2+n))
        ratio           B_eq/B_me = (2/n)^(1/(2+n))

    Measuring n numerically rather than assuming (s+1)/2 is important, because
    near the gamma_min / gamma_max cut-offs -- or with an aged or broken
    spectrum -- the effective exponent departs from the textbook value, and
    the departure is itself a useful diagnostic (it is reported).

    This calibration turns every subsequent Monte-Carlo sample into three
    floating-point operations instead of a ~50-evaluation bisection, which is
    what makes a full error budget affordable interactively.  The closed-form
    result is cross-checked against a direct pysynch solve by the caller.
    """
    require_pysynch()
    probes: List[Tuple[float, float]] = []
    for b in b_probe:
        # method='fixed' takes a `bfield` kwarg that pysynch_normalise does not
        # forward, so the probe points call pysynch directly.
        source = make_synch_source(
            volume_m3, alpha_inj, redshift, gmin, gmax,
            spectrum=spectrum, spectrum_kwargs=spectrum_kwargs,
            cosmology=cosmology,
        )
        try:
            source.normalize(
                float(norm_freq_hz), float(norm_flux_jy),
                method="fixed", bfield=float(b), zeta=1.0,
            )
        except Exception:                                      # noqa: BLE001
            continue
        u_e = float(source.electron_energy_density)
        if np.isfinite(u_e) and u_e > 0:
            probes.append((float(b), u_e))

    if len(probes) < 2:
        raise RuntimeError(
            "Could not calibrate the u_e(B) scaling: pysynch produced fewer "
            "than two valid probe points."
        )

    b_arr = np.asarray([p[0] for p in probes], dtype=float)
    u_arr = np.asarray([p[1] for p in probes], dtype=float)
    slope, intercept = np.polyfit(np.log(b_arr), np.log(u_arr), 1)
    n_eff = float(-slope)
    a_coeff = float(math.exp(intercept))

    predicted_n = 0.5 * (alpha_signed_to_s(alpha_inj) + 1.0)
    residual = np.log(u_arr) - (intercept + slope * np.log(b_arr))
    return {
        "A": a_coeff,
        "n": n_eff,
        "n_textbook": float(predicted_n),
        "n_departure": float(n_eff - predicted_n),
        "max_log_residual": float(np.max(np.abs(residual))),
        "n_probe_points": len(probes),
        "reference_flux_Jy": float(norm_flux_jy),
        "reference_volume_m3": float(volume_m3),
        "reference_frequency_Hz": float(norm_freq_hz),
        "note": (
            "u_e(B) = A B^-n measured directly from pysynch. A departure of n "
            "from the textbook (s+1)/2 indicates that the gamma limits or the "
            "spectral shape matter at this normalisation frequency."
        ),
    }


def fields_from_scaling(
    scaling: Dict[str, float], zeta: float,
    flux_jy: Optional[float] = None, volume_m3: Optional[float] = None,
) -> Dict[str, float]:
    """
    Closed-form equipartition and minimum-energy solutions from a calibrated
    u_e(B) = A B^-n scaling.

    u_e is linear in the observed flux and inversely proportional to the
    volume, so A can be rescaled analytically for Monte-Carlo samples without
    re-running pysynch.
    """
    a_coeff = float(scaling["A"])
    n = float(scaling["n"])
    if flux_jy is not None:
        a_coeff *= float(flux_jy) / float(scaling["reference_flux_Jy"])
    if volume_m3 is not None:
        a_coeff *= float(scaling["reference_volume_m3"]) / float(volume_m3)

    if not (np.isfinite(n) and n > 0 and np.isfinite(a_coeff) and a_coeff > 0):
        raise ValueError("Invalid u_e(B) scaling for a closed-form solution.")

    exponent = 1.0 / (2.0 + n)
    b_eq = (2.0 * MU_0 * zeta * a_coeff) ** exponent
    b_me = (MU_0 * n * zeta * a_coeff) ** exponent

    def budget(b: float) -> Dict[str, float]:
        u_e = a_coeff * b ** (-n)
        u_b = b * b / (2.0 * MU_0)
        u_tot = zeta * u_e + u_b
        return {
            "B_T": float(b),
            "B_uG": float(b / TESLA_PER_MICROGAUSS),
            "electron_energy_density_J_m3": float(u_e),
            "particle_energy_density_J_m3": float(zeta * u_e),
            "magnetic_energy_density_J_m3": float(u_b),
            "total_energy_density_J_m3": float(u_tot),
            "pressure_Pa": float(u_tot / 3.0),
        }

    return {
        "equipartition": budget(b_eq),
        "minimum_energy": budget(b_me),
        "B_eq_over_B_me": float(b_eq / b_me),
        "B_eq_over_B_me_predicted": float((2.0 / n) ** (1.0 / (2.0 + n))),
        "n": n,
        "A": a_coeff,
    }


# ===========================================================================
# PART 12.  EQUIPARTITION / MINIMUM-ENERGY ANALYSIS WITH A FULL ERROR BUDGET
# ===========================================================================

def equipartition_analysis(
    volume_m3: float,
    alpha_inj: float,
    redshift: float,
    norm_freq_hz: float,
    norm_flux_jy: float,
    *,
    volume_ln_sigma: float = 0.20,
    alpha_err: float = 0.05,
    norm_flux_err_jy: float = 0.0,
    kappa: float = DEFAULT_KAPPA,
    gmin: float = DEFAULT_GAMMA_MIN,
    gmax: float = DEFAULT_GAMMA_MAX,
    filling_factor: float = DEFAULT_FILLING_FACTOR,
    mc_samples: int = DEFAULT_BEQ_MC,
    seed: int = DEFAULT_SEED,
    spectrum: str = "powerlaw",
    spectrum_kwargs: Optional[Dict[str, Any]] = None,
    cosmology=DEFAULT_COSMOLOGY,
    gamma_sensitivity: bool = True,
    verify_closed_form: bool = True,
) -> Dict[str, Any]:
    """
    Equipartition AND minimum-energy fields with a propagated error budget.

    zeta = 1 + kappa is the ratio of total particle to electron energy
    density that pysynch balances against the field.  kappa = 0 (the default)
    is the electron-positron assumption standard for FR-II lobes; kappa = 100
    reproduces the classical proton-dominated Pacholczyk convention and will
    raise B by roughly a factor (1+kappa)^(1/(2+n)) ~ 3.

    Uncertainties propagated by Monte Carlo:
        flux density        Gaussian, truncated at zero
        injection index     Gaussian
        volume              log-normal (the geometry error is multiplicative)
    Reported separately, because they are choices rather than measurements:
        gamma_min/gamma_max sensitivity
        filling factor      (B scales as phi^(-2/7) for n = 3/2)
        kappa               (stated in every output)
    """
    require_pysynch()
    zeta = 1.0 + float(kappa)
    volume_m3 = float(volume_m3)
    rng = np.random.default_rng(int(seed))

    out: Dict[str, Any] = {
        "inputs": {
            "volume_m3": volume_m3,
            "volume_ln_sigma": float(volume_ln_sigma),
            "volume_fractional_error": float(math.expm1(volume_ln_sigma)),
            "alpha_inj_signed": float(alpha_inj),
            "alpha_inj_positive": float(-alpha_inj),
            "alpha_inj_err": float(alpha_err),
            "injection_index_s": float(alpha_signed_to_s(alpha_inj)),
            "redshift": float(redshift),
            "normalisation_frequency_Hz": float(norm_freq_hz),
            "normalisation_flux_Jy": float(norm_flux_jy),
            "normalisation_flux_err_Jy": float(norm_flux_err_jy),
            "kappa_proton_electron_energy_ratio": float(kappa),
            "zeta_uB_over_uparticles": float(zeta),
            "gamma_min": float(gmin),
            "gamma_max": float(gmax),
            "filling_factor": float(filling_factor),
            "electron_spectrum": str(spectrum),
            "electron_spectrum_parameters": dict(spectrum_kwargs or {}),
        }
    }

    # ---- direct pysynch solutions ---------------------------------------
    def _solve_direct(g_max: float) -> Dict[str, Any]:
        res: Dict[str, Any] = {}
        for method in ("equipartition", "minimum_energy"):
            res[method] = pysynch_normalise(
                volume_m3, alpha_inj, redshift, norm_freq_hz, norm_flux_jy,
                zeta=zeta, gmin=gmin, gmax=g_max, method=method,
                spectrum=spectrum, spectrum_kwargs=spectrum_kwargs,
                cosmology=cosmology,
            )
            res[method]["total_energy_J"] = (
                res[method]["total_energy_density_J_m3"] * volume_m3
            )
        return res

    direct = _solve_direct(gmax)

    # If pysynch_normalise had to pull gamma_max down to normalise an aged
    # spectrum, EVERYTHING below must use that same value. The u_e(B)
    # calibration, the Monte Carlo and the closed-form check all call pysynch
    # again; handed the original gamma_max they hit the same underflow, the
    # calibration found "fewer than two valid probe points", and the caller
    # threw the whole aged solve away and fell back to an unbroken power law
    # -- biasing B_eq high by a factor of ~3 on J021926. Re-solving both
    # methods at the one gamma_max that works also keeps equipartition and
    # minimum energy on the same electron population.
    recovered_g = [float(r["gamma_max_used"]) for r in direct.values()
                   if r.get("gamma_max_used")]
    if recovered_g:
        g_req = float(gmax)
        gmax = float(min(recovered_g))
        direct = _solve_direct(gmax)
        out["inputs"]["gamma_max_requested"] = g_req
        out["inputs"]["gamma_max"] = gmax
        out["inputs"]["gamma_max_note"] = (
            f"gamma_max lowered from {g_req:g} to {gmax:g}: the aged electron "
            "spectrum underflows to zero over the requested grid. Electrons "
            "far above the ageing break have radiated away, so this discards "
            "no physical population. Every quantity in this solution uses "
            f"gamma_max = {gmax:g}."
        )
    out["direct"] = direct

    b_eq = direct["equipartition"]["B_T"]
    b_me = direct["minimum_energy"]["B_T"]

    # ---- calibrated scaling and closed-form cross-check ------------------
    scaling = calibrate_energy_scaling(
        volume_m3, alpha_inj, redshift, norm_freq_hz, norm_flux_jy,
        gmin, gmax, spectrum=spectrum, spectrum_kwargs=spectrum_kwargs,
        cosmology=cosmology,
    )
    out["ue_scaling"] = scaling
    closed = fields_from_scaling(scaling, zeta)
    out["closed_form"] = closed

    n_eff = float(scaling["n"])
    qa = {
        "B_eq_direct_uG": b_eq / TESLA_PER_MICROGAUSS,
        "B_eq_closed_form_uG": closed["equipartition"]["B_uG"],
        "B_eq_relative_difference": float(
            abs(closed["equipartition"]["B_T"] - b_eq) / max(b_eq, 1e-30)
        ),
        "B_me_direct_uG": b_me / TESLA_PER_MICROGAUSS,
        "B_me_closed_form_uG": closed["minimum_energy"]["B_uG"],
        "B_me_relative_difference": float(
            abs(closed["minimum_energy"]["B_T"] - b_me) / max(b_me, 1e-30)
        ),
        "B_eq_over_B_me_measured": float(b_eq / b_me),
        "B_eq_over_B_me_analytic": float((2.0 / n_eff) ** (1.0 / (2.0 + n_eff))),
        "B_eq_over_B_me_classical_p2": float((4.0 / 3.0) ** (2.0 / 7.0)),
        "u_min_over_u_eq": float(
            direct["minimum_energy"]["total_energy_density_J_m3"]
            / max(direct["equipartition"]["total_energy_density_J_m3"], 1e-300)
        ),
        "u_eq_equals_2uB": float(
            direct["equipartition"]["total_energy_density_J_m3"]
            / max(2.0 * direct["equipartition"]["magnetic_energy_density_J_m3"], 1e-300)
        ),
        "explanation": (
            "The minimum-energy field must be slightly BELOW the "
            "equipartition field, by (2/n)^(1/(2+n)) ~ 1.05-1.09 for realistic "
            "spectra (exactly (4/3)^(2/7) = 1.0851 for s = 2). u_min/u_eq must "
            "be slightly below 1 -- the energy minimum is shallow, so the two "
            "conventions differ by ~1% in total energy but ~8% in B. Large "
            "deviations here mean the u_e(B) power-law approximation has "
            "broken down, usually because gamma_min or gamma_max is biting at "
            "the normalisation frequency."
        ),
    }
    tolerance = 0.05
    qa["closed_form_validated"] = bool(
        qa["B_eq_relative_difference"] < tolerance
        and qa["B_me_relative_difference"] < tolerance
    )
    qa["ratio_consistent"] = bool(
        abs(qa["B_eq_over_B_me_measured"] - qa["B_eq_over_B_me_analytic"])
        < 0.02 * qa["B_eq_over_B_me_analytic"]
    )
    qa["minimum_energy_below_equipartition"] = bool(b_me < b_eq)
    qa["u_min_not_above_u_eq"] = bool(qa["u_min_over_u_eq"] <= 1.0 + 1e-6)
    out["internal_consistency"] = qa

    if verify_closed_form and not qa["closed_form_validated"]:
        out["monte_carlo_engine"] = "direct pysynch solves (closed form failed validation)"
    else:
        out["monte_carlo_engine"] = "calibrated closed form (validated against pysynch)"

    # ---- alpha-dependence of the scaling ---------------------------------
    # A and n both depend on the injection index, so the Monte Carlo needs
    # them as functions of alpha rather than as constants.
    alpha_sigma = float(max(abs(alpha_err), 1e-4))
    alpha_nodes = np.linspace(
        alpha_inj - 3.0 * alpha_sigma, alpha_inj + 3.0 * alpha_sigma, 7
    )
    ln_a_nodes: List[float] = []
    n_nodes: List[float] = []
    valid_alpha: List[float] = []
    for a in alpha_nodes:
        try:
            sc = calibrate_energy_scaling(
                volume_m3, float(a), redshift, norm_freq_hz, norm_flux_jy,
                gmin, gmax, spectrum=spectrum,
                spectrum_kwargs=spectrum_kwargs, cosmology=cosmology,
            )
        except Exception:                                      # noqa: BLE001
            continue
        if sc["A"] > 0 and sc["n"] > 0:
            valid_alpha.append(float(a))
            ln_a_nodes.append(math.log(sc["A"]))
            n_nodes.append(float(sc["n"]))

    if len(valid_alpha) >= 2:
        alpha_grid = np.asarray(valid_alpha)
        ln_a_grid = np.asarray(ln_a_nodes)
        n_grid = np.asarray(n_nodes)
    else:
        alpha_grid = np.asarray([alpha_inj])
        ln_a_grid = np.asarray([math.log(scaling["A"])])
        n_grid = np.asarray([scaling["n"]])

    # ---- Monte Carlo -----------------------------------------------------
    n_mc = int(max(mc_samples, 0))
    samples: Dict[str, List[float]] = {
        k: [] for k in (
            "B_eq_uG", "B_me_uG", "u_e_eq", "u_B_eq", "u_tot_eq",
            "u_e_me", "u_B_me", "u_min", "E_min_J", "p_min_Pa", "volume_m3",
        )
    }
    ln_sigma_v = float(max(volume_ln_sigma, 1e-6))
    flux_sigma = float(max(norm_flux_err_jy, 0.0))

    n_rejected = 0
    for _ in range(n_mc):
        a = float(rng.normal(alpha_inj, alpha_sigma))
        # keep the injection index inside the physical prior
        a = float(np.clip(a, s_to_alpha_signed(4.0), s_to_alpha_signed(1.5)))
        f = float(rng.normal(norm_flux_jy, flux_sigma)) if flux_sigma > 0 else norm_flux_jy
        if not np.isfinite(f) or f <= 0:
            n_rejected += 1
            continue
        v = volume_m3 * float(rng.lognormal(
            mean=-0.5 * ln_sigma_v ** 2, sigma=ln_sigma_v
        ))
        if not np.isfinite(v) or v <= 0:
            n_rejected += 1
            continue

        if alpha_grid.size > 1:
            ln_a = float(np.interp(a, alpha_grid, ln_a_grid))
            n_i = float(np.interp(a, alpha_grid, n_grid))
        else:
            ln_a = float(ln_a_grid[0])
            n_i = float(n_grid[0])

        a_coeff = math.exp(ln_a) * (f / norm_flux_jy) * (volume_m3 / v)
        if not (np.isfinite(a_coeff) and a_coeff > 0 and n_i > 0):
            n_rejected += 1
            continue

        exponent = 1.0 / (2.0 + n_i)
        beq = (2.0 * MU_0 * zeta * a_coeff) ** exponent
        bme = (MU_0 * n_i * zeta * a_coeff) ** exponent

        ue_eq = a_coeff * beq ** (-n_i)
        ub_eq = beq * beq / (2.0 * MU_0)
        ue_me = a_coeff * bme ** (-n_i)
        ub_me = bme * bme / (2.0 * MU_0)
        u_min = zeta * ue_me + ub_me

        samples["B_eq_uG"].append(beq / TESLA_PER_MICROGAUSS)
        samples["B_me_uG"].append(bme / TESLA_PER_MICROGAUSS)
        samples["u_e_eq"].append(ue_eq)
        samples["u_B_eq"].append(ub_eq)
        samples["u_tot_eq"].append(zeta * ue_eq + ub_eq)
        samples["u_e_me"].append(ue_me)
        samples["u_B_me"].append(ub_me)
        samples["u_min"].append(u_min)
        samples["E_min_J"].append(u_min * v)
        samples["p_min_Pa"].append(u_min / 3.0)
        samples["volume_m3"].append(v)

    mc: Dict[str, Any] = {
        "n_requested": n_mc,
        "n_success": len(samples["B_eq_uG"]),
        "n_rejected": int(n_rejected),
        "engine": out["monte_carlo_engine"],
        "sampled": [
            "normalisation flux (Gaussian, positive)",
            "injection index (Gaussian, clipped to the physical prior)",
            "volume (log-normal from the measured geometry spread)",
        ],
        "not_sampled": [
            "gamma_min / gamma_max (reported as a separate sensitivity)",
            "filling factor (a stated assumption, B ~ phi^(-1/(2+n)))",
            "kappa (a stated assumption)",
            "redshift (negligible for a spectroscopic value)",
        ],
    }
    if len(samples["B_eq_uG"]) >= max(50, int(0.05 * max(n_mc, 1))):
        for key, values in samples.items():
            arr = np.asarray(values, dtype=float)
            mc[key] = {
                "median": float(np.median(arr)),
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0,
                "p16": float(np.percentile(arr, 16)),
                "p84": float(np.percentile(arr, 84)),
                "p2.5": float(np.percentile(arr, 2.5)),
                "p97.5": float(np.percentile(arr, 97.5)),
            }
        mc["status"] = "ok"
    elif n_mc > 0:
        mc["status"] = "insufficient_successful_samples"
    else:
        mc["status"] = "disabled"
    out["monte_carlo"] = mc

    # ---- gamma_min / gamma_max sensitivity --------------------------------
    if gamma_sensitivity:
        sens: Dict[str, Any] = {}
        for gm, gx, tag in (
            (1.0, gmax, "gamma_min=1"),
            (10.0, gmax, "gamma_min=10"),
            (100.0, gmax, "gamma_min=100"),
            (gmin, 1.0e6, "gamma_max=1e6"),
            (gmin, 1.0e8, "gamma_max=1e8"),
        ):
            try:
                res = pysynch_normalise(
                    volume_m3, alpha_inj, redshift, norm_freq_hz, norm_flux_jy,
                    zeta=zeta, gmin=gm, gmax=gx, method="equipartition",
                    spectrum=spectrum, spectrum_kwargs=spectrum_kwargs,
                    cosmology=cosmology,
                )
                sens[tag] = {
                    "gamma_min": gm, "gamma_max": gx,
                    "B_eq_uG": res["B_uG"],
                    "ratio_to_adopted": float(res["B_T"] / b_eq),
                }
            except Exception as exc:                           # noqa: BLE001
                sens[tag] = {"error": f"{type(exc).__name__}: {exc}"}
        ratios = [
            v["ratio_to_adopted"] for v in sens.values()
            if isinstance(v, dict) and "ratio_to_adopted" in v
        ]
        sens["summary"] = {
            "max_ratio": float(max(ratios)) if ratios else None,
            "min_ratio": float(min(ratios)) if ratios else None,
            "note": (
                "gamma_min is the single most influential unmeasured "
                "parameter: lowering it adds low-energy electrons that "
                "radiate nowhere in the observed band but carry energy, "
                "raising u_e and hence B_eq. Quote the assumed value."
            ),
        }
        out["gamma_sensitivity"] = sens

    # ---- filling-factor scaling ------------------------------------------
    out["filling_factor_scaling"] = {
        "filling_factor_applied": float(filling_factor),
        "exponent_on_B": float(-1.0 / (2.0 + n_eff)),
        "note": (
            f"B scales as phi^(-1/(2+n)) = phi^({-1.0/(2.0+n_eff):.4f}) at the "
            "measured n. The adopted volume already includes the filling "
            "factor, so no further correction is needed."
        ),
    }

    # ---- headline numbers -------------------------------------------------
    eq, me = direct["equipartition"], direct["minimum_energy"]
    headline = {
        "B_eq_uG": eq["B_uG"],
        "B_min_energy_uG": me["B_uG"],
        "u_e_equipartition_J_m3": eq["electron_energy_density_J_m3"],
        "u_B_equipartition_J_m3": eq["magnetic_energy_density_J_m3"],
        "u_total_equipartition_J_m3": eq["total_energy_density_J_m3"],
        "u_e_minimum_energy_J_m3": me["electron_energy_density_J_m3"],
        "u_B_minimum_energy_J_m3": me["magnetic_energy_density_J_m3"],
        "u_min_J_m3": me["total_energy_density_J_m3"],
        "E_total_equipartition_J": eq["total_energy_density_J_m3"] * volume_m3,
        "E_min_J": me["total_energy_density_J_m3"] * volume_m3,
        "p_min_Pa": me["total_energy_density_J_m3"] / 3.0,
        "p_min_dyn_cm2": me["total_energy_density_J_m3"] / 3.0 * 10.0,
        "volume_m3": volume_m3,
        "volume_kpc3": float(
            volume_m3 / (1000.0 * 3.0856775814913673e16) ** 3
        ),
    }
    if mc.get("status") == "ok":
        headline.update({
            "B_eq_uG_err": mc["B_eq_uG"]["std"],
            "B_eq_uG_p16": mc["B_eq_uG"]["p16"],
            "B_eq_uG_p84": mc["B_eq_uG"]["p84"],
            "B_min_energy_uG_err": mc["B_me_uG"]["std"],
            "u_min_J_m3_p16": mc["u_min"]["p16"],
            "u_min_J_m3_p84": mc["u_min"]["p84"],
            "E_min_J_p16": mc["E_min_J"]["p16"],
            "E_min_J_p84": mc["E_min_J"]["p84"],
            "p_min_Pa_p16": mc["p_min_Pa"]["p16"],
            "p_min_Pa_p84": mc["p_min_Pa"]["p84"],
        })
    out["headline"] = headline

    # legacy convenience field, kept so existing downstream scripts still work
    out["Beq_uG"] = eq["B_uG"]
    out["B_T"] = eq["B_T"]
    out["volume_m3"] = volume_m3
    out["electron_energy_density_J_m3"] = eq["electron_energy_density_J_m3"]
    out["magnetic_energy_density_J_m3"] = eq["magnetic_energy_density_J_m3"]
    out["total_energy_density_J_m3"] = eq["total_energy_density_J_m3"]
    return out


def sub_equipartition_budget(
    fraction: float,
    b_eq_t: float,
    volume_m3: float,
    alpha_inj: float,
    redshift: float,
    norm_freq_hz: float,
    norm_flux_jy: float,
    zeta: float,
    gmin: float,
    gmax: float,
    spectrum: str = "powerlaw",
    spectrum_kwargs: Optional[Dict[str, Any]] = None,
    break_frequency_hz: Optional[float] = None,
    age_model: str = "JP",
    cosmology=DEFAULT_COSMOLOGY,
) -> Dict[str, Any]:
    """
    Energy budget at a FIXED sub-equipartition field, B = f * B_eq.

    Equipartition is an assumption, not a measurement, and inverse-Compton
    work on radio galaxies routinely finds fields below it. Quoting a single
    equipartition number hides how strongly the derived quantities depend on
    that assumption, so the same budget is recomputed at a prescribed
    fraction of B_eq and reported alongside.

    Nothing is re-fitted: the synchrotron flux is held to the observed value
    and the electron normalisation is re-solved at the fixed field, which is
    exactly what pysynch's method='fixed' does. The consequences are large
    and worth seeing explicitly:

        u_B  scales as B^2            -> f^2   (0.16 at f = 0.4)
        u_e  scales roughly as B^-3/2 -> rises as the field falls, because
                                         fewer, more energetic electrons are
                                         needed to produce the same flux
        tau  scales as sqrt(B)/(B^2 + B_CMB^2)

    The age scaling is NOT monotonic. tau(B) = sqrt(B) / (B^2 + B_CMB^2)
    peaks exactly at B = B_CMB/sqrt(3): setting dtau/dB = 0 gives
    B_CMB^2 = 3B^2. ABOVE that peak a weaker field makes the source look
    older; BELOW it inverse-Compton losses on the CMB dominate the electron
    energy loss and a weaker field makes it look YOUNGER again. Which side
    of the peak each field falls on is therefore reported, because the sign
    of the effect cannot be guessed from the field alone.
    """
    require_pysynch()
    f = float(fraction)
    if not np.isfinite(f) or f <= 0:
        raise ValueError("The sub-equipartition fraction must be positive.")
    b_sub = f * float(b_eq_t)

    source = make_synch_source(
        volume_m3, alpha_inj, redshift, gmin, gmax,
        spectrum=spectrum, spectrum_kwargs=spectrum_kwargs,
        cosmology=cosmology,
    )
    source.normalize(
        float(norm_freq_hz), float(norm_flux_jy),
        method="fixed", bfield=float(b_sub), zeta=float(zeta),
    )
    u_e = float(source.electron_energy_density)
    if not np.isfinite(u_e) or u_e <= 0:
        raise RuntimeError(
            "pysynch returned a non-physical electron energy density at the "
            f"fixed field {b_sub:.4e} T."
        )

    u_b = b_sub ** 2 / (2.0 * MU_0)
    u_part = float(zeta) * u_e
    u_tot = u_part + u_b

    b_cmb = b_cmb_tesla(redshift)
    out: Dict[str, Any] = {
        "fraction_of_B_eq": f,
        "B_T": float(b_sub),
        "B_uG": float(b_sub / TESLA_PER_MICROGAUSS),
        "B_eq_T": float(b_eq_t),
        "B_eq_uG": float(b_eq_t / TESLA_PER_MICROGAUSS),
        "zeta": float(zeta),
        "electron_energy_density_J_m3": u_e,
        "particle_energy_density_J_m3": u_part,
        "magnetic_energy_density_J_m3": u_b,
        "total_energy_density_J_m3": u_tot,
        "total_energy_J": u_tot * float(volume_m3),
        "pressure_Pa": u_tot / 3.0,
        "pressure_dyn_cm2": (u_tot / 3.0) * 10.0,
        "u_B_over_u_particles": (u_b / u_part) if u_part > 0 else None,
        "B_CMB_uG": float(b_cmb / TESLA_PER_MICROGAUSS),
        "B_over_B_CMB": float(b_sub / b_cmb) if b_cmb > 0 else None,
        "B_CMB_over_sqrt3_uG": float(b_cmb / math.sqrt(3.0) / TESLA_PER_MICROGAUSS),
        # tau(B) peaks at B_CMB/sqrt(3); below it inverse-Compton losses win.
        "inverse_compton_dominated": bool(b_sub < b_cmb / math.sqrt(3.0)),
    }

    if break_frequency_hz is not None and np.isfinite(break_frequency_hz) \
            and break_frequency_hz > 0:
        age_eq = synchrotron_age_from_break(
            float(break_frequency_hz), float(b_eq_t), redshift, model=age_model
        )
        age_sub = synchrotron_age_from_break(
            float(break_frequency_hz), float(b_sub), redshift, model=age_model
        )
        out.update({
            "age_model": age_model,
            "break_frequency_Hz": float(break_frequency_hz),
            "spectral_age_Myr": float(age_sub),
            "spectral_age_at_B_eq_Myr": float(age_eq),
            "age_ratio_sub_over_eq": (
                float(age_sub / age_eq) if age_eq > 0 else None
            ),
        })

    out["interpretation"] = (
        f"Budget recomputed at B = {f:g} B_eq with the observed flux held "
        "fixed. u_B falls as B^2 while u_e rises to keep the synchrotron "
        "emission unchanged, so the total energy is NOT minimised here -- "
        "that is the point: it shows how far from minimum the source sits "
        "if the field really is sub-equipartition."
        + (
            " This field lies below B_CMB/sqrt(3) = "
            f"{out['B_CMB_uG'] / math.sqrt(3.0):.4f} uG, where the spectral "
            "age peaks, so inverse-Compton losses on the CMB dominate the "
            "electron energy loss and the age is SHORTER here than at the "
            "peak -- weakening the field further shortens it again."
            if out["inverse_compton_dominated"] else
            " This field lies above B_CMB/sqrt(3) = "
            f"{out['B_CMB_uG'] / math.sqrt(3.0):.4f} uG, so synchrotron "
            "losses still dominate and lowering the field lengthens the age."
        )
    )
    return out

def adopted_field_from_beq(
    beq_uG: float, beq_err_uG: Optional[float] = None,
    fraction: float = ADOPTED_FIELD_FRACTION,
) -> Dict[str, float]:
    """
    B_adopted = fraction * B_eq.

    Kept because inverse-Compton measurements of FR-II lobes systematically
    return fields around 0.3-0.5 B_eq (Croston et al. 2005), so a scaled field
    is a common and defensible modelling choice.  It is a CHOICE, not a
    measurement, and it is labelled as such everywhere it appears.
    """
    if not np.isfinite(beq_uG) or beq_uG <= 0:
        raise ValueError("B_eq must be finite and positive.")
    if not 0.0 < fraction <= 1.0:
        raise ValueError("fraction must lie in (0, 1].")
    out = {
        "fraction_of_Beq": float(fraction),
        "B_adopted_uG": float(fraction * beq_uG),
        "B_adopted_T": float(fraction * beq_uG * TESLA_PER_MICROGAUSS),
        "justification": (
            "Inverse-Compton studies of radio lobes typically find "
            "B ~ 0.3-0.5 B_eq; this is an assumption, not a measurement."
        ),
    }
    if beq_err_uG is not None and np.isfinite(beq_err_uG) and beq_err_uG >= 0:
        out["B_adopted_err_uG"] = float(fraction * beq_err_uG)
        out["B_adopted_err_T"] = float(
            fraction * beq_err_uG * TESLA_PER_MICROGAUSS
        )
    return out


# ===========================================================================
# PART 13.  SELF-CONSISTENT alpha_inj <-> B <-> AGED-SPECTRUM SOLUTION
# ===========================================================================
#
# The three quantities are mutually dependent:
#
#     alpha_inj  ---->  B_eq          (a flatter injected spectrum needs fewer
#                                      electrons, so B_eq changes)
#     B_eq       ---->  Tribble fits  (TJP/TKP are conditional on B)
#     B_eq       ---->  spectral age  (tau ~ B^0.5/(B^2+B_CMB^2))
#     age        ---->  electron spectrum used for the equipartition
#                       normalisation (an aged spectrum has fewer high-energy
#                       electrons than a power law, so a given observed flux
#                       implies a LARGER low-energy population and a different
#                       energy budget)
#
# v1 cut this loop after a single provisional pass.  v2 iterates to a fixed
# point in ln B.  Convergence is typically reached in 2-4 iterations; the full
# trace is written to the output so the reader can see it converge.
# ---------------------------------------------------------------------------

def gamma_break_from_frequency(
    nu_break_obs_hz: float, b_field_t: float, redshift: float
) -> float:
    """
    Lorentz factor of the spectral break.

        nu_c = (3/2) gamma^2 e B sin(theta) / (2 pi m_e)

    evaluated at sin(theta) = 1 and in the SOURCE rest frame.  Different
    authors place the break at the critical frequency of the break Lorentz
    factor or at a factor of a few away from it; the convention used here is
    stated so the number can be rescaled if you prefer another.
    """
    nu_rest = float(nu_break_obs_hz) * (1.0 + float(redshift))
    if not (np.isfinite(nu_rest) and nu_rest > 0):
        raise ValueError("Break frequency must be finite and positive.")
    if not (np.isfinite(b_field_t) and b_field_t > 0):
        raise ValueError("Magnetic field must be finite and positive.")
    return float(math.sqrt(nu_rest / (SYNCH_NU_C * b_field_t)))


def spectrum_from_model(
    model_name: Optional[str],
    break_frequency_hz: Optional[float],
    age_myr: Optional[float],
    b_field_t: float,
    redshift: float,
    allow_aged: bool = True,
) -> Tuple[str, Dict[str, Any], str]:
    """
    Translate a fitted ageing model into a pysynch electron spectrum.

    Returns (spectrum_name, spectrum_kwargs, rationale).

      JP / KP / Tribble  -> pysynch spectrum='aged', which is a JP aged
                            electron distribution implemented in C.  KP is not
                            separately exposed by pysynch, so a KP-preferred
                            fit is mapped onto the JP aged spectrum and the
                            approximation is recorded: KP retains more
                            high-energy electrons, so the JP mapping slightly
                            UNDER-estimates the electron energy content and
                            therefore B.
      CI / CI-OFF        -> pysynch spectrum='broken' with dpow = 1.0, the
                            exact continuous-injection steepening
                            (delta alpha = 0.5).
      anything else      -> 'powerlaw'
    """
    if (
        not allow_aged
        or model_name is None
        or break_frequency_hz is None
        or not np.isfinite(break_frequency_hz)
        or break_frequency_hz <= 0
    ):
        return "powerlaw", {}, "no usable break frequency; unbroken power law"

    name = str(model_name)

    if name.startswith("CI"):
        try:
            gbreak = gamma_break_from_frequency(
                break_frequency_hz, b_field_t, redshift
            )
        except Exception as exc:                               # noqa: BLE001
            return "powerlaw", {}, f"break Lorentz factor failed: {exc}"
        return (
            "broken",
            {"gbreak": float(gbreak), "dpow": 1.0},
            (
                f"{name}: continuous injection, so the electron spectrum is a "
                f"broken power law steepening by delta-s = 1 above "
                f"gamma_break = {gbreak:.3g}."
            ),
        )

    if age_myr is None or not np.isfinite(age_myr) or age_myr <= 0:
        return "powerlaw", {}, "no usable spectral age; unbroken power law"

    approximation = ""
    if name.startswith("KP"):
        approximation = (
            " NOTE: pysynch exposes only the JP aged spectrum, so this "
            "KP-preferred fit is mapped onto JP. KP keeps more high-energy "
            "electrons, so this slightly under-estimates u_e and hence B."
        )
    return (
        "aged",
        {
            "age": float(age_myr) * MYR,
            "ageb": float(b_field_t),
            "cmbage": True,
        },
        (
            f"{name}: single-burst ageing, mapped to the pysynch JP aged "
            f"electron spectrum with tau = {age_myr:.3g} Myr and an ageing "
            f"field of sqrt(B^2 + B_CMB^2)." + approximation
        ),
    )


def self_consistent_solution(
    nu_hz: np.ndarray,
    flux_jy: np.ndarray,
    err_jy: np.ndarray,
    detected: np.ndarray,
    volume_m3: float,
    volume_ln_sigma: float,
    redshift: float,
    norm_index: int,
    cfg: Dict[str, Any],
    observed: Dict[str, Any],
    alpha_override: Optional[float] = None,
    alpha_override_err: Optional[float] = None,
    max_iterations: int = 6,
    tolerance_ln_b: float = 0.01,
    damping: float = 0.5,
) -> Dict[str, Any]:
    """
    Iterate injection index, magnetic field and electron spectrum to a fixed
    point, then run the full equipartition error budget at the solution.
    """
    nu = np.asarray(nu_hz, dtype=float)
    fl = np.asarray(flux_jy, dtype=float)
    er = np.asarray(err_jy, dtype=float)
    det = np.asarray(detected, dtype=bool)

    nu_d, fl_d, er_d = nu[det], fl[det], er[det]
    norm_freq = float(nu[norm_index])
    norm_flux = float(fl[norm_index])
    norm_err = float(er[norm_index])

    zeta = 1.0 + float(cfg["kappa"])

    # --- seed ------------------------------------------------------------
    if alpha_override is not None:
        alpha = float(alpha_override)
        alpha_err = float(
            alpha_override_err if alpha_override_err is not None
            else DEFAULT_ALPHA_ERR_FLOOR
        )
        alpha_source = "user_override"
    else:
        seed_alpha = observed.get("alpha_inj_upper_bound_signed")
        if seed_alpha is None or not np.isfinite(seed_alpha):
            seed_alpha = observed.get("alpha_all", -0.7)
        alpha = float(seed_alpha)
        alpha_err = float(
            observed.get("alpha_low_err") or DEFAULT_ALPHA_ERR_FLOOR
        )
        alpha_source = "observed_bound_seed"

    trace: List[Dict[str, Any]] = []
    spectrum, spectrum_kwargs, rationale = "powerlaw", {}, "initial iterate"
    ensemble: Optional[Dict[str, Any]] = None
    b_current: Optional[float] = None
    converged = False
    # Carried across iterations so the spectrum can be re-derived AT THE
    # CONVERGED FIELD after the loop exits, rather than inheriting whatever
    # `spectrum` happened to be left as by the specific iteration that
    # satisfied the tolerance -- see the note where these are consumed below.
    last_best_model: Optional[str] = None
    last_break_hz: Optional[float] = None
    last_age_myr: Optional[float] = None

    equip_kwargs = dict(
        volume_ln_sigma=volume_ln_sigma,
        kappa=cfg["kappa"], gmin=cfg["gmin"], gmax=cfg["gmax"],
        filling_factor=cfg["filling_factor"],
        cosmology=cfg.get("cosmology", DEFAULT_COSMOLOGY),
    )

    for iteration in range(int(max_iterations)):
        # ---- field at the current alpha and spectrum --------------------
        try:
            solve = pysynch_normalise(
                volume_m3, alpha, redshift, norm_freq, norm_flux,
                zeta=zeta, gmin=cfg["gmin"], gmax=cfg["gmax"],
                method="equipartition", spectrum=spectrum,
                spectrum_kwargs=spectrum_kwargs,
                cosmology=cfg.get("cosmology", DEFAULT_COSMOLOGY),
            )
        except Exception as exc:                               # noqa: BLE001
            # An aged or broken spectrum can fail to normalise part-way
            # through the loop: once the fitted break falls below gamma_min
            # pysynch's electron integral is zero and it divides by it. That
            # is a property of ONE trial spectrum, not of the source, so
            # abandoning the whole iteration for it throws away a converging
            # solution -- here it killed the loop at iteration 1 of 20 and
            # left the field unconverged. Retry this step on an unbroken
            # power law and carry on, recording that the substitution was
            # made so the iterate is not mistaken for an aged-spectrum one.
            retry_error = f"{type(exc).__name__}: {exc}"
            try:
                solve = pysynch_normalise(
                    volume_m3, alpha, redshift, norm_freq, norm_flux,
                    zeta=zeta, gmin=cfg["gmin"], gmax=cfg["gmax"],
                    method="equipartition", spectrum="powerlaw",
                    spectrum_kwargs=None,
                    cosmology=cfg.get("cosmology", DEFAULT_COSMOLOGY),
                )
            except Exception as exc2:                          # noqa: BLE001
                trace.append({
                    "iteration": iteration, "status": "field_solve_failed",
                    "error": retry_error,
                    "powerlaw_retry_error": f"{type(exc2).__name__}: {exc2}",
                    "alpha_inj_signed": alpha, "spectrum": spectrum,
                })
                break
            spectrum, spectrum_kwargs = "powerlaw", None
            trace.append({
                "iteration": iteration,
                "status": "aged_spectrum_failed_used_powerlaw",
                "error": retry_error,
                "alpha_inj_signed": alpha,
                "note": (
                    "The aged spectrum could not be normalised at this "
                    "iteration; an unbroken power law was substituted and "
                    "the loop continued. A power law over-estimates the "
                    "electron count above the break, so this iterate biases "
                    "u_e and B_eq high."
                ),
            })

        b_raw = float(solve["B_T"])

        # -- under-relaxation ------------------------------------------------
        #
        # This fixed-point loop can oscillate instead of converging, and it
        # is not a physical oscillation: alpha_inj comes back from a fit on a
        # DISCRETE grid, so successive iterations snap between neighbouring
        # grid points and B follows them. Measured on a real source, the
        # alpha_inj steps were +/-0.03 against a grid spacing of 0.035 -- one
        # step -- and B bounced by ~2 per cent about a 1 per cent tolerance
        # forever. Adding iterations cannot fix that; each new one just lands
        # on the other side again.
        #
        # Averaging the new estimate with the previous one in LOG space damps
        # the alternation without moving the fixed point: any B that solves
        # the system is unchanged by averaging it with itself. The damping is
        # applied from the second iteration onward, so the first step is free
        # to move as far as it needs to.
        if b_current and b_current > 0 and iteration > 0:
            b_new = float(math.exp(
                (1.0 - damping) * math.log(b_current) + damping * math.log(b_raw)
            ))
        else:
            b_new = b_raw

        delta_ln_b = (
            abs(math.log(b_new) - math.log(b_current))
            if b_current and b_current > 0 else float("inf")
        )

        step: Dict[str, Any] = {
            "iteration": iteration,
            "alpha_inj_signed": float(alpha),
            "alpha_inj_positive": float(-alpha),
            "alpha_inj_err": float(alpha_err),
            "alpha_source": alpha_source,
            "electron_spectrum": spectrum,
            "electron_spectrum_rationale": rationale,
            "B_eq_uG": float(b_new / TESLA_PER_MICROGAUSS),
            "B_eq_uG_undamped": float(b_raw / TESLA_PER_MICROGAUSS),
            "damping": float(damping),
            "delta_ln_B": float(delta_ln_b) if np.isfinite(delta_ln_b) else None,
        }

        if np.isfinite(delta_ln_b) and delta_ln_b < tolerance_ln_b:
            step["status"] = "converged"
            trace.append(step)
            b_current = b_new
            converged = True
            break

        b_current = b_new

        # ---- refit the ensemble conditioned on the new field -------------
        if HAVE_SYNCHROFIT and cfg.get("use_synchrofit", True) and nu_d.size >= 2:
            b_err = None
            if cfg.get("thorough"):
                # a crude but adequate conditioning width for the Tribble refits
                b_err = 0.30 * b_new
            ensemble = fit_spectral_ageing_ensemble(
                nu_d, fl_d, er_d,
                bfield_t=b_new, redshift=redshift, cfg=cfg,
                bfield_err_t=b_err, observed=observed,
            )
            robust = ensemble.get("robust_injection_index", {})
            if (
                alpha_override is None
                and robust.get("alpha_inj_radio") is not None
                and np.isfinite(robust["alpha_inj_radio"])
            ):
                alpha = float(robust["alpha_inj_radio"])
                alpha_err = float(robust.get("alpha_inj_radio_err", alpha_err))
                alpha_source = robust.get("method", "ensemble")

            restricted = ensemble.get("restricted_consensus") or {}
            best_model = restricted.get("best_model") or (
                ensemble.get("free_s_consensus", {}).get("best_model")
            )
            break_hz = restricted.get("break_frequency_Hz")
            if break_hz is None:
                fs = ensemble.get("free_s_consensus", {})
                if fs.get("log10_break_frequency_Hz") is not None:
                    break_hz = 10.0 ** float(fs["log10_break_frequency_Hz"])

            age_myr = None
            if best_model and break_hz:
                model_record = (
                    ensemble.get("restricted_models_fixed_s", {}).get(best_model)
                    or ensemble.get("models", {}).get(best_model)
                )
                if model_record and model_record.get("spectral_age"):
                    age_myr = model_record["spectral_age"].get("tau_Myr")

            spectrum, spectrum_kwargs, rationale = spectrum_from_model(
                best_model, break_hz, age_myr, b_new, redshift,
                allow_aged=cfg.get("use_aged_spectrum", True),
            )
            last_best_model, last_break_hz, last_age_myr = (
                best_model, break_hz, age_myr
            )
            step.update({
                "best_model": best_model,
                "break_frequency_Hz": break_hz,
                "spectral_age_Myr": age_myr,
                "next_alpha_inj_signed": float(alpha),
                "next_electron_spectrum": spectrum,
            })
            step["status"] = "iterated"
        else:
            step["status"] = "no_ageing_backend_single_pass"
            trace.append(step)
            converged = True
            break

        trace.append(step)

    if b_current is None:
        raise RuntimeError(
            "The self-consistent loop could not obtain any magnetic-field "
            "solution. Check the normalisation flux, volume and gamma limits."
        )

    # --- final full analysis at the converged point ------------------------
    #
    # `spectrum` here may still say "powerlaw" for a reason that has nothing
    # to do with the converged field: the convergence check breaks out of the
    # loop BEFORE the per-iteration ensemble refit / spectrum_from_model call
    # that would normally choose the electron spectrum for the NEXT
    # iteration. If the iteration that happened to satisfy the tolerance was
    # also the one where the "aged" attempt failed and fell back, that stale
    # "powerlaw" choice would otherwise survive into the final analysis --
    # not because "aged" is wrong AT THE CONVERGED FIELD, but because it was
    # never tried there. (Seen in practice: an iteration failed on "aged",
    # fell back to "powerlaw" for that one trial B, and that same trial
    # happened to satisfy the 1% tolerance -- so the reported B_eq came from
    # a power law that over-estimates the electron count above the break,
    # even though every recent iterate up to that point had normalised the
    # aged spectrum successfully.)
    #
    # So re-derive the intended spectrum from the last successful ensemble
    # fit, evaluated AT THE FINAL b_current, and give it a genuine attempt
    # here regardless of what the loop's last iterate fell back to.
    if last_best_model is not None:
        spectrum, spectrum_kwargs, rationale = spectrum_from_model(
            last_best_model, last_break_hz, last_age_myr, b_current, redshift,
            allow_aged=cfg.get("use_aged_spectrum", True),
        )
    #
    # The aged/broken electron spectrum can fail to normalise at this last
    # step even when every earlier iterate succeeded: once the fitted break
    # falls below gamma_min, pysynch's electron integral is zero and it
    # divides by it. Losing the whole run to that would be absurd when a
    # perfectly good power-law solution is available, so the failure is
    # recorded and the analysis falls back to an unbroken spectrum -- loudly,
    # because that IS a different physical assumption and over-estimates the
    # electron count above the break.
    spectrum_fallback = None
    try:
        final = equipartition_analysis(
            volume_m3, alpha, redshift, norm_freq, norm_flux,
            alpha_err=alpha_err, norm_flux_err_jy=norm_err,
            mc_samples=int(cfg["mc_samples"]), seed=int(cfg["seed"]),
            spectrum=spectrum, spectrum_kwargs=spectrum_kwargs,
            gamma_sensitivity=True, **equip_kwargs,
        )
    except RuntimeError as exc:
        if spectrum == "powerlaw":
            raise
        spectrum_fallback = {
            "requested_spectrum": spectrum,
            "used_spectrum": "powerlaw",
            "reason": str(exc),
            "warning": (
                "The aged electron spectrum could not be normalised, so the "
                "energetics assume an UNBROKEN power law. That over-estimates "
                "the number of radiating electrons above the break and "
                "therefore biases u_e and B_eq -- usually high, but the size "
                "and even the sign depend on where the break sits relative to "
                "the normalisation frequency (J021926, 6 bands: 3-10x high; "
                "5 bands: 3% low). Treat these values as "
                "provisional. The usual cause is a gamma_max far above the "
                "ageing break, where the aged spectrum underflows to zero: "
                "LOWER --gamma-max towards the break; do not widen it."
            ),
        }
        warnings.warn(
            spectrum_fallback["warning"], RuntimeWarning, stacklevel=2
        )
        spectrum, spectrum_kwargs = "powerlaw", None
        final = equipartition_analysis(
            volume_m3, alpha, redshift, norm_freq, norm_flux,
            alpha_err=alpha_err, norm_flux_err_jy=norm_err,
            mc_samples=int(cfg["mc_samples"]), seed=int(cfg["seed"]),
            spectrum=spectrum, spectrum_kwargs=spectrum_kwargs,
            gamma_sensitivity=True, **equip_kwargs,
        )

    # A power-law reference, so the reader can see how much the aged spectrum
    # actually changed the answer.
    powerlaw_reference = None
    if spectrum != "powerlaw":
        try:
            powerlaw_reference = pysynch_normalise(
                volume_m3, alpha, redshift, norm_freq, norm_flux,
                zeta=zeta, gmin=cfg["gmin"], gmax=cfg["gmax"],
                method="equipartition", spectrum="powerlaw",
                cosmology=cfg.get("cosmology", DEFAULT_COSMOLOGY),
            )
        except Exception:                                      # noqa: BLE001
            powerlaw_reference = None

    return {
        "converged": bool(converged),
        "spectrum_fallback": spectrum_fallback,
        "n_iterations": len(trace),
        "tolerance_ln_B": float(tolerance_ln_b),
        "trace": trace,
        "alpha_inj_signed": float(alpha),
        "alpha_inj_positive": float(-alpha),
        "alpha_inj_err": float(alpha_err),
        "alpha_source": alpha_source,
        "electron_spectrum": spectrum,
        "electron_spectrum_parameters": spectrum_kwargs,
        "electron_spectrum_rationale": rationale,
        "normalisation_band_index": int(norm_index),
        "normalisation_frequency_Hz": norm_freq,
        "normalisation_flux_Jy": norm_flux,
        "ensemble": ensemble,
        "equipartition": final,
        "powerlaw_reference": powerlaw_reference,
        "aged_vs_powerlaw_B_ratio": (
            float(final["direct"]["equipartition"]["B_T"] / powerlaw_reference["B_T"])
            if powerlaw_reference else None
        ),
        "note": (
            "The normalisation band is the lowest-frequency significant "
            "detection, which is the least affected by radiative ageing and "
            "therefore the most robust anchor for the electron energy content."
        ),
    }


# ===========================================================================
# PART 14.  RESOLVED SPECTRAL-INDEX, CURVATURE AND ERROR MAPS
# ===========================================================================
#
# The fit is  ln S_b = ln S_0 + alpha * z_b (+ 0.5 * c * z_b^2),  z = ln(nu/nu_0)
#
# TWO ERROR BUDGETS, DELIBERATELY KEPT APART
# ------------------------------------------
#   statistical : image noise. Independent between bands AND between pixels,
#                 so it averages down when you bin regions and it is what sets
#                 whether a pixel-to-pixel gradient is real.
#   systematic  : the absolute flux scale of each map. Independent between
#                 bands but perfectly COHERENT across every pixel of a given
#                 band, so it shifts the whole alpha map up or down as a rigid
#                 body and NEVER averages down.
#
# v1 added the calibration term into the per-pixel weights, which both
# distorted the weighting and made the resulting error map look like noise
# when in fact most of it was a single global offset.  Structure in an alpha
# map must be judged against sigma_stat; the absolute level must be judged
# against sigma_tot.
# ---------------------------------------------------------------------------

def _weighted_linear_maps(
    logn: np.ndarray,          # (nb,)
    y: np.ndarray,             # (nb, ny, nx)  ln S
    w: np.ndarray,             # (nb, ny, nx)  1/var(ln S), zero where invalid
) -> Dict[str, np.ndarray]:
    """Vectorised weighted straight-line fit over the band axis."""
    with np.errstate(invalid="ignore", divide="ignore"):
        x = logn[:, None, None]
        sw = np.sum(w, axis=0)
        swx = np.sum(w * x, axis=0)
        swy = np.sum(w * y, axis=0)
        swxx = np.sum(w * x * x, axis=0)
        swxy = np.sum(w * x * y, axis=0)

        xbar = np.where(sw > 0, swx / np.where(sw > 0, sw, 1.0), np.nan)
        sxx = swxx - np.where(sw > 0, swx * swx / np.where(sw > 0, sw, 1.0), np.nan)
        sxy = swxy - np.where(sw > 0, swx * swy / np.where(sw > 0, sw, 1.0), np.nan)

        good = np.isfinite(sxx) & (sxx > 0)
        alpha = np.where(good, sxy / np.where(good, sxx, 1.0), np.nan)
        ybar = np.where(sw > 0, swy / np.where(sw > 0, sw, 1.0), np.nan)
        var_alpha = np.where(good, 1.0 / np.where(good, sxx, 1.0), np.nan)

        # residual chi-square
        model = ybar[None, :, :] + alpha[None, :, :] * (x - xbar[None, :, :])
        resid = np.where(w > 0, y - model, 0.0)
        chi2 = np.sum(w * resid * resid, axis=0)

    return {
        "alpha": alpha,
        "var_alpha": var_alpha,
        "sxx": sxx,
        "xbar": xbar,
        "ybar": ybar,
        "chi2": chi2,
        "sum_w": sw,
    }


def spectral_index_maps(
    maps: Sequence[RadioMap],
    source_mask: np.ndarray,
    min_snr: float = DEFAULT_MIN_SNR_MAP,
    min_bands: int = DEFAULT_MIN_FIT_BANDS,
    compute_curvature: bool = True,
    mc_realisations: int = 0,
    seed: int = DEFAULT_SEED,
) -> Dict[str, np.ndarray]:
    """
    Per-pixel spectral index, curvature and error maps -- fully vectorised.

    Returns a dictionary of 2-D maps:
        alpha, alpha_err_stat, alpha_err_sys, alpha_err_total,
        chi2, chi2_reduced, nbands,
        curvature, curvature_err, curvature_significance   (>= 4 bands)
        alpha_err_mc                                       (if requested)

    Only pixels with `min_bands` bands above `min_snr` are fitted.  The
    signal-to-noise floor matters: taking the logarithm of a low-significance
    positive noise excursion biases alpha, because the S > 0 requirement
    truncates the noise distribution asymmetrically.  Three sigma is the
    conventional compromise and the residual bias there is well under
    0.05 in alpha for four bands.
    """
    ordered = sorted(maps, key=lambda m: m.freq_hz)
    freqs = np.asarray([m.freq_hz for m in ordered], dtype=float)
    cube = np.stack([np.asarray(m.data, dtype=float) for m in ordered], axis=0)
    rms = np.asarray([m.rms_jybeam for m in ordered], dtype=float)
    cal = np.asarray([m.cal_frac for m in ordered], dtype=float)

    nband, ny, nx = cube.shape
    if nband < 2:
        raise RuntimeError("At least two frequency maps are required.")
    if min_bands > nband:
        raise RuntimeError(
            f"--spectral-min-bands={min_bands} exceeds the number of maps ({nband})."
        )

    sigma_stat = np.broadcast_to(rms[:, None, None], cube.shape)
    sigma_cal = cal[:, None, None] * np.abs(cube)
    sigma_tot = np.sqrt(sigma_stat ** 2 + sigma_cal ** 2)

    mask3 = np.broadcast_to(source_mask[None, :, :], cube.shape)
    valid = (
        mask3
        & np.isfinite(cube)
        & (cube > 0)
        & np.isfinite(sigma_tot)
        & (sigma_tot > 0)
        & ((cube / np.where(sigma_tot > 0, sigma_tot, np.inf)) >= float(min_snr))
    )

    nvalid = np.count_nonzero(valid, axis=0).astype(np.int16)
    fit_here = nvalid >= int(min_bands)

    safe_cube = np.where(valid, cube, 1.0)
    y = np.log(safe_cube)
    # statistical weights only: w = (S/sigma_stat)^2 = 1/var(ln S)_stat
    w = np.where(valid, (safe_cube / sigma_stat) ** 2, 0.0)
    logn = np.log(freqs)

    fit = _weighted_linear_maps(logn, y, w)
    alpha = np.where(fit_here, fit["alpha"], np.nan)
    var_stat = np.where(fit_here, fit["var_alpha"], np.nan)
    alpha_err_stat = np.sqrt(np.maximum(var_stat, 0.0))

    # --- coherent calibration systematic -----------------------------------
    # alpha = sum_b c_b * y_b   with   c_b = w_b (x_b - xbar) / Sxx
    with np.errstate(invalid="ignore", divide="ignore"):
        x = logn[:, None, None]
        sxx_safe = np.where(fit["sxx"] > 0, fit["sxx"], np.nan)
        coeff = w * (x - fit["xbar"][None, :, :]) / sxx_safe[None, :, :]
        # sigma of ln S from a fractional flux-scale error f is ln(1+f) ~= f
        sigma_ln_cal = np.log1p(cal)[:, None, None]
        var_sys = np.sum(np.where(valid, (coeff * sigma_ln_cal) ** 2, 0.0), axis=0)
    alpha_err_sys = np.where(fit_here, np.sqrt(np.maximum(var_sys, 0.0)), np.nan)
    alpha_err_total = np.sqrt(
        np.maximum(alpha_err_stat ** 2 + alpha_err_sys ** 2, 0.0)
    )

    # Per-band coherent sensitivity  d(alpha) / d ln(flux scale of band b).
    #
    # These coefficients satisfy  sum_b c_b = 0  exactly at every pixel, which
    # says that scaling EVERY band by the same factor cannot change a spectral
    # index -- only the RELATIVE flux scales between bands matter. So the
    # figure to worry about is the scale agreement between the surveys, not
    # each survey's absolute accuracy. (The medians below are taken over
    # pixels with slightly different weights, so they need not sum to exactly
    # zero.)
    # A single realisation of the flux-scale errors moves the WHOLE map by
    # sum_b (sensitivity_b * ln(1+f_b)); this is the quantity to quote when
    # asking "could a 5% scale error between these two surveys explain the
    # spectral index I measured?".  The coefficients depend weakly on position
    # through the weights, so the median over fitted pixels is reported.
    cal_sensitivity = []
    for b, m in enumerate(ordered):
        c_b = np.where(fit_here & valid[b], coeff[b], np.nan)
        good_c = c_b[np.isfinite(c_b)]
        cal_sensitivity.append({
            "label": m.label,
            "frequency_GHz": m.freq_hz / 1e9,
            "cal_frac": float(m.cal_frac),
            "d_alpha_d_lnscale_median": (
                float(np.median(good_c)) if good_c.size else float("nan")
            ),
            "d_alpha_d_lnscale_spread": (
                float(np.std(good_c)) if good_c.size > 1 else 0.0
            ),
            "alpha_shift_for_one_sigma_scale_error": (
                float(np.median(good_c) * math.log1p(m.cal_frac))
                if good_c.size else float("nan")
            ),
        })

    dof = np.maximum(nvalid - 2, 0)
    chi2 = np.where(fit_here, fit["chi2"], np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        chi2_red = np.where(dof > 0, chi2 / np.where(dof > 0, dof, 1), np.nan)

    products: Dict[str, Any] = {
        "alpha": alpha,
        "alpha_err_stat": alpha_err_stat,
        "alpha_err_sys": alpha_err_sys,
        "alpha_err_total": alpha_err_total,
        "chi2": chi2,
        "chi2_reduced": chi2_red,
        "nbands": nvalid,
        "_cal_sensitivity": cal_sensitivity,
    }

    # --- curvature ---------------------------------------------------------
    if compute_curvature and nband >= 4:
        fit4 = nvalid >= 4
        z = (logn[:, None, None] - fit["xbar"][None, :, :])
        z = np.where(valid, z, 0.0)
        s0 = np.sum(w, axis=0)
        s1 = np.sum(w * z, axis=0)
        s2 = np.sum(w * z ** 2, axis=0)
        s3 = np.sum(w * z ** 3, axis=0)
        s4 = np.sum(w * z ** 4, axis=0)
        t0 = np.sum(w * y, axis=0)
        t1 = np.sum(w * z * y, axis=0)
        t2 = np.sum(w * z ** 2 * y, axis=0)

        mat = np.stack([
            np.stack([s0, s1, s2], axis=-1),
            np.stack([s1, s2, s3], axis=-1),
            np.stack([s2, s3, s4], axis=-1),
        ], axis=-2)                                   # (ny, nx, 3, 3)
        vec = np.stack([t0, t1, t2], axis=-1)         # (ny, nx, 3)

        det = np.linalg.det(mat)
        solvable = fit4 & np.isfinite(det) & (np.abs(det) > 1e-30)

        curvature = np.full((ny, nx), np.nan)
        curvature_err = np.full((ny, nx), np.nan)
        if np.any(solvable):
            idx = np.where(solvable)
            sub_m = mat[idx]
            sub_v = vec[idx]
            try:
                coeffs = np.linalg.solve(sub_m, sub_v[..., None])[..., 0]
                cov = np.linalg.inv(sub_m)
                # ln S = a0 + a1 z + a2 z^2  ->  d2 lnS/d(ln nu)^2 = 2 a2
                curvature[idx] = 2.0 * coeffs[:, 2]
                curvature_err[idx] = 2.0 * np.sqrt(
                    np.maximum(cov[:, 2, 2], 0.0)
                )
            except np.linalg.LinAlgError:
                pass

        with np.errstate(invalid="ignore", divide="ignore"):
            significance = np.abs(curvature) / curvature_err
        products.update({
            "curvature": curvature,
            "curvature_err": curvature_err,
            "curvature_significance": significance,
        })

    # --- optional Monte-Carlo error map ------------------------------------
    if int(mc_realisations) > 0:
        rng = np.random.default_rng(int(seed))
        acc = np.zeros((ny, nx), dtype=float)
        acc2 = np.zeros((ny, nx), dtype=float)
        count = np.zeros((ny, nx), dtype=float)
        for _ in range(int(mc_realisations)):
            noisy = cube + rng.normal(0.0, 1.0, cube.shape) * sigma_stat
            ok = valid & (noisy > 0)
            safe = np.where(ok, noisy, 1.0)
            yy = np.log(safe)
            ww = np.where(ok, (safe / sigma_stat) ** 2, 0.0)
            trial = _weighted_linear_maps(logn, yy, ww)
            a = trial["alpha"]
            finite = np.isfinite(a) & fit_here
            acc[finite] += a[finite]
            acc2[finite] += a[finite] ** 2
            count[finite] += 1.0
        with np.errstate(invalid="ignore", divide="ignore"):
            mean = np.where(count > 1, acc / np.maximum(count, 1), np.nan)
            var = np.where(
                count > 1,
                acc2 / np.maximum(count, 1) - mean ** 2,
                np.nan,
            )
        products["alpha_err_mc"] = np.sqrt(np.maximum(var, 0.0))
        products["alpha_mc_mean"] = mean
        products["alpha_mc_bias"] = mean - alpha
        products["alpha_mc_n"] = count

    return products


def spectral_index_map_summary(
    products: Dict[str, np.ndarray],
    maps: Sequence[RadioMap],
    common_beam: Beam,
    wcs: WCS,
) -> Dict[str, Any]:
    """Scalar diagnostics that belong with the alpha maps in a paper."""
    alpha = products["alpha"]
    finite = np.isfinite(alpha)
    n_fitted = int(np.count_nonzero(finite))
    pix_per_beam = beam_area_pixels(common_beam, wcs)

    summary: Dict[str, Any] = {
        "n_pixels_fitted": n_fitted,
        "n_independent_beams": float(n_fitted / max(pix_per_beam, 1e-12)),
        "pixels_per_beam": float(pix_per_beam),
        "correlation_warning": (
            f"Adjacent pixels are NOT independent: the common beam spans "
            f"{pix_per_beam:.1f} pixels. Any statistic computed over pixels "
            f"(a histogram width, a fitted gradient) must use the "
            f"{n_fitted / max(pix_per_beam, 1e-12):.0f} independent beams, not "
            f"the {n_fitted} pixels, or its significance will be overstated by "
            f"roughly sqrt({pix_per_beam:.0f})."
        ),
        "alpha_convention": "S_nu ~ nu^alpha with alpha negative",
    }

    if n_fitted:
        vals = alpha[finite]
        summary.update({
            "alpha_median": float(np.median(vals)),
            "alpha_mean": float(np.mean(vals)),
            "alpha_std": float(np.std(vals, ddof=1)) if vals.size > 1 else 0.0,
            "alpha_p16": float(np.percentile(vals, 16)),
            "alpha_p84": float(np.percentile(vals, 84)),
            "alpha_min": float(np.min(vals)),
            "alpha_max": float(np.max(vals)),
            "alpha_range": float(np.ptp(vals)),
        })
        for key, label in (
            ("alpha_err_stat", "statistical"),
            ("alpha_err_sys", "calibration_systematic"),
            ("alpha_err_total", "total"),
        ):
            arr = products.get(key)
            if arr is None:
                continue
            good = np.isfinite(arr)
            if np.any(good):
                summary[f"median_{label}_error"] = float(np.median(arr[good]))

        sys_arr = products.get("alpha_err_sys")
        if sys_arr is not None and np.any(np.isfinite(sys_arr)):
            good = np.isfinite(sys_arr)
            entry = {
                "median": float(np.median(sys_arr[good])),
                "spread": float(np.std(sys_arr[good])),
                "per_band_sensitivity": products.get("_cal_sensitivity"),
                "note": (
                    "One realisation of the flux-scale errors shifts the whole "
                    "alpha map in the SAME DIRECTION, by "
                    "sum_b (d alpha / d ln scale_b) * ln(1+f_b). The magnitude "
                    "varies mildly across the map because the fit weights do, "
                    "but the shift is not independent between pixels and does "
                    "not average down. Quote it with any ABSOLUTE spectral "
                    "index; exclude it when judging spatial structure."
                ),
            }
            sens = products.get("_cal_sensitivity") or []
            shifts = [
                s["alpha_shift_for_one_sigma_scale_error"] for s in sens
                if np.isfinite(s.get("alpha_shift_for_one_sigma_scale_error", np.nan))
            ]
            if shifts:
                entry["worst_case_coherent_shift"] = float(
                    sum(abs(v) for v in shifts)
                )
                entry["rms_coherent_shift"] = float(
                    math.sqrt(sum(v * v for v in shifts))
                )
            summary["calibration_systematic_is_coherent"] = entry

        stat = products.get("alpha_err_stat")
        if stat is not None and np.any(np.isfinite(stat)):
            good = finite & np.isfinite(stat)
            spread = float(np.std(alpha[good], ddof=1)) if np.count_nonzero(good) > 1 else 0.0
            typical = float(np.median(stat[good]))
            summary["structure_significance"] = {
                "alpha_spatial_spread": spread,
                "typical_statistical_error": typical,
                "spread_over_error": float(spread / max(typical, 1e-12)),
                "interpretation": (
                    "Spatial variation is significant relative to the noise"
                    if spread > 2.0 * typical else
                    "The observed spread is comparable to the noise: the map "
                    "may be consistent with a single spectral index"
                ),
            }

    curv = products.get("curvature")
    if curv is not None and np.any(np.isfinite(curv)):
        sig = products.get("curvature_significance")
        good = np.isfinite(curv)
        summary["curvature"] = {
            "median": float(np.median(curv[good])),
            "fraction_significant_negative": float(
                np.mean((curv[good] < 0) & (sig[good] >= 2.0))
                if sig is not None else float("nan")
            ),
            "interpretation": (
                "Negative curvature is the resolved signature of radiative "
                "ageing; a positive value indicates absorption or a second, "
                "flatter component in the beam."
            ),
        }
    return summary


# ===========================================================================
# PART 15.  LOCAL DEPTH, ADAPTIVE BINNING AND RESOLVED AGEING MAPS
# ===========================================================================

def local_width_map(
    mask: np.ndarray,
    wcs: WCS,
    beam: Beam,
    direction_pixel: Tuple[float, float],
    direction_sky_pa_deg: float,
    deconv: str = "quadrature",
    max_steps: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    """
    Transverse extent of the source through every masked pixel.

    For each pixel the mask is walked in both directions along
    `direction_pixel` until it leaves the mask; the run length is the local
    width of the structure that pixel belongs to.  Working in the ORIGINAL
    pixel frame (rather than rotating the image) avoids resampling artefacts
    and keeps disconnected lobes strictly separate: a ray that exits the mask
    stops, so a gap between two lobes is never filled in.

    Returns the observed width map, the beam-deconvolved width map, and the
    effective line-of-sight depth  (pi/4) * d_deconvolved,  which is the
    factor that turns a summed pixel area into the volume of a chain of
    circular-cross-section cylinders:

        V = sum_pixels  A_pixel * (pi/4) * d_deconvolved
    """
    ny, nx = mask.shape
    sx, sy = pixel_scales_arcsec(wcs)

    dx, dy = float(direction_pixel[0]), float(direction_pixel[1])
    scale = max(abs(dx), abs(dy))
    if scale <= 0:
        raise ValueError("Degenerate direction vector for the width scan.")
    dx, dy = dx / scale, dy / scale                # one pixel per step
    step_arcsec = math.hypot(dx * sx, dy * sy)

    if max_steps is None:
        max_steps = int(max(ny, nx)) + 2

    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        empty = np.full(mask.shape, np.nan)
        return {"width_obs_arcsec": empty, "width_dec_arcsec": empty.copy(),
                "depth_arcsec": empty.copy()}

    counts = np.zeros(ys.size, dtype=np.int32)
    for sign in (+1.0, -1.0):
        alive = np.ones(ys.size, dtype=bool)
        for k in range(1, max_steps):
            if not alive.any():
                break
            px = np.rint(xs + sign * dx * k).astype(np.int64)
            py = np.rint(ys + sign * dy * k).astype(np.int64)
            inside = (
                (px >= 0) & (px < nx) & (py >= 0) & (py < ny)
            )
            hit = np.zeros(ys.size, dtype=bool)
            idx = np.where(alive & inside)[0]
            if idx.size:
                hit[idx] = mask[py[idx], px[idx]]
            alive &= hit
            counts += alive
    # +1 for the pixel itself
    width_obs = (counts + 1).astype(float) * step_arcsec

    theta_beam = beam_extent_arcsec(beam, direction_sky_pa_deg)
    if deconv == "quadrature":
        width_dec = np.sqrt(np.maximum(width_obs ** 2 - theta_beam ** 2, 0.0))
    elif deconv == "linear":
        width_dec = np.maximum(width_obs - theta_beam, 0.0)
    else:
        raise ValueError(f"Unknown deconvolution mode {deconv!r}")
    width_dec = np.maximum(width_dec, step_arcsec)

    obs_map = np.full(mask.shape, np.nan)
    dec_map = np.full(mask.shape, np.nan)
    obs_map[ys, xs] = width_obs
    dec_map[ys, xs] = width_dec

    # Effective per-pixel depth.
    #
    # A transverse segment of OBSERVED width d_obs occupies d_obs/pix pixels
    # in its column, but the physical cylinder we want to build has the
    # DECONVOLVED diameter d_dec, so its column volume is (pi/4) d_dec^2 pix.
    # Distributing that over the d_obs/pix pixels actually present gives
    #
    #     depth = (pi/4) * d_dec^2 / d_obs
    #
    # so that  sum_pixels A_pixel * depth  reproduces  (pi/4) d_dec^2 L
    # exactly.  Using (pi/4) d_dec here instead -- the obvious but wrong
    # choice -- over-estimates the volume by the factor d_obs/d_dec, which is
    # 9% for a structure twice the beam width and much worse near the beam.
    depth_map = (math.pi / 4.0) * dec_map ** 2 / np.where(
        obs_map > 0, obs_map, np.nan
    )

    return {
        "width_obs_arcsec": obs_map,
        "width_dec_arcsec": dec_map,
        "depth_arcsec": depth_map,
        "beam_extent_arcsec": float(theta_beam),
        "step_arcsec": float(step_arcsec),
    }


def integrated_volume_raytrace(
    mask: np.ndarray, wcs: WCS, beam: Beam,
    axes: Optional[Dict[str, Any]] = None,
    deconv: str = "quadrature",
) -> Dict[str, Any]:
    """
    Volume as the sum over pixels of  A_pixel * (pi/4) * d_local.

    Equivalent to the column-integration of Part 6 but computed without
    rotating the image, and it yields a per-pixel depth map that the resolved
    equipartition maps reuse.
    """
    axes = axes or principal_axes(mask, wcs)
    widths = local_width_map(
        mask, wcs, beam,
        direction_pixel=axes["minor_vec_pixel"],
        direction_sky_pa_deg=axes["minor_pa_deg"],
        deconv=deconv,
    )
    pix_area = pixel_area_arcsec2(wcs)
    depth = widths["depth_arcsec"]
    finite = np.isfinite(depth)
    volume_arcsec3 = float(np.sum(depth[finite]) * pix_area)
    return {
        "volume_arcsec3": volume_arcsec3,
        "deconvolution": deconv,
        "depth_map_arcsec": depth,
        "width_obs_map_arcsec": widths["width_obs_arcsec"],
        "width_dec_map_arcsec": widths["width_dec_arcsec"],
        "beam_extent_arcsec": widths["beam_extent_arcsec"],
        "median_width_arcsec": float(np.nanmedian(widths["width_obs_arcsec"])),
        "method": "ray-traced local width, circular cross-section",
    }


# ---------------------------------------------------------------------------
# Adaptive binning
# ---------------------------------------------------------------------------

@dataclass
class Region:
    index: int
    pixels: Tuple[np.ndarray, np.ndarray]     # (y indices, x indices)
    bbox: Tuple[int, int, int, int]           # y0, y1, x0, x1
    n_pixels: int
    flux_jy: np.ndarray                       # per band
    flux_err_jy: np.ndarray
    snr_min: float          # minimum THERMAL S/N across bands
    volume_m3: float = float("nan")
    centroid: Tuple[float, float] = (float("nan"), float("nan"))


def _region_photometry(
    cube: np.ndarray, rms: np.ndarray, cal: np.ndarray,
    pix_per_beam: float, sel: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Integrated flux, TOTAL error, and STATISTICAL signal-to-noise per band
    inside a boolean pixel selection.

    The two error concepts are needed for different jobs and must not be
    interchanged:

      * The returned error includes the flux-scale term, because when a
        region's spectrum is fitted across bands each band's calibration is an
        independent nuisance and does belong in the fit.
      * The returned signal-to-noise is THERMAL ONLY, because that is what
        the binning decision needs.  Folding the calibration term into the
        binning criterion caps the achievable signal-to-noise at 1/cal_frac
        (only 20 for a 5% flux scale), so any target at or above that becomes
        silently unreachable and the binner returns nothing however bright the
        source is.  A coherent flux-scale error does not limit how finely the
        map can be divided.
    """
    npix = int(np.count_nonzero(sel))
    if npix == 0:
        n = cube.shape[0]
        return np.zeros(n), np.full(n, np.inf), 0.0
    nbeam = npix / max(pix_per_beam, 1e-12)
    flux = np.nansum(np.where(sel[None, :, :], cube, np.nan), axis=(1, 2)) / max(
        pix_per_beam, 1e-12
    )
    stat = rms * math.sqrt(max(nbeam, 1.0))
    err = np.sqrt(stat ** 2 + (cal * np.abs(flux)) ** 2)
    with np.errstate(invalid="ignore", divide="ignore"):
        snr_stat = np.where(stat > 0, flux / stat, 0.0)
    return flux, err, float(np.nanmin(snr_stat))


def adaptive_bin(
    mask: np.ndarray,
    common_maps: Sequence[RadioMap],
    common_beam: Beam,
    wcs: WCS,
    target_snr: float = 10.0,
    min_beams_per_region: float = 1.0,
    max_depth: int = 8,
) -> List[Region]:
    """
    Quadtree binning to a target THERMAL signal-to-noise in the WORST band.

    Per-pixel ageing fits are hopeless with three or four frequencies, but
    binned regions are perfectly tractable.  The tree splits only while every
    child still reaches `target_snr` in the least sensitive band, so regions
    are as small as the data allow and no region is ever below the target.
    The recursion floor is one synthesised beam: subdividing below the beam
    would create regions that are not independent measurements.

    The worst band is the binding constraint on purpose -- a break frequency
    is determined by the high-frequency points, so a region that is bright at
    150 MHz but undetected at 1.4 GHz constrains nothing.

    `target_snr` is measured against THERMAL noise only (see
    `_region_photometry`); the per-region errors used for the subsequent
    spectral fits do include the flux-scale term.
    """
    ordered = sorted(common_maps, key=lambda m: m.freq_hz)
    cube = np.stack([np.asarray(m.data, dtype=float) for m in ordered], axis=0)
    cube = np.where(np.isfinite(cube), cube, 0.0)
    rms = np.asarray([m.rms_jybeam for m in ordered], dtype=float)
    cal = np.asarray([m.cal_frac for m in ordered], dtype=float)

    pix_per_beam = beam_area_pixels(common_beam, wcs)
    min_pixels = max(int(round(min_beams_per_region * pix_per_beam)), 4)
    min_side = max(int(round(math.sqrt(min_pixels))), 2)

    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return []
    root = (int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1)

    leaves: List[Tuple[int, int, int, int]] = []

    def recurse(box: Tuple[int, int, int, int], depth: int) -> None:
        y0, y1, x0, x1 = box
        sel = np.zeros(mask.shape, dtype=bool)
        sel[y0:y1, x0:x1] = mask[y0:y1, x0:x1]
        if np.count_nonzero(sel) < min_pixels:
            return
        _, _, snr = _region_photometry(cube, rms, cal, pix_per_beam, sel)
        if snr < target_snr:
            return
        if depth >= max_depth or (y1 - y0) < 2 * min_side or (x1 - x0) < 2 * min_side:
            leaves.append(box)
            return

        ym, xm = (y0 + y1) // 2, (x0 + x1) // 2
        children = [
            (y0, ym, x0, xm), (y0, ym, xm, x1),
            (ym, y1, x0, xm), (ym, y1, xm, x1),
        ]
        # Split into whichever children can stand on their own.
        #
        # Requiring ALL FOUR children to reach the target before splitting
        # anything sounds conservative, but on a real source it is
        # catastrophic: a wide-angle tail is a bright head plus a thin
        # diagonal tail, so at the very first split one quadrant is nearly
        # empty sky, fails the test, and blocks the subdivision of the other
        # three. The whole source then comes back as a SINGLE region -- which
        # is exactly what it did here -- and the resolved ageing map carries
        # one age with no spatial information at all.
        #
        # Each child is judged on its own instead. Children that reach the
        # target are recursed into (and will subdivide further while they
        # can); children that cannot are dropped, which is the same rule the
        # leaf test above already applies. If nothing survives, the parent
        # stays whole.
        viable = []
        for child in children:
            cy0, cy1, cx0, cx1 = child
            csel = np.zeros(mask.shape, dtype=bool)
            csel[cy0:cy1, cx0:cx1] = mask[cy0:cy1, cx0:cx1]
            npix = int(np.count_nonzero(csel))
            if npix < min_pixels:
                continue
            _, _, csnr = _region_photometry(cube, rms, cal, pix_per_beam, csel)
            if csnr >= target_snr:
                viable.append(child)

        if viable:
            for child in viable:
                recurse(child, depth + 1)
        else:
            leaves.append(box)

    recurse(root, 0)

    regions: List[Region] = []
    for i, (y0, y1, x0, x1) in enumerate(leaves):
        sel = np.zeros(mask.shape, dtype=bool)
        sel[y0:y1, x0:x1] = mask[y0:y1, x0:x1]
        npix = int(np.count_nonzero(sel))
        if npix < min_pixels:
            continue
        flux, err, snr = _region_photometry(cube, rms, cal, pix_per_beam, sel)
        pys, pxs = np.nonzero(sel)
        regions.append(Region(
            index=i, pixels=(pys, pxs), bbox=(y0, y1, x0, x1),
            n_pixels=npix, flux_jy=flux, flux_err_jy=err, snr_min=snr,
            centroid=(float(np.mean(pxs)), float(np.mean(pys))),
        ))
    return regions


# ---------------------------------------------------------------------------
# pysynch emissivity templates
# ---------------------------------------------------------------------------

def build_pysynch_templates(
    kind: str,
    frequencies_obs_hz: np.ndarray,
    alpha_inj: float,
    b_field_t: float,
    redshift: float,
    gmin: float,
    gmax: float,
    n_nodes: int = 60,
    age_range_myr: Tuple[float, float] = (0.05, 500.0),
    log_gamma_break_range: Tuple[float, float] = (2.0, 7.0),
    cosmology=DEFAULT_COSMOLOGY,
) -> Dict[str, Any]:
    """
    Precompute a family of model spectra ONCE.

    Because the injection index and the magnetic field are held fixed across
    the map, every region shares the same set of model SHAPES and differs only
    in normalisation.  Building the shapes once (n_nodes pysynch evaluations
    per frequency, of order a few hundred in total) and then fitting each
    region with a single closed-form normalisation reduces a resolved ageing
    analysis from hours to seconds.

    kind:
        'JP' -> pysynch spectrum='aged' on a grid of synchrotron ages
        'CI' -> pysynch spectrum='broken' (dpow = 1) on a grid of break
                Lorentz factors
    """
    require_pysynch()
    freqs_rest = np.asarray(frequencies_obs_hz, dtype=float) * (1.0 + float(redshift))
    volume_dummy = 1.0e60      # arbitrary: only the spectral SHAPE is used

    if kind == "JP":
        nodes = np.logspace(
            math.log10(age_range_myr[0]), math.log10(age_range_myr[1]), int(n_nodes)
        )
        node_name = "age_Myr"
    elif kind == "CI":
        nodes = np.logspace(
            log_gamma_break_range[0], log_gamma_break_range[1], int(n_nodes)
        )
        node_name = "gamma_break"
    else:
        raise ValueError("Template kind must be 'JP' or 'CI'.")

    templates = np.full((nodes.size, freqs_rest.size), np.nan)
    for i, node in enumerate(nodes):
        if kind == "JP":
            spectrum, kwargs = "aged", {
                "age": float(node) * MYR,
                "ageb": float(b_field_t),
                "cmbage": True,
            }
        else:
            spectrum, kwargs = "broken", {"gbreak": float(node), "dpow": 1.0}
        try:
            source = make_synch_source(
                volume_dummy, alpha_inj, redshift, gmin, gmax,
                spectrum=spectrum, spectrum_kwargs=kwargs, cosmology=cosmology,
            )
            source.B = float(b_field_t)
            source.synchnorm = 1.0
            values = np.asarray(source.emiss(freqs_rest), dtype=float).ravel()
            if values.size == freqs_rest.size and np.all(np.isfinite(values)):
                templates[i, :] = values
        except Exception:                                      # noqa: BLE001
            continue

    usable = np.all(np.isfinite(templates) & (templates > 0), axis=1)
    if np.count_nonzero(usable) < 5:
        raise RuntimeError(
            f"Only {int(np.count_nonzero(usable))} usable {kind} templates were "
            "produced by pysynch; check gamma_min/gamma_max and the field."
        )

    return {
        "kind": kind,
        "node_name": node_name,
        "nodes": nodes[usable],
        "templates": templates[usable, :],
        "frequencies_obs_Hz": np.asarray(frequencies_obs_hz, dtype=float),
        "frequencies_rest_Hz": freqs_rest,
        "b_field_T": float(b_field_t),
        "b_field_uG": float(b_field_t / TESLA_PER_MICROGAUSS),
        "alpha_inj_signed": float(alpha_inj),
        "injection_index_s": float(alpha_signed_to_s(alpha_inj)),
        "gamma_min": float(gmin),
        "gamma_max": float(gmax),
        "n_usable": int(np.count_nonzero(usable)),
        "engine": "pysynch (GSL) emissivity",
    }


def fit_region_to_templates(
    flux: np.ndarray, err: np.ndarray, templates: Dict[str, Any],
) -> Optional[Dict[str, float]]:
    """
    Fit one region against a template family.

    The normalisation is eliminated analytically at every node
    (A = sum(f m / e^2) / sum(m^2 / e^2)), so the search reduces to a 1-D
    chi-square scan.  The confidence interval is the Delta chi2 = 1 crossing
    interpolated between nodes.
    """
    f = np.asarray(flux, dtype=float)
    e = np.asarray(err, dtype=float)
    good = np.isfinite(f) & np.isfinite(e) & (e > 0) & (f > 0)
    if np.count_nonzero(good) < 3:
        return None

    models = np.asarray(templates["templates"], dtype=float)[:, good]
    nodes = np.asarray(templates["nodes"], dtype=float)
    fg, eg = f[good], e[good]

    inv_var = 1.0 / eg ** 2
    denom = np.sum(models ** 2 * inv_var, axis=1)
    numer = np.sum(models * fg * inv_var, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        amp = np.where(denom > 0, numer / denom, np.nan)
    resid = fg[None, :] - amp[:, None] * models
    chi2 = np.sum((resid ** 2) * inv_var[None, :], axis=1)

    valid = np.isfinite(chi2) & (amp > 0)
    if np.count_nonzero(valid) < 3:
        return None
    chi2 = np.where(valid, chi2, np.inf)

    i_best = int(np.argmin(chi2))
    chi2_min = float(chi2[i_best])
    delta = chi2 - chi2_min

    def crossing(direction: int) -> float:
        rng_ = (
            range(i_best, len(nodes) - 1) if direction > 0
            else range(i_best, 0, -1)
        )
        for i in rng_:
            j = i + 1 if direction > 0 else i - 1
            if np.isfinite(delta[j]) and delta[j] >= 1.0 > delta[i]:
                t = (1.0 - delta[i]) / max(delta[j] - delta[i], 1e-12)
                return float(nodes[i] + t * (nodes[j] - nodes[i]))
        return float(nodes[-1] if direction > 0 else nodes[0])

    lo, hi = crossing(-1), crossing(+1)
    n_data = int(np.count_nonzero(good))
    dof = max(n_data - 2, 1)                # normalisation + node
    bounded = bool(
        min(lo, hi) > nodes[0] * 1.001 and max(lo, hi) < nodes[-1] * 0.999
    )

    return {
        "best_node": float(nodes[i_best]),
        "node_lo": float(min(lo, hi)),
        "node_hi": float(max(lo, hi)),
        "node_err": float(0.5 * abs(hi - lo)),
        "amplitude": float(amp[i_best]),
        "chi2": chi2_min,
        "dof": int(dof),
        "reduced_chi2": float(chi2_min / dof),
        "n_points": n_data,
        "bounded": bounded,
        "at_lower_edge": bool(i_best == 0),
        "at_upper_edge": bool(i_best == len(nodes) - 1),
    }


def synchrotron_age_from_break(
    break_frequency_hz: float, b_field_t: float, redshift: float,
    model: str = "JP", synchrofit_compatible_bcmb: bool = False,
) -> float:
    """
    Spectral age in Myr:

        tau = v sqrt(B) / (B^2 + B_CMB^2) * (nu_b (1+z))^(-1/2)

    with v = sqrt(243 pi m_e^5 c^2 / (4 mu_0^2 e^7)), identical to
    synchrofit's `const_synage`, reduced by 1/2.25 for KP.

    THREE CONVENTION WARNINGS, because this is where cross-code comparisons
    usually go wrong:

    1. The 1/2.25 KP factor is applied AT FIXED BREAK FREQUENCY, so this
       function returns a SMALLER age for KP than for JP given the same
       nu_b.  That is not the same statement as "KP fits give younger ages":
       synchrofit's KP model defines its break through a sin^3(alpha)
       scaling (JP uses sin(alpha)), so a KP fit to the same data returns a
       DIFFERENT nu_b.  Compare JP and KP ages only through their own fits;
       never by feeding one model's break frequency into the other's age
       relation.

    2. synchrofit's own Tribble model code computes an internal t_syn with
       `const_synage` and NO KP factor, while its public spectral_ages()
       applies the 1/2.25.  TKP ages are therefore not guaranteed to be
       internally consistent inside synchrofit; treat TKP ages with caution.

    3. synchrofit hardcodes B_CMB = 0.318 nT (T_0 ~ 2.70 K). The default
       here is the exact 0.32392 nT for T_0 = 2.72548 K. Pass
       synchrofit_compatible_bcmb=True to reproduce synchrofit exactly; the
       difference is a few per cent in age when B << B_CMB.
    """
    b_ic = b_cmb_tesla(redshift, synchrofit_compatible=synchrofit_compatible_bcmb)
    v = math.sqrt(
        (243.0 * math.pi * M_ELECTRON ** 5 * C_LIGHT ** 2)
        / (4.0 * MU_0 ** 2 * E_CHARGE ** 7)
    )
    if str(model).upper().startswith("KP"):
        v /= 2.25
    tau_s = (
        v * math.sqrt(b_field_t) / (b_field_t ** 2 + b_ic ** 2)
        * (float(break_frequency_hz) * (1.0 + float(redshift))) ** -0.5
    )
    return float(tau_s / MYR)


def break_frequency_from_age(
    age_myr: float, b_field_t: float, redshift: float, model: str = "JP",
    synchrofit_compatible_bcmb: bool = False,
) -> float:
    """Inverse of `synchrotron_age_from_break`."""
    b_ic = b_cmb_tesla(redshift, synchrofit_compatible=synchrofit_compatible_bcmb)
    v = math.sqrt(
        (243.0 * math.pi * M_ELECTRON ** 5 * C_LIGHT ** 2)
        / (4.0 * MU_0 ** 2 * E_CHARGE ** 7)
    )
    if str(model).upper().startswith("KP"):
        v /= 2.25
    tau_s = float(age_myr) * MYR
    root = v * math.sqrt(b_field_t) / ((b_field_t ** 2 + b_ic ** 2) * tau_s)
    return float(root ** 2 / (1.0 + float(redshift)))


def resolved_ageing_maps(
    common_maps: Sequence[RadioMap],
    source_mask: np.ndarray,
    common_beam: Beam,
    wcs: WCS,
    alpha_inj: float,
    b_field_t: float,
    redshift: float,
    cfg: Dict[str, Any],
    depth_map_arcsec: Optional[np.ndarray] = None,
    energy_scaling: Optional[Dict[str, float]] = None,
    target_snr: float = 10.0,
    template_kinds: Sequence[str] = ("JP", "CI"),
) -> Dict[str, Any]:
    """
    Resolved spectral-age, break-frequency, equipartition-field and pressure
    maps, built on adaptively binned regions.

    The injection index is FIXED at the value determined from the integrated
    ensemble.  That is the standard BRATS-style approach and is what makes the
    problem identifiable: with s fixed, each region has only a normalisation
    and a break to determine, so three usable bands already give one degree of
    freedom.  Letting s float per region with four bands would produce maps
    that are pure noise dressed as physics.
    """
    require_pysynch()
    ordered = sorted(common_maps, key=lambda m: m.freq_hz)
    freqs = np.asarray([m.freq_hz for m in ordered], dtype=float)

    regions = adaptive_bin(
        source_mask, ordered, common_beam, wcs,
        target_snr=float(target_snr),
        min_beams_per_region=float(cfg.get("min_beams_per_region", 1.0)),
    )
    if not regions:
        return {
            "status": "no_regions",
            "reason": (
                f"No region reached the target signal-to-noise of {target_snr} "
                "in every band. Lower --resolved-snr, or accept that the data "
                "do not support a resolved ageing analysis."
            ),
        }

    templates: Dict[str, Any] = {}
    for kind in template_kinds:
        try:
            templates[kind] = build_pysynch_templates(
                kind, freqs, alpha_inj, b_field_t, redshift,
                cfg["gmin"], cfg["gmax"],
                n_nodes=int(cfg.get("template_nodes", 60)),
                cosmology=cfg.get("cosmology", DEFAULT_COSMOLOGY),
            )
        except Exception as exc:                               # noqa: BLE001
            templates[kind] = {"error": f"{type(exc).__name__}: {exc}"}

    usable_kinds = [k for k, v in templates.items() if "error" not in v]
    if not usable_kinds:
        return {
            "status": "no_templates",
            "reason": "; ".join(
                f"{k}: {v['error']}" for k, v in templates.items()
            ),
        }

    shape = source_mask.shape
    out_maps: Dict[str, np.ndarray] = {
        name: np.full(shape, np.nan) for name in (
            "age_Myr", "age_err_Myr", "log10_break_Hz", "break_Hz",
            "chi2_reduced", "region_id", "region_snr",
            "B_eq_uG", "u_min_J_m3", "p_min_Pa", "volume_m3",
            "preferred_model_code",
        )
    }

    pix_area_arcsec2 = pixel_area_arcsec2(wcs)
    zeta = 1.0 + float(cfg["kappa"])
    model_codes = {"JP": 1.0, "CI": 2.0}

    records: List[Dict[str, Any]] = []
    for region in regions:
        ys, xs = region.pixels

        # --- region volume from the local depth map ----------------------
        volume_m3 = float("nan")
        if depth_map_arcsec is not None:
            depths = depth_map_arcsec[ys, xs]
            depths = depths[np.isfinite(depths)]
            if depths.size:
                volume_m3 = arcsec3_to_m3(
                    float(np.sum(depths) * pix_area_arcsec2), redshift,
                    cfg.get("cosmology", DEFAULT_COSMOLOGY),
                ) * float(cfg.get("filling_factor", 1.0))
        region.volume_m3 = volume_m3

        # --- template fits ------------------------------------------------
        fits_by_kind: Dict[str, Any] = {}
        for kind in usable_kinds:
            res = fit_region_to_templates(
                region.flux_jy, region.flux_err_jy, templates[kind]
            )
            if res is not None:
                fits_by_kind[kind] = res

        if not fits_by_kind:
            continue

        best_kind = min(fits_by_kind, key=lambda k: fits_by_kind[k]["chi2"])
        best = fits_by_kind[best_kind]

        if best_kind == "JP":
            age = best["best_node"]
            age_err = best["node_err"]
            nu_b = break_frequency_from_age(age, b_field_t, redshift, "JP")
        else:
            gamma_b = best["best_node"]
            nu_b = SYNCH_NU_C * b_field_t * gamma_b ** 2 / (1.0 + redshift)
            age = synchrotron_age_from_break(nu_b, b_field_t, redshift, "CI")
            gb_lo, gb_hi = best["node_lo"], best["node_hi"]
            nu_lo = SYNCH_NU_C * b_field_t * gb_lo ** 2 / (1.0 + redshift)
            nu_hi = SYNCH_NU_C * b_field_t * gb_hi ** 2 / (1.0 + redshift)
            age_err = 0.5 * abs(
                synchrotron_age_from_break(max(nu_lo, 1e-30), b_field_t, redshift, "CI")
                - synchrotron_age_from_break(max(nu_hi, 1e-30), b_field_t, redshift, "CI")
            )

        # --- resolved equipartition ---------------------------------------
        b_region = u_min_region = p_min_region = float("nan")
        if (
            energy_scaling is not None
            and np.isfinite(volume_m3) and volume_m3 > 0
            and np.isfinite(region.flux_jy[0]) and region.flux_jy[0] > 0
        ):
            try:
                fields = fields_from_scaling(
                    energy_scaling, zeta,
                    flux_jy=float(region.flux_jy[0]), volume_m3=volume_m3,
                )
                b_region = fields["equipartition"]["B_uG"]
                u_min_region = fields["minimum_energy"]["total_energy_density_J_m3"]
                p_min_region = fields["minimum_energy"]["pressure_Pa"]
            except Exception:                                  # noqa: BLE001
                pass

        for name, value in (
            ("age_Myr", age), ("age_err_Myr", age_err),
            ("log10_break_Hz", math.log10(nu_b) if nu_b > 0 else np.nan),
            ("break_Hz", nu_b),
            ("chi2_reduced", best["reduced_chi2"]),
            ("region_id", float(region.index)),
            ("region_snr", region.snr_min),
            ("B_eq_uG", b_region),
            ("u_min_J_m3", u_min_region),
            ("p_min_Pa", p_min_region),
            ("volume_m3", volume_m3),
            ("preferred_model_code", model_codes.get(best_kind, np.nan)),
        ):
            out_maps[name][ys, xs] = value

        records.append({
            "region_id": int(region.index),
            "n_pixels": int(region.n_pixels),
            "centroid_pixel": list(region.centroid),
            "min_band_snr": float(region.snr_min),
            "flux_Jy": region.flux_jy.tolist(),
            "flux_err_Jy": region.flux_err_jy.tolist(),
            "volume_m3": float(volume_m3),
            "preferred_model": best_kind,
            "age_Myr": float(age),
            "age_err_Myr": float(age_err),
            "break_frequency_Hz": float(nu_b),
            "reduced_chi2": float(best["reduced_chi2"]),
            "interval_bounded": bool(best["bounded"]),
            "B_eq_uG": float(b_region),
            "u_min_J_m3": float(u_min_region),
            "p_min_Pa": float(p_min_region),
            "all_model_chi2": {
                k: float(v["chi2"]) for k, v in fits_by_kind.items()
            },
        })

    ages = out_maps["age_Myr"][np.isfinite(out_maps["age_Myr"])]
    summary = {
        "status": "ok",
        "n_regions": len(records),
        "target_snr": float(target_snr),
        "fixed_injection_index_s": float(alpha_signed_to_s(alpha_inj)),
        "fixed_alpha_inj_signed": float(alpha_inj),
        "assumed_uniform_B_uG": float(b_field_t / TESLA_PER_MICROGAUSS),
        "template_kinds": usable_kinds,
        "template_diagnostics": {
            k: ({"error": v["error"]} if "error" in v else {
                "n_usable": v["n_usable"],
                "node_name": v["node_name"],
                "node_min": float(np.min(v["nodes"])),
                "node_max": float(np.max(v["nodes"])),
            })
            for k, v in templates.items()
        },
        "model_code_legend": {"1": "JP (aged)", "2": "CI (broken)"},
        "caveats": [
            "The magnetic field is assumed UNIFORM across the source. Because "
            "tau ~ B^0.5/(B^2+B_CMB^2), a real field gradient maps directly "
            "into an apparent age gradient; ages are therefore relative "
            "within the source far more reliably than they are absolute.",
            "The injection index is fixed globally. Any genuine spatial "
            "variation in s is absorbed into the fitted break.",
            "Adjacent regions share the common beam, so neighbouring values "
            "are correlated at roughly the beam scale.",
            "pysynch exposes the JP aged spectrum but not KP, so the 'JP' "
            "templates here are strictly JP; a KP interpretation would give "
            "systematically older ages.",
        ],
    }
    if ages.size:
        summary.update({
            "age_median_Myr": float(np.median(ages)),
            "age_min_Myr": float(np.min(ages)),
            "age_max_Myr": float(np.max(ages)),
            "age_range_Myr": float(np.ptp(ages)),
        })
    beq = out_maps["B_eq_uG"][np.isfinite(out_maps["B_eq_uG"])]
    if beq.size:
        summary.update({
            "B_eq_median_uG": float(np.median(beq)),
            "B_eq_min_uG": float(np.min(beq)),
            "B_eq_max_uG": float(np.max(beq)),
        })

    return {"maps": out_maps, "regions": records, "summary": summary,
            "region_objects": regions, "status": "ok"}


# ===========================================================================
# PART 15B.  BRATS ENGINE
# ===========================================================================
#
# BRATS (Broadband Radio Astronomy ToolS; Harwood et al. 2013, 2015;
# ascl:1806.025) is an independent, peer-reviewed implementation of exactly the
# same physics this pipeline needs.  Running both it and synchrofit over the
# same maps is the strongest robustness test available: the two codes share no
# source, integrate the synchrotron kernel differently, and fit by different
# search strategies, so where they agree the answer is a property of the data
# rather than of one implementation.
#
# HOW BRATS IS DRIVEN
# -------------------
# BRATS is an interactive terminal program.  Its own Python wrapper
# (JeremyHarwood/bratswrapper) works by piping a text file of commands into
# the process's stdin, one input per line, and that is exactly what is done
# here -- no reimplementation, no patched binary.  Every command, parameter
# and menu index below was taken from the BRATS v2.6.x source
# (commandlist.h, main.c) and the BRATS Cookbook, not from memory.
#
# WHAT BRATS CAN AND CANNOT DO  (verified against the source, not assumed)
# ------------------------------------------------------------------------
#   fitjpmodel     JP,  resolved, per region          -- available
#   fitkpmodel     KP,  resolved, per region          -- available
#   fitjptribble   Tribble (JP), resolved             -- available
#   fitkptribble   Tribble (KP), resolved             -- NOT AVAILABLE.
#                  The command is commented out in main.c (cmdnum 100) and
#                  absent from commandlist.h. `plotkptribble` exists but can
#                  only plot values that nothing can compute. The exportdata
#                  menu likewise has no TRIBKP entries. KP-Tribble is
#                  therefore routed to synchrofit, which does implement it.
#   fitcimodel     CI,     integrated, from text files -- available
#   fitcioff       CI-off, integrated, from text files -- available
#   findinject     injection index by chi-square minimisation over all regions
#   specindex      resolved spectral index (weighted GSL least squares)
#   specindexerrors / specchisquared    resolved error and chi-square maps
#
# The CI models take a different input path from the single-injection ones:
# they read two comma-delimited text files (a header file of
# identifier/redshift/B/injection index, and a data file of
# identifier/frequency/flux/error/upper-limit-flag) rather than FITS maps.
# That suits us: we already have a fully characterised integrated SED.
#
# A FITS-HEADER TRAP
# ------------------
# BRATS looks for the observing frequency under REFFREQ, then FREQ, then
# CTYPE3='FREQ' with CRVAL3.  It does NOT read RESTFRQ, which is what the
# common-beam products carry.  Maps staged for BRATS therefore get REFFREQ,
# CTYPE3 and CRVAL3 written explicitly; without this, `load` fails with an
# unhelpful error.
# ---------------------------------------------------------------------------

#: name -> (menu index, fittable on maps, fittable integrated, exportdata codes)
BRATS_MODELS: Dict[str, Dict[str, Any]] = {
    "JP": {
        "index": 1, "fit_command": "fitjpmodel", "maps": True,
        "integrated": True, "age": 0, "chi2": 3, "errors": 6, "norm": 9,
        "token": "JP",
    },
    "KP": {
        "index": 2, "fit_command": "fitkpmodel", "maps": True,
        "integrated": True, "age": 1, "chi2": 4, "errors": 7, "norm": 10,
        "token": "KP",
    },
    "JP_Tribble": {
        "index": 3, "fit_command": "fitjptribble", "maps": True,
        "integrated": True, "age": 2, "chi2": 5, "errors": 8, "norm": 11,
        "token": "TRIBJP",
    },
    "KP_Tribble": {
        "index": 4, "fit_command": None, "maps": False,
        "integrated": True, "age": None, "chi2": None, "errors": None,
        "norm": None, "token": "TRIBKP",
        "unavailable_reason": (
            "BRATS cannot fit KP-Tribble on maps: fitkptribble is commented "
            "out in main.c (cmdnum 100), absent from commandlist.h, and the "
            "exportdata menu has no Tribble (KP) entries, so there is no way "
            "to compute or retrieve resolved values. fitintegrated does offer "
            "model index 4 and will write IntergratedFit_Mod4_*.txt, so the "
            "integrated fit is attempted, but it is unverified and should be "
            "treated as provisional. Use --engine synchrofit for a supported "
            "KP-Tribble fit."
        ),
    },
    "CI": {
        "index": None, "fit_command": "fitcimodel", "maps": False,
        "integrated": True, "age": None, "chi2": None, "errors": None,
        "norm": None, "token": "CI",
    },
    "CI-OFF": {
        "index": None, "fit_command": "fitcioff", "maps": False,
        "integrated": True, "age": None, "chi2": None, "errors": None,
        "norm": None, "token": "CIOFF",
    },
}

#: exportdata menu, verified against the table printed by main.c
BRATS_EXPORT_CODES: Dict[int, str] = {
    0: "JP", 1: "KP", 2: "TRIBJP",
    3: "JPX2", 4: "KPX2", 5: "TRIBJPX2",
    6: "JPERRORS", 7: "KPERRORS", 8: "TRIBJPERRORS",
    9: "JPNORM", 10: "KPNORM", 11: "TRIBJPNORM",
    12: "SPECIND", 13: "INJIND_MOD",
    14: "INJECTBYREGION", 15: "INJECTCHISQUAREDBYREGION",
    16: "FLUX", 17: "REGARRAY",
}


class BratsUnavailable(RuntimeError):
    """Raised when the BRATS executable cannot be located or run."""


#: Built from the Singularity recipe BRATS ships. Kept in the tree so the
#: workspace is self-sufficient: a collaborator who has the workspace can
#: rebuild the exact environment the results came from.
BRATS_DOCKERFILE = """\
# BRATS in a container.
#
# BRATS needs PGPLOT and FUNTOOLS. Both are packaged for Debian/Ubuntu but
# neither has a maintained macOS build, so on a Mac a container is by far the
# shortest path to a working BRATS -- and on any machine it pins the exact
# environment the numbers came from.
#
#   docker build -t brats:latest -f Dockerfile .
#
# The pipeline will then find it automatically with:
#   --brats-container docker --brats-image brats:latest
FROM ubuntu:20.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \\
        funtools gfortran git libfuntools-dev libreadline-dev \\
        libgsl-dev libpng-dev libx11-dev make pgplot5 ca-certificates \\
    && rm -rf /var/lib/apt/lists/*
RUN mkdir -p /soft \\
    && git clone https://github.com/JeremyHarwood/BRATS.git /BRATS \\
    && sed -i "s|#include <funtools.h>|#include <funtools/funtools.h>|g" \\
         /BRATS/brats/main.c \\
    && make -C /BRATS/brats INSTALLFROM=/BRATS/brats/ INSTALLTO=/soft/brats \\
    && rm -rf /BRATS
ENV LD_LIBRARY_PATH=/usr/local/lib
ENV PGPLOT_DIR=/usr/lib/pgplot5
WORKDIR /work
ENTRYPOINT ["/soft/brats/brats"]
"""

#: where BRATS lives inside the image built from the recipe above
BRATS_CONTAINER_EXECUTABLE = "/soft/brats/brats"
#: mount point the workspace is bound to inside the container
BRATS_CONTAINER_WORKDIR = "/work"


def write_brats_dockerfile(directory: Path) -> Path:
    """Drop the container recipe next to the workspace."""
    path = Path(directory) / "Dockerfile"
    path.write_text(BRATS_DOCKERFILE)
    return path


class BratsRunner:
    """
    Locate, script and drive a BRATS installation.

    Nothing here patches or reimplements BRATS: commands are piped to its
    stdin exactly as the official Python wrapper does, so results are
    bit-identical to typing the same commands by hand.

    BRATS can be driven either as a native executable or inside a container.
    The container path exists because BRATS depends on PGPLOT and FUNTOOLS,
    which are packaged on Linux but effectively unavailable on macOS -- so on
    a Mac a container is usually the only way to run BRATS at all, and the
    stdin-piping contract is identical either way.
    """

    def __init__(
        self, executable: Optional[str] = None,
        container: Optional[str] = None,
        image: Optional[str] = None,
    ) -> None:
        self.image = image or os.environ.get("BRATS_IMAGE") or "brats:latest"
        requested = container or "auto"

        # A native BRATS is preferred whenever one exists: it avoids the
        # container start-up cost on every call and sidesteps the bind-mount
        # entirely. A container is only reached for when asked for by name,
        # or when nothing native could be found.
        self.executable = (
            self._locate(executable) if requested in {"auto", "none"} else None
        )
        if self.executable is not None:
            self.container = None
            self.available = True
            self.mode = "native"
        elif requested == "none":
            self.container = None
            self.available = False
            self.mode = "unavailable"
        else:
            self.container = self._resolve_container(requested)
            if self.container:
                self.available = True
                self.mode = "container"
            else:
                # An explicitly named runtime that is not usable should still
                # fall back to a native install rather than reporting nothing.
                self.executable = self._locate(executable)
                self.available = self.executable is not None
                self.mode = "native" if self.available else "unavailable"
        self.version = self._version() if self.available else None

    # -- discovery ---------------------------------------------------------
    @staticmethod
    def _resolve_container(container: Optional[str]) -> Optional[str]:
        """
        Pick a container runtime, verifying it can actually start one.

        `shutil.which("docker")` is not enough on macOS: the Docker CLI is
        installed by Docker Desktop and stays on PATH whether or not the
        daemon is running, so a `which` hit alone would send every BRATS call
        into a confusing connection error. `docker info` is the cheap check
        that the daemon is really there.
        """
        if container in {None, "", "auto"}:
            candidates = ["docker", "apptainer", "singularity", "podman"]
        elif container == "none":
            return None
        else:
            candidates = [container]
        for name in candidates:
            if shutil.which(name) is None:
                continue
            if name in {"docker", "podman"}:
                try:
                    probe = subprocess.run(
                        [name, "info"], capture_output=True, text=True,
                        timeout=30, check=False,
                    )
                    if probe.returncode != 0:
                        continue
                except Exception:                              # noqa: BLE001
                    continue
            return name
        return None

    @staticmethod
    def _locate(explicit: Optional[str]) -> Optional[str]:
        candidates: List[str] = []
        if explicit:
            # An explicitly named --brats-path that does not resolve is a
            # mistake worth surfacing. Falling back to some other BRATS found
            # elsewhere would run the analysis with a different binary from
            # the one asked for, and say nothing about it.
            explicit_path = Path(explicit).expanduser()
            if not (explicit_path.is_file() and os.access(explicit_path, os.X_OK)):
                warnings.warn(
                    f"--brats-path {explicit!r} is not an executable file. "
                    "Falling back to the usual search (PATH, the BRATS "
                    "environment variable, the interpreter's own directory, "
                    "the standard install locations); check which executable "
                    "is reported before trusting the results.",
                    RuntimeWarning, stacklevel=2,
                )
            candidates.append(explicit)
        env = os.environ.get("BRATS") or os.environ.get("BRATS_PATH")
        if env:
            candidates.append(env)
        found = shutil.which("brats")
        if found:
            candidates.append(found)
        # The interpreter's own bin directory. BRATS is commonly installed
        # into the same conda environment as the Python stack, and that
        # directory is only on PATH while the environment is ACTIVATED --
        # running the script as `/path/to/envs/foo/bin/python3 PySynch...`
        # is enough to import everything and still leave `which brats`
        # empty. Looking next to sys.executable finds it either way.
        candidates.append(str(Path(sys.executable).resolve().parent / "brats"))
        conda_prefix = os.environ.get("CONDA_PREFIX")
        if conda_prefix:
            candidates.append(str(Path(conda_prefix) / "bin" / "brats"))
        # A source checkout sitting next to the data is the common case for
        # someone who has just downloaded BRATS, so look there before the
        # system-wide locations.
        here = Path(__file__).resolve().parent
        for root in (Path.cwd(), here):
            candidates += [
                str(root / "brats" / "brats"),
                str(root / "BRATS" / "brats" / "brats"),
                str(root / "BRATS-master" / "brats" / "brats"),
            ]
        candidates += [
            os.path.expanduser("~/brats/brats"),
            "/usr/local/brats/brats",
            "/soft/brats/brats",
            "/opt/brats/brats",
        ]
        for candidate in candidates:
            path = Path(candidate).expanduser()
            if path.is_file() and os.access(path, os.X_OK):
                return str(path.resolve())
        return None

    # -- command construction ----------------------------------------------
    @staticmethod
    def x11_forwarding() -> Dict[str, Any]:
        """
        Work out how to give a containerised BRATS a display.

        BRATS draws through PGPLOT, so its most useful interactive commands
        (plotting a model against the data, inspecting a region) need an X
        server.  A plain `docker run` has none, and the failure mode is
        confusing: BRATS starts, accepts commands, and then dies or silently
        does nothing the moment you ask it to plot.

        On Linux the host socket can be bind-mounted directly.  On macOS the
        container reaches the host through the special DNS name
        host.docker.internal, and XQuartz must be running with network
        clients allowed -- which is a deliberate user action, not something
        to switch on behind their back, so it is reported rather than done.
        """
        info: Dict[str, Any] = {"available": False, "args": [], "notes": []}
        if sys.platform == "darwin":
            xquartz = (
                shutil.which("xquartz")
                or shutil.which("Xquartz")
                or Path("/opt/X11/bin/Xquartz").is_file()
                or Path("/Applications/Utilities/XQuartz.app").is_dir()
            )
            if not xquartz:
                info["notes"].append(
                    "XQuartz is not installed, so a containerised BRATS has "
                    "no X server to draw on. Its plotting commands will fail; "
                    "everything that writes to file still works. Install it "
                    "with 'brew install --cask xquartz' if you want the "
                    "PGPLOT displays."
                )
                return info
            info["available"] = True
            info["args"] = [
                "-e", "DISPLAY=host.docker.internal:0",
                "-v", "/tmp/.X11-unix:/tmp/.X11-unix",
            ]
            info["notes"].append(
                "XQuartz must be running and set to allow network clients "
                "(XQuartz > Settings > Security > 'Allow connections from "
                "network clients'), then run 'xhost +localhost' once per "
                "login. Without that, PGPLOT cannot open a window."
            )
            return info

        display = os.environ.get("DISPLAY")
        if not display:
            info["notes"].append(
                "DISPLAY is not set, so a containerised BRATS has no X "
                "server. Plotting commands will fail; file output is "
                "unaffected."
            )
            return info
        info["available"] = True
        info["args"] = ["-e", f"DISPLAY={display}",
                        "-v", "/tmp/.X11-unix:/tmp/.X11-unix"]
        info["notes"].append(
            "The host X socket is bind-mounted. If BRATS cannot open a "
            "window, run 'xhost +local:' once."
        )
        return info

    def _argv(self, cwd: Path, interactive: bool = False) -> List[str]:
        """The argv that starts BRATS, native or containerised."""
        if not self.container:
            return [str(self.executable)]
        mount = f"{Path(cwd).resolve()}:{BRATS_CONTAINER_WORKDIR}"
        if self.container in {"docker", "podman"}:
            flags = ["-it"] if interactive else ["-i"]
            display_args = (
                self.x11_forwarding().get("args", []) if interactive else []
            )
            return [
                self.container, "run", "--rm", *flags, *display_args,
                "-v", mount, "-w", BRATS_CONTAINER_WORKDIR,
                "--entrypoint", BRATS_CONTAINER_EXECUTABLE,
                self.image,
            ]
        # apptainer / singularity pass the host environment and mount /tmp by
        # default, so an existing DISPLAY already works without extra flags.
        return [
            self.container, "exec",
            "--bind", mount, "--pwd", BRATS_CONTAINER_WORKDIR,
            self.image, BRATS_CONTAINER_EXECUTABLE,
        ]

    @staticmethod
    def filter_environment() -> Dict[str, str]:
        """
        Environment BRATS needs for FUNTOOLS region filters to compile.

        BRATS applies its source and background regions through FUNTOOLS,
        which does not evaluate them directly: it GENERATES a small C program
        per filter and compiles it at run time.  That compile uses whatever
        `cc` it finds first, and on macOS that is frequently a Homebrew GCC
        whose bundled fixed headers are incompatible with the system SDK.
        The result is hundreds of lines of errors inside stdio.h ending in

            ERROR: filter compilation failed

        after which `load` fails and every later command runs against an
        empty dataset -- a failure that looks nothing like a region problem
        and is easy to mistake for one.

        Apple's own clang compiles the generated code without complaint, so
        it is selected explicitly on macOS. An explicit FILTER_CC set by the
        user always wins: they may have a working toolchain deliberately
        chosen, and second-guessing it would be worse than the default.
        """
        env = dict(os.environ)
        if env.get("FILTER_CC"):
            return env
        if sys.platform == "darwin":
            for candidate in ("/usr/bin/clang", shutil.which("clang")):
                if candidate and Path(candidate).is_file():
                    env["FILTER_CC"] = candidate
                    break
        return env

    def describe(self) -> str:
        if self.mode == "container":
            return f"{self.container} image {self.image}"
        if self.executable:
            return str(self.executable)
        return "not found"

    def _version(self) -> Optional[str]:
        try:
            if self.container:
                argv = self._argv(Path.cwd()) + ["-version"]
            else:
                argv = [str(self.executable), "-version"]
            proc = subprocess.run(
                argv, capture_output=True, text=True, timeout=120, check=False,
            )
            text = (proc.stdout or "") + (proc.stderr or "")
            match = re.search(r"(\d+\.\d+\.\d+)", text)
            if match:
                return match.group(1)
            # No version number in the banner. Fall back to the first
            # non-empty line only: this string is printed inline in the
            # report, and a multi-line banner would break its layout.
            for line in text.splitlines():
                line = line.strip()
                if line:
                    return line[:80]
            return None
        except Exception:                                      # noqa: BLE001
            return None

    def require(self) -> None:
        if not self.available:
            raise BratsUnavailable(
                "BRATS could not be found.\n"
                "\n"
                "  Native install: build it from "
                "https://www.askanastronomer.co.uk/brats/ and either put it "
                "on your PATH, set the BRATS environment variable, or pass "
                "--brats-path /full/path/to/brats.\n"
                "\n"
                "  Container (recommended on macOS, where PGPLOT and FUNTOOLS "
                "have no maintained build): a Dockerfile is written into the "
                "BRATS workspace. Build it once with\n"
                "      docker build -t brats:latest -f Dockerfile .\n"
                "  then re-run with --brats-container docker.\n"
                "\n"
                "  The workspace, region files and command scripts are "
                "prepared either way, so nothing computed so far is lost."
            )

    # -- execution --------------------------------------------------------
    def run_script(
        self, script_path: Path, cwd: Path, timeout: Optional[float] = None,
        log_path: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """
        Pipe a command file into BRATS and capture everything it prints.

        BRATS is driven entirely through stdin, so a run is reproducible: the
        same command file always produces the same analysis, and the file is
        kept alongside the results as a record of exactly what was done.
        """
        self.require()
        commands = Path(script_path).read_text()
        argv = self._argv(Path(cwd), interactive=False)
        started = time.time()
        try:
            proc = subprocess.run(
                argv, input=commands, cwd=str(cwd),
                capture_output=True, text=True, timeout=timeout, check=False,
                env=self.filter_environment(),
            )
            stdout, stderr, code = proc.stdout, proc.stderr, proc.returncode
            timed_out = False
        except subprocess.TimeoutExpired as exc:
            stdout = exc.stdout.decode() if isinstance(exc.stdout, bytes) else (exc.stdout or "")
            stderr = exc.stderr.decode() if isinstance(exc.stderr, bytes) else (exc.stderr or "")
            code, timed_out = -1, True

        if log_path is not None:
            Path(log_path).write_text(
                f"# BRATS command file: {script_path}\n"
                f"# invocation: {' '.join(argv)}\n"
                f"# mode: {self.mode}   ({self.describe()})\n"
                f"# cwd: {cwd}\n"
                f"# return code: {code}   timed out: {timed_out}\n"
                f"{'=' * 72}\nCOMMANDS SENT\n{'=' * 72}\n{commands}\n"
                f"{'=' * 72}\nSTDOUT\n{'=' * 72}\n{stdout}\n"
                f"{'=' * 72}\nSTDERR\n{'=' * 72}\n{stderr}\n"
            )
        return {
            "return_code": code, "timed_out": timed_out,
            "stdout": stdout, "stderr": stderr,
            "elapsed_seconds": time.time() - started,
            "script": str(script_path), "log": str(log_path) if log_path else None,
            "invocation": " ".join(argv), "mode": self.mode,
        }

    def launch_interactive(
        self, cwd: Path, banner_file: Optional[Path] = None
    ) -> Dict[str, Any]:
        """
        Open BRATS in a NEW terminal window, sitting in a prepared workspace.

        BRATS is genuinely interactive -- it draws through PGPLOT and several
        of its most useful tasks want a human looking at the plot -- so a
        scripted run is not always what you want.  This drops you into a
        directory where the maps, region files and a ready-to-paste command
        script are already in place, with BRATS running and waiting.

        If no terminal emulator can be launched (a headless node, an ssh
        session with no X forwarding), this does not fail: it returns the
        exact command to run by hand.
        """
        self.require()
        cwd = Path(cwd).resolve()
        argv = self._argv(cwd, interactive=True)
        launch_cmd = " ".join(argv)
        manual = f"cd {cwd} && {launch_cmd}"
        result: Dict[str, Any] = {
            "workspace": str(cwd),
            "executable": self.executable,
            "mode": self.mode,
            "invocation": launch_cmd,
            "manual_command": manual,
            "launched": False,
        }

        hello = ""
        if banner_file and Path(banner_file).is_file():
            hello = f"echo; cat {banner_file}; echo; "

        # The interactive session needs the same working compiler for its
        # region filters as the scripted one, and it is a fresh shell, so the
        # setting has to be exported into it rather than inherited.
        filter_cc = self.filter_environment().get("FILTER_CC")
        if filter_cc:
            hello = f"export FILTER_CC={filter_cc}; " + hello
            result["filter_cc"] = filter_cc

        try:
            if sys.platform == "darwin":
                # The command is embedded in an AppleScript string literal, so
                # quotes and backslashes in it have to be escaped or the
                # script silently truncates at the first stray quote.
                inner = f"{hello}cd {cwd} && {launch_cmd}"
                escaped = inner.replace("\\", "\\\\").replace('"', '\\"')
                script = (
                    'tell application "Terminal"\n'
                    f'  do script "{escaped}"\n'
                    "  activate\n"
                    "end tell"
                )
                proc = subprocess.run(
                    ["osascript", "-e", script], capture_output=True,
                    text=True, timeout=30, check=False,
                )
                if proc.returncode == 0:
                    result.update(launched=True, method="Terminal.app via osascript")
                    return result
                result["error"] = (proc.stderr or "").strip()
            else:
                inner = (
                    f"{hello}cd {cwd} && {launch_cmd}; "
                    "exec ${SHELL:-/bin/bash}"
                )
                emulators = [
                    (["gnome-terminal", "--", "bash", "-lc", inner], "gnome-terminal"),
                    (["konsole", "-e", "bash", "-lc", inner], "konsole"),
                    (["xfce4-terminal", "-e", f"bash -lc '{inner}'"], "xfce4-terminal"),
                    (["mate-terminal", "--", "bash", "-lc", inner], "mate-terminal"),
                    (["tilix", "-e", "bash", "-lc", inner], "tilix"),
                    (["x-terminal-emulator", "-e", "bash", "-lc", inner],
                     "x-terminal-emulator"),
                    (["xterm", "-e", "bash", "-lc", inner], "xterm"),
                ]
                if not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
                    result["error"] = (
                        "No DISPLAY or WAYLAND_DISPLAY is set, so no terminal "
                        "window can be opened (headless session or ssh without "
                        "X forwarding)."
                    )
                    return result
                for emulator_argv, name in emulators:
                    if shutil.which(emulator_argv[0]) is None:
                        continue
                    try:
                        subprocess.Popen(
                            emulator_argv, cwd=str(cwd),
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                            start_new_session=True,
                        )
                        result.update(launched=True, method=name)
                        return result
                    except Exception:                          # noqa: BLE001
                        continue
                result["error"] = (
                    "No supported terminal emulator was found (looked for "
                    "gnome-terminal, konsole, xfce4-terminal, mate-terminal, "
                    "tilix, x-terminal-emulator, xterm)."
                )
        except Exception as exc:                               # noqa: BLE001
            result["error"] = f"{type(exc).__name__}: {exc}"
        return result


# ---------------------------------------------------------------------------
# Workspace preparation
# ---------------------------------------------------------------------------

def write_ds9_box_region(
    path: Path, x_centre: float, y_centre: float, width: float, height: float,
    comment: str = "",
) -> None:
    """
    DS9-format region in IMAGE coordinates.

    Image coordinates are used deliberately.  BRATS reads regions through
    FUNTOOLS, and a sky-coordinate region depends on FUNTOOLS parsing the WCS
    identically to astropy; in image coordinates the region means exactly what
    we computed it to mean.  FITS pixel indexing is 1-based, so the centres
    written here are the 0-based array indices plus one.
    """
    lines = [
        "# Region file format: DS9 version 4.1",
        f"# {comment}" if comment else "# generated by PySynch",
        "image",
        f"box({x_centre + 1.0:.3f},{y_centre + 1.0:.3f},"
        f"{width:.3f},{height:.3f},0)",
    ]
    Path(path).write_text("\n".join(lines) + "\n")


def choose_background_box(
    data: np.ndarray, source_mask: np.ndarray, box_fraction: float = 0.18,
    margin: int = 4,
) -> Tuple[float, float, float, float]:
    """
    Find an emission-free box for the BRATS background/noise region.

    BRATS derives its detection threshold and per-region errors from the RMS
    of this box, so a badly chosen one silently corrupts every downstream
    error bar.  Candidates are scored on robust scatter and on being flat and
    source-free: a box containing any masked pixel is rejected outright, and
    among the rest the one with the smallest MAD is chosen, with the median
    offset used to break ties.
    """
    ny, nx = data.shape
    bw = max(int(round(nx * box_fraction)), 16)
    bh = max(int(round(ny * box_fraction)), 16)
    bw, bh = min(bw, nx - 2 * margin), min(bh, ny - 2 * margin)
    if bw < 8 or bh < 8:
        raise RuntimeError(
            "The common-beam image is too small to place a background region."
        )

    grown = binary_dilation(source_mask, iterations=max(margin, 2))
    best: Optional[Tuple[float, Tuple[float, float, float, float]]] = None
    #: source-free candidates whose robust sigma came out non-positive
    degenerate: List[Tuple[float, Tuple[float, float, float, float]]] = []

    ys = np.linspace(margin, ny - bh - margin, 6, dtype=int)
    xs = np.linspace(margin, nx - bw - margin, 6, dtype=int)
    for y0 in ys:
        for x0 in xs:
            sub = data[y0:y0 + bh, x0:x0 + bw]
            sub_mask = grown[y0:y0 + bh, x0:x0 + bw]
            if sub_mask.any():
                continue
            finite = sub[np.isfinite(sub)]
            if finite.size < 0.8 * sub.size:
                continue
            box = (x0 + bw / 2.0, y0 + bh / 2.0, float(bw), float(bh))
            sigma = _robust_mad_sigma(finite)
            if not np.isfinite(sigma) or sigma <= 0:
                # A source-free box whose robust scatter is exactly zero: a
                # blanked region, or noiseless simulated data. Keep it as a
                # fallback rather than discarding it, but prefer any box with
                # real noise, because BRATS needs a measurable RMS here.
                degenerate.append((abs(float(np.nanmedian(finite))), box))
                continue
            score = sigma * (1.0 + abs(float(np.nanmedian(finite))) / sigma)
            if best is None or score < best[0]:
                best = (score, box)

    if best is not None:
        return best[1]

    if degenerate:
        # Every candidate had zero scatter. Placing the flattest one is still
        # the right move -- it is genuinely source-free -- but BRATS will
        # derive a zero or meaningless RMS from it, so warn loudly. Passing
        # --brats-keep-own-noise off (the default) means the pipeline's own
        # per-map RMS values are pushed into BRATS anyway, which rescues this
        # case for simulated data.
        degenerate.sort(key=lambda item: item[0])
        warnings.warn(
            "Every candidate BRATS background box has zero robust scatter "
            "(blanked, masked or noiseless data). A box has still been "
            "placed, but BRATS cannot measure a meaningful off-source RMS "
            "from it. The pipeline's own per-map RMS values are supplied to "
            "BRATS explicitly, which covers this; do not pass "
            "--brats-keep-own-noise with data like this.",
            RuntimeWarning, stacklevel=2,
        )
        return degenerate[0][1]

    raise RuntimeError(
        "No source-free background box could be placed. The cutout is "
        "probably too tight around the source: BRATS needs blank sky to "
        "measure the off-source noise. Re-cut the images with more "
        "surrounding field, or pass --brats-background-region with your "
        "own DS9 region file."
    )


def square_pad_for_brats(
    data: np.ndarray, header: fits.Header,
) -> Tuple[np.ndarray, fits.Header, int, int]:
    """
    Pad an image to a square, keeping its WCS correct.

    BRATS refuses to load a non-square image outright:

        *** Error: The image ... is not square (this is currently a
        requirement). ***

    and since `load` is the first real command in the script, everything
    after it then fails against an empty dataset.  A common-beam product is
    almost never square -- it is the intersection of several footprints --
    so this affects essentially every real run.

    BRATS suggests its own `resizeimages`, but padding here is better: the
    reference pixel is shifted by exactly the amount padded, so the WCS of
    the staged map stays truthful, and the offsets are returned so that
    per-region results can be mapped back onto the pipeline's own grid
    afterwards.

    The pad value is NaN, matching the blanking already present outside the
    common footprint.  BRATS selects pixels by a sigma threshold, and a NaN
    fails that comparison, so padded pixels are excluded exactly as blanked
    ones are.

    Returns (padded, header, x_offset, y_offset), where the offsets are where
    the original array's origin sits inside the padded one.
    """
    array = np.asarray(data, dtype=float)
    ny, nx = array.shape
    if ny == nx:
        return array, header, 0, 0

    size = max(ny, nx)
    y0 = (size - ny) // 2
    x0 = (size - nx) // 2
    padded = np.full((size, size), np.nan, dtype=float)
    padded[y0:y0 + ny, x0:x0 + nx] = array

    out = header.copy()
    # FITS CRPIX is 1-based, but a shift of the array origin is a shift of
    # the reference pixel by the same number of pixels either way.
    for key, shift in (("CRPIX1", x0), ("CRPIX2", y0)):
        if key in out:
            out[key] = float(out[key]) + float(shift)
    out["NAXIS1"] = size
    out["NAXIS2"] = size
    out["PSPADX"] = (x0, "pixels padded left (square for BRATS)")
    out["PSPADY"] = (y0, "pixels padded below (square for BRATS)")
    return padded, out, x0, y0

def prepare_brats_workspace(
    outdir: Path,
    common_maps: Sequence[RadioMap],
    source_mask: np.ndarray,
    redshift: float,
    background_region: Optional[Path] = None,
    source_region: Optional[Path] = None,
    source_margin_beams: float = 2.0,
    common_beam: Optional[Beam] = None,
    target_name: str = "PYSYNCH_SOURCE",
) -> Dict[str, Any]:
    """
    Build a complete BRATS working directory from the common-beam products.

    Layout follows the structure the BRATS Cookbook expects:

        brats_workspace/
            maps/        one FITS per frequency, already on a common grid
            regions/     source and background DS9 regions
            images/      BRATS writes maps and plots here
            data/        BRATS writes exported tables here
            commands/    the generated command scripts

    The maps are re-written rather than symlinked so the frequency keywords
    BRATS actually reads (REFFREQ, CTYPE3/CRVAL3) can be added without
    touching the pipeline's own products.
    """
    workspace = Path(outdir) / "brats_workspace"
    for sub in ("maps", "regions", "images", "data", "commands"):
        (workspace / sub).mkdir(parents=True, exist_ok=True)

    # Always drop the container recipe in, whether or not BRATS was found on
    # this machine. It costs nothing, and it means the workspace is a
    # complete, self-contained record: anyone who receives it can rebuild the
    # exact BRATS environment the results came from.
    write_brats_dockerfile(workspace)

    ordered = sorted(common_maps, key=lambda m: m.freq_hz)
    staged: List[Dict[str, Any]] = []
    for m in ordered:
        header = m.header.copy()
        # BRATS reads REFFREQ, then FREQ, then CTYPE3=='FREQ' + CRVAL3.
        # It never reads RESTFRQ, which is what our own products carry.
        header["REFFREQ"] = float(m.freq_hz)
        header["FREQ"] = float(m.freq_hz)
        header["CTYPE3"] = "FREQ"
        header["CRVAL3"] = float(m.freq_hz)
        header["CDELT3"] = 1.0
        header["CRPIX3"] = 1.0
        header["OBJECT"] = target_name
        header["BUNIT"] = "JY/BEAM"

        # BRATS reads the astrometry through a fixed list of keywords and
        # ABORTS the load if any of them is missing (load.h: "Error: Unable
        # to obtain the EQUINOX/EPOCH of ..."). Modern radio products
        # describe their frame with RADESYS instead, which BRATS does not
        # know about, so the equinox has to be written explicitly or nothing
        # loads at all.
        equinox = header.get("EQUINOX", header.get("EPOCH"))
        if equinox is None:
            radesys = str(header.get("RADESYS", "ICRS")).upper()
            # ICRS and FK5/J2000 agree to far better than any radio beam, so
            # 2000.0 is the honest value for both; FK4/B1950 is flagged
            # rather than silently relabelled.
            if radesys.startswith("FK4") or radesys.startswith("B"):
                equinox = 1950.0
                warnings.warn(
                    f"{m.label} is on a B1950/FK4 frame. EQUINOX=1950 has "
                    "been written for BRATS, but the pipeline's own products "
                    "are on the common J2000 grid; check the astrometry "
                    "before trusting any BRATS overlay.",
                    RuntimeWarning, stacklevel=2,
                )
            else:
                equinox = 2000.0
        header["EQUINOX"] = float(equinox)
        header["EPOCH"] = float(equinox)      # older BRATS reads EPOCH only

        # CROTA1/2 are only a warning when absent, but the assumed value is
        # 0.0 and the common grid genuinely has no rotation, so stating it
        # removes an alarming message that means nothing here.
        header.setdefault("CROTA1", 0.0)
        header.setdefault("CROTA2", 0.0)
        name = f"{m.label}.fits"
        padded, header, pad_x, pad_y = square_pad_for_brats(m.data, header)
        write_fits_map(workspace / "maps" / name, padded, header)
        staged.append({
            "label": m.label, "file": name,
            "frequency_Hz": float(m.freq_hz),
            "rms_Jy_beam": float(m.rms_jybeam),
            "cal_frac": float(m.cal_frac),
            "pad_x": int(pad_x), "pad_y": int(pad_y),
            "padded_shape": [int(padded.shape[0]), int(padded.shape[1])],
        })

    detection = ordered[0]

    # Every staged map was padded by the same amount (they share a grid), so
    # one offset covers them all. Regions are written in IMAGE coordinates,
    # which are the PADDED image's coordinates as far as BRATS is concerned,
    # so both boxes have to move with the data. Getting this wrong would put
    # the background box over blank padding and the source box off the source.
    pad_x = int(staged[0].get("pad_x", 0)) if staged else 0
    pad_y = int(staged[0].get("pad_y", 0)) if staged else 0

    if source_region is None:
        ys, xs = np.nonzero(source_mask)
        margin_pix = 4.0
        if common_beam is not None:
            margin_pix = max(
                margin_pix,
                source_margin_beams
                * math.sqrt(max(beam_area_pixels(common_beam, detection.wcs), 1.0)),
            )
        x0, x1 = float(xs.min()) - margin_pix, float(xs.max()) + margin_pix
        y0, y1 = float(ys.min()) - margin_pix, float(ys.max()) + margin_pix
        ny, nx = source_mask.shape
        x0, y0 = max(x0, 0.0), max(y0, 0.0)
        x1, y1 = min(x1, nx - 1.0), min(y1, ny - 1.0)
        source_region_path = workspace / "regions" / "source.reg"
        write_ds9_box_region(
            source_region_path,
            0.5 * (x0 + x1) + pad_x, 0.5 * (y0 + y1) + pad_y,
            x1 - x0, y1 - y0,
            comment=(
                "loose bounding box around the detected source; BRATS applies "
                "its own sigma threshold and adaptive region selection inside it"
            ),
        )
    else:
        source_region_path = workspace / "regions" / Path(source_region).name
        shutil.copyfile(source_region, source_region_path)

    if background_region is None:
        bx, by, bw, bh = choose_background_box(detection.data, source_mask)
        background_region_path = workspace / "regions" / "background.reg"
        write_ds9_box_region(
            background_region_path, bx + pad_x, by + pad_y, bw, bh,
            comment="automatically placed source-free box for off-source RMS",
        )
    else:
        background_region_path = workspace / "regions" / Path(background_region).name
        shutil.copyfile(background_region, background_region_path)

    return {
        "workspace": workspace,
        "maps_dir": workspace / "maps",
        "regions_dir": workspace / "regions",
        "images_dir": workspace / "images",
        "data_dir": workspace / "data",
        "commands_dir": workspace / "commands",
        "source_region": source_region_path,
        "background_region": background_region_path,
        "staged_maps": staged,
        "redshift": float(redshift),
        "target_name": target_name,
        "pad_x": pad_x,
        "pad_y": pad_y,
        "pipeline_shape": [
            int(np.shape(source_mask)[0]), int(np.shape(source_mask)[1]),
        ],
    }


def write_brats_ci_files(
    workspace: Dict[str, Any],
    frequency_hz: Sequence[float],
    flux_jy: Sequence[float],
    err_jy: Sequence[float],
    detected: Sequence[bool],
    redshift: float,
    b_field_t: float,
    alpha_inj_positive: float,
    identifier: str = "PYSYNCH_SOURCE",
) -> Dict[str, Path]:
    """
    Write the two comma-delimited files that BRATS' CI/CI-off fitting reads.

        header : identifier, redshift, magnetic field (T), injection index
        data   : identifier, frequency (Hz), flux (Jy), error (Jy), upper-limit

    The upper-limit flag is where the pipeline's non-detection handling pays
    off: a band that failed the 3-sigma test is passed to BRATS as a genuine
    upper limit (flag 1) at 3 sigma, rather than being dropped or, worse,
    passed as a low-significance detection that would drag the fitted break.
    """
    data_dir = Path(workspace["data_dir"])
    header_path = data_dir / "ci_header.dat"
    data_path = data_dir / "ci_data.dat"

    header_path.write_text(
        f"{identifier}, {float(redshift):.6f}, {float(b_field_t):.6e}, "
        f"{float(alpha_inj_positive):.4f}\n"
    )

    lines: List[str] = []
    for nu, s, e, det in zip(frequency_hz, flux_jy, err_jy, detected):
        if not np.isfinite(nu) or nu <= 0 or not np.isfinite(e) or e <= 0:
            continue
        if det and np.isfinite(s) and s > 0:
            lines.append(f"{identifier}, {nu:.6e}, {s:.6e}, {e:.6e}, 0")
        else:
            lines.append(f"{identifier}, {nu:.6e}, {3.0 * e:.6e}, {e:.6e}, 1")
    if not lines:
        raise RuntimeError("No usable SED points for a BRATS CI fit.")
    data_path.write_text("\n".join(lines) + "\n")

    return {"header_file": header_path, "data_file": data_path}


# ---------------------------------------------------------------------------
# Command-script construction
# ---------------------------------------------------------------------------

def build_brats_map_script(
    workspace: Dict[str, Any],
    models: Sequence[str],
    alpha_inj_positive: float,
    b_field_t: float,
    *,
    sigma: float = 5.0,
    onsource: float = 3.0,
    sed_plot_models: Sequence[str] = (),
    plot_symbol: int = 4,
    plot_titles: bool = False,
    searcharea: float = 1.0,
    signaltonoise: float = 1.0,
    max_myears: float = 50.0,
    min_myears: float = 0.0,
    ageres: int = 10,
    levels: int = 3,
    gmin: float = 10.0,
    gmax: float = 1.0e6,
    find_injection: bool = False,
    inject_min: float = 0.5,
    inject_max: float = 1.0,
    inject_intervals: int = 10,
    inject_by_model: Optional[Dict[str, float]] = None,
    findinject_models: Optional[Sequence[str]] = None,
    export_fits: bool = True,
    set_noise: bool = True,
    dataset: int = 0,
    tag: str = "pysynch",
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Generate a BRATS command file for the resolved (map-based) analysis.

    Every line is one input to BRATS: a command line, then one line per
    parameter it prompts for.  The ordering below follows the Cookbook's
    quick-start sequence exactly.

    Returns the script text and a manifest describing which output files each
    step is expected to produce, so the results can be located afterwards
    without guessing.
    """
    ws = workspace
    lines: List[str] = []
    manifest: List[Dict[str, Any]] = []

    def cmd(*entries: Any) -> None:
        lines.extend(str(e) for e in entries)

    rel = lambda p: os.path.relpath(str(p), str(ws["workspace"]))  # noqa: E731

    # --- locations and global switches ---------------------------------
    cmd("imageloc", rel(ws["images_dir"]))
    cmd("dataloc", rel(ws["data_dir"]))
    if export_fits:
        # `export` toggles writing to file instead of screen; `exportasfits`
        # makes the map products FITS rather than PNG (BRATS >= 2.5.0).
        cmd("exportasfits")
        cmd("export")
    # Suppress the chi-square confidence tables: with tens of thousands of
    # single-pixel regions the CDF call can overflow and abort the run.
    cmd("suppressconf")

    # --- plot styling -------------------------------------------------------
    # `symbol` picks the PGPLOT marker for data points (0-126); 4 is the open
    # circle, which stays legible on a printed SED where the default filled
    # dot (1) merges with its own error bar.
    cmd("symbol", int(plot_symbol))
    # `titles` is a TOGGLE, not a set-value command, and BRATS starts with
    # titles ON -- so issuing it exactly once turns them off. Issuing it twice
    # would silently turn them back on, which is why this is not in a loop.
    if not plot_titles:
        cmd("titles")

    # --- load ------------------------------------------------------------
    cmd("sigma", f"{sigma:g}")
    cmd("load",
        rel(ws["maps_dir"]),
        rel(ws["background_region"]),
        rel(ws["source_region"]),
        f"{ws['redshift']:.6f}")

    # targetname MUST come after load. With no dataset loaded BRATS still
    # prompts for a dataset number, rejects it as out of range, and re-prompts
    # -- silently eating the following lines of the script and desynchronising
    # everything after it. The exported filenames all carry this name, so it
    # is worth setting, but only once there is something to name.
    cmd("targetname", dataset, ws["target_name"])

    # --- hand BRATS our own noise and flux-scale numbers -------------------
    #
    # Both `rmsnoise` and `fluxcalerror` prompt for: dataset, map selection
    # (0-based index, or 888 for all), then the value. Setting them
    # explicitly means BRATS and synchrofit see IDENTICAL uncertainties, which
    # is what makes a comparison between the two engines meaningful -- left to
    # itself BRATS would re-derive the noise from the background box alone,
    # and any engine difference would then partly just be a difference in
    # noise estimate.
    if set_noise:
        for i, entry in enumerate(ws["staged_maps"]):
            cmd("rmsnoise", dataset, i, f"{entry['rms_Jy_beam']:.6e}")

    cal_fracs = [entry["cal_frac"] for entry in ws["staged_maps"]]
    if len(set(round(c, 6) for c in cal_fracs)) == 1:
        cmd("fluxcalerror", dataset, 888, f"{cal_fracs[0]:.4f}")
    else:
        for i, frac in enumerate(cal_fracs):
            cmd("fluxcalerror", dataset, i, f"{frac:.4f}")

    # --- region selection --------------------------------------------------
    cmd("searcharea", f"{searcharea:g}")
    cmd("signaltonoise", f"{signaltonoise:g}")
    cmd("onsource", f"{onsource:g}")
    cmd("setregions", dataset)

    # --- spectral index ----------------------------------------------------
    cmd("specindex", dataset)
    cmd("specindexerrors", dataset)
    cmd("specchisquared", dataset)
    manifest.append({"step": "specindex", "kind": "map",
                     "expect_fits": ["SpectralIndex", "SpecIndex"]})
    cmd("exportdata", dataset, 12, tag)
    manifest.append({"step": "specindex", "kind": "table",
                     "token": "SPECIND", "tag": tag})

    # --- model parameters ---------------------------------------------------
    cmd("injectionindex", f"{alpha_inj_positive:.4f}")
    cmd("bfield", f"{b_field_t:.6e}")
    cmd("myears", f"{max_myears:g}")
    cmd("minmyears", f"{min_myears:g}")
    cmd("ageres", int(ageres))
    cmd("levels", int(levels))
    cmd("gmin", f"{gmin:g}")
    cmd("gmax", f"{gmax:g}")

    # --- injection-index minimisation ---------------------------------------
    if find_injection:
        cmd("mininject", f"{inject_min:.4f}")
        cmd("maxinject", f"{inject_max:.4f}")
        cmd("injectintervals", int(inject_intervals))
        for name in (findinject_models if findinject_models is not None
                     else models):
            spec = BRATS_MODELS.get(name, {})
            if not spec.get("maps"):
                continue
            cmd("findinject", dataset, spec["index"])
            cmd("exportdata", dataset, 13, spec["index"], f"{tag}_{name}")
            manifest.append({
                "step": "findinject", "model": name, "kind": "table",
                "token": f"INJIND_MOD{spec['index']}", "tag": f"{tag}_{name}",
            })

    # --- model fitting ------------------------------------------------------
    for name in models:
        spec = BRATS_MODELS.get(name)
        if spec is None or not spec.get("maps"):
            continue
        # `findinject` in BRATS is DIAGNOSTIC ONLY. Reading its source, it
        # fills the chi-square arrays for export and prints the bracketing
        # grid nodes, but never assigns the working `inject` variable -- its
        # own closing message says the results "can be output using the
        # exportdata command for the purposes of analysis". So unless the
        # measured value is set here explicitly, every fit below silently
        # uses whatever injection index was handed in, and the age maps come
        # out at the wrong spectral slope.
        measured = (inject_by_model or {}).get(name)
        if measured is not None and np.isfinite(measured) and measured > 0:
            cmd("injectionindex", f"{float(measured):.4f}")
        cmd(spec["fit_command"], dataset)
        cmd("specagemap", dataset, spec["index"])
        cmd("chisquaredmap", dataset, spec["index"])
        # errormap asks dataset, model, then the error sign.
        # BRATS uses 0 = POSITIVE, 1 = negative (main.c: "Use positive or
        # negative errors? (0 for positive, 1 for negative)"). Positive
        # errors are the ones normally quoted, and getting this index the
        # wrong way round silently produces a map of the other error wing.
        cmd("errormap", dataset, spec["index"], 0)
        manifest.append({"step": "fit", "model": name, "kind": "map"})
        for code_key, label in (
            ("age", "ages"), ("chi2", "chi2"),
            ("errors", "errors"), ("norm", "normalisation"),
        ):
            code = spec.get(code_key)
            if code is None:
                continue
            cmd("exportdata", dataset, code, f"{tag}_{name}")
            manifest.append({
                "step": "fit", "model": name, "quantity": label,
                "kind": "table", "token": BRATS_EXPORT_CODES[code],
                "tag": f"{tag}_{name}",
            })

    # --- the same maps again, as PGPLOT images -------------------------------
    #
    # `exportasfits` OVERRIDES `imagetype` (help.h: "Exports maps in FITS
    # format. Overrides imagetype"), so while it is on BRATS writes FITS and
    # no picture at all. It is a plain toggle, so flipping it back off and
    # re-issuing the same map commands makes BRATS render each one through
    # PGPLOT to `<name>.png/png` -- its own view of the data, with its own
    # scaling and region boundaries, which is what you want to look at
    # alongside the quantitative FITS.
    if export_fits and models:
        cmd("exportasfits")
        for name in models:
            spec = BRATS_MODELS.get(name)
            if spec is None or not spec.get("maps"):
                continue
            cmd("specagemap", dataset, spec["index"])
            cmd("chisquaredmap", dataset, spec["index"])
            cmd("errormap", dataset, spec["index"], 0)
            manifest.append({"step": "fit", "model": name, "kind": "png"})
        cmd("specindex", dataset)
        cmd("specindexerrors", dataset)
        manifest.append({"step": "specindex", "kind": "png"})
        # leave it as we found it, so anything after still gets FITS
        cmd("exportasfits")

    # --- region geometry, needed to paint tables back onto pixels ------------
    cmd("exportdata", dataset, 17, tag)
    manifest.append({"step": "regions", "kind": "table",
                     "token": "REGARRAY", "tag": tag})
    cmd("exportdata", dataset, 16, tag)
    manifest.append({"step": "regionflux", "kind": "table",
                     "token": "FLUX", "tag": tag})

    # --- integrated SED: model against the observed points ------------------
    #
    # `plotmodelobs` draws one plot PER REGION. With the adaptive regions
    # above that is thousands of them, and BRATS stops to ask "You are about
    # to export N plots! Are you sure? (1 Yes, 0 No)". A scripted run has no
    # answer for that prompt, so the NEXT command in the file is swallowed as
    # the reply, the whole stream shifts by one, and every later command runs
    # against the wrong arguments -- which silently cost the JP_Tribble age
    # export on the previous run.
    #
    # So collapse the source to a single region first. That yields exactly
    # one plot -- the integrated SED, which is what is actually wanted -- and
    # the confirmation is answered explicitly anyway so the stream cannot
    # drift again. This is done LAST, after every map export is finished,
    # because it replaces the adaptive regions.
    if sed_plot_models:
        cmd("setsingleregion", dataset)
        for name in sed_plot_models:
            spec = BRATS_MODELS.get(name)
            if spec is None or not spec.get("fit_command") or not spec.get("maps"):
                continue
            cmd(spec["fit_command"], dataset)
            cmd("plotmodelobs", dataset, spec["index"])
            cmd(1)                      # confirm the (single) plot export
            manifest.append({"step": "sed", "model": name, "kind": "sed_plot"})

    # --- full backup then exit ------------------------------------------------
    cmd("fullexport", dataset, f"{tag}_backup")
    cmd("quit")
    return "\n".join(lines) + "\n", manifest


def build_brats_integrated_script(
    workspace: Dict[str, Any],
    models: Sequence[str],
    frequency_hz: Sequence[float],
    flux_jy: Sequence[float],
    err_jy: Sequence[float],
    detected: Sequence[bool],
    redshift: float,
    alpha_inj_positive: float,
    b_field_t: float,
    *,
    max_myears: float = 50.0,
    min_myears: float = 0.0,
    ageres: int = 10,
    levels: int = 5,
    gmin: float = 10.0,
    gmax: float = 1.0e6,
    min_off: float = 0.0,
    max_off: float = 1.0,
    ci_files: Optional[Dict[str, Path]] = None,
    tag: str = "pysynch",
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Generate a BRATS command file for the INTEGRATED SED.

    Two different BRATS pathways are used, because BRATS itself separates them:

      * single-injection models (JP, KP, Tribble JP) go through
        `fitintegrated`, which prompts for the model, redshift, the number of
        points, and then frequency/flux/error for each one;
      * CI and CI-off go through `fitcimodel` / `fitcioff`, which read the two
        comma-delimited text files written by `write_brats_ci_files`.

    Only significant detections are passed to `fitintegrated`, which has no
    upper-limit flag. The CI pathway does support upper limits and receives
    the non-detections as 3-sigma limits.
    """
    ws = workspace
    lines: List[str] = []
    manifest: List[Dict[str, Any]] = []

    def cmd(*entries: Any) -> None:
        lines.extend(str(e) for e in entries)

    rel = lambda p: os.path.relpath(str(p), str(ws["workspace"]))  # noqa: E731

    cmd("imageloc", rel(ws["images_dir"]))
    cmd("dataloc", rel(ws["data_dir"]))
    cmd("suppressconf")
    # `export` switches BRATS from printing results at the terminal to
    # writing them to file. Without it `fitcimodel` reports its fit only on
    # stdout and never writes CI_ModelData_*.txt, so the CI model curve --
    # the thing needed to draw a CI SED -- is simply lost.
    cmd("export")
    cmd("symbol", 4)
    cmd("titles")          # toggle: BRATS starts with titles ON

    cmd("injectionindex", f"{alpha_inj_positive:.4f}")
    cmd("bfield", f"{b_field_t:.6e}")
    cmd("myears", f"{max_myears:g}")
    cmd("minmyears", f"{min_myears:g}")
    cmd("ageres", int(ageres))
    cmd("levels", int(levels))
    cmd("gmin", f"{gmin:g}")
    cmd("gmax", f"{gmax:g}")

    nu = np.asarray(frequency_hz, dtype=float)
    fl = np.asarray(flux_jy, dtype=float)
    er = np.asarray(err_jy, dtype=float)
    det = np.asarray(detected, dtype=bool)
    order = np.argsort(nu)
    nu, fl, er, det = nu[order], fl[order], er[order], det[order]
    use = det & np.isfinite(fl) & (fl > 0) & np.isfinite(er) & (er > 0)

    single = [
        n for n in models
        if BRATS_MODELS.get(n, {}).get("integrated")
        and BRATS_MODELS[n].get("index") is not None
    ]
    if single and np.count_nonzero(use) >= 2:
        for name in single:
            spec = BRATS_MODELS[name]
            cmd("fitintegrated", spec["index"], f"{redshift:.6f}",
                int(np.count_nonzero(use)))
            for f_hz, s_jy, e_jy in zip(nu[use], fl[use], er[use]):
                cmd(f"{f_hz:.6e}", f"{s_jy:.6e}", f"{e_jy:.6e}")
            cmd(f"{tag}_{name}")
            manifest.append({
                "step": "fitintegrated", "model": name, "kind": "table",
                "expect_name": f"IntergratedFit_Mod{spec['index']}_{tag}_{name}",
            })

    if ci_files is not None:
        for name in models:
            if name not in {"CI", "CI-OFF"}:
                continue
            if name == "CI-OFF":
                cmd("minoff", f"{min_off:g}")
                cmd("maxoff", f"{max_off:g}")
            # With `export` on, fitcimodel asks for an OUTPUT NAME after the
            # header and data files -- a third prompt it does not issue when
            # results go to the terminal. Omitting it makes BRATS swallow the
            # next command as the filename: on the previous run that ate the
            # trailing `quit`, and the fit results and the model curve were
            # both written to one file called "CI_Fitting_Results_quit.dat",
            # the curve overwriting the results. Naming it explicitly keeps
            # the stream aligned and gives the two products distinct files.
            cmd(BRATS_MODELS[name]["fit_command"],
                rel(ci_files["header_file"]), rel(ci_files["data_file"]),
                f"{tag}_{name}")
            # BRATS names these from its own timestamp, not from our tag
            # (main.c: "%s/CI_ModelData_%s.txt" with strftime), so they have
            # to be found by prefix. Without an expect_name the harvest has
            # nothing to glob for and the CI results are silently lost.
            # Two products come out of this, and the names depend on
            # whether an output name was given. With one (which we now always
            # supply, see above) BRATS writes the FIT RESULTS to
            # "CI_Fitting_Results_<outname>.dat" and the model curve to
            # "CI_ModelData_<outname>.txt"; the timestamped "CI_ModelData_"
            # form only appears when it is naming the file itself. Expecting
            # the timestamped name meant the results file was never matched
            # and every run reported "MISSING: fitcimodel CI" while the file
            # sat in the workspace, fully written.
            manifest.append({
                "step": BRATS_MODELS[name]["fit_command"], "model": name,
                "kind": "table", "token": BRATS_MODELS[name]["token"],
                "expect_name": (
                    ("CIOff_Fitting_Results_" if name == "CI-OFF"
                     else "CI_Fitting_Results_") + f"{tag}_{name}"
                ),
            })

    cmd("quit")
    return "\n".join(lines) + "\n", manifest


# ---------------------------------------------------------------------------
# Result harvesting
# ---------------------------------------------------------------------------
# Reading what BRATS exported
#
# COLUMN LAYOUTS ARE NOT UNIFORM.  This matters enormously and is the single
# easiest way to get a silently wrong answer out of BRATS.  The layouts below
# were read straight out of the `exportdata` writer in main.c (the block that
# switches on `dataexporttype`), not inferred from the menu labels:
#
#   0,1,2    JP / KP / Tribble-JP ages ......... 1 col   age (Myr)
#   3,4,5    chi-squared ....................... 1 col   chi^2
#   6,7,8    age errors ........................ 2 cols  "+plus -minus"
#   9,10,11  normalisation ..................... 1 col   norm
#   12       spectral index .................... 2 cols  alpha, sigma_alpha
#   13       injection index ................... 2 cols  inject, SUM chi^2
#   14       injection index by region ......... 2 cols  region_id, inject
#   15       injection chi^2 by region ......... 2 cols  region_id, chi^2
#   16       region flux (one file per band) ... 1 col   flux (Jy/beam)
#   17       region array ...................... 3 cols  x, y, region_id
#
# Reading column 0 of every file -- which is what a naive one-value-per-line
# parser does -- is therefore correct for only about half of them.  For code
# 13 it returns the *trial injection index grid* instead of the chi-squared
# curve, so taking its minimum returns the first grid node every single time
# and the "BRATS injection index" is then just `mininject`, independent of the
# data.  For codes 14/15 it returns region ID numbers.  For 17 it returns x
# pixel coordinates.  Each of those is a wrong number that looks entirely
# plausible in a summary table, which is exactly why the layout is encoded
# explicitly here rather than assumed.
# ---------------------------------------------------------------------------

#: token -> (column names, dtype hint).  Tokens are the strings BRATS itself
#: writes into the filename, so they cannot disagree with the data.
BRATS_TABLE_COLUMNS: Dict[str, Tuple[str, ...]] = {
    "JP": ("age_Myr",),
    "KP": ("age_Myr",),
    "TRIBJP": ("age_Myr",),
    "JPX2": ("chi2",),
    "KPX2": ("chi2",),
    "TRIBJPX2": ("chi2",),
    "JPERRORS": ("age_err_plus_Myr", "age_err_minus_Myr"),
    "KPERRORS": ("age_err_plus_Myr", "age_err_minus_Myr"),
    "TRIBJPERRORS": ("age_err_plus_Myr", "age_err_minus_Myr"),
    "JPNORM": ("normalisation",),
    "KPNORM": ("normalisation",),
    "TRIBJPNORM": ("normalisation",),
    "SPECIND": ("alpha_positive", "alpha_err"),
    "FLUX": ("flux_Jy_beam",),
    "REGARRAY": ("x", "y", "region_id"),
    "INJECTBYREGION": ("region_id", "injection_index"),
    "INJECTCHISQUAREDBYREGION": ("region_id", "chi2"),
    # fitcimodel writes ONE comma-separated line per dataset, opening with the
    # dataset name rather than a number -- a different shape from every
    # exportdata table above. Only the fields verified against the values
    # BRATS itself prints on the CI figure (injection index, B field,
    # spectral age, reduced chi-square) and those forced by arithmetic
    # (chi2 / dof = chi2_reduced) are named. The rest are left positional on
    # purpose: guessing a name here would put an authoritative-looking but
    # unverified quantity into the results JSON, which is precisely the
    # failure mode the explicit-layout table exists to prevent.
    "CI": (
        "redshift", "b_field_T", "alpha_inj_positive",
        "field_03", "field_04", "field_05", "field_06", "field_07",
        "age_Myr",
        "field_09", "field_10", "field_11", "field_12", "field_13",
        "field_14", "field_15", "field_16",
        "chi2", "chi2_reduced", "dof",
        "normalisation", "break_frequency_Hz",
    ),
}

#: the value column to summarise for each token, when one is meaningful
BRATS_PRIMARY_COLUMN: Dict[str, str] = {
    "JP": "age_Myr", "KP": "age_Myr", "TRIBJP": "age_Myr",
    "JPX2": "chi2", "KPX2": "chi2", "TRIBJPX2": "chi2",
    "JPERRORS": "age_err_plus_Myr", "KPERRORS": "age_err_plus_Myr",
    "TRIBJPERRORS": "age_err_plus_Myr",
    "JPNORM": "normalisation", "KPNORM": "normalisation",
    "TRIBJPNORM": "normalisation",
    "SPECIND": "alpha_positive",
    "FLUX": "flux_Jy_beam",
    "INJECTBYREGION": "injection_index",
    "INJECTCHISQUAREDBYREGION": "chi2",
    "CI": "age_Myr",
}


def brats_table_columns(token: Optional[str]) -> Tuple[str, ...]:
    """
    Column names for an export token.

    `INJIND_MOD<n>` carries the model index in the token itself, so it is
    matched by prefix rather than by an exact key.
    """
    if not token:
        return ()
    if token.startswith("INJIND_MOD"):
        return ("injection_index", "sum_chi2")
    return BRATS_TABLE_COLUMNS.get(token, ())


def parse_brats_table(
    path: Path, token: Optional[str] = None,
) -> Dict[str, np.ndarray]:
    """
    Read a BRATS exportdata table into named columns.

    The age-error files are written as `"+%f -%f"`, so the tokens carry
    leading signs that must not be read as a subtraction; `float("+1.2")` and
    `float("-0.3")` both parse correctly, and the minus column is stored as a
    positive magnitude because that is how BRATS means it (it is the size of
    the downward error bar, not a negative age).

    Rows whose column count does not match the expected layout are skipped
    and counted, rather than being silently coerced -- a file that does not
    look like what its token promises is a sign that BRATS wrote something
    unexpected, and that is worth surfacing.
    """
    names = brats_table_columns(token)
    rows: List[List[float]] = []
    n_bad = 0
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = [p for p in re.split(r"[,\s]+", line) if p]
        # fitcimodel opens its line with the dataset NAME. Every other BRATS
        # table is numeric throughout, so a leading non-numeric field used to
        # make the whole row unparseable and the CI fit was reported as a bad
        # row. Drop one leading label, and one only: a row that is still not
        # numeric after that is genuinely not what its token promised.
        if parts:
            try:
                float(parts[0])
            except ValueError:
                parts = parts[1:]
        try:
            values = [float(p) for p in parts]
        except ValueError:
            n_bad += 1
            continue
        if names and len(values) < len(names):
            n_bad += 1
            continue
        rows.append(values)

    if not rows:
        return {"_n_rows": np.asarray([0]), "_n_bad_rows": np.asarray([n_bad])}

    width = min(len(r) for r in rows)
    array = np.asarray([r[:width] for r in rows], dtype=float)

    out: Dict[str, np.ndarray] = {}
    if names:
        for i, name in enumerate(names[:width]):
            column = array[:, i]
            # BRATS writes the minus error as "-x"; the magnitude is meant.
            if name.endswith("_minus_Myr"):
                column = np.abs(column)
            out[name] = column
    else:
        for i in range(width):
            out[f"col{i}"] = array[:, i]
    out["_n_rows"] = np.asarray([array.shape[0]])
    out["_n_bad_rows"] = np.asarray([n_bad])
    return out


def parse_brats_value_table(
    path: Path, token: Optional[str] = None,
) -> np.ndarray:
    """
    The single scientifically meaningful column of a BRATS export.

    Kept as the convenience entry point, but it now selects the column that
    actually holds the measurement rather than assuming it is the first one.
    Without a token it falls back to column 0, which is all that can be done
    with no way to know the layout.
    """
    table = parse_brats_table(path, token)
    if token and token.startswith("INJIND_MOD"):
        key = "sum_chi2"
    else:
        key = BRATS_PRIMARY_COLUMN.get(token or "", "")
    if key and key in table:
        return np.asarray(table[key], dtype=float)
    for name, column in table.items():
        if not name.startswith("_"):
            return np.asarray(column, dtype=float)
    return np.asarray([], dtype=float)

def harvest_brats_outputs(
    workspace: Dict[str, Any], manifest: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Locate and read whatever BRATS actually wrote.

    Files are found by GLOBBING for the datasettype token BRATS embeds in the
    filename rather than by reconstructing the name from the menu index.  That
    matters: the exportdata menu numbering has been renumbered between BRATS
    releases, so a hardcoded index-to-quantity map would silently mislabel
    results on a different version, whereas the token in the filename is
    written by the same branch that produced the data and cannot disagree with
    it.  Anything requested but not produced is reported as missing rather
    than quietly skipped.
    """
    data_dir = Path(workspace["data_dir"])
    images_dir = Path(workspace["images_dir"])

    harvest: Dict[str, Any] = {
        "tables": {}, "maps": [], "missing": [],
        "data_dir": str(data_dir), "images_dir": str(images_dir),
    }

    for entry in manifest:
        if entry.get("kind") != "table":
            continue
        token, tag = entry.get("token"), entry.get("tag")
        if token and tag:
            matches = sorted(data_dir.glob(f"*_{token}_{tag}.*"))
        elif entry.get("expect_name"):
            matches = sorted(data_dir.glob(f"{entry['expect_name']}*"))
        else:
            # A manifest entry with no way to identify its output is a bug in
            # the script builder, not something to pass over in silence: it
            # would look exactly like a successful run that produced nothing.
            harvest["missing"].append({
                "step": entry.get("step"), "model": entry.get("model"),
                "quantity": entry.get("quantity"), "token": None,
                "reason": (
                    "no export token or filename was recorded for this step, "
                    "so its output could not be located"
                ),
            })
            continue
        if not matches:
            harvest["missing"].append({
                "step": entry.get("step"), "model": entry.get("model"),
                "quantity": entry.get("quantity"), "token": token,
            })
            continue
        path = matches[0]
        key = "_".join(
            str(x) for x in (
                entry.get("step"), entry.get("model"), entry.get("quantity")
            ) if x
        )
        try:
            table = parse_brats_table(path, token)
            values = parse_brats_value_table(path, token)
            record: Dict[str, Any] = {
                "file": str(path), "token": token,
                "step": entry.get("step"),
                "model": entry.get("model"),
                "quantity": entry.get("quantity"),
                "columns": [c for c in table if not c.startswith("_")],
                "n_values": int(values.size),
                "n_bad_rows": int(table.get("_n_bad_rows", [0])[0]),
                "median": float(np.nanmedian(values)) if values.size else None,
                "min": float(np.nanmin(values)) if values.size else None,
                "max": float(np.nanmax(values)) if values.size else None,
                "values": values.tolist() if values.size <= 20000 else None,
                "values_truncated": bool(values.size > 20000),
            }
            # Keep every column, not just the summarised one: the second
            # column of SPECIND is the spectral-index error map, the second
            # of INJIND_MOD* is the chi-squared curve the injection index is
            # actually read from, and REGARRAY is the region -> pixel
            # mapping that lets any per-region table be painted back onto
            # the sky. Discarding them would throw away real measurements.
            record["table"] = {
                name: (column.tolist() if column.size <= 200000 else None)
                for name, column in table.items()
                if not name.startswith("_")
            }
            harvest["tables"][key] = record
        except Exception as exc:                               # noqa: BLE001
            harvest["tables"][key] = {"file": str(path), "error": str(exc)}

    for path in sorted(images_dir.glob("*.fits")):
        harvest["maps"].append({"file": str(path), "name": path.name})
    # PGPLOT renderings written by the second pass, kept as BRATS' own view
    harvest["png_maps"] = [
        {"file": str(path), "name": path.name}
        for path in sorted(images_dir.glob("*.png"))
    ]
    harvest["n_png_maps"] = len(harvest["png_maps"])
    harvest["n_maps"] = len(harvest["maps"])
    harvest["n_tables"] = len(harvest["tables"])
    return harvest


def summarise_brats_injection(
    harvest: Dict[str, Any], beam_area_pixels_value: float = 1.0,
) -> Dict[str, Any]:
    """
    Turn a BRATS `findinject` chi-square curve into an actual injection index.

    The exported file has TWO columns -- trial injection index, and the
    chi-square summed over every region -- so the minimum has to be taken in
    the second column and read off in the first.  Reading column 0 and
    minimising it returns `mininject` every time, whatever the data say.

    Two corrections are applied to get from that curve to a quotable number:

    (1) PARABOLIC REFINEMENT.  The grid is typically 10 nodes across a range
        of 0.5, i.e. a spacing of 0.05, which is coarser than the precision
        the curve actually supports.  A parabola through the minimum node and
        its two neighbours locates the vertex to well inside one spacing.
        The refinement is rejected if it falls outside the bracketing nodes,
        which is what happens when the minimum is at a grid edge and the
        curve is not really minimised at all.

    (2) BEAM-AREA CORRECTION.  As the BRATS Cookbook stresses, the summed
        chi-square is over-weighted because neighbouring regions inside one
        beam are not independent measurements.  The sum behaves as though
        there were N_pix constraints when there are really about
        N_pix / A_beam.  Dividing Delta chi-square by the beam area before
        applying the Delta chi-square = 1 rule undoes this; skipping it
        produces an error bar roughly sqrt(A_beam) too small, which for a
        typical map is a factor of several.

    An interval is only quoted when the curve genuinely brackets its minimum.
    A monotonic curve means the best injection index lies outside
    [mininject, maxinject] and the honest answer is a limit, not a value.
    """
    beam_area = float(beam_area_pixels_value)
    if not np.isfinite(beam_area) or beam_area <= 0:
        beam_area = 1.0

    out: Dict[str, Any] = {"models": {}, "beam_area_pixels": beam_area}
    for key, table in (harvest.get("tables") or {}).items():
        if "findinject" not in key:
            continue
        columns = table.get("table") or {}
        grid = columns.get("injection_index")
        chi2 = columns.get("sum_chi2")
        if grid is None or chi2 is None:
            continue
        grid = np.asarray(grid, dtype=float)
        chi2 = np.asarray(chi2, dtype=float)
        good = np.isfinite(grid) & np.isfinite(chi2)
        grid, chi2 = grid[good], chi2[good]
        if grid.size < 3:
            continue
        order = np.argsort(grid)
        grid, chi2 = grid[order], chi2[order]

        model = key.split("_", 1)[1] if "_" in key else key
        node = int(np.argmin(chi2))
        best = float(grid[node])
        at_edge = node in (0, grid.size - 1)

        # -- parabolic vertex through the three nodes around the minimum ----
        refined = best
        if not at_edge:
            x0, x1, x2 = grid[node - 1], grid[node], grid[node + 1]
            y0, y1, y2 = chi2[node - 1], chi2[node], chi2[node + 1]
            # General (unequally spaced) three-point parabolic vertex. BRATS'
            # grid is uniform, but nothing guarantees the exported nodes stay
            # that way, and the general form costs nothing.
            denom = (x1 - x0) * (y1 - y2) - (x1 - x2) * (y1 - y0)
            if np.isfinite(denom) and denom != 0:
                vertex = x1 - 0.5 * (
                    (x1 - x0) ** 2 * (y1 - y2) - (x1 - x2) ** 2 * (y1 - y0)
                ) / denom
                if np.isfinite(vertex) and x0 <= vertex <= x2:
                    refined = float(vertex)

        # -- beam-corrected Delta chi-square = 1 interval --------------------
        #
        # Interpolating the Delta chi-square curve linearly between grid
        # nodes is not good enough here.  `findinject` grids are coarse (10
        # nodes over a range of 0.5 is the default, so a spacing of 0.05)
        # while the summed chi-square is enormous and steeply curved, so the
        # curve climbs by hundreds between adjacent nodes.  A straight line
        # drawn across that gap crosses Delta chi-square = 1 almost
        # immediately and returns an error bar that can be an order of
        # magnitude too small.
        #
        # The curve near its minimum is a parabola, so the curvature is what
        # should be used:  chi2 = chi2_min + A (x - x_v)^2  gives, after the
        # beam-area correction,  sigma = sqrt(A_beam / A).  A is fitted over
        # up to five nodes centred on the minimum, which is steadier against
        # a single noisy node than the bare three-point form.
        delta = (chi2 - np.nanmin(chi2)) / beam_area
        sigma = None
        curvature = None
        lower = upper = None
        grid_limited = False

        if not at_edge:
            lo = max(node - 2, 0)
            hi = min(node + 3, grid.size)
            if hi - lo < 3:
                lo, hi = max(node - 1, 0), min(node + 2, grid.size)
            try:
                coeffs = np.polyfit(grid[lo:hi], chi2[lo:hi], 2)
                curvature = float(coeffs[0])
            except Exception:                                  # noqa: BLE001
                curvature = None
            if curvature is not None and np.isfinite(curvature) and curvature > 0:
                sigma = float(math.sqrt(beam_area / curvature))
                lower, upper = refined - sigma, refined + sigma

            # The grid can only localise the minimum to about its own
            # spacing. A formal error far below that is a statement about the
            # parabola, not about a sampled curve, so it is flagged rather
            # than quietly quoted.
            spacing = float(np.median(np.diff(grid))) if grid.size > 1 else 0.0
            if sigma is not None and spacing > 0 and sigma < 0.5 * spacing:
                grid_limited = True

        # Straight interpolation is kept purely as a cross-check on the
        # parabola; it is not what gets quoted.
        interp_lower = interp_upper = None
        if not at_edge:
            left_x, left_d = grid[:node + 1], delta[:node + 1]
            right_x, right_d = grid[node:], delta[node:]
            if left_d.size > 1 and np.nanmax(left_d) >= 1.0:
                interp_lower = float(np.interp(1.0, left_d[::-1], left_x[::-1]))
            if right_d.size > 1 and np.nanmax(right_d) >= 1.0:
                interp_upper = float(np.interp(1.0, right_d, right_x))

        out["models"][model] = {
            "n_nodes": int(grid.size),
            "grid_min": float(grid[0]),
            "grid_max": float(grid[-1]),
            "grid_spacing": float(np.median(np.diff(grid))) if grid.size > 1 else None,
            "chi2_min": float(np.nanmin(chi2)),
            "argmin_node": node,
            "alpha_inj_positive_gridpoint": best,
            "alpha_inj_positive": float(refined),
            "alpha_inj_lower": lower,
            "alpha_inj_upper": upper,
            "alpha_inj_error": sigma,
            "chi2_curvature": curvature,
            "alpha_inj_lower_interp": interp_lower,
            "alpha_inj_upper_interp": interp_upper,
            "grid_resolution_limited": bool(grid_limited),
            "minimum_is_at_grid_edge": bool(at_edge),
            "beam_area_pixels": beam_area,
            "file": table.get("file"),
            "status": (
                "minimum_at_grid_edge" if at_edge
                else ("bracketed" if sigma is not None else "curvature_undefined")
            ),
            "note": (
                "Injection index read from the chi-square column (column 2 "
                "of the export), refined by a parabola through the minimum. "
                "The error is sqrt(A_beam / A) from the fitted curvature A, "
                "which applies the Delta chi-square = 1 rule after dividing "
                f"the summed chi-square by the beam area ({beam_area:.1f} "
                "pixels) to account for the non-independence of regions "
                "within one beam."
                + (
                    " The minimum sits on the edge of the search range, so "
                    "this is a LIMIT, not a measurement: widen "
                    "--brats-inject-min / --brats-inject-max and re-run."
                    if at_edge else ""
                )
                + (
                    " The formal error is smaller than half the grid spacing, "
                    "so it is set by the assumed parabola rather than by the "
                    "sampled curve; re-run with more --brats-inject-intervals "
                    "over a narrower range to measure it directly."
                    if grid_limited else ""
                )
            ),
        }
    return out

# ---------------------------------------------------------------------------
# Painting BRATS' per-region tables back onto the sky
#
# BRATS reports resolved quantities as one value per REGION, plus a separate
# REGARRAY export giving the region ID of every pixel.  Its own FITS writer
# exists (`specagemap`, `errormap`, `chisquaredmap` with `exportasfits`), but
# it is worth reconstructing the maps here as well, for three reasons:
#
#   * the FITS BRATS writes carries only a minimal header, so it cannot be
#     overlaid on anything without re-attaching the WCS by hand;
#   * `exportasfits` is a relatively recent switch and silently produces PNGs
#     on older builds, which cannot be used quantitatively at all;
#   * the error maps are the quantity most often wanted and are exported as
#     a two-column "+plus -minus" table, so both wings are available here
#     whereas the FITS map holds only whichever sign was requested.
#
# Reconstructing from REGARRAY also means every exported quantity -- including
# ones BRATS has no map writer for, such as the per-region injection index --
# can be turned into an image on exactly the pipeline's own pixel grid.
# ---------------------------------------------------------------------------

def paint_brats_region_map(
    region_array: Dict[str, np.ndarray],
    values: np.ndarray,
    shape: Tuple[int, int],
    source_mask: Optional[np.ndarray] = None,
    pad_x: int = 0,
    pad_y: int = 0,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Build a pixel map from a BRATS region array and one value per region.

    BRATS numbers regions from 1, so `values[region_id - 1]` is the value of
    the pixel's region.  Pixels not assigned to a region stay NaN.

    ORIENTATION.  The BRATS source carries an explicit admission next to this
    export -- "The x/y --> i/j mapping has become crossed over somewhere ...
    have just swapped them over for now" -- so the axis order is not something
    to take on trust.  Both interpretations are tried and scored on how much
    of the painted area lands inside the pipeline's own source mask; the
    orientation that agrees with the segmentation is used.  For a
    non-square image the wrong one simply falls outside the array and is
    rejected on bounds alone.
    """
    ny, nx = int(shape[0]), int(shape[1])
    # BRATS worked on the SQUARE-PADDED copy of the maps, so its pixel
    # coordinates are in that frame. Undo the padding to get back to the
    # pipeline's own grid; without this every map comes out shifted by the
    # pad, which is a silent registration error rather than a visible one.
    xs = np.asarray(region_array.get("x"), dtype=float) - float(pad_x)
    ys = np.asarray(region_array.get("y"), dtype=float) - float(pad_y)
    ids = np.asarray(region_array.get("region_id"), dtype=float)
    values = np.asarray(values, dtype=float)

    if xs.size == 0 or ids.size != xs.size or ys.size != xs.size:
        raise ValueError("REGARRAY export is empty or has mismatched columns.")

    diagnostics: Dict[str, Any] = {
        "pad_x": int(pad_x), "pad_y": int(pad_y),
        "n_region_pixels": int(xs.size),
        "n_regions_in_array": int(np.nanmax(ids)) if ids.size else 0,
        "n_values": int(values.size),
    }

    def attempt(col: np.ndarray, row: np.ndarray) -> Optional[np.ndarray]:
        col_i = np.rint(col).astype(int)
        row_i = np.rint(row).astype(int)
        idx = np.rint(ids).astype(int) - 1
        # A region can legitimately extend into the padding (BRATS grew it
        # there), so out-of-range pixels are dropped rather than taken as
        # proof that this orientation is wrong. An orientation that is
        # genuinely wrong loses almost everything and loses the mask overlap
        # test below, which is the real discriminator.
        ok = (
            (idx >= 0) & (idx < values.size)
            & (col_i >= 0) & (col_i < nx)
            & (row_i >= 0) & (row_i < ny)
        )
        if not ok.any():
            return None
        out = np.full((ny, nx), np.nan, dtype=float)
        out[row_i[ok], col_i[ok]] = values[idx[ok]]
        return out

    candidates: List[Tuple[str, np.ndarray]] = []
    direct = attempt(xs, ys)
    if direct is not None:
        candidates.append(("x=column, y=row", direct))
    swapped = attempt(ys, xs)
    if swapped is not None:
        candidates.append(("x=row, y=column (axes swapped)", swapped))

    if not candidates:
        raise ValueError(
            "The REGARRAY pixel coordinates do not fit the common-beam image "
            f"({ny} x {nx}). BRATS was probably run on different maps from "
            "the ones this pipeline produced."
        )

    if len(candidates) == 1 or source_mask is None:
        label, painted = candidates[0]
    else:
        mask = np.asarray(source_mask, dtype=bool)
        scored = []
        for label, painted in candidates:
            filled = np.isfinite(painted)
            n = int(filled.sum())
            overlap = float((filled & mask).sum() / n) if n else 0.0
            scored.append((overlap, label, painted))
        scored.sort(key=lambda item: item[0], reverse=True)
        diagnostics["orientation_scores"] = {
            label: round(score, 4) for score, label, _ in scored
        }
        _, label, painted = scored[0]

    diagnostics["orientation"] = label
    filled = np.isfinite(painted)
    diagnostics["n_pixels_painted"] = int(filled.sum())
    if source_mask is not None:
        mask = np.asarray(source_mask, dtype=bool)
        diagnostics["fraction_inside_source_mask"] = (
            float((filled & mask).sum() / max(int(filled.sum()), 1))
        )
    return painted, diagnostics


def source_contour_mask(
    reference: np.ndarray, lowest_level: float, source_mask: np.ndarray,
    grow_pixels: int = 0,
) -> np.ndarray:
    """
    Region within which reference contours may be drawn.

    Contours are there to show WHERE the derived quantity sits on the source,
    so they must not wander off it. Following connected emission was still
    too permissive: at low surface brightness the source is bridged to its
    neighbours, so the outermost contour swept across the whole field and
    left the reader unable to tell the source from the background.

    The rule here is strictly local -- the segmented source, grown by about
    one restoring beam so the lowest level closes just outside it rather than
    being cut along the mask edge, and nothing else. Anything beyond that is
    not part of this source and is not drawn.
    """
    mask = np.asarray(source_mask, dtype=bool)
    grow = int(max(grow_pixels, 1))

    # The region contours may occupy is defined by the REFERENCE EMISSION,
    # not by the segmentation mask.
    #
    # Bounding it by a dilated segmentation mask makes every derived map
    # inherit the segmentation's own outline: the hysteresis mask is a
    # generous, ragged polygon (7687 px on this source, reaching well past
    # the bright emission and throwing a lobe to the north), and clipping the
    # contours to it stamps that same ragged edge -- and its stray northern
    # excursion -- onto maps that have nothing to do with segmentation. The
    # segmentation outline belongs in the QA figure that documents it and
    # nowhere else.
    #
    # So the mask is used ONLY to identify WHICH emission component is the
    # target; the drawn extent follows the emission itself.
    try:
        ref_arr = np.asarray(reference, dtype=float)
        lit = np.isfinite(ref_arr) & (ref_arr >= float(lowest_level))
        if lit.any():
            labels, n = connected_components(lit)
            if n > 0:
                keep = np.unique(labels[mask & lit])
                keep = keep[keep > 0]
                if keep.size:
                    # Touching the mask is NOT enough to be the target. The
                    # hysteresis mask is generous enough to swallow a faint
                    # neighbour whole -- on this source it engulfs all 53 px
                    # of an unrelated 5.9-sigma component to the north -- and
                    # every such component then gets its own contour drawn on
                    # a map that is only about the target. Keep the single
                    # component that dominates the mask; everything else is a
                    # neighbour and is excised below.
                    counts = [int(np.count_nonzero(labels[mask] == c))
                              for c in keep]
                    keep = keep[[int(np.argmax(counts))]]
                    target = np.isin(labels, keep)
                    other = lit & ~target
                    allowed = binary_dilation(target, iterations=grow)
                    if other.any():
                        allowed &= ~binary_dilation(other, iterations=grow)
                    return allowed
    except Exception:                                          # noqa: BLE001
        pass

    halo = binary_dilation(mask, iterations=grow)

    # Dilating alone is not enough. The halo is an annulus of blank sky one
    # beam wide, and any unrelated noise peak or field source that happens to
    # fall inside it is still above the lowest contour level -- so it gets
    # contoured too, and the figure ends up with short arcs floating in empty
    # sky beside the source, with no emission under them to explain what they
    # are. (This was visible in the WAT spectral-index map: two orphan arcs,
    # one north of the core and one south of the tail.)
    #
    # So intersect the halo with the connected emission that actually belongs
    # to this source: label the pixels above the lowest contour level, keep
    # only the components that touch the segmented source, and drop the rest.
    # A contour is then drawn only where there is source emission to draw it
    # around.
    try:
        ref = np.asarray(reference, dtype=float)
        above = np.isfinite(ref) & (ref >= float(lowest_level))
        if above.any():
            labels, n = connected_components(above)
            if n > 0:
                keep = np.unique(labels[mask & above])
                keep = keep[keep > 0]
                if keep.size:
                    connected = np.isin(labels, keep)
                    # Grow the kept emission so the lowest contour closes
                    # just outside it rather than being sliced along the
                    # level set it is drawn at.
                    allowed = halo & binary_dilation(connected,
                                                     iterations=grow)
                    # ...then subtract a beam-wide exclusion zone around
                    # every OTHER emission component. Keeping only the
                    # source's own component is not sufficient on a crowded
                    # field: dilating it by a beam can push the allowed
                    # region back over a neighbour that sits just outside
                    # the segmentation, and the neighbour is then contoured
                    # as a set of open arcs floating in blank sky beside the
                    # source. (Seen on J021926-051535: a companion ~20 arcsec
                    # east produced exactly that.) Excluding the neighbours
                    # explicitly is the only version that holds when the
                    # field is busy.
                    other = above & ~connected
                    if other.any():
                        allowed &= ~binary_dilation(other, iterations=grow)
                    return allowed
    except Exception:                                          # noqa: BLE001
        pass
    return halo

def smooth_reference_for_contours(
    data: np.ndarray, beam, wcs, fraction: float = 8.0
) -> np.ndarray:
    """
    Cosmetic smoothing applied to the reference image BEFORE contouring.

    Contours trace a level set pixel by pixel, so residual pixel-to-pixel
    noise turns what should be a clean closed curve into a crenellated one,
    and the ragged edge is the single most common reason overlaid contours
    look wrong in a figure. Convolving with a Gaussian of one EIGHTH of the
    restoring-beam FWHM suppresses it.

    This is a rendering choice and nothing else. It touches only the array
    handed to `ax.contour`; every flux, spectral index, age and error in this
    pipeline is measured from the unsmoothed data. The kernel FWHM is 1/8 of
    the beam, so the drawn contour's effective resolution grows by
    sqrt(1 + (1/8)^2) = 0.8 per cent -- less than the width of the line.

    Measured honestly, it does nothing on well-sampled data at a high contour
    level: on the 1284 MHz MIGHTEE map (5.0 pixels per beam) the 5 sigma
    level set has an identical perimeter and area before and after, because
    at 5 sigma there is no pixel noise left to remove. It earns its place
    only at low levels or coarse sampling -- a 3 sigma contour, or a map with
    2-3 pixels across the beam -- where the level set does crawl along
    individual noisy pixels. It is kept for those cases and is a no-op here.

    Returns the input unchanged if the beam or WCS is unknown, rather than
    guessing a kernel.
    """
    arr = np.asarray(data, dtype=float)
    if beam is None or wcs is None or fraction <= 0:
        return arr
    try:
        from scipy.ndimage import gaussian_filter
        sx, sy = pixel_scales_arcsec(wcs)
        fwhm_pix = math.sqrt(
            (float(beam.major.to(u.arcsec).value) / max(sy, 1e-12))
            * (float(beam.minor.to(u.arcsec).value) / max(sx, 1e-12))
        )
        sigma_pix = (fwhm_pix / float(fraction)) / 2.3548200450309493
        if not np.isfinite(sigma_pix) or sigma_pix <= 0.05:
            return arr
        finite = np.isfinite(arr)
        if not finite.all():
            # Convolve with NaNs held out, then renormalise, so blanked
            # pixels do not bleed zeros into the edge of the source.
            filled = np.where(finite, arr, 0.0)
            num = gaussian_filter(filled, sigma_pix, mode="nearest")
            den = gaussian_filter(finite.astype(float), sigma_pix,
                                  mode="nearest")
            out = np.where(den > 1e-8, num / np.maximum(den, 1e-8), np.nan)
            return np.where(finite, out, np.nan)
        return gaussian_filter(arr, sigma_pix, mode="nearest")
    except Exception:                                          # noqa: BLE001
        return arr

def reference_contour_levels(
    data: np.ndarray, rms: float, start_sigma: float = 3.0,
    factor: float = 2.0, max_levels: int = 9, drop_lowest: int = 0,
    explicit_sigma: Optional[Sequence[float]] = None,
) -> List[float]:
    """
    Contour levels for overlaying a total-intensity reference image.

    The radio convention is a geometric ladder anchored on the noise:
    3 sigma, 6, 12, 24 ... up to the peak.  Starting at 3 sigma keeps noise
    peaks out of the plot, and doubling keeps the number of contours readable
    across the very large dynamic range a radio map usually has.

    Levels above the image peak are dropped rather than drawn, because
    matplotlib silently ignores them and an unwary reader counting contours
    would otherwise mis-read the dynamic range.

    `drop_lowest` discards that many levels from the BOTTOM of the finished
    ladder. The lowest level is the detection isophote, and it is very
    nearly the same curve as the segmentation boundary -- measured on this
    source the two run within 1.1 pixels of each other, against a 17-pixel
    beam. Drawn on a derived map it therefore reads as the segmentation
    outline stamped over the data rather than as a contour of the emission,
    which is exactly the thing that outline must not do. Dropping it leaves
    the ladder anchored on the same noise level and the same geometric
    spacing; only the outermost line goes. The count is made up from the
    top, so the number of contours the reader sees does not change.

    `explicit_sigma` replaces the ladder outright with the multiples of
    the noise given, in order. It exists because a geometric ladder
    cannot express every set a figure may need -- a run of doublings
    finished off by a close pair near the peak, say -- and faking one by
    tuning `factor` would silently move every other level with it.
    `start_sigma`, `factor`, `max_levels` and `drop_lowest` are all
    ignored when it is supplied; the peak cut still applies.
    """
    finite = np.asarray(data, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0 or not np.isfinite(rms) or rms <= 0:
        return []
    peak = float(np.nanmax(finite))
    if not np.isfinite(peak) or peak <= start_sigma * rms:
        return []
    if explicit_sigma is not None:
        return [float(x) * float(rms) for x in explicit_sigma
                if np.isfinite(x) and float(x) > 0
                and float(x) * float(rms) <= peak]

    drop = max(int(drop_lowest), 0)
    wanted = int(max_levels) + drop
    levels: List[float] = []
    level = float(start_sigma) * float(rms)
    while level <= peak and len(levels) < wanted:
        levels.append(level)
        level *= float(factor)
    return levels[drop:]

#: BRATS failures worth lifting out of a few hundred lines of prompt echo.
#: The text is matched against its own error strings in main.c / load.h.
BRATS_ERROR_PATTERNS: Tuple[Tuple[str, str], ...] = (
    ("is not square",
     "BRATS refuses non-square images. The pipeline now pads the staged maps "
     "to a square before writing them, so seeing this means an older "
     "workspace was reused: delete brats_workspace/ and re-run."),
    ("Data failed to load",
     "`load` failed, so every later command ran against an empty dataset. "
     "The real cause is the message immediately before this one."),
    ("No data sets have yet been loaded",
     "Commands were sent before any dataset existed -- `load` did not "
     "succeed."),
    ("No datasets have yet been loaded",
     "Commands were sent before any dataset existed -- `load` did not "
     "succeed."),
    ("filter compilation failed",
     "FUNTOOLS compiles a small C program for each region filter, and that "
     "compile failed -- usually because the default `cc` on this machine is "
     "a Homebrew GCC whose headers clash with the macOS SDK. The pipeline "
     "sets FILTER_CC to Apple clang on macOS; if you overrode FILTER_CC, "
     "point it at a compiler that can build against the system headers."),
    ("Unable to obtain the EQUINOX",
     "The staged FITS header has no EQUINOX or EPOCH. BRATS does not read "
     "RADESYS, so the equinox must be stated explicitly; the pipeline writes "
     "it, so seeing this means an older workspace was reused -- delete "
     "brats_workspace/ and re-run."),
    ("Unable to obtain the",
     "BRATS could not find a keyword it requires in the FITS header "
     "(RA, Dec, reference pixel, frequency or equinox) and aborted the load."),
    ("Unable to locate or access",
     "A file BRATS was pointed at is missing or unreadable."),
    ("unable to define any regions",
     "`setregions` could not build a single region. The signal-to-noise "
     "target is unreachable: no region can grow large enough to reach it "
     "before running out of source. Lower --brats-signaltonoise (at a "
     "38-arcsec beam a source only a few beams across cannot support a "
     "target of 30), or drop the resolved BRATS analysis with "
     "--brats-no-maps and use a higher-resolution band set for it. NOTE: "
     "every command after this one desynchronises, because the failed "
     "prompt swallows the following script lines -- so all later output in "
     "this run is unreliable, not just the regions."),
    ("No regions",
     "Region selection produced nothing: the sigma threshold is above the "
     "data, or the source region does not overlap the emission."),
)


def diagnose_brats_run(run: Dict[str, Any]) -> List[Dict[str, str]]:
    """
    Lift real failures out of a BRATS transcript.

    BRATS exits 0 even when `load` failed, and it echoes every prompt it
    printed, so a broken run looks like a few hundred lines of ordinary
    dialogue with two fatal lines buried in the middle. Reporting only the
    downstream symptom -- "no REGARRAY was exported" -- sends you looking at
    the sigma threshold when the actual message was about image geometry.
    """
    text = (run.get("stdout") or "") + "\n" + (run.get("stderr") or "")
    found: List[Dict[str, str]] = []
    seen: set = set()
    for line in text.splitlines():
        stripped = line.strip().strip("*").strip()
        if not stripped:
            continue
        for needle, explanation in BRATS_ERROR_PATTERNS:
            if needle.lower() in stripped.lower() and needle not in seen:
                seen.add(needle)
                found.append({
                    "message": stripped[:300],
                    "meaning": explanation,
                })
                break
    return found

def run_brats_analysis(
    outdir: Path,
    common_maps: Sequence[RadioMap],
    source_mask: np.ndarray,
    redshift: float,
    alpha_inj_signed: float,
    b_field_t: float,
    sed: Dict[str, np.ndarray],
    cfg: Dict[str, Any],
    runner: BratsRunner,
    models: Sequence[str] = ("JP", "KP", "JP_Tribble", "CI", "CI-OFF"),
    do_maps: bool = True,
    do_integrated: bool = True,
    find_injection: bool = False,
    interactive: bool = False,
    timeout: Optional[float] = None,
    common_beam: Optional[Beam] = None,
    beam_area_pixels_value: float = 1.0,
    launch: bool = True,
) -> Dict[str, Any]:
    """
    Run the whole BRATS side of the analysis and return a structured record.

    `interactive=True` prepares the workspace and scripts and then opens BRATS
    in a new terminal window instead of running it, which is the right mode
    when you want to drive the PGPLOT displays yourself.
    """
    alpha_positive = float(-alpha_inj_signed)
    requested = list(models)

    # Every exported filename carries this tag. It is timestamped because
    # BRATS prompts "File already exists, are you sure you want to overwrite?"
    # when an export collides, and that extra prompt would consume the next
    # line of the command file and desynchronise everything after it. A unique
    # tag also means a re-run never destroys the output of a previous fit,
    # which can represent hours of computation.
    tag = f"pysynch_{time.strftime('%Y%m%d_%H%M%S')}"

    unsupported = []
    supported = []
    for name in requested:
        spec = BRATS_MODELS.get(name)
        if spec is None:
            unsupported.append({"model": name, "reason": "not a BRATS model"})
        elif do_maps and not spec.get("maps") and name not in {"CI", "CI-OFF"}:
            unsupported.append({
                "model": name,
                "reason": spec.get("unavailable_reason", "not fittable on maps"),
            })
            if spec.get("integrated"):
                supported.append(name)
        else:
            supported.append(name)

    workspace = prepare_brats_workspace(
        outdir, common_maps, source_mask, redshift,
        background_region=cfg.get("brats_background_region"),
        source_region=cfg.get("brats_source_region"),
        common_beam=common_beam,
        target_name=str(cfg.get("brats_target_name", "PYSYNCH_SOURCE")),
    )

    result: Dict[str, Any] = {
        "engine": "BRATS",
        "executable": runner.executable,
        "runtime": runner.describe(),
        "run_mode": runner.mode,
        "container": runner.container,
        "container_image": runner.image if runner.container else None,
        "version": runner.version,
        "workspace": str(workspace["workspace"]),
        "requested_models": requested,
        "unsupported_models": unsupported,
        "redshift": redshift,
        "injection_index_positive": alpha_positive,
        "b_field_T": float(b_field_t),
        "b_field_uG": float(b_field_t / TESLA_PER_MICROGAUSS),
        "export_tag": tag,
        "pad_x": int(workspace.get("pad_x", 0)),
        "pad_y": int(workspace.get("pad_y", 0)),
        "runs": [],
    }

    scripts: List[Tuple[str, Path, List[Dict[str, Any]]]] = []

    if do_maps:
        map_models = [m for m in supported if BRATS_MODELS[m].get("maps")]
        if map_models and find_injection:
            # A separate, minimal first pass: load, select regions, run
            # findinject, export the chi-square curves, quit. Splitting it
            # out is what makes the measured value available before the
            # models are fitted -- in one script there is no way to read an
            # export back and act on it.
            text, manifest = build_brats_map_script(
                workspace, [], alpha_positive, float(b_field_t),
                sigma=float(cfg.get("source_sigma", 5.0)),
                onsource=float(cfg.get("brats_onsource", 3.0)),
                searcharea=float(cfg.get("brats_searcharea", 1.0)),
                signaltonoise=float(cfg.get("brats_signaltonoise", 1.0)),
                max_myears=float(cfg.get("brats_max_myears", 50.0)),
                ageres=int(cfg.get("brats_ageres", 10)),
                levels=int(cfg.get("brats_levels", 3)),
                gmin=float(cfg.get("gmin", 10.0)),
                gmax=float(cfg.get("gmax", 1.0e6)),
                find_injection=True,
                inject_min=float(cfg.get("brats_inject_min", 0.5)),
                inject_max=float(cfg.get("brats_inject_max", 1.0)),
                inject_intervals=int(cfg.get("brats_inject_intervals", 10)),
                set_noise=bool(cfg.get("brats_set_noise", True)),
                findinject_models=map_models,
                export_fits=False,
                tag=tag,
            )
            path = Path(workspace["commands_dir"]) / "brats_findinject.txt"
            path.write_text(text)
            scripts.append(("findinject", path, manifest))
        if map_models:
            text, manifest = build_brats_map_script(
                workspace, map_models, alpha_positive, float(b_field_t),
                sigma=float(cfg.get("source_sigma", 5.0)),
                onsource=float(cfg.get("brats_onsource", 3.0)),
                searcharea=float(cfg.get("brats_searcharea", 1.0)),
                sed_plot_models=tuple(models),
                signaltonoise=float(cfg.get("brats_signaltonoise", 1.0)),
                max_myears=float(cfg.get("brats_max_myears", 50.0)),
                ageres=int(cfg.get("brats_ageres", 10)),
                levels=int(cfg.get("brats_levels", 3)),
                gmin=float(cfg.get("gmin", 10.0)),
                gmax=float(cfg.get("gmax", 1.0e6)),
                find_injection=False,
                set_noise=bool(cfg.get("brats_set_noise", True)),
                tag=tag,
            )
            path = Path(workspace["commands_dir"]) / "brats_maps.txt"
            path.write_text(text)
            scripts.append(("maps", path, manifest))

    if do_integrated:
        ci_files = None
        if any(m in supported for m in ("CI", "CI-OFF")):
            try:
                ci_files = write_brats_ci_files(
                    workspace, sed["frequency_Hz"], sed["flux_Jy"],
                    sed["flux_err_Jy"], sed["detected"], redshift,
                    float(b_field_t), alpha_positive,
                    identifier=str(cfg.get("brats_target_name", "PYSYNCH_SOURCE")),
                )
            except Exception as exc:                           # noqa: BLE001
                result["ci_file_error"] = str(exc)
        text, manifest = build_brats_integrated_script(
            workspace, supported, sed["frequency_Hz"], sed["flux_Jy"],
            sed["flux_err_Jy"], sed["detected"], redshift, alpha_positive,
            float(b_field_t),
            max_myears=float(cfg.get("brats_max_myears", 50.0)),
            ageres=int(cfg.get("brats_ageres", 10)),
            levels=int(cfg.get("brats_levels", 5)),
            gmin=float(cfg.get("gmin", 10.0)),
            gmax=float(cfg.get("gmax", 1.0e6)),
            ci_files=ci_files,
            tag=tag,
        )
        path = Path(workspace["commands_dir"]) / "brats_integrated.txt"
        path.write_text(text)
        scripts.append(("integrated", path, manifest))
        if ci_files:
            result["ci_files"] = {k: str(v) for k, v in ci_files.items()}

    result["scripts"] = [
        {"stage": stage, "file": str(path)} for stage, path, _ in scripts
    ]

    banner = _write_brats_banner(workspace, scripts, result)
    result["readme"] = str(banner)

    if interactive:
        result["mode"] = "interactive"
        if not launch:
            # The caller will open the session itself once the rest of the
            # pipeline has finished, so preparing the workspace is all that is
            # wanted here. Trying to launch now would only report the same
            # outcome twice.
            result["interactive_launch"] = {
                "launched": False,
                "deferred": True,
                "workspace": str(workspace["workspace"]),
                "manual_command": (
                    f"cd {workspace['workspace']} && "
                    f"{runner.describe() if runner.available else '/path/to/brats'}"
                ),
            }
            return result
        if runner.available:
            result["interactive_launch"] = runner.launch_interactive(
                Path(workspace["workspace"]), banner_file=banner
            )
        else:
            # The workspace is still worth having: it is everything BRATS
            # needs, so the user can install BRATS later and start from here
            # without rerunning any of this pipeline.
            result["interactive_launch"] = {
                "launched": False,
                "workspace": str(workspace["workspace"]),
                "executable": None,
                "manual_command": (
                    f"cd {workspace['workspace']} && /path/to/brats"
                ),
                "error": (
                    "The BRATS executable was not found, so nothing could be "
                    "launched. The workspace, region files and command scripts "
                    "are all prepared: install BRATS, then run it from this "
                    "directory and paste in commands/brats_maps.txt."
                ),
            }
        return result

    result["mode"] = "scripted"

    # -- pass 1: let BRATS measure its own injection index -------------------
    #
    # The injection index handed in comes from the synchrofit ensemble, and
    # when that ensemble is degenerate -- a break railed at the edge of its
    # prior, which is what happens whenever the bands do not bracket the
    # break -- the "injection index" collapses onto the OBSERVED spectral
    # index. That is physically impossible: an aged spectrum can only be
    # steeper than the one injected. Feeding it to BRATS biases every fitted
    # age. BRATS can measure the quantity itself from the resolved data, so
    # where it has, its own value is used for the fits and the supplied one
    # is kept only as a record of what was proposed.
    inject_by_model: Dict[str, float] = {}
    if do_maps and find_injection:
        inject_stage = [s for s in scripts if s[0] == "findinject"]
        for stage, path, manifest in inject_stage:
            run = runner.run_script(
                path, cwd=Path(workspace["workspace"]), timeout=timeout,
                log_path=Path(workspace["workspace"]) / f"brats_{stage}.log",
            )
            run["stage"] = stage
            run["problems"] = diagnose_brats_run(run)
            harvest = harvest_brats_outputs(workspace, manifest)
            run["harvest"] = harvest
            run["injection"] = summarise_brats_injection(
                harvest, beam_area_pixels_value=float(beam_area_pixels_value)
            )
            result["runs"].append(run)
            for model, record in (run["injection"].get("models") or {}).items():
                value = record.get("alpha_inj_positive")
                if value is None or not np.isfinite(value):
                    continue
                if record.get("minimum_is_at_grid_edge"):
                    # A minimum on the edge of the search range is a limit,
                    # not a measurement; adopting it would just move the bias.
                    continue
                inject_by_model[model] = float(value)
        result["injection_index_measured_by_brats"] = dict(inject_by_model)
        result["injection_index_supplied"] = alpha_positive

        # This pass has already been executed, so drop it unconditionally --
        # leaving it in the list when no usable minimum came back would run
        # the whole findinject scan a second time.
        scripts = [s for s in scripts if s[0] != "findinject"]

        if inject_by_model:
            # Rebuild the map script now that the measured values are known.
            map_models = [m for m in supported if BRATS_MODELS[m].get("maps")]
            if map_models:
                text, manifest = build_brats_map_script(
                    workspace, map_models, alpha_positive, float(b_field_t),
                    sigma=float(cfg.get("source_sigma", 5.0)),
                    onsource=float(cfg.get("brats_onsource", 3.0)),
                    searcharea=float(cfg.get("brats_searcharea", 1.0)),
                    sed_plot_models=tuple(models),
                    signaltonoise=float(cfg.get("brats_signaltonoise", 1.0)),
                    max_myears=float(cfg.get("brats_max_myears", 50.0)),
                    ageres=int(cfg.get("brats_ageres", 10)),
                    levels=int(cfg.get("brats_levels", 3)),
                    gmin=float(cfg.get("gmin", 10.0)),
                    gmax=float(cfg.get("gmax", 1.0e6)),
                    find_injection=False,
                    set_noise=bool(cfg.get("brats_set_noise", True)),
                    inject_by_model=inject_by_model,
                    tag=tag,
                )
                path = Path(workspace["commands_dir"]) / "brats_maps.txt"
                path.write_text(text)
                scripts = [("maps", path, manifest)] + [
                    s for s in scripts if s[0] != "maps"
                ]

    for stage, path, manifest in scripts:
        run = runner.run_script(
            path, cwd=Path(workspace["workspace"]), timeout=timeout,
            log_path=Path(workspace["workspace"]) / f"brats_{stage}.log",
        )
        run["stage"] = stage
        run["problems"] = diagnose_brats_run(run)
        harvest = harvest_brats_outputs(workspace, manifest)
        run["harvest"] = harvest
        if stage == "maps" and find_injection:
            run["injection"] = summarise_brats_injection(
                harvest, beam_area_pixels_value=float(beam_area_pixels_value)
            )
        result["runs"].append(run)

    result["n_output_maps"] = sum(
        r["harvest"].get("n_maps", 0) for r in result["runs"]
    )
    result["n_output_tables"] = sum(
        r["harvest"].get("n_tables", 0) for r in result["runs"]
    )
    result["problems"] = [
        dict(problem, stage=r.get("stage"))
        for r in result["runs"] for problem in (r.get("problems") or [])
    ]
    # BRATS exits 0 even when `load` failed, so the exit code alone is not
    # evidence that anything worked.
    result["succeeded"] = bool(
        result["runs"]
        and all(r["return_code"] in (0, None) and not r["timed_out"]
                for r in result["runs"])
        and (result["n_output_maps"] + result["n_output_tables"]) > 0
        and not result["problems"]
    )
    return result


def find_brats_session_backup(workspace: Path) -> Optional[Path]:
    """
    The .brats session file a scripted run left behind, if there is one.

    `fullexport` writes the entire fitted state -- regions, model fits, ages,
    errors -- as `{target}_{timestamp}_{set}_{tag}.brats`.  Handing that path
    to `fullimport` in a fresh session restores everything in seconds, which
    matters because a resolved BRATS fit can represent hours of computation
    and is otherwise lost the moment the scripted process exits.
    """
    candidates = sorted(
        Path(workspace).glob("data/*.brats"),
        key=lambda p: p.stat().st_mtime, reverse=True,
    )
    return candidates[0] if candidates else None


def open_brats_session(
    outdir: Path,
    runner: BratsRunner,
    brats_result: Optional[Dict[str, Any]],
    *,
    common_maps: Sequence[RadioMap],
    source_mask: np.ndarray,
    redshift: float,
    alpha_inj_signed: float,
    b_field_t: float,
    sed: Dict[str, Any],
    cfg: Dict[str, Any],
    models: Sequence[str],
    common_beam: Optional[Beam] = None,
) -> Dict[str, Any]:
    """
    Open an interactive BRATS window on the finished results.

    This runs at the very end, after every product has been written, so the
    pipeline's own output is complete and safe whatever happens to the
    interactive session.  Two situations are handled:

      * BRATS already ran (scripted).  The workspace is reused as-is and the
        session is pointed at the .brats backup that run left, so the fits
        are restored rather than recomputed.
      * BRATS did not run -- not installed, or --engine synchrofit.  A full
        workspace is prepared first (staged maps, regions, command scripts,
        Dockerfile), so the window opens on something immediately usable.

    Failure here is never fatal.  The window is a convenience; the science
    products are already on disk.
    """
    info: Dict[str, Any] = {"requested": True}

    workspace_dir: Optional[Path] = None
    banner: Optional[Path] = None
    if brats_result and brats_result.get("workspace"):
        workspace_dir = Path(brats_result["workspace"])
        readme = brats_result.get("readme")
        banner = Path(readme) if readme and Path(readme).is_file() else None
        info["workspace_source"] = "reused from the BRATS stage"

    if workspace_dir is None or not workspace_dir.is_dir():
        # No BRATS stage ran, so build the workspace now. This is what makes
        # --open-brats useful with --engine synchrofit: the injection index
        # and magnetic field measured by this pipeline are already baked into
        # the command scripts that get written.
        try:
            prepared = run_brats_analysis(
                outdir=outdir,
                common_maps=common_maps,
                source_mask=source_mask,
                redshift=redshift,
                alpha_inj_signed=alpha_inj_signed,
                b_field_t=b_field_t,
                sed=sed,
                cfg=cfg,
                runner=runner,
                models=list(models),
                do_maps=True,
                do_integrated=True,
                find_injection=bool(cfg.get("brats_findinject", False)),
                interactive=True,
                common_beam=common_beam,
            )
        except Exception as exc:                               # noqa: BLE001
            info["error"] = f"{type(exc).__name__}: {exc}"
            return info
        info["workspace_source"] = "prepared now"
        info["launch"] = prepared.get("interactive_launch", {})
        info["workspace"] = prepared.get("workspace")
        info["session_backup"] = None
        return info

    info["workspace"] = str(workspace_dir)
    backup = find_brats_session_backup(workspace_dir)
    info["session_backup"] = str(backup) if backup else None

    if not runner.available:
        info["launch"] = {
            "launched": False,
            "workspace": str(workspace_dir),
            "manual_command": f"cd {workspace_dir} && /path/to/brats",
            "error": (
                "BRATS is not installed here, so no window could be opened. "
                "The workspace is complete and includes a Dockerfile."
            ),
        }
        return info

    info["launch"] = runner.launch_interactive(workspace_dir, banner_file=banner)
    return info


def _report_brats_session(info: Dict[str, Any], runner: BratsRunner) -> None:
    """Tell the user what just opened, and what to type once they are in."""
    if info.get("error"):
        print(f"  Could not open BRATS: {info['error']}")
        return

    launch = info.get("launch") or {}
    workspace = info.get("workspace") or launch.get("workspace")
    if launch.get("launched"):
        print(f"  BRATS opened in a new window via {launch['method']}.")
        print(f"  Working directory: {workspace}")
    else:
        print("  Could not open a terminal window automatically"
              + (f": {launch.get('error')}" if launch.get("error") else "")
              + "\n  Open one yourself and run:\n"
              f"      {launch.get('manual_command')}")

    backup = info.get("session_backup")
    if backup:
        rel = os.path.relpath(backup, workspace) if workspace else backup
        print(
            "\n  The scripted run saved its complete fitted state. Restore it\n"
            "  in the new session instead of refitting -- type:\n"
            "      fullimport\n"
            f"      {rel}\n"
            "  and every region, model fit, age and error comes back."
        )
    else:
        print(
            "\n  Nothing has been fitted in this workspace yet. To reproduce\n"
            "  the scripted analysis, paste in commands/brats_maps.txt, or\n"
            "  work through it a line at a time to watch each step."
        )

    if runner.mode == "container":
        display = runner.x11_forwarding()
        if not display.get("available"):
            print("\n  NOTE: " + " ".join(display.get("notes", [])))
        else:
            for note in display.get("notes", []):
                print(f"\n  NOTE: {note}")

def _write_brats_banner(
    workspace: Dict[str, Any],
    scripts: Sequence[Tuple[str, Path, List[Dict[str, Any]]]],
    context: Dict[str, Any],
) -> Path:
    """Human-readable orientation note dropped into the BRATS workspace."""
    ws = Path(workspace["workspace"])
    rel = lambda p: os.path.relpath(str(p), str(ws))            # noqa: E731

    lines = [
        "=" * 74,
        f" BRATS workspace prepared by PySynch v{__version__}",
        "=" * 74,
        "",
        "Everything BRATS needs is already here. You are in the working",
        "directory, so all paths below are relative and can be pasted as-is.",
        "",
        "  maps/       common-beam FITS, one per frequency, on a shared grid",
        "              (REFFREQ / CTYPE3 / CRVAL3 written for BRATS)",
        "  regions/    source.reg and background.reg in IMAGE coordinates",
        "  images/     BRATS will write maps and plots here",
        "  data/       BRATS will write exported tables here",
        "  commands/   ready-to-run command files",
        "  Dockerfile  a container recipe, if you do not have BRATS built",
        "",
        "IF YOU DO NOT HAVE BRATS YET",
        "-" * 74,
        "BRATS needs PGPLOT and FUNTOOLS. Both are packaged on Linux; neither",
        "has a maintained macOS build, so on a Mac the container is normally",
        "the only practical route. From this directory:",
        "",
        "    docker build -t brats:latest -f Dockerfile .",
        "",
        "then re-run the pipeline with --brats-container docker, or start an",
        "interactive session yourself with:",
        "",
        "    docker run --rm -it -v \"$PWD\":/work -w /work \\",
        "        --entrypoint /soft/brats/brats brats:latest",
        "",
        "The bind mount means BRATS sees this directory as /work, so every",
        "relative path in the command files below resolves unchanged.",
        "",
        "Inputs already measured for you:",
        f"  redshift            {workspace['redshift']:.6f}",
        f"  injection index     {context.get('injection_index_positive', float('nan')):.4f}"
        "   (positive convention, as BRATS expects)",
        f"  magnetic field      {context.get('b_field_T', float('nan')):.4e} T"
        f"  = {context.get('b_field_uG', float('nan')):.3f} uG",
        "",
        "Frequencies staged:",
    ]
    for entry in workspace["staged_maps"]:
        lines.append(
            f"  {entry['label']:<20} {entry['frequency_Hz']/1e9:9.4f} GHz   "
            f"rms {entry['rms_Jy_beam']*1e6:8.2f} uJy/beam"
        )
    lines += [
        "",
        "TO REPRODUCE THE SCRIPTED RUN BY HAND",
        "-" * 74,
        "Paste the contents of a command file into BRATS one block at a time",
        "(BRATS accepts inline input, so a whole file can be pasted at once):",
    ]
    for stage, path, _ in scripts:
        lines.append(f"  {stage:<12} {rel(path)}")
    lines += [
        "",
        "USEFUL COMMANDS ONCE YOU ARE IN",
        "-" * 74,
        "  help                  full command reference",
        "  list                  show loaded datasets",
        "  specindex 0           spectral index map",
        "  fitjpmodel 0          fit JP; then specagemap 0 1",
        "  fitkpmodel 0          fit KP; then specagemap 0 2",
        "  fitjptribble 0        fit Tribble (JP); then specagemap 0 3",
        "  findinject 0 1        minimise the injection index for JP",
        "  export / exportasfits toggle writing maps to images/ as FITS",
        "  exportdata            write numerical tables to data/",
        "  fullexport 0 mybackup save the whole session",
        "",
        "NOTE ON KP-TRIBBLE",
        "-" * 74,
    ]
    lines += textwrap.wrap(
        BRATS_MODELS["KP_Tribble"]["unavailable_reason"], width=72,
        initial_indent="  ", subsequent_indent="  ",
    )
    lines += [
        "  Use the synchrofit engine (--engine synchrofit) if you need it.",
        "",
        "=" * 74,
    ]
    path = ws / "README_BRATS.txt"
    path.write_text("\n".join(lines) + "\n")
    return path


# ---------------------------------------------------------------------------
# Turning a BRATS run into finished science products
# ---------------------------------------------------------------------------

#: BRATS table key -> (output stem, colour map, colour-bar label, FITS BUNIT)
BRATS_MAP_PRODUCTS: Dict[str, Tuple[str, str, str, str]] = {
    "ages": ("age", "plasma", r"Spectral age $\tau$ (Myr)", "Myr"),
    "errors": ("age_error", "viridis",
               r"$\sigma_\tau^{+}$ (Myr)", "Myr"),
    "chi2": ("chi2", "cividis", r"$\chi^2$", "chi2"),
    "normalisation": ("normalisation", "magma", "Normalisation", "arbitrary"),
}


def brats_table_full(table: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """
    Every column of a harvested table, re-reading the file when needed.

    The harvest keeps columns inline only up to a size cap, so that a JSON
    summary of the run stays readable.  That cap is easy to exceed here: one
    BRATS region per pixel over a source a few hundred pixels across is
    already hundreds of thousands of rows, and the REGARRAY export has one
    line per pixel by construction.  Reading the file again is cheap and is
    the only way to be sure a column is complete -- silently painting a
    truncated or absent column would produce a map that looks finished and
    is wrong.
    """
    columns = table.get("table") or {}
    complete = all(v is not None for v in columns.values())
    if columns and complete:
        return {k: np.asarray(v, dtype=float) for k, v in columns.items()}
    path = table.get("file")
    if not path or not Path(path).is_file():
        return {k: np.asarray(v, dtype=float)
                for k, v in columns.items() if v is not None}
    parsed = parse_brats_table(Path(path), table.get("token"))
    return {k: v for k, v in parsed.items() if not k.startswith("_")}


def _brats_region_array(harvest: Dict[str, Any]) -> Optional[Dict[str, np.ndarray]]:
    """Pull the REGARRAY export out of a harvest, as numpy columns."""
    for key, table in (harvest.get("tables") or {}).items():
        if table.get("token") != "REGARRAY":
            continue
        columns = brats_table_full(table)
        if not all(
            c in columns and np.asarray(columns[c]).size
            for c in ("x", "y", "region_id")
        ):
            continue
        return {c: np.asarray(columns[c], dtype=float)
                for c in ("x", "y", "region_id")}
    return None


def render_brats_products(
    outdir: Path,
    brats_result: Dict[str, Any],
    common_maps: Sequence[RadioMap],
    source_mask: np.ndarray,
    target_wcs: WCS,
    common_beam: Optional[Beam],
    reference_map: Optional[RadioMap] = None,
    contour_start_sigma: float = 5.0,
    contour_factor: float = 2.0,
    contour_max_levels: int = 5,
    contour_drop_lowest: int = 0,
    contour_levels_sigma: Optional[Sequence[float]] = None,
    contour_mask: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Convert a finished BRATS run into FITS maps and publication figures.

    Everything BRATS measured per region is painted back onto the pipeline's
    own pixel grid using the REGARRAY export, written out as FITS with the
    real WCS attached, and rendered with total-intensity contours from the
    reference band drawn over it.

    The products are written into a `brats_maps/` subdirectory rather than
    mixed in with the synchrofit ones.  Keeping them separate is deliberate:
    when both engines run, two files called `spectral_index.fits` that came
    from different codes would be an easy and serious mistake to make, and
    the whole point of running both is to be able to compare them.
    """
    products: Dict[str, Any] = {
        "written": [], "skipped": [], "figures": [], "diagnostics": {},
    }
    dest = Path(outdir) / "brats_maps"

    shape = tuple(np.asarray(source_mask).shape)
    base_header = common_maps[0].header if common_maps else None
    pad_x = int(brats_result.get("pad_x", 0))
    pad_y = int(brats_result.get("pad_y", 0))
    products["pad_x"], products["pad_y"] = pad_x, pad_y
    if contour_mask is None and source_mask is not None and reference_map is not None:
        _lv = reference_contour_levels(
            reference_map.data, float(reference_map.rms_jybeam),
            start_sigma=float(contour_start_sigma),
            factor=float(contour_factor),
            max_levels=int(contour_max_levels),
            drop_lowest=int(contour_drop_lowest),
            explicit_sigma=contour_levels_sigma,
        )
        if _lv:
            contour_mask = source_contour_mask(
                reference_map.data,
                # the detection isophote, not the lowest drawn level
                float(contour_start_sigma) * float(reference_map.rms_jybeam),
                source_mask,
                # HALF a beam, matching the derived-map path. A full beam
                # reaches far enough past the segmentation to pick up
                # neighbouring emission and draw it as open arcs in blank
                # sky; the two paths must agree or the BRATS figures carry
                # stray fragments the PySynch figures do not.
                grow_pixels=int(max(2, round(0.5 * math.sqrt(max(
                    beam_area_pixels(common_beam, target_wcs), 1.0
                ))))) if common_beam is not None else 2,
            )

    # -- reference contours ---------------------------------------------------
    ref_image = ref_levels = None
    ref_label = None
    if reference_map is not None:
        ref_image = np.asarray(reference_map.data, dtype=float)
        # The ladder MUST be built with the same factor and level cap the
        # PySynch figures used. Falling back to this helper's own defaults
        # gave the BRATS maps 9 contours where every other figure in the
        # same run had 6, so the two engines' maps could not be compared
        # by eye -- which is the entire point of producing both.
        ref_levels = reference_contour_levels(
            ref_image, float(reference_map.rms_jybeam),
            start_sigma=float(contour_start_sigma),
            factor=float(contour_factor),
            max_levels=int(contour_max_levels),
            drop_lowest=int(contour_drop_lowest),
            explicit_sigma=contour_levels_sigma,
        )
        # Deliberately unlabelled: the contour provenance belongs in the
        # figure caption, not stamped over the map. It is recorded in the
        # products dictionary and the run report instead.
        ref_label = None
        products["reference_contours"] = {
            "label": reference_map.label,
            "frequency_Hz": float(reference_map.freq_hz),
            "rms_Jy_beam": float(reference_map.rms_jybeam),
            "levels_Jy_beam": [float(l) for l in ref_levels],
            "start_sigma": float(contour_start_sigma),
            "drop_lowest": int(contour_drop_lowest),
            "lowest_drawn_sigma": (
                float(ref_levels[0]) / float(reference_map.rms_jybeam)
                if ref_levels and reference_map.rms_jybeam else None
            ),
        }
        if not ref_levels:
            products["skipped"].append({
                "product": "reference_contours",
                "reason": (
                    "The reference image peak does not reach "
                    f"{contour_start_sigma:g} sigma, so no contour ladder "
                    "could be built."
                ),
            })

    def _emit(stem: str, array: np.ndarray, cmap: str, label: str,
              bunit: str, model: str = "", **kwargs: Any) -> None:
        dest.mkdir(parents=True, exist_ok=True)
        if base_header is not None:
            # Provenance in the header, so a file that gets separated from
            # this run can still say which engine, model, field and injection
            # index produced it. Ages in particular are meaningless without
            # the assumed B.
            extra: Dict[str, Any] = {
                "ENGINE": ("BRATS", "fitting engine"),
                "BRATSVER": (
                    str(brats_result.get("version") or "unknown"),
                    "BRATS version",
                ),
                "BFIELD_T": (
                    float(brats_result.get("b_field_T", float("nan"))),
                    "assumed B field (T)",
                ),
                "ALPHAINJ": (
                    float(brats_result.get(
                        "injection_index_positive", float("nan")
                    )),
                    "injection index (positive)",
                ),
                "REDSHIFT": (
                    float(brats_result.get("redshift", float("nan"))),
                    "redshift used",
                ),
            }
            if model:
                extra["AGEMODEL"] = (str(model), "spectral ageing model")
            save_fits(
                dest / f"{stem}.fits", np.asarray(array, dtype=np.float32),
                product_header(base_header, bunit, common_beam, extra=extra),
            )
            products["written"].append(str(dest / f"{stem}.fits"))
        save_png_map(
            dest / f"{stem}.png", array, label, cmap=cmap, wcs=target_wcs,
            reference_image=ref_image, reference_levels=ref_levels,
            reference_label=ref_label, reference_mask=contour_mask,
            # Black, heavier than the default. This was red for a while, on
            # the argument that a sequential age ramp runs through dark blues
            # at the young end and black would vanish into them. Measured on
            # this source, that argument is backwards: sampling the colormap
            # along the drawn contours gives black a median contrast ratio of
            # 5.5:1 against 1.9:1 for red, and only 27% of the contour length
            # falls below 3:1 where red left 92% below it. Red's luminance
            # sits in the MIDDLE of plasma/viridis, so it collides with the
            # mid-tones that cover most of a source; black only loses the
            # dark young end. Keep the weight -- it is what carries the line
            # across that end.
            reference_color="black", reference_linewidth=1.9,
            beam=common_beam, **kwargs,
        )
        products["figures"].append(str(dest / f"{stem}.png"))

    for run in brats_result.get("runs", []):
        harvest = run.get("harvest") or {}
        region_array = _brats_region_array(harvest)
        if region_array is None:
            # Only the map-based stage has regions at all. The integrated
            # stage fits the SED from text files and legitimately exports no
            # region array, so warning about it there is noise that makes a
            # perfectly good run look broken.
            if run.get("stage") == "maps" and harvest.get("tables"):
                products["skipped"].append({
                    "stage": run.get("stage"),
                    "reason": (
                        "No REGARRAY export was found, so per-region values "
                        "cannot be placed on the sky. This is what BRATS "
                        "writes when `setregions` produced no regions -- "
                        "usually the sigma threshold is too high for the "
                        "data, or the source region does not overlap the "
                        "emission."
                    ),
                })
            continue

        for key, table in (harvest.get("tables") or {}).items():
            token = table.get("token")
            if token in {"REGARRAY", "FLUX"}:
                continue
            columns = brats_table_full(table)
            if not columns:
                continue

            # -- resolved spectral index (and its error) --------------------
            if token == "SPECIND":
                for column, stem, cmap, label, bunit, extra in (
                    ("alpha_positive", "brats_spectral_index", "coolwarm",
                     r"$\alpha$   ($S_\nu \propto \nu^{-\alpha}$, positive)",
                     "spectral_index", {}),
                    ("alpha_err", "brats_spectral_index_error", "viridis",
                     r"$\sigma_\alpha$", "spectral_index_error", {}),
                ):
                    if column not in columns or not np.asarray(
                        columns[column]
                    ).size:
                        continue
                    try:
                        painted, diag = paint_brats_region_map(
                            region_array,
                            np.asarray(columns[column], dtype=float),
                            shape, source_mask, pad_x=pad_x, pad_y=pad_y,
                        )
                    except ValueError as exc:
                        products["skipped"].append(
                            {"product": stem, "reason": str(exc)}
                        )
                        continue
                    products["diagnostics"][stem] = diag
                    _emit(stem, painted, cmap, label, bunit, **extra)
                continue

            # -- per-model ageing products -----------------------------------
            quantity = table.get("quantity") or ""
            model = table.get("model") or ""
            if not quantity and "_" in key:
                parts = key.split("_")
                model = parts[1] if len(parts) > 1 else ""
                quantity = parts[-1]
            spec = BRATS_MAP_PRODUCTS.get(quantity)
            if spec is None:
                continue
            stem_base, cmap, label, bunit = spec

            # The error export carries both wings; write each, because a
            # strongly asymmetric age error is itself the diagnostic that the
            # fit is running into the age grid boundary.
            wanted = [(c, "") for c in columns if not c.startswith("_")]
            for column, _suffix in wanted:
                values = np.asarray(columns[column], dtype=float)
                if values.size == 0:
                    continue
                suffix = ""
                this_label = label
                if column.endswith("_plus_Myr"):
                    suffix, this_label = "_plus", r"$\sigma_\tau^{+}$ (Myr)"
                elif column.endswith("_minus_Myr"):
                    suffix, this_label = "_minus", r"$\sigma_\tau^{-}$ (Myr)"
                stem = f"brats_{model}_{stem_base}{suffix}" if model else \
                       f"brats_{stem_base}{suffix}"
                try:
                    painted, diag = paint_brats_region_map(
                        region_array, values, shape, source_mask,
                        pad_x=pad_x, pad_y=pad_y,
                    )
                except ValueError as exc:
                    products["skipped"].append(
                        {"product": stem, "reason": str(exc)}
                    )
                    continue
                products["diagnostics"][stem] = diag
                _emit(stem, painted, cmap, this_label, bunit, model=model)

    # -- BRATS' own FITS maps, with a usable WCS attached ---------------------
    #
    # BRATS writes these with a minimal header. Re-attaching the WCS from the
    # common-beam grid makes them overlayable; it is legitimate only because
    # BRATS was handed exactly that grid, so the pixel arrays correspond
    # one-to-one. The shape is checked before anything is copied, and a map
    # that does not match is reported rather than silently re-gridded.
    native: List[Dict[str, Any]] = []
    for run in brats_result.get("runs", []):
        for entry in (run.get("harvest") or {}).get("maps", []):
            path = Path(entry["file"])
            try:
                with fits.open(path) as hdul:
                    array = np.squeeze(np.asarray(hdul[0].data, dtype=float))
            except Exception as exc:                           # noqa: BLE001
                native.append({"file": str(path), "error": str(exc)})
                continue
            # BRATS worked on the square-padded maps, so its own FITS output
            # is that size. Crop the padding back off so the array lines up
            # with the pipeline grid the WCS describes.
            if array.ndim == 2 and array.shape != shape:
                side = max(shape)
                if array.shape == (side, side):
                    array = array[pad_y:pad_y + shape[0],
                                  pad_x:pad_x + shape[1]]
            if array.ndim != 2 or array.shape != shape:
                native.append({
                    "file": str(path), "shape": list(np.shape(array)),
                    "skipped": (
                        f"shape {np.shape(array)} does not match the "
                        f"common-beam grid {shape} even after removing the "
                        f"BRATS square padding ({pad_x}, {pad_y}), so the "
                        "pipeline WCS cannot be attached to it"
                    ),
                })
                continue
            if base_header is not None:
                dest.mkdir(parents=True, exist_ok=True)
                out = dest / f"native_{path.stem}.fits"
                save_fits(
                    out, array.astype(np.float32),
                    product_header(base_header, "BRATS", common_beam),
                )
                native.append({"file": str(path), "rewritten": str(out)})
                products["written"].append(str(out))
    products["native_maps"] = native

    # -- BRATS' own PGPLOT renderings ---------------------------------------
    # Copied next to the pipeline's figures so both views of the same fit sit
    # together; BRATS draws its own region boundaries and scaling, which is a
    # useful independent check on the painted maps.
    pgplot: List[str] = []
    # De-duplicated by DESTINATION. `harvest_brats_outputs` re-globs the whole
    # images/ directory once per run stage, so a file written during the
    # findinject pass is listed again by the maps pass and the raw count came
    # out at twice the number of files that actually exist. The copy itself
    # was always idempotent -- same destination name, overwritten -- so only
    # the reported count was ever wrong, but it was wrong by 2x.
    seen: Set[str] = set()
    for run in brats_result.get("runs", []):
        for entry in (run.get("harvest") or {}).get("png_maps", []):
            src_path = Path(entry["file"])
            if not src_path.is_file():
                continue
            out = dest / f"pgplot_{src_path.name}"
            if str(out) in seen:
                continue
            dest.mkdir(parents=True, exist_ok=True)
            try:
                shutil.copyfile(src_path, out)
                seen.add(str(out))
                pgplot.append(str(out))
            except Exception:                                  # noqa: BLE001
                continue
    products["pgplot_figures"] = pgplot
    products["n_pgplot_figures"] = len(pgplot)

    products["n_written"] = len(products["written"])
    products["n_figures"] = len(products["figures"])
    return products

def compare_engines(
    synchrofit_result: Optional[Dict[str, Any]],
    brats_result: Optional[Dict[str, Any]],
    beam_area_pixels_value: float = 1.0,
) -> Dict[str, Any]:
    """
    Cross-compare the two independent engines.

    This is the payoff for running both.  Agreement between synchrofit and
    BRATS on a quantity means it is a property of the data; disagreement
    larger than the quoted errors means it is a property of the fitting code,
    and should be reported as a systematic rather than hidden by picking a
    favourite.
    """
    comparison: Dict[str, Any] = {
        "synchrofit_available": synchrofit_result is not None,
        "brats_available": brats_result is not None,
        "quantities": {},
        "interpretation": (
            "synchrofit and BRATS share no source code and integrate the "
            "synchrotron kernel differently. Treat any difference that "
            "exceeds the individual statistical errors as a model-"
            "implementation systematic on the final quoted value."
        ),
    }
    if not (synchrofit_result and brats_result):
        comparison["status"] = "only_one_engine_ran"
        return comparison

    sf_alpha = (
        (synchrofit_result.get("robust_injection_index") or {})
        .get("alpha_inj_positive")
    )
    brats_inject = None
    for run in brats_result.get("runs", []):
        if run.get("injection", {}).get("models"):
            brats_inject = run["injection"]["models"]
            break

    # alpha_inj_positive = -alpha_inj_radio, so the error magnitude carries
    # over unchanged. The total is the statistical error and the between-model
    # systematic already combined in quadrature.
    _robust = synchrofit_result.get("robust_injection_index") or {}
    sf_err = _robust.get("alpha_inj_radio_err")
    sf_stat = _robust.get("alpha_inj_stat_err")
    sf_sys = _robust.get("alpha_inj_model_systematic")

    if sf_alpha is not None and brats_inject:
        entry: Dict[str, Any] = {
            "synchrofit": float(sf_alpha),
            "synchrofit_error": float(sf_err) if sf_err else None,
            "synchrofit_error_statistical": float(sf_stat) if sf_stat else None,
            "synchrofit_error_model_systematic": (
                float(sf_sys) if sf_sys else None
            ),
            "brats_models": brats_inject,
        }
        # Turn the per-model BRATS values into an actual head-to-head number.
        # A comparison that stops at "here are both results" leaves the only
        # interesting question -- whether they agree -- to be eyeballed.
        usable = {
            name: rec for name, rec in brats_inject.items()
            if rec.get("alpha_inj_positive") is not None
            and not rec.get("minimum_is_at_grid_edge")
        }
        for name, rec in usable.items():
            b_alpha = float(rec["alpha_inj_positive"])
            b_err = rec.get("alpha_inj_error")
            diff = b_alpha - float(sf_alpha)
            # Combine in quadrature: the two engines are independent, so the
            # tension is the difference over the joint uncertainty. Where an
            # error is missing the difference is still reported, without a
            # significance, rather than inventing one.
            joint = None
            if b_err and sf_err:
                joint = math.sqrt(float(b_err) ** 2 + float(sf_err) ** 2)
            entry.setdefault("per_model", {})[name] = {
                "brats": b_alpha,
                "brats_error": float(b_err) if b_err else None,
                "difference_brats_minus_synchrofit": float(diff),
                "joint_error": joint,
                "tension_sigma": (
                    float(abs(diff) / joint) if joint and joint > 0 else None
                ),
                "grid_resolution_limited": bool(
                    rec.get("grid_resolution_limited")
                ),
            }
        if usable:
            values = [
                float(r["alpha_inj_positive"]) for r in usable.values()
            ]
            entry["brats_mean"] = float(np.mean(values))
            entry["brats_model_scatter"] = (
                float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            )
            tensions = [
                v["tension_sigma"] for v in entry.get("per_model", {}).values()
                if v.get("tension_sigma") is not None
            ]
            if tensions:
                worst = max(tensions)
                entry["max_tension_sigma"] = float(worst)
                entry["verdict"] = (
                    "consistent" if worst < 2.0
                    else ("marginal" if worst < 3.0 else "discrepant")
                )
                entry["interpretation"] = (
                    "The two engines agree within their joint uncertainty; "
                    "the injection index is a property of the data."
                    if worst < 2.0 else
                    "The engines differ by more than their joint uncertainty. "
                    "Quote the spread between them as a model-implementation "
                    "systematic rather than picking one."
                )
        else:
            entry["verdict"] = "brats_minimum_unbracketed"
            entry["interpretation"] = (
                "Every BRATS injection-index minimum sits on the edge of its "
                "search range, so BRATS has produced limits rather than "
                "measurements. Widen --brats-inject-min / --brats-inject-max "
                "before comparing."
            )
        entry["note"] = (
            "The BRATS value is the minimum of its summed chi-square curve, "
            "refined parabolically, with the summed chi-square divided by the "
            f"beam area ({beam_area_pixels_value:.1f} pixels) before the "
            "Delta chi-square = 1 interval is read off, because regions "
            "within one beam are not independent."
        )
        comparison["quantities"]["injection_index_positive"] = entry

    # -- a known defect in the BRATS spectral-index errors -------------------
    #
    # BRATS fits ln S against ln nu, but builds the weight for each band as
    #
    #     w = 1 / (ln(S) * sigma / S)^2          (calcspecindex.h:223)
    #
    # where correct propagation of sigma through the logarithm gives
    #
    #     w = 1 / (sigma / S)^2
    #
    # The stray factor of ln(S) is not dimensionless and does not cancel: for
    # fluxes in Jy it is of order |ln S| ~ 7-9 across these bands, so every
    # per-pixel sigma_alpha BRATS reports is inflated by roughly that factor.
    # Checked numerically on this source: correct propagation gives 0.127,
    # the BRATS weighting gives 0.803, BRATS itself reports 0.670, and this
    # pipeline's own fit gives 0.135 -- i.e. PySynch agrees with correct
    # propagation to 6 per cent while BRATS is high by a factor of ~5.
    #
    # It affects the ERRORS only. The spectral index VALUES are unaffected,
    # because a weight that is the same multiplicative factor for every band
    # cancels out of a weighted mean -- and the two engines' alpha maps do
    # agree. So BRATS' alpha is used as an independent check as intended, and
    # only its sigma_alpha is flagged as not directly comparable.
    comparison["caveats"] = [
        {
            "quantity": "brats_spectral_index_error",
            "severity": "do_not_compare_directly",
            "summary": (
                "BRATS inflates sigma_alpha by a factor of order |ln S| "
                "(~7-9 for fluxes in Jy) because calcspecindex.h weights the "
                "log-space fit as 1/(lnS*sigma/S)^2 instead of 1/(sigma/S)^2. "
                "Spectral index VALUES are unaffected -- a common factor "
                "cancels from the weighted mean -- but the quoted errors are "
                "not on the same footing as this pipeline's and a tension "
                "computed against them will be spuriously small."
            ),
            "source": "BRATS 2.6.3 calcspecindex.h:223",
        }
    ]
    comparison["status"] = (
        "compared" if comparison["quantities"] else "no_common_quantities"
    )
    return comparison


def _format_brats_summary(result: Dict[str, Any]) -> str:
    """Human-readable account of what the BRATS stage did and produced."""
    lines = [
        f"BRATS STAGE SUMMARY  (PySynch v{__version__})",
        "=" * 74,
        f"Generated  : {time.strftime('%Y-%m-%d %H:%M:%S %Z')}",
        f"Executable : {result.get('executable') or 'NOT FOUND'}",
        f"Version    : {result.get('version') or 'unknown'}",
        f"Mode       : {result.get('mode', 'n/a')}",
        f"Workspace  : {result.get('workspace')}",
        "",
        "INPUTS HANDED TO BRATS",
        "-" * 74,
        f"  redshift            {result.get('redshift')}",
        f"  injection index     {result.get('injection_index_positive')}"
        "   (positive convention)",
        f"  magnetic field      {result.get('b_field_T')} T"
        f"  ({result.get('b_field_uG')} uG)",
        f"  field provenance    {result.get('b_field_source', 'n/a')}",
        "",
        "MODELS",
        "-" * 74,
        f"  requested : {', '.join(result.get('requested_models', []))}",
    ]
    for entry in result.get("unsupported_models", []):
        lines.append(f"  UNAVAILABLE: {entry['model']}")
        lines += textwrap.wrap(
            entry["reason"], width=70, initial_indent="      ",
            subsequent_indent="      ",
        )

    lines += ["", "COMMAND FILES", "-" * 74]
    for entry in result.get("scripts", []):
        lines.append(f"  {entry['stage']:<12} {entry['file']}")
    lines += [
        "",
        "These are plain text, one BRATS input per line. They can be pasted",
        "straight into an interactive BRATS session, which is the recommended",
        "way to check a scripted run or to carry on by hand.",
        "",
    ]

    if result.get("mode") == "interactive":
        launch = result.get("interactive_launch", {})
        lines += [
            "INTERACTIVE LAUNCH",
            "-" * 74,
            f"  launched : {launch.get('launched')}",
            f"  method   : {launch.get('method', 'n/a')}",
        ]
        if launch.get("error"):
            lines.append(f"  error    : {launch['error']}")
        lines += [
            "  run by hand with:",
            f"      {launch.get('manual_command')}",
            "",
        ]
    else:
        lines += ["RUNS", "-" * 74]
        for run in result.get("runs", []):
            harvest = run.get("harvest", {})
            lines += [
                f"  stage {run.get('stage')}: return code "
                f"{run.get('return_code')}, "
                f"{run.get('elapsed_seconds', 0):.1f} s"
                + (", TIMED OUT" if run.get("timed_out") else ""),
                f"    FITS maps produced : {harvest.get('n_maps', 0)}",
                f"    data tables read   : {harvest.get('n_tables', 0)}",
            ]
            for key, table in sorted((harvest.get("tables") or {}).items()):
                if "error" in table:
                    lines.append(f"      {key:<34} ERROR: {table['error']}")
                else:
                    lines.append(
                        f"      {key:<34} n={table['n_values']:<7} "
                        f"median={table['median']}"
                    )
            for miss in harvest.get("missing", []):
                lines.append(
                    f"      MISSING: {miss.get('step')} "
                    f"{miss.get('model') or ''} {miss.get('quantity') or ''} "
                    f"(token {miss.get('token')})"
                )

            for problem in (run.get("problems") or []):
                lines.append(f"      ERROR: {problem['message']}")
                lines += textwrap.wrap(
                    problem["meaning"], width=66,
                    initial_indent="             ",
                    subsequent_indent="             ",
                )

            injection = (run.get("injection") or {}).get("models") or {}
            if injection:
                lines += [
                    "",
                    "    INJECTION INDEX (chi-square minimisation over all "
                    "regions)",
                ]
                for model, rec in sorted(injection.items()):
                    value = rec.get("alpha_inj_positive")
                    err = rec.get("alpha_inj_error")
                    quoted = (
                        f"{value:.4f}" if value is not None else "n/a"
                    ) + (f" +/- {err:.4f}" if err else "")
                    lines.append(
                        f"      {model:<14} alpha_inj = {quoted}   "
                        f"[{rec.get('status')}]"
                    )
                    if rec.get("minimum_is_at_grid_edge"):
                        lines.append(
                            "        LIMIT ONLY: the minimum is on the edge "
                            "of the search range. Widen "
                            "--brats-inject-min / --brats-inject-max."
                        )
                    elif rec.get("grid_resolution_limited"):
                        lines.append(
                            "        The error comes from the fitted "
                            "curvature and is below half the grid spacing; "
                            "increase --brats-inject-intervals to measure it."
                        )
            lines.append("")

    rendered = result.get("rendered_products") or {}
    if rendered and not rendered.get("error"):
        lines += ["MAPS AND FIGURES BUILT FROM THE BRATS RESULTS", "-" * 74]
        reference = rendered.get("reference_contours")
        if reference:
            levels = reference.get("levels_Jy_beam") or []
            lines.append(
                f"  contours : {reference['label']} at "
                f"{reference['frequency_Hz']/1e9:.3f} GHz, "
                f"{len(levels)} levels from "
                f"{(reference.get('lowest_drawn_sigma') or reference['start_sigma']):g}"
                " sigma"
                + (f" ({levels[0]*1e3:.4f} mJy/beam)" if levels else "")
                + (
                    f", ladder anchored at {reference['start_sigma']:g} sigma "
                    f"with the lowest {reference['drop_lowest']} level(s) "
                    "not drawn"
                    if reference.get("drop_lowest") else ""
                )
            )
        lines.append(
            f"  written  : {rendered.get('n_written', 0)} FITS, "
            f"{rendered.get('n_figures', 0)} figures"
        )
        for name, diag in sorted((rendered.get("diagnostics") or {}).items()):
            frac = diag.get("fraction_inside_source_mask")
            lines.append(
                f"    {name:<38} {diag.get('n_pixels_painted', 0):>7} px"
                + (f", {frac:.0%} on source" if frac is not None else "")
            )
        for entry in rendered.get("skipped", []):
            lines += textwrap.wrap(
                "SKIPPED: " + str(entry.get("reason")), width=70,
                initial_indent="    ", subsequent_indent="      ",
            )
        lines += [
            "",
            "  These were rebuilt from the REGARRAY export, so they sit on",
            "  exactly the pipeline's pixel grid and carry its WCS. They are",
            "  kept in brats_maps/ rather than beside the synchrofit products",
            "  so the two engines' maps can never be confused with one",
            "  another.",
            "",
        ]
    elif rendered.get("error"):
        lines += [
            "MAPS AND FIGURES BUILT FROM THE BRATS RESULTS",
            "-" * 74,
            f"  FAILED: {rendered['error']}",
            "",
        ]

    lines += [
        "INTERPRETATION",
        "-" * 74,
    ]
    for note in (
        "BRATS ages are conditional on the magnetic field it was given. The "
        "field above came from the PySynch equipartition solution, so a BRATS "
        "age and a synchrofit age computed at the same field are directly "
        "comparable; ages from a different assumed field are not.",
        "BRATS was handed the same per-map RMS and flux-calibration errors "
        "the rest of the pipeline used (unless --brats-keep-own-noise was "
        "set), so a difference between the two engines reflects the fitting "
        "code and not a difference in the assumed uncertainties.",
        "BRATS' findinject sums chi-square over all regions, which "
        "over-weights by roughly the beam area because single-pixel regions "
        "are not independent. Divide by the beam area before reading a "
        "Delta chi-square = 1 interval off the exported curve.",
        "The source and background regions were placed automatically. Check "
        "them in DS9 before trusting the numbers: the background box sets "
        "the detection threshold and every error bar BRATS reports.",
    ):
        lines += textwrap.wrap(
            note, width=72, initial_indent="  * ", subsequent_indent="    "
        )
    return "\n".join(lines) + "\n"


# ===========================================================================
# PART 16.  PUBLICATION PRODUCTS
# ===========================================================================

PLOT_DPI = 350

#: Legend entry for the data points. Naming what the bar actually contains
#: matters: `integrated_flux_for_mask` adds THREE terms in quadrature --
#: thermal noise over the aperture, the zero-level/background uncertainty,
#: and the per-band flux-scale (calibration) error -- and a reader who
#: assumes the bar is thermal-only will judge the fit far too harshly, since
#: the calibration term dominates in every band here.
MEASURED_LABEL = (
    r"Measured ($1\sigma$: thermal $\oplus$ zero-level $\oplus$ scale)"
)

#: Figure style. The target is a journal page (A&A / MNRAS / ApJ), so the
#: text is serif and the maths is set in STIX, which is metrically close to
#: Times and matches what LaTeX puts in the surrounding caption and body --
#: a sans-serif axis label next to a serif caption is the usual giveaway
#: that a figure was not prepared for the paper it appears in.
#:
#: STIXGeneral ships with matplotlib, so this needs no system font and
#: renders identically on any machine that runs the pipeline. DejaVu Serif
#: and Times New Roman follow only as fallbacks.
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["STIXGeneral", "DejaVu Serif", "Times New Roman"],
    "mathtext.fontset": "stix",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "xtick.labelsize": 10.5,
    "ytick.labelsize": 10.5,
    "legend.fontsize": 9.5,
    "axes.linewidth": 1.0,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
    # Ticks long enough to read after the figure is scaled into a column.
    "xtick.major.size": 5.5, "ytick.major.size": 5.5,
    "xtick.minor.size": 3.0, "ytick.minor.size": 3.0,
    "xtick.major.width": 1.0, "ytick.major.width": 1.0,
    "legend.frameon": False,
    "savefig.facecolor": "white",
    "savefig.bbox": "tight",
})


def json_safe_copy(obj: Any) -> Any:
    """Recursively convert to JSON-serialisable types, dropping private keys."""
    if isinstance(obj, dict):
        return {
            k: json_safe_copy(v) for k, v in obj.items()
            if not str(k).startswith("_")
        }
    if isinstance(obj, (list, tuple)):
        return [json_safe_copy(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return json_safe_copy(obj.tolist())
    if isinstance(obj, (np.floating, np.integer, np.bool_)):
        obj = obj.item()
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    if isinstance(obj, Path):
        return str(obj)
    return obj


def _plain_number(value: float) -> str:
    """
    Format one tick value the way a reader expects to see it.

    Matplotlib labels the minor ticks of a log axis through
    LogFormatterSciNotation, which renders 20 as "2 x 10^1" and 300 as
    "3 x 10^2". That is correct but hard to read on an axis whose values are
    ordinary numbers, and it invites the eye to misread the exponent.

    Plain decimals are used across the range where they stay short. Outside
    it -- volume emissivity sits near 1e-41 -- plain notation would be
    unreadable, so a proper power of ten is used instead.
    """
    if not np.isfinite(value) or value == 0:
        return ""
    magnitude = abs(value)
    if 1e-3 <= magnitude < 1e5:
        if magnitude >= 1 and float(value).is_integer():
            return f"{int(round(value)):d}"
        text = f"{value:.10g}"
        return text
    exponent = int(math.floor(math.log10(magnitude)))
    mantissa = value / (10.0 ** exponent)
    if abs(mantissa - round(mantissa)) < 1e-6:
        mantissa_text = f"{int(round(mantissa)):d}"
        if mantissa_text == "1":
            return rf"$10^{{{exponent}}}$"
        return rf"${mantissa_text}\times10^{{{exponent}}}$"
    return rf"${mantissa:.1f}\times10^{{{exponent}}}$"


def axis_pow10_factor(values: np.ndarray) -> Tuple[float, str]:
    """
    Pull a common power of ten out of an axis into its label.

    Quantities like L_nu (~1e24 W/Hz) and the volume emissivity (~1e-41
    W m^-3 Hz^-1) cannot be written as plain digits on a tick. Rather than
    label every tick "4 x 10^-41", the shared exponent is stated once in the
    axis label and the ticks carry ordinary numbers.

    Returns (divisor, latex_suffix); the suffix is empty when the values are
    already in a comfortable range and no factoring is needed.
    """
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite) & (finite != 0)]
    if finite.size == 0:
        return 1.0, ""
    exponent = int(math.floor(math.log10(np.nanmedian(np.abs(finite)))))
    if -3 <= exponent < 5:
        return 1.0, ""
    return 10.0 ** exponent, rf"$10^{{{exponent}}}\,$"

def style_log_axis(ax, which: str = "both") -> None:
    """
    Put readable numbers on the log axes of a figure.

    Both the major and the minor formatter have to be set: on a log axis
    spanning less than one decade -- which every one of these SED panels
    does -- every visible label is a MINOR tick, so setting only the major
    formatter changes nothing at all.
    """
    from matplotlib.ticker import FuncFormatter, LogLocator

    formatter = FuncFormatter(lambda v, _pos: _plain_number(v))
    axes = []
    if which in ("x", "both"):
        axes.append(ax.xaxis)
    if which in ("y", "both"):
        axes.append(ax.yaxis)
    for axis in axes:
        if axis.get_scale() != "log":
            continue
        axis.set_major_formatter(formatter)

        # How many labels a decade can carry depends on how many decades are
        # on show. These SED panels span well under one decade, so a sparse
        # set leaves the ends of the axis bare -- the 650 MHz point had no
        # tick at all. A wide axis with the same dense set would collide.
        # get_xlim/get_ylim force an autoscale pass; get_view_interval can
        # still hold the default (0, 1) if nothing has triggered one yet,
        # which would pick the tick density from a fictitious range.
        lo, hi = (
            ax.get_xlim() if axis is ax.xaxis else ax.get_ylim()
        )
        try:
            decades = abs(math.log10(max(hi, 1e-300)) - math.log10(max(lo, 1e-300)))
        except ValueError:
            decades = 1.0
        if decades <= 1.5:
            subs = (1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0)
        elif decades <= 4.0:
            subs = (2.0, 3.0, 5.0)
        else:
            subs = (3.0,)
        axis.set_minor_locator(
            LogLocator(base=10.0, subs=subs, numticks=100)
        )
        axis.set_minor_formatter(formatter)


def pad_log_axis(ax, which: str = "both", factor: float = 1.12) -> None:
    """
    Keep markers and error bars clear of the axis spines.

    Matplotlib fits a log axis tightly to the data, so the lowest-frequency
    point routinely lands ON the left spine with half its marker and the
    lower half of its error bar clipped off. Widening the limits by a fixed
    multiplicative factor -- the log-axis equivalent of a margin -- gives
    every point room without changing any value.
    """
    if which in ("x", "both") and ax.get_xscale() == "log":
        lo, hi = ax.get_xlim()
        if lo > 0 and hi > 0:
            ax.set_xlim(lo / factor, hi * factor)
    if which in ("y", "both") and ax.get_yscale() == "log":
        lo, hi = ax.get_ylim()
        if lo > 0 and hi > 0:
            ax.set_ylim(lo / factor, hi * factor)

def style_legend(ax, **kwargs: Any):
    """
    One legend style everywhere, chosen so it never hides data.

    A solid white box at partial opacity keeps the entries readable when the
    legend has to sit over a curve, which `loc="best"` will do on a crowded
    panel.
    """
    options: Dict[str, Any] = {
        "loc": "best",
        "fontsize": 9.5,
        "frameon": True,
        "framealpha": 0.92,
        "edgecolor": "0.6",
        "facecolor": "white",
        "borderpad": 0.55,
        "handlelength": 1.9,
        "handletextpad": 0.6,
        "labelspacing": 0.4,
        "columnspacing": 1.2,
        # One marker per entry. Matplotlib draws several for a line-plus-
        # marker artist by default, which makes an SED legend look cluttered
        # and widens it enough to start covering the data.
        "numpoints": 1,
        "scatterpoints": 1,
        "markerscale": 1.0,
    }
    options.update(kwargs)
    legend = ax.legend(**options)
    if legend is not None:
        frame = legend.get_frame()
        frame.set_linewidth(0.7)
        # Sit above the data, never under it.
        legend.set_zorder(20)
    return legend

def robust_display_limits(arr: np.ndarray, lo: float = 2.0, hi: float = 98.0):
    v = np.asarray(arr)[np.isfinite(arr)]
    if v.size < 10:
        return -1.0, 1.0
    return float(np.nanpercentile(v, lo)), float(np.nanpercentile(v, hi))


def _format_sky_axes(ax) -> None:
    ax.set_xlabel("Right Ascension (J2000)")
    ax.set_ylabel("Declination (J2000)")
    try:
        ax.coords[0].set_format_unit(u.hourangle)
        ax.coords[1].set_format_unit(u.deg)
        ax.coords[0].set_major_formatter("hh:mm:ss")
        ax.coords[1].set_major_formatter("dd:mm:ss")
    except Exception:                                          # noqa: BLE001
        pass
    ax.grid(False)


def draw_beam_ellipse(ax, beam: Optional[Beam], wcs: Optional[WCS],
                      color: str = "black") -> None:
    """
    Draw the restoring beam in the lower-RIGHT corner, unboxed.

    A resolved map without a beam marker cannot be read: every structure in
    it is only meaningful relative to the resolution element, so the beam is
    part of the measurement, not decoration.

    It sits bottom-right because these sources trail to the lower left, and a
    marker there overlapped the emission. No surrounding box: the ellipse is
    conventional enough to be unambiguous on its own, and the frame was
    competing with the contours for the reader's attention.
    """
    if beam is None or wcs is None:
        return
    try:
        from matplotlib.patches import Ellipse
        sx, sy = pixel_scales_arcsec(wcs)
        major = float(beam.major.to(u.arcsec).value) / max(sy, 1e-12)
        minor = float(beam.minor.to(u.arcsec).value) / max(sx, 1e-12)
        pa = float(beam.pa.to(u.deg).value)
        xlim, ylim = ax.get_xlim(), ax.get_ylim()
        span_x = xlim[1] - xlim[0]
        span_y = ylim[1] - ylim[0]
        # Inset by the beam's own size plus a margin, so a large beam does
        # not hang off the axes on a small field.
        cx = xlim[1] - 0.06 * span_x - 0.5 * major
        cy = ylim[0] + 0.06 * span_y + 0.5 * major
        ax.add_patch(Ellipse(
            (cx, cy), width=minor, height=major, angle=pa,
            facecolor=color, edgecolor=color, alpha=0.9, zorder=7,
        ))
    except Exception:                                          # noqa: BLE001
        # A missing or malformed beam must never take a figure down; the map
        # itself is still the science product.
        pass

def save_png_map(
    path: Path, data: np.ndarray, cbar_label: str, cmap: str = "viridis",
    vmin: Optional[float] = None, vmax: Optional[float] = None,
    diverging_zero: bool = False, wcs: Optional[WCS] = None,
    contour: Optional[np.ndarray] = None,
    contour_levels: Optional[Sequence[float]] = None,
    reference_image: Optional[np.ndarray] = None,
    reference_levels: Optional[Sequence[float]] = None,
    reference_color: str = "black",
    reference_linewidth: Optional[float] = None,
    reference_label: Optional[str] = None,
    reference_mask: Optional[np.ndarray] = None,
    beam: Optional[Beam] = None,
    title: Optional[str] = None,
) -> None:
    """
    Write one science map as a publication-ready figure.

    `reference_image` / `reference_levels` overlay total-intensity contours
    from a chosen reference band.  This is the standard way these maps are
    presented, and it is not cosmetic: a spectral-index or spectral-age map
    is a derived quantity with no morphology of its own, so without the total
    intensity drawn over it there is no way to tell which structure a given
    age or index belongs to -- the hotspot, the lobe, or the tail.
    """
    fig = plt.figure(figsize=(8.6, 7.0), constrained_layout=True)
    ax = fig.add_subplot(111, projection=wcs.celestial if wcs is not None else None)
    finite = np.asarray(data)[np.isfinite(data)]
    if diverging_zero and finite.size:
        lim = float(np.nanmax(np.abs(finite)))
        vmin, vmax = -lim, lim
    elif vmin is None or vmax is None:
        vmin, vmax = robust_display_limits(data)
    im = ax.imshow(
        data, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax,
        interpolation="nearest",
    )
    cb = fig.colorbar(im, ax=ax, pad=0.02)
    cb.set_label(cbar_label)

    if reference_image is not None and reference_levels is not None:
        levels = [float(l) for l in reference_levels if np.isfinite(l)]
        if levels:
            ref = smooth_reference_for_contours(
                np.asarray(reference_image, dtype=float), beam, wcs
            )
            if reference_mask is not None:
                # Confine the contours to the source. Drawn on the raw image
                # they also trace unrelated field sources and noise ridges
                # outside the segmentation, which makes the figure look as
                # though the derived quantity has simply failed to cover the
                # emission -- when in fact that emission is not part of this
                # source at all.
                ref = np.where(np.asarray(reference_mask, dtype=bool),
                               ref, np.nan)
            ax.contour(
                ref, levels=sorted(levels), colors=reference_color,
                # 0.6 pt is too fine to read once a figure is scaled down to
                # a journal column; the lowest level is drawn heavier so the
                # 5 sigma boundary of the source is unambiguous.
                linewidths=(
                    [float(reference_linewidth)] * len(levels)
                    if reference_linewidth
                    else [1.15] + [0.75] * (len(levels) - 1)
                ),
                alpha=0.9, zorder=4,
            )
            # The contour provenance is deliberately NOT stamped inside the
            # axes by default. In a paper that information belongs in the
            # figure caption, and a text box floating over the map is the
            # first thing a referee objects to. It is recorded in the FITS
            # header and the run report instead, and can be turned back on
            # with --contour-annotate.
            if reference_label:
                ax.text(
                    0.02, 0.98, reference_label, transform=ax.transAxes,
                    va="top", ha="left", fontsize=8, zorder=7,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                              edgecolor="0.7", alpha=0.85, linewidth=0.5),
                )

    if contour is not None:
        ax.contour(
            np.asarray(contour, dtype=float),
            levels=list(contour_levels) if contour_levels else [0.5],
            colors="k", linewidths=0.8, zorder=5,
        )

    # The beam marker stays neutral even when the contours are coloured:
    # it is an instrumental annotation, not part of the data, and a red beam
    # reads as though it belonged to the contour set.
    draw_beam_ellipse(ax, beam, wcs, color="black")

    if title:
        ax.set_title(title, fontsize=10)
    if wcs is not None:
        _format_sky_axes(ax)
    else:
        ax.set_xlabel("Pixel X")
        ax.set_ylabel("Pixel Y")
        ax.grid(False)
    fig.savefig(path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)

def save_common_beam_png(
    output: Path, data: np.ndarray, label: str, freq_hz: float, wcs: WCS
) -> None:
    finite = data[np.isfinite(data)]
    if finite.size < 10:
        return
    fig = plt.figure(figsize=(8.4, 7.0), constrained_layout=True)
    ax = fig.add_subplot(111, projection=wcs.celestial)
    im = ax.imshow(
        data, origin="lower", cmap="inferno",
        vmin=float(np.nanpercentile(finite, 1)),
        vmax=float(np.nanpercentile(finite, 99.7)),
        interpolation="nearest",
    )
    cb = fig.colorbar(im, ax=ax, pad=0.02)
    cb.set_label(r"$S_\nu$ (Jy beam$^{-1}$)")
    _format_sky_axes(ax)
    fig.savefig(output, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)


def save_segmentation_png(
    output: Path, detection_map: RadioMap, source_mask: np.ndarray,
    regions: Optional[Sequence[Region]] = None,
) -> None:
    finite = detection_map.data[np.isfinite(detection_map.data)]
    if finite.size < 10:
        return
    fig = plt.figure(figsize=(9.4, 7.6), constrained_layout=True)
    ax = fig.add_subplot(111, projection=detection_map.wcs.celestial)
    ax.imshow(
        detection_map.data, origin="lower", cmap="gray_r",
        vmin=float(np.nanpercentile(finite, 5)),
        vmax=float(np.nanpercentile(finite, 99.7)),
        interpolation="nearest",
    )
    ax.contour(
        source_mask.astype(float), levels=[0.5], colors="crimson", linewidths=1.3,
    )
    if regions:
        for region in regions:
            y0, y1, x0, x1 = region.bbox
            ax.plot(
                [x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0],
                lw=0.5, color="tab:blue", alpha=0.7,
            )
    _format_sky_axes(ax)
    fig.savefig(output, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)


def save_science_sed(
    output: Path,
    freq_hz: np.ndarray, flux_jy: np.ndarray, flux_err_jy: np.ndarray,
    detected: np.ndarray, redshift: float, alpha_for_k: float,
    model_curves: Optional[List[Dict[str, Any]]] = None,
    luminosity_output: Optional[Path] = None,
    alpha_err: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Observed SED and rest-frame luminosity, as two SEPARATE figures.

    They were one two-panel figure, which is convenient on screen and wrong
    for a paper: the two panels carry different physics (an observable and a
    derived, k-corrected quantity), are cited in different places, and a
    journal will size them independently. `luminosity_output` receives the
    L_nu panel; pass None to keep the combined layout.
    """
    nu = np.asarray(freq_hz, dtype=float)
    f = np.asarray(flux_jy, dtype=float)
    e = np.asarray(flux_err_jy, dtype=float)
    det = np.asarray(detected, dtype=bool)
    order = np.argsort(nu)
    nu, f, e, det = nu[order], f[order], e[order], det[order]

    # Two independent figures when a luminosity path is given, so each can be
    # sized and placed on its own in a paper.
    if luminosity_output is not None:
        fig = plt.figure(figsize=(8.0, 6.0), constrained_layout=True)
        fig_lum = plt.figure(figsize=(8.0, 6.0), constrained_layout=True)
        axes = [fig.add_subplot(111), fig_lum.add_subplot(111)]
    else:
        fig = plt.figure(figsize=(12.6, 5.6), constrained_layout=True)
        fig_lum = None
        axes = [fig.add_subplot(121), fig.add_subplot(122)]

    if np.any(det):
        axes[0].errorbar(
            nu[det] / 1e9, f[det] * 1e3, yerr=e[det] * 1e3,
            fmt="o", ms=6, mew=0.9, capsize=3, lw=1.2,
            color="k", label=MEASURED_LABEL,
        )
    if np.any(~det):
        axes[0].errorbar(
            nu[~det] / 1e9, np.maximum(3.0 * e[~det] * 1e3, 1e-12),
            uplims=True, yerr=0.25 * np.maximum(3.0 * e[~det] * 1e3, 1e-12),
            fmt="v", ms=6, color="0.45", label=r"$3\sigma$ upper limit",
        )

    for curve in (model_curves or []):
        axes[0].plot(
            np.asarray(curve["nu_Hz"]) / 1e9,
            np.asarray(curve["flux_Jy"]) * 1e3,
            lw=curve.get("lw", 1.8), ls=curve.get("ls", "-"),
            color=curve.get("color"), alpha=curve.get("alpha", 1.0),
            label=curve.get("label"),
        )

    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    style_log_axis(axes[0])
    pad_log_axis(axes[0])
    axes[0].set_xlabel(r"Observed frequency, $\nu_{\rm obs}$ (GHz)")
    axes[0].set_ylabel(r"Integrated flux density, $S_\nu$ (mJy)")
    style_legend(axes[0])

    lnu = rest_lnu_from_flux(f, redshift, alpha=alpha_for_k)
    elnu = rest_lnu_from_flux(e, redshift, alpha=alpha_for_k)
    nurest = nu * (1.0 + redshift)
    ok = det & np.isfinite(lnu) & (lnu > 0)
    kcorr_ln_sigma = kcorrection_ln_sigma(redshift, alpha_err)
    # Frequencies in GHz and a common power of ten pulled out of L_nu, so
    # every tick on this panel is an ordinary number.
    _lnu_scale, _lnu_suffix = axis_pow10_factor(lnu[ok] if np.any(ok) else lnu)
    if np.any(ok):
        axes[1].errorbar(
            nurest[ok] / 1e9, lnu[ok] / _lnu_scale, yerr=elnu[ok] / _lnu_scale,
            fmt="o", ms=6, mew=0.9, capsize=3, lw=1.2, color="k",
            label=MEASURED_LABEL,
        )
    for curve in (model_curves or []):
        axes[1].plot(
            np.asarray(curve["nu_Hz"]) * (1.0 + redshift) / 1e9,
            rest_lnu_from_flux(
                np.asarray(curve["flux_Jy"]), redshift, alpha=alpha_for_k
            ) / _lnu_scale,
            lw=curve.get("lw", 1.8), ls=curve.get("ls", "-"),
            color=curve.get("color"), alpha=curve.get("alpha", 1.0),
            label=curve.get("label"),
        )
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    style_log_axis(axes[1])
    pad_log_axis(axes[1])
    axes[1].set_xlabel(r"Rest-frame frequency, $\nu_{\rm rest}$ (GHz)")
    axes[1].set_ylabel(rf"$L_\nu$ ({_lnu_suffix}W Hz$^{{-1}}$)")
    # The error bars are per-point flux errors. The K-correction adds a
    # CORRELATED term, ln(1+z)*sigma_alpha, which slides every point together
    # and so is quoted here rather than folded into the bars.
    if kcorr_ln_sigma > 0:
        axes[1].text(
            0.03, 0.03,
            rf"$\pm{100 * kcorr_ln_sigma:.0f}\%$ correlated "
            rf"$K$-correction ($\sigma_\alpha$)",
            transform=axes[1].transAxes, va="bottom", ha="left",
            fontsize=8.5, color="0.25",
        )
    style_legend(axes[1])

    fig.savefig(output, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    if fig_lum is not None:
        fig_lum.savefig(luminosity_output, dpi=PLOT_DPI, bbox_inches="tight")
        plt.close(fig_lum)

    return {
        "frequency_Hz": nu.tolist(),
        "flux_Jy": f.tolist(),
        "flux_err_Jy": e.tolist(),
        "rest_frequency_Hz": nurest.tolist(),
        "rest_luminosity_W_Hz": lnu.tolist(),
        "rest_luminosity_err_W_Hz": elnu.tolist(),
        "k_correction_alpha": float(alpha_for_k),
        "n_model_curves": len(model_curves or []),
    }


def smooth_model_curve(
    nu_hz: np.ndarray, flux: np.ndarray, window: int = 21
) -> np.ndarray:
    """
    Remove quadrature noise from a model curve before drawing it.

    synchrofit evaluates the Tribble models on a fixed integration grid
    (spectral_models.py: `nalpha, nfields = 32, 128`) and the discretisation
    error changes from one frequency to the next, so the line comes out
    visibly ragged -- 0.63 per cent rms for TJP against 0.28 per cent for JP,
    far inside the error bars. A synchrotron spectrum is smooth by
    construction, so those wiggles are known artefacts and removing them
    hides no structure.

    Two details make this safe rather than cosmetic:

    * the filter runs in LOG-LOG space, where these spectra are nearly
      straight, and

    * it is a Savitzky-Golay filter of polynomial order 2, which reproduces
      any quadratic exactly. Spectral CURVATURE is the quantity this whole
      analysis is measuring, and a quadratic-preserving filter cannot bias
      it. A plain boxcar can and does: measured against a known curved
      spectrum it was accurate to 0.35 per cent through the middle but
      distorted the ENDS by up to 9 per cent, because mirror padding flattens
      the slope where the curve is steepest. `mode="interp"` fits the
      polynomial to the end windows instead of padding at all.
    """
    y = np.asarray(flux, dtype=float)
    if y.size < 7:
        return y
    if not (np.all(np.isfinite(y)) and np.all(y > 0)):
        return y
    win = int(min(window, y.size if y.size % 2 else y.size - 1))
    if win % 2 == 0:
        win -= 1
    if win < 7:
        return y
    try:
        from scipy.signal import savgol_filter
        return np.exp(savgol_filter(np.log(y), win, 2, mode="interp"))
    except Exception:                                          # noqa: BLE001
        return y

def build_ensemble_curves(
    ensemble: Dict[str, Any], nu_grid: np.ndarray,
    redshift: float, b_field_t: Optional[float],
    which: str = "models",
) -> List[Dict[str, Any]]:
    """Evaluate every successfully fitted ensemble model on a frequency grid."""
    curves: List[Dict[str, Any]] = []
    palette = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    records = ensemble.get(which) or {}
    for i, (name, record) in enumerate(sorted(records.items())):
        if not isinstance(record, dict) or not record.get("fit_success"):
            continue
        model_flux, _ = SYNCHROFIT.model(
            {
                "fit_type": record.get("synchrofit_fit_type", record.get("fit_type")),
                "_params8": record.get("_params8"),
                "_grid": record.get("_grid"),
            },
            nu_grid,
            # Prefer the field this record was fitted at; fall back to the
            # caller's only when the record does not carry one.
            b_field_t=(
                (record.get("b_field_T_used") or b_field_t)
                if record.get("tribble") else None
            ),
            redshift=redshift if record.get("tribble") else None,
            # Only the central curve is drawn; the Monte-Carlo error array
            # this would otherwise build is discarded on the next line. It
            # does not change the returned model at all -- verified identical
            # to five significant figures against mc_length=100 -- but it
            # costs about eleven times as long, per model, per figure.
            mc_length=1, err_width=1,
        )
        if model_flux is None or not np.all(np.isfinite(model_flux)):
            continue
        if not np.all(model_flux > 0):
            continue
        model_flux = smooth_model_curve(nu_grid, model_flux)
        weight = None
        for key in ("akaike_weights",):
            src = ensemble.get("free_s_consensus", {}).get(key) or {}
            if name in src:
                weight = src[name]
        label = name if weight is None else f"{name} (w={weight:.2f})"
        curves.append({
            "nu_Hz": nu_grid,
            "flux_Jy": model_flux,
            "label": label,
            "color": palette[i % len(palette)] if palette else None,
            "lw": 1.6,
            "alpha": 0.9,
        })
    return curves


def save_ensemble_sed_plot(
    output: Path,
    freq_hz: np.ndarray, flux_jy: np.ndarray, err_jy: np.ndarray,
    detected: np.ndarray, ensemble: Dict[str, Any],
    redshift: float, b_field_t: Optional[float],
) -> None:
    """
    Every ageing model on one axis, with normalised residuals underneath.

    Showing all six curves together is the honest way to present a
    four-point SED: if the models are indistinguishable over the observed
    range, the reader can see that immediately rather than inferring it from
    a table of near-identical AICc values.
    """
    nu = np.asarray(freq_hz, dtype=float)
    f = np.asarray(flux_jy, dtype=float)
    e = np.asarray(err_jy, dtype=float)
    det = np.asarray(detected, dtype=bool)
    order = np.argsort(nu)
    nu, f, e, det = nu[order], f[order], e[order], det[order]

    pos = nu[nu > 0]
    if pos.size < 2:
        return
    grid = np.logspace(
        math.log10(float(np.min(pos)) * 0.6),
        math.log10(float(np.max(pos)) * 1.8), 300,
    )
    curves = build_ensemble_curves(ensemble, grid, redshift, b_field_t)

    fig, axes = plt.subplots(
        2, 1, figsize=(8.0, 7.6), sharex=True,
        gridspec_kw={"height_ratios": [3.0, 1.3]}, constrained_layout=True,
    )
    if np.any(det):
        axes[0].errorbar(
            nu[det] / 1e9, f[det] * 1e3, yerr=e[det] * 1e3,
            fmt="o", ms=6, mew=0.9, capsize=3, lw=1.2, color="k",
            label=MEASURED_LABEL, zorder=5,
        )
    for curve in curves:
        axes[0].plot(
            curve["nu_Hz"] / 1e9, np.asarray(curve["flux_Jy"]) * 1e3,
            lw=curve["lw"], color=curve["color"], alpha=curve["alpha"],
            label=curve["label"],
        )
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    style_log_axis(axes[0])
    axes[0].set_ylabel(r"$S_\nu$ (mJy)")
    style_legend(axes[0], ncol=2)

    axes[1].axhline(0.0, lw=0.8, color="k")
    for level, style in ((1.0, ":"), (-1.0, ":")):
        axes[1].axhline(level, lw=0.6, ls=style, color="0.6")
    for curve in curves:
        interp = np.exp(np.interp(
            np.log(nu[det]), np.log(curve["nu_Hz"]),
            np.log(np.asarray(curve["flux_Jy"])),
        ))
        axes[1].plot(
            nu[det] / 1e9, (f[det] - interp) / e[det],
            "o", ms=5, color=curve["color"], alpha=0.9,
        )
    axes[1].set_xscale("log")
    # The panels share this axis, so its labels are the ones the reader sees.
    style_log_axis(axes[1], which="x")
    axes[1].set_xlabel(r"Observed frequency, $\nu$ (GHz)")
    axes[1].set_ylabel(r"$(S-S_{\rm model})/\sigma$")

    fig.savefig(output, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)


def select_best_model(
    ensemble: Dict[str, Any], force_model: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """
    The single best-supported ageing model, by AICc.

    AICc is the right criterion here rather than raw chi-square: the models
    carry different numbers of free parameters (CI-off has four, JP has
    three), the sample is tiny, and AICc penalises both. The Akaike weight is
    reported alongside, because "best" is only meaningful next to how much
    better it is than the runners-up -- with five SED points the winner is
    frequently not significant, and saying so is part of the answer.
    """
    models = (ensemble or {}).get("models") or {}

    # An explicitly requested model overrides the ranking. The ranking is
    # still computed and reported, so the figure can say where the chosen
    # model actually sits among the alternatives rather than implying it won.
    forced = None
    if force_model and force_model not in {"Ensemble", "Power-law"}:
        for name, rec in models.items():
            if name.replace("_", "-").upper() == force_model.replace("_", "-").upper():
                if rec.get("fit_success"):
                    forced = name
                break

    scored = [
        (float(rec["AICc"]), name, rec)
        for name, rec in models.items()
        if rec.get("fit_success") and rec.get("AICc") is not None
        and np.isfinite(rec.get("AICc", np.nan))
    ]
    criterion = "AICc"

    if not scored:
        # AICc is undefined whenever n <= k + 1, which is the normal state
        # with four SED points and three free parameters -- exactly the case
        # after dropping a bad band. Ranking is still possible on BIC, which
        # only needs n > 0, and failing that on raw chi-square. Both are
        # weaker evidence than AICc and the criterion actually used is
        # reported, because a "best model" chosen on chi-square alone takes
        # no account of the models having different numbers of parameters.
        scored = [
            (float(rec["BIC"]), name, rec)
            for name, rec in models.items()
            if rec.get("fit_success") and rec.get("BIC") is not None
            and np.isfinite(rec.get("BIC", np.nan))
        ]
        criterion = "BIC"
    if not scored:
        scored = [
            (float(rec["chi2"]), name, rec)
            for name, rec in models.items()
            if rec.get("fit_success") and rec.get("chi2") is not None
            and np.isfinite(rec.get("chi2", np.nan))
        ]
        criterion = "chi2"
    if not scored:
        return None
    scored.sort(key=lambda t: t[0])

    aicc = np.asarray([s[0] for s in scored], dtype=float)
    delta = aicc - aicc[0]
    weights = np.exp(-0.5 * delta)
    weights = weights / weights.sum()

    if forced is not None:
        # Re-order so the requested model is first; weights and deltas below
        # are then quoted relative to it.
        scored.sort(key=lambda t: (t[1] != forced, t[0]))
    best_aicc, best_name, best_rec = scored[0]
    n_pts = best_rec.get("n_points")
    k_free = best_rec.get("k_free")
    ranking = [
        {
            "model": name,
            "criterion": criterion,
            "criterion_value": float(a),
            "delta_criterion": float(a - best_aicc),
            "akaike_weight": float(w),
            "injection_index_s": rec.get("injection_index_s"),
            "log10_break_frequency_Hz": rec.get("log10_break_frequency_Hz"),
            "chi2": rec.get("chi2"),
            "railed": bool(
                rec.get("log10_break_frequency_Hz") is not None
                and rec.get("break_at_grid_edge", False)
            ),
        }
        for (a, name, rec), w in zip(scored, weights)
    ]

    second = delta[1] if delta.size > 1 else float("inf")
    # Burnham & Anderson: Delta AICc < 2 means the models are essentially
    # indistinguishable on this data.
    if second < 2.0:
        verdict = (
            "NOT significant: the runner-up is within Delta AICc = 2, so the "
            "data do not distinguish these models. Quote the ensemble "
            "average, not this single model."
        )
    elif second < 4.0:
        verdict = (
            "Weak preference: Delta AICc = %.1f over the runner-up. Treat "
            "the model choice as provisional." % second
        )
    else:
        verdict = (
            "Preferred: Delta AICc = %.1f over the runner-up, which is "
            "substantial support on this data." % second
        )

    if criterion != "AICc":
        verdict = (
            f"Ranked on {criterion}, NOT AICc: with {n_pts} SED points and "
            f"{k_free} free parameters AICc is undefined (it needs "
            f"n > k + 1). " + verdict
        )
    if forced is not None:
        verdict = (
            f"{forced} was REQUESTED, not selected by {criterion}. " + verdict
        )
    return {
        "best_model": best_name,
        "forced_model": forced,
        "record": best_rec,
        "criterion": criterion,
        "akaike_weight": float(weights[0]),
        "delta_criterion_to_second": float(second),
        "significant": bool(second >= 2.0 and criterion == "AICc"),
        "verdict": verdict,
        "ranking": ranking,
    }


def save_best_model_sed(
    output: Path,
    freq_hz: np.ndarray, flux_jy: np.ndarray, err_jy: np.ndarray,
    detected: np.ndarray, ensemble: Dict[str, Any],
    redshift: float, b_field_t: Optional[float],
    force_model: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """
    The SED with a single ageing model drawn through it.

    The ensemble figure answers "which models are viable"; this one answers
    "what does the favoured model actually say", which is the plot a paper
    puts in the results section. The ranking is printed on the figure so the
    reader can see immediately whether the preference is meaningful or the
    models are tied.
    """
    best = select_best_model(ensemble, force_model=force_model)
    if best is None:
        return None

    nu = np.asarray(freq_hz, dtype=float)
    f = np.asarray(flux_jy, dtype=float)
    e = np.asarray(err_jy, dtype=float)
    det = np.asarray(detected, dtype=bool)
    order = np.argsort(nu)
    nu, f, e, det = nu[order], f[order], e[order], det[order]
    pos = nu[nu > 0]
    if pos.size < 2:
        return None

    grid = np.logspace(
        math.log10(float(np.min(pos)) * 0.6),
        math.log10(float(np.max(pos)) * 1.8), 300,
    )
    name = best["best_model"]
    single = {"models": {name: best["record"]},
              "free_s_consensus": ensemble.get("free_s_consensus", {})}
    curves = build_ensemble_curves(single, grid, redshift, b_field_t)
    if not curves:
        return None
    curve = curves[0]
    model_flux = np.asarray(curve["flux_Jy"], dtype=float)

    fig, axes = plt.subplots(
        2, 1, figsize=(8.0, 7.6), sharex=True,
        gridspec_kw={"height_ratios": [3.0, 1.3]}, constrained_layout=True,
    )
    if np.any(det):
        axes[0].errorbar(
            nu[det] / 1e9, f[det] * 1e3, yerr=e[det] * 1e3,
            fmt="o", ms=6, mew=0.9, capsize=3, lw=1.2, color="k",
            label=MEASURED_LABEL, zorder=5,
        )
    rec = best["record"]
    s_val = rec.get("injection_index_s")
    lb = rec.get("log10_break_frequency_Hz")
    # The legend carries the model name only; every fitted quantity is
    # listed in the parameter box instead, so the two do not duplicate
    # each other and the legend stays narrow.
    label = name.replace("_", " ")
    axes[0].plot(grid / 1e9, model_flux * 1e3, lw=2.0, color="crimson",
                 label=label, zorder=4)

    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    style_log_axis(axes[0])
    axes[0].set_ylabel(r"$S_\nu$ (mJy)")

    interp = np.exp(np.interp(
        np.log(nu[det]), np.log(grid), np.log(model_flux)
    ))
    resid = (f[det] - interp) / e[det]

    # -- goodness of fit, stated on the figure ---------------------------
    #
    # chi^2/dof alone is misleading at the dof this analysis has: with four
    # points and three free parameters dof = 1, where chi^2/dof has an rms
    # scatter of sqrt(2/dof) = 1.41 about its expectation. A value of 2.5
    # there corresponds to p = 0.11 and is a perfectly acceptable fit. The
    # p-value is therefore quoted next to it, because that is the number that
    # actually says whether the model is rejected.
    n_fit = int(np.count_nonzero(det))
    k_free = int(rec.get("k_free", 3) or 3)
    dof = max(n_fit - k_free, 1)
    chi2 = float(np.sum(resid ** 2))
    chi2_red = chi2 / dof
    try:
        from scipy import stats as _stats
        p_value = float(1.0 - _stats.chi2.cdf(chi2, dof))
    except Exception:                                          # noqa: BLE001
        p_value = float("nan")

    age_myr = None
    sp = rec.get("spectral_age") or {}
    if isinstance(sp, dict):
        age_myr = sp.get("tau_Myr")

    # Fitted quantities, in the order a reader of the paper wants them.
    # t_on and t_off are only defined for the CI-off family, where the
    # remnant fraction splits the total age into an active and a switched-off
    # phase; for the single-injection models they are omitted rather than
    # invented.
    alpha_inj_pos = rec.get("alpha_inj_positive")
    if alpha_inj_pos is None and s_val is not None:
        alpha_inj_pos = 0.5 * (float(s_val) - 1.0)
    b_used = None
    if isinstance(sp, dict):
        b_used = sp.get("b_field_used_T")
    if b_used is None:
        b_used = rec.get("b_field_T_used") or rec.get("b_field_input_T")
    t_on = sp.get("t_on_Myr") if isinstance(sp, dict) else None
    t_off = sp.get("t_off_Myr") if isinstance(sp, dict) else None
    nu_b_ghz = (10.0 ** float(lb)) / 1e9 if lb is not None else None

    info = [rf"$\mathbf{{{name.replace('_', chr(92)+'_')}}}$"]
    if alpha_inj_pos is not None and np.isfinite(alpha_inj_pos):
        info.append(rf"$\alpha_{{\rm inj}} = {alpha_inj_pos:.3f}$")
    if b_used is not None and np.isfinite(b_used):
        info.append(rf"$B = {b_used:.3e}$ T")
    if age_myr is not None and np.isfinite(age_myr):
        info.append(rf"$\tau = {age_myr:.2f}$ Myr")
    if t_on is not None and np.isfinite(t_on):
        info.append(rf"$t_{{\rm on}} = {t_on:.2f}$ Myr")
    if t_off is not None and np.isfinite(t_off):
        info.append(rf"$t_{{\rm off}} = {t_off:.2f}$ Myr")
    if nu_b_ghz is not None and np.isfinite(nu_b_ghz):
        info.append(rf"$\nu_{{\rm b}} = {nu_b_ghz:.3f}$ GHz")
    info.append(rf"$\chi^2_{{\rm red}} = {chi2_red:.2f}$")
    axes[0].text(
        0.03, 0.05, "\n".join(info), transform=axes[0].transAxes,
        va="bottom", ha="left", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.45", facecolor="white",
                  edgecolor="0.7", alpha=0.92, linewidth=0.6),
    )
    style_legend(axes[0])
    axes[1].axhline(0.0, lw=0.8, color="k")
    for level in (1.0, -1.0):
        axes[1].axhline(level, lw=0.6, ls=":", color="0.6")
    axes[1].plot(nu[det] / 1e9, resid, "o", ms=6, color="crimson")
    axes[1].set_xscale("log")
    style_log_axis(axes[1], which="x")
    axes[1].set_xlabel(r"Observed frequency, $\nu$ (GHz)")
    axes[1].set_ylabel(r"$(S-S_{\rm model})/\sigma$")

    fig.savefig(output, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)

    best["chi2"] = chi2
    best["chi2_reduced"] = chi2_red
    best["dof"] = dof
    best["p_value"] = p_value
    best["spectral_age_Myr"] = age_myr
    best["goodness_of_fit_note"] = (
        f"chi2/dof = {chi2_red:.2f} on dof = {dof}, p = {p_value:.3f}. At low "
        "dof chi2/dof scatters by sqrt(2/dof) about 1, so it should be read "
        "with the p-value rather than compared to 1 directly; p > 0.05 means "
        "the model is not rejected."
    )
    return best

def smooth_through_nodes(
    x: np.ndarray, y: np.ndarray, n: int = 400
) -> Tuple[np.ndarray, np.ndarray]:
    """
    A smooth curve through scan nodes, for drawing only.

    A grid scan gives a handful of nodes joined by straight segments, which
    reads as a jagged line even though the underlying chi-square surface is
    smooth. This interpolates it.

    PCHIP, not a spline: PCHIP is shape-preserving and cannot overshoot
    between nodes. That matters here because a cubic spline through a
    chi-square profile will happily dip BELOW the minimum between two nodes,
    drawing a Delta chi2 < 0 that does not exist and inventing a second
    minimum where there is none. PCHIP passes through every node exactly and
    stays inside their range, so the drawn curve cannot claim anything the
    scan did not find.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    good = np.isfinite(x) & np.isfinite(y)
    x, y = x[good], y[good]
    if x.size < 4:
        return x, y
    order = np.argsort(x)
    x, y = x[order], y[order]
    keep = np.concatenate(([True], np.diff(x) > 0))
    x, y = x[keep], y[keep]
    if x.size < 4:
        return x, y
    try:
        from scipy.interpolate import PchipInterpolator
        fine = np.linspace(x[0], x[-1], n)
        return fine, PchipInterpolator(x, y)(fine)
    except Exception:                                          # noqa: BLE001
        return x, y

def save_injection_scan_plot(output: Path, scans: Dict[str, Any]) -> None:
    """Profile chi-square in the injection index, with the Delta chi2 = 1 band."""
    usable = {k: v for k, v in (scans or {}).items() if v and v.get("nodes")}
    if not usable:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.9), constrained_layout=True)
    for name, scan in sorted(usable.items()):
        s = np.asarray([n["s"] for n in scan["nodes"]], dtype=float)
        chi2 = np.asarray([n["chi2"] for n in scan["nodes"]], dtype=float)
        alpha = -np.asarray(
            [n["alpha_inj_signed"] for n in scan["nodes"]], dtype=float
        )
        dchi2 = chi2 - np.nanmin(chi2)
        # Nodes as markers, the smooth interpolant as the line, so the reader
        # can see where the scan actually sampled.
        line, = axes[0].plot(*smooth_through_nodes(alpha, dchi2), lw=1.7,
                             label=name)
        axes[0].plot(alpha, dchi2, "o", ms=3.4, color=line.get_color(),
                     alpha=0.85)
        brk = np.asarray(
            [n.get("log10_break_frequency_Hz", np.nan) for n in scan["nodes"]],
            dtype=float,
        )
        line2, = axes[1].plot(*smooth_through_nodes(alpha, brk), lw=1.7,
                              label=name)
        axes[1].plot(alpha, brk, "o", ms=3.4, color=line2.get_color(),
                     alpha=0.85)
    axes[0].axhline(1.0, ls="--", lw=0.9, color="0.4")
    axes[0].text(
        0.02, 0.94, r"$\Delta\chi^2=1$", transform=axes[0].transAxes,
        fontsize=9, color="0.35",
    )
    axes[0].set_xlabel(r"Injection index $\alpha_{\rm inj}$ (positive convention)")
    axes[0].set_ylabel(r"$\Delta\chi^2$")
    axes[0].set_ylim(bottom=0)
    style_legend(axes[0])

    axes[1].set_xlabel(r"Injection index $\alpha_{\rm inj}$ (positive convention)")
    axes[1].set_ylabel(r"$\log_{10}(\nu_{\rm break}/{\rm Hz})$")
    style_legend(axes[1])

    fig.savefig(output, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)


def save_energy_budget_png(output: Path, analysis: Dict[str, Any]) -> None:
    """Side-by-side energy budget at equipartition and at minimum energy."""
    eq = analysis["direct"]["equipartition"]
    me = analysis["direct"]["minimum_energy"]
    labels = [r"$u_e$", r"$u_{\rm part}=\zeta u_e$", r"$u_B$", r"$u_{\rm tot}$"]
    eq_vals = [
        eq["electron_energy_density_J_m3"],
        eq["particle_energy_density_J_m3"],
        eq["magnetic_energy_density_J_m3"],
        eq["total_energy_density_J_m3"],
    ]
    me_vals = [
        me["electron_energy_density_J_m3"],
        me["particle_energy_density_J_m3"],
        me["magnetic_energy_density_J_m3"],
        me["total_energy_density_J_m3"],
    ]
    x = np.arange(len(labels))
    width = 0.38

    fig, ax = plt.subplots(figsize=(8.2, 5.4), constrained_layout=True)
    # Energy densities here are ~1e-14 J/m^3; factor the exponent into the
    # axis label so the ticks read as ordinary numbers.
    _u_scale, _u_suffix = axis_pow10_factor(np.asarray(eq_vals + me_vals))
    eq_vals = [v / _u_scale for v in eq_vals]
    me_vals = [v / _u_scale for v in me_vals]
    ax.bar(x - width / 2, eq_vals, width,
           label=fr"Equipartition ($B={eq['B_uG']:.2f}\,\mu$G)", color="tab:blue")
    ax.bar(x + width / 2, me_vals, width,
           label=fr"Minimum energy ($B={me['B_uG']:.2f}\,\mu$G)", color="tab:orange")
    ax.set_yscale("log")
    style_log_axis(ax, which="y")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel(rf"Energy density ({_u_suffix}J m$^{{-3}}$)")
    style_legend(ax)
    fig.savefig(output, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)


def save_volume_emissivity(
    output: Path, freq_hz: np.ndarray, flux_jy: np.ndarray,
    flux_err_jy: np.ndarray, detected: np.ndarray, redshift: float,
    volume_m3: float, alpha_for_k: float,
    ensemble: Optional[Dict[str, Any]] = None,
    b_field_t: Optional[float] = None,
    alpha_err: Optional[float] = None,
    volume_ln_sigma: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Rest-frame volume emissivity, with the fitted models drawn through it.

    The measured points alone show only that the emissivity falls with
    frequency. What makes the figure worth publishing is the fitted spectrum
    over them: the same models fitted to the integrated SED, converted point
    by point through the identical k-correction and volume, so any curvature
    the ageing models predict can be read directly against the data.
    """
    if not np.isfinite(volume_m3) or volume_m3 <= 0:
        raise ValueError("Volume must be positive for an emissivity plot.")
    nu = np.asarray(freq_hz, dtype=float)
    f = np.asarray(flux_jy, dtype=float)
    e = np.asarray(flux_err_jy, dtype=float)
    det = np.asarray(detected, dtype=bool)
    good = det & np.isfinite(nu) & (nu > 0) & (f > 0) & (e > 0)
    nu, f, e = nu[good], f[good], e[good]
    order = np.argsort(nu)
    nu, f, e = nu[order], f[order], e[order]

    nurest = nu * (1.0 + redshift)
    j = rest_lnu_from_flux(f, redshift, alpha=alpha_for_k) / volume_m3
    ej = rest_lnu_from_flux(e, redshift, alpha=alpha_for_k) / volume_m3

    fig, ax = plt.subplots(figsize=(8.0, 5.9), constrained_layout=True)
    _j_scale, _j_suffix = axis_pow10_factor(j)
    kcorr_ln_sigma = kcorrection_ln_sigma(redshift, alpha_err)

    curves: List[Dict[str, Any]] = []
    if ensemble:
        grid = np.logspace(
            math.log10(float(nu.min()) * 0.6),
            math.log10(float(nu.max()) * 1.8), 300,
        )   # observed-frame Hz; converted to rest-frame GHz when plotted
        for curve in build_ensemble_curves(
            ensemble, grid, redshift, b_field_t
        ):
            model_j = rest_lnu_from_flux(
                np.asarray(curve["flux_Jy"], dtype=float),
                redshift, alpha=alpha_for_k,
            ) / volume_m3
            ax.plot(
                grid * (1.0 + redshift) / 1e9, model_j / _j_scale,
                lw=curve["lw"], color=curve["color"], alpha=curve["alpha"],
                label=curve["label"], zorder=3,
            )
            curves.append({"label": curve["label"]})

    ax.errorbar(
        nurest / 1e9, j / _j_scale, yerr=ej / _j_scale,
        fmt="o", ms=6, capsize=3, lw=1.2, color="k",
        label=MEASURED_LABEL, zorder=5,
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    style_log_axis(ax)
    ax.set_xlabel(r"Rest-frame frequency, $\nu_{\rm rest}$ (GHz)")
    ax.set_ylabel(
        rf"Volume emissivity, $j_\nu$ ({_j_suffix}W m$^{{-3}}$ Hz$^{{-1}}$)"
    )
    # j_nu = L_nu / V carries TWO correlated systematics that the per-point
    # flux errors do not contain, and both are larger than the error bars:
    #
    #   K-correction  ln(1+z) * sigma_alpha   -- one alpha for every band
    #   source volume sigma_V / V             -- one volume for every band
    #
    # Neither scatters the points against one another; they slide the whole
    # curve. Folding them into the bars would misrepresent them as noise and
    # make the spectrum look far less well determined than it is, so they are
    # stated separately.
    _sys = []
    if kcorr_ln_sigma > 0:
        _sys.append(rf"$\pm{100 * kcorr_ln_sigma:.0f}\%$ $K$-corr.")
    if volume_ln_sigma and np.isfinite(volume_ln_sigma) and volume_ln_sigma > 0:
        _sys.append(rf"$\pm{100 * float(volume_ln_sigma):.0f}\%$ volume")
    if _sys:
        ax.text(
            0.03, 0.03, "correlated: " + ", ".join(_sys),
            transform=ax.transAxes, va="bottom", ha="left",
            fontsize=8.5, color="0.25",
        )
    pad_log_axis(ax)
    style_legend(ax, ncol=2 if len(curves) > 3 else 1)
    fig.savefig(output, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)

    return {
        "rest_frequency_Hz": nurest.tolist(),
        "volume_emissivity_W_m3_Hz": j.tolist(),
        "volume_emissivity_err_W_m3_Hz": ej.tolist(),
        "volume_m3": float(volume_m3),
        "models_overplotted": [c["label"] for c in curves],
        "correlated_systematics": {
            "kcorrection_fractional": float(kcorr_ln_sigma),
            "volume_fractional": (
                float(volume_ln_sigma) if volume_ln_sigma else None
            ),
            "note": (
                "Error bars are per-point flux errors only. These two terms "
                "are correlated across all bands -- one alpha and one volume "
                "are used for every point -- so they translate the whole "
                "curve rather than scattering it, and must be quoted "
                "separately from the bars."
            ),
        },
    }


def save_model_comparison_plot(output: Path, ensemble: Dict[str, Any]) -> None:
    """Injection index, break frequency and Akaike weight per model."""
    models = ensemble.get("models") or {}
    names, alphas, aerr, breaks, berr, weights = [], [], [], [], [], []
    akaike = ensemble.get("free_s_consensus", {}).get("akaike_weights", {})
    for name in SPECTRAL_AGE_MODELS:
        record = models.get(name)
        if not record or not record.get("fit_success"):
            continue
        names.append(name)
        alphas.append(-record["alpha_inj_radio"])
        aerr.append(record["alpha_inj_radio_err"])
        breaks.append(record["log10_break_frequency_Hz"])
        berr.append(record["log10_break_frequency_err"])
        weights.append(akaike.get(name, 0.0))
    if not names:
        return

    x = np.arange(len(names))
    fig, axes = plt.subplots(3, 1, figsize=(8.0, 8.6), sharex=True,
                             constrained_layout=True)
    axes[0].errorbar(x, alphas, yerr=aerr, fmt="o", ms=6, capsize=3,
                     lw=1.2, color="k")
    axes[0].set_ylabel(r"$\alpha_{\rm inj}$")
    axes[1].errorbar(x, breaks, yerr=berr, fmt="s", ms=6, capsize=3,
                     lw=1.2, color="tab:blue")
    axes[1].set_ylabel(r"$\log_{10}(\nu_{\rm break}/{\rm Hz})$")
    axes[2].bar(x, weights, color="tab:green")
    axes[2].set_ylabel("Akaike weight")
    axes[2].set_ylim(0, 1.05)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(names, rotation=25, ha="right")
    fig.savefig(output, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)


def save_source_properties_text(
    output: Path,
    source_label: str,
    geometry: Dict[str, Any],
    solution: Optional[Dict[str, Any]],
    observed: Dict[str, Any],
    ensemble: Optional[Dict[str, Any]],
    config: Dict[str, Any],
) -> None:
    """Human-readable summary of every headline number, with its provenance."""
    lines: List[str] = [
        "PySynch v{} -- SOURCE PHYSICAL PROPERTIES".format(__version__),
        "=" * 72,
        f"Source: {source_label}",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S %Z')}",
        "",
        "CONVENTIONS",
        "-" * 72,
        "  S_nu ~ nu^alpha, alpha NEGATIVE for optically thin synchrotron.",
        "  alpha_positive = -alpha is also quoted throughout.",
        "  N(E) ~ E^-s ;  alpha_inj_positive = (s-1)/2 ;  s = 1 - 2*alpha.",
        f"  zeta = u_B/u_particles at equipartition = 1 + kappa = "
        f"{1.0 + float(config.get('kappa', 0.0)):.4g}"
        f"  (kappa = {config.get('kappa', 0.0):.4g}).",
        "",
        "GEOMETRY AND VOLUME",
        "-" * 72,
        f"  Solid-model classification    : {geometry.get('geometry')}",
        f"  Adopted volume estimator      : {geometry.get('volume_estimator')}",
        f"  Volume                        : "
        f"{geometry.get('adopted_volume_m3'):.6e} m^3  "
        f"({geometry.get('adopted_volume_kpc3'):.6g} kpc^3)",
        f"  Volume fractional uncertainty : "
        f"{geometry.get('volume_fractional_error'):.3f}"
        f"  (from {len(geometry.get('volume_realisations_arcsec3', {}))} "
        f"independent realisations)",
        f"  Filling factor assumed        : {geometry.get('filling_factor')}",
        f"  Largest linear size           : "
        f"{geometry.get('largest_linear_size_kpc'):.4g} kpc",
        f"  Minor axis                    : "
        f"{geometry.get('minor_axis_kpc'):.4g} kpc",
        f"  Projected area                : "
        f"{geometry.get('projected_area_kpc2'):.4g} kpc^2",
        f"  Independent beams in source   : "
        f"{geometry.get('beams_per_source'):.1f}",
        "",
        "  Individual volume realisations (arcsec^3):",
    ]
    for key, value in sorted(
        geometry.get("volume_realisations_arcsec3", {}).items()
    ):
        lines.append(f"      {key:38s} {value:.6e}")
    lines += ["", f"  {geometry.get('assumption_note')}", ""]

    lines += ["OBSERVED SPECTRUM (model-independent)", "-" * 72]
    lines.append(
        f"  Bands / significant detections : {observed.get('n_bands')} / "
        f"{observed.get('n_detections')}"
    )
    if np.isfinite(observed.get("alpha_all", np.nan)):
        lines.append(
            f"  alpha (all bands)              : {observed['alpha_all']:+.4f} "
            f"+/- {observed['alpha_all_err']:.4f}   "
            f"(alpha_positive = {-observed['alpha_all']:.4f}, "
            f"chi2_red = {observed.get('chi2_red_all', float('nan')):.2f})"
        )
    if np.isfinite(observed.get("alpha_low", np.nan)):
        lines.append(
            f"  alpha (3 lowest bands)         : {observed['alpha_low']:+.4f} "
            f"+/- {observed['alpha_low_err']:.4f}"
        )
    if "curvature" in observed:
        lines.append(
            f"  Spectral curvature             : {observed['curvature']:+.4f} "
            f"+/- {observed['curvature_err']:.4f}  "
            f"({observed['curvature_significance']:.1f} sigma)"
        )
        lines.append(f"      {observed.get('curvature_interpretation','')}")
    for tp in observed.get("two_point_indices", []):
        lines.append(
            f"      {tp['nu_low_GHz']:8.3f} - {tp['nu_high_GHz']:8.3f} GHz : "
            f"alpha = {tp['alpha']:+.4f} +/- {tp['alpha_err']:.4f}"
        )
    if "alpha_inj_upper_bound_signed" in observed:
        lines.append(
            f"  Model-free bound on alpha_inj  : flatter than or equal to "
            f"{observed['alpha_inj_upper_bound_signed']:+.4f} "
            f"(alpha_inj_positive <= "
            f"{observed['alpha_inj_upper_bound_positive']:.4f})"
        )
    lines.append("")

    lines += ["INJECTION INDEX", "-" * 72]
    robust = (ensemble or {}).get("robust_injection_index", {}) if ensemble else {}

    # When --alpha-inj was supplied, THAT is the number the equipartition
    # solve, the resolved maps and BRATS all ran on, and it must be the first
    # thing in this section. The profile-scan value below is then a
    # diagnostic -- what the SED alone would have preferred -- and the two
    # routinely disagree. Printing only the scan value, as this section used
    # to, left the file contradicting its own HEADLINE RESULTS block with
    # nothing to say which had been used.
    if solution is not None and solution.get("alpha_source") == "user_override":
        lines += [
            f"  alpha_inj USED (positive)      : "
            f"{solution.get('alpha_inj_positive', float('nan')):.4f}"
            f" +/- {solution.get('alpha_inj_err', float('nan')):.4f}",
            f"  alpha_inj USED (signed)        : "
            f"{solution.get('alpha_inj_signed', float('nan')):+.4f}",
            "  Provenance                     : fixed by the user "
            "(--alpha-inj); NOT fitted",
            "",
            "    This is the value the equipartition solution, the resolved",
            "    ageing maps and BRATS were all run with. Everything below in",
            "    this section is a DIAGNOSTIC: it is what the integrated SED",
            "    on its own would have preferred, and it is free to disagree.",
            "",
        ]

    if robust.get("alpha_inj_radio") is not None:
        lines += [
            f"  alpha_inj (signed)             : "
            f"{robust['alpha_inj_radio']:+.4f}",
            f"  alpha_inj (positive)           : "
            f"{robust['alpha_inj_positive']:.4f}",
            f"  Injection energy index s       : "
            f"{robust['injection_index_s']:.4f}",
            f"  Statistical uncertainty        : "
            f"{robust.get('alpha_inj_stat_err', float('nan')):.4f}",
            f"  Model systematic               : "
            f"{robust.get('alpha_inj_model_systematic', float('nan')):.4f}",
            f"  Total uncertainty              : "
            f"{robust.get('alpha_inj_radio_err', float('nan')):.4f}",
            f"  Method                         : {robust.get('method')}",
            f"  Confidence class               : {robust.get('confidence')}",
        ]
        if robust.get("models_used"):
            lines.append(
                f"  Models averaged                : "
                f"{', '.join(robust['models_used'])}"
            )
        if robust.get("warning"):
            lines += ["", "  WARNING:"]
            lines += [
                "    " + w
                for w in textwrap.wrap(robust["warning"], width=68)
            ]
    else:
        lines.append("  Not determined (no ageing backend or no usable fit).")
    lines.append("")

    if ensemble and ensemble.get("models"):
        lines += ["MODEL ENSEMBLE", "-" * 72,
                  f"  {'Model':<14}{'s':>9}{'+/-':>8}{'log nu_b':>11}"
                  f"{'chi2_r':>9}{'AICc':>10}{'w':>7}  status"]
        akaike = ensemble.get("free_s_consensus", {}).get("akaike_weights", {})
        for name in SPECTRAL_AGE_MODELS:
            record = ensemble["models"].get(name)
            if not record:
                continue
            if not record.get("fit_success"):
                lines.append(f"  {name:<14}{'--':>9}{'--':>8}{'--':>11}"
                             f"{'--':>9}{'--':>10}{'--':>7}  {record.get('status')}")
                continue
            # dict.get's default does NOT fire when the key is present and
            # holds None, which is exactly what AICc is whenever the model is
            # underconstrained (n <= k + 1) -- and what it becomes for every
            # NaN once these results have been through JSON. Coerce instead.
            aicc = record.get("AICc")
            aicc = float("nan") if aicc is None else float(aicc)
            lines.append(
                f"  {name:<14}{record['injection_index_s']:>9.3f}"
                f"{record['injection_index_s_err']:>8.3f}"
                f"{record['log10_break_frequency_Hz']:>11.3f}"
                f"{float(record.get('reduced_chi2') or float('nan')):>9.2f}"
                f"{aicc if np.isfinite(aicc) else float('nan'):>10.2f}"
                f"{akaike.get(name, 0.0):>7.2f}  {record.get('fit_quality','')}"
            )
            if record.get("spectral_age"):
                age = record["spectral_age"]
                lines.append(
                    f"      spectral age = {age['tau_Myr']:.3g} +/- "
                    f"{age.get('tau_err_Myr', float('nan')):.2g} Myr "
                    f"(t_on = {age['t_on_Myr']:.3g}, "
                    f"t_off = {age['t_off_Myr']:.3g} Myr)"
                )
        restricted = ensemble.get("restricted_consensus")
        if restricted:
            lines += [
                "",
                f"  Fixed-s consensus break        : "
                f"log10(nu_b/Hz) = "
                f"{restricted['log10_break_frequency_Hz']:.3f} +/- "
                f"{restricted['log10_break_frequency_stat_err']:.3f} (stat) "
                f"+/- {restricted['log10_break_frequency_model_systematic']:.3f}"
                f" (model)",
                f"  Preferred model (fixed s)      : {restricted['best_model']}",
            ]
        lines.append("")

    if solution:
        eq = solution["equipartition"]
        head = eq["headline"]
        qa = eq["internal_consistency"]
        mc = eq.get("monte_carlo", {})
        lines += [
            "MAGNETIC FIELD AND ENERGY CONTENT",
            "-" * 72,
            f"  Electron spectrum used         : "
            f"{solution['electron_spectrum']}"
            + ("   ** FALLBACK -- the aged spectrum could not be normalised **"
               if solution.get("spectrum_fallback") else ""),
            # The rationale describes the spectrum that was REQUESTED. Under a
            # fallback it is not the one used, and printed bare beneath
            # "powerlaw" it read as though the aged spectrum had been applied.
            (f"      requested: {solution['electron_spectrum_rationale']}"
             if solution.get("spectrum_fallback")
             else f"      {solution['electron_spectrum_rationale']}"),
            *([f"      reason   : {solution['spectrum_fallback'].get('reason')}"]
              if solution.get("spectrum_fallback") else []),
            f"  Self-consistent loop           : "
            f"{'converged' if solution['converged'] else 'NOT converged'} "
            f"in {solution['n_iterations']} iterations",
            "",
            f"  B_equipartition                : {head['B_eq_uG']:.5g} uG"
            + (f"  [{mc['B_eq_uG']['p16']:.4g}, {mc['B_eq_uG']['p84']:.4g}] "
               f"(68% CI)" if mc.get("status") == "ok" else ""),
            f"  B_minimum-energy               : "
            f"{head['B_min_energy_uG']:.5g} uG",
            f"  B_eq / B_min-energy            : "
            f"{qa['B_eq_over_B_me_measured']:.5f}  "
            f"(analytic {qa['B_eq_over_B_me_analytic']:.5f}, "
            f"classical s=2 value {qa['B_eq_over_B_me_classical_p2']:.5f})",
            "",
            f"  u_e   (equipartition)          : "
            f"{head['u_e_equipartition_J_m3']:.6e} J m^-3",
            f"  u_B   (equipartition)          : "
            f"{head['u_B_equipartition_J_m3']:.6e} J m^-3",
            f"  u_tot (equipartition)          : "
            f"{head['u_total_equipartition_J_m3']:.6e} J m^-3",
            "",
            f"  u_e   (minimum energy)         : "
            f"{head['u_e_minimum_energy_J_m3']:.6e} J m^-3",
            f"  u_B   (minimum energy)         : "
            f"{head['u_B_minimum_energy_J_m3']:.6e} J m^-3",
            f"  u_min (MINIMUM ENERGY DENSITY) : {head['u_min_J_m3']:.6e} J m^-3"
            + (f"  [{mc['u_min']['p16']:.4e}, {mc['u_min']['p84']:.4e}]"
               if mc.get("status") == "ok" else ""),
            "",
            f"  Total energy at minimum        : {head['E_min_J']:.6e} J",
            f"  Minimum pressure p = u_min/3   : {head['p_min_Pa']:.6e} Pa "
            f"({head['p_min_dyn_cm2']:.6e} dyn cm^-2)",
            f"  Volume used                    : {head['volume_m3']:.6e} m^3",
            "",
        ]

    # Which field the analysis actually RAN at. B_eq above is the derived
    # equipartition value; when --analysis-at-sub-equipartition is set, every
    # age in this run -- resolved maps, BRATS, everything -- was computed at a
    # FRACTION of it, and a reader given only B_eq would reasonably assume the
    # ages belong to that field. They do not, and ages are not transferable
    # between fields, so the number actually used has to appear here.
    sub = (solution or {}).get("sub_equipartition") if solution else None
    if sub and not sub.get("applied_to_analysis", False):
        b_eq_used = (solution.get("equipartition", {}).get("direct", {})
                     .get("equipartition", {}).get("B_uG"))
        lines += [
            "  FIELD USED FOR THE ANALYSIS",
            f"    B = B_eq                     : {b_eq_used:.4f} uG"
            if b_eq_used is not None else "    B = B_eq",
            "    The resolved ageing maps and BRATS were run at the FULL",
            "    equipartition field. The block below is a what-if budget at",
            f"    {sub['fraction_of_B_eq']:g} B_eq; no map or BRATS age in this "
            "run used it.",
            "    (--analysis-at-sub-equipartition applies it to the analysis.)",
            "",
        ]
    if sub:
        lines += [
            ("  FIELD USED FOR THE ANALYSIS" if sub.get("applied_to_analysis")
             else f"  SUB-EQUIPARTITION BUDGET ({sub['fraction_of_B_eq']:g} B_eq,"
                  " reported only)"),
            f"    B = {sub['fraction_of_B_eq']:g} B_eq            "
            f"     : {sub['B_uG']:.4f} uG",
            f"    u_tot at that field          : "
            f"{sub['total_energy_density_J_m3']:.6e} J m^-3",
            f"    pressure at that field       : {sub['pressure_Pa']:.6e} Pa",
            f"    B_CMB                        : {sub['B_CMB_uG']:.4f} uG "
            f"(B/B_CMB = {sub['B_over_B_CMB']:.4f})",
            f"    B_CMB/sqrt(3) (age maximum)  : "
            f"{sub['B_CMB_over_sqrt3_uG']:.4f} uG",
        ]
        if sub.get("inverse_compton_dominated"):
            lines.append(
                "    NOTE: this field is BELOW the age maximum, so inverse-"
                "Compton"
            )
            lines.append(
                "          losses on the CMB dominate and the age is shorter "
                "here."
            )
        # The integrated age quoted in this record inherits the ensemble's
        # break frequency. When that break railed at the edge of its prior --
        # which is what happens when the observing bands do not bracket it --
        # the age is a projection far outside the data and must not be read
        # as a measurement. Say so next to the number rather than in a note
        # somewhere else.
        age_sub = sub.get("spectral_age_Myr")
        nub = sub.get("break_frequency_Hz")
        if age_sub is not None and nub:
            lines.append(
                f"    integrated age at B          : {age_sub:.4g} Myr "
                f"(from nu_b = {nub:.4g} Hz)"
            )
            lines.append(
                "          CONDITIONAL on that break. Compare the resolved "
                "and BRATS"
            )
            lines.append(
                "          ages before quoting it; a break above the highest "
                "band is"
            )
            lines.append(
                "          an extrapolation, not a measurement."
            )
        lines.append("")

    if solution and solution.get("equipartition"):
        lines += [
            "  INTERNAL CONSISTENCY CHECKS",
            f"    closed form vs pysynch B_eq  : "
            f"{qa['B_eq_relative_difference']*100:.3f}% "
            f"({'PASS' if qa['closed_form_validated'] else 'FAIL'})",
            f"    B_me < B_eq                  : "
            f"{'PASS' if qa['minimum_energy_below_equipartition'] else 'FAIL'}",
            f"    u_min <= u_eq                : "
            f"{'PASS' if qa['u_min_not_above_u_eq'] else 'FAIL'} "
            f"(ratio {qa['u_min_over_u_eq']:.5f})",
            f"    B_eq/B_me matches analytic   : "
            f"{'PASS' if qa['ratio_consistent'] else 'CHECK'}",
            f"    measured u_e(B) exponent n   : "
            f"{eq['ue_scaling']['n']:.4f} "
            f"(textbook (s+1)/2 = {eq['ue_scaling']['n_textbook']:.4f})",
            "",
        ]
        gs = eq.get("gamma_sensitivity", {}).get("summary", {})
        if gs.get("max_ratio"):
            lines.append(
                f"  gamma_min/gamma_max sensitivity: B_eq varies by a factor "
                f"{gs['min_ratio']:.3f} - {gs['max_ratio']:.3f} over the "
                f"tested range"
            )
        if solution.get("aged_vs_powerlaw_B_ratio"):
            lines.append(
                f"  Aged vs power-law spectrum     : B_eq(aged)/B_eq(power law)"
                f" = {solution['aged_vs_powerlaw_B_ratio']:.4f}"
            )
        lines.append("")

    lines += [
        "SCIENTIFIC QUALIFICATIONS",
        "-" * 72,
    ]
    for note in (
        "An observed low-frequency spectral index is not by itself an "
        "injection index; only a model fit that is formally identifiable at "
        "the available number of frequencies can measure one.",
        "The line-of-sight depth is a geometric model assumption, not a "
        "measurement, and it dominates the uncertainty on the total energy.",
        "gamma_min is unconstrained by these data and directly scales the "
        "electron energy density; the assumed value is recorded above.",
        "zeta = 1 + kappa assumes a specific non-radiating particle content. "
        "kappa = 0 (electron-positron) and kappa = 100 (classical "
        "proton-dominated) differ in B_eq by roughly a factor of three.",
        "Calibration systematics are coherent across each map: they shift the "
        "whole spectral-index map together and must not be used when judging "
        "spatial structure within it.",
        "Adjacent pixels in every map are correlated on the scale of the "
        "common beam.",
    ):
        lines += textwrap.wrap(note, width=72, initial_indent="  * ",
                               subsequent_indent="    ")
    output.write_text("\n".join(lines) + "\n")


# ===========================================================================
# PART 17.  COMMAND LINE AND INTERACTIVE INPUT
# ===========================================================================

def parse_label_values(items: Sequence[str], cast) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Expected LABEL=VALUE, got {item!r}")
        key, value = item.split("=", 1)
        result[key.strip()] = cast(value.strip())
    return result


def parse_images(items: Sequence[str]) -> List[Tuple[str, Path]]:
    result: List[Tuple[str, Path]] = []
    for item in items:
        if "=" not in item:
            raise ValueError(f"Expected LABEL=FITS, got {item!r}")
        label_, filename = item.split("=", 1)
        path = Path(filename).expanduser()
        if not path.is_file():
            raise FileNotFoundError(path)
        result.append((label_.strip(), path))
    return result


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="PySynch.py",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "Publication-grade radio-source synchrotron analysis: common-beam "
            "regridding, integrated SEDs, a six-model spectral-ageing "
            "ensemble, equipartition and minimum-energy fields, and resolved "
            "spectral-index / ageing maps."
        ),
        epilog=textwrap.dedent("""\
            Examples
            --------
            Four-band analysis with the full ensemble and resolved maps:

              python PySynch.py \\
                --image L150=lofar_150.fits --image G400=ugmrt_400.fits \\
                --image G650=ugmrt_650.fits --image L1400=vla_1400.fits \\
                --reference-label L1400 --redshift 0.1469 \\
                --resolved-ageing --output-dir Products

            Reproduce a classical proton-dominated equipartition field:

              ... --kappa 100

            Quick iteration (no Monte Carlo, no resolved ageing):

              ... --beq-mc 0
        """),
    )

    g = p.add_argument_group("input")
    g.add_argument("--image", action="append", default=[], metavar="LABEL=FITS",
                   help="Cropped radio FITS image; repeat for every frequency.")
    g.add_argument("--rms", action="append", default=[], metavar="LABEL=JYBEAM",
                   help="Override the automatic noise estimate for a map.")
    g.add_argument("--cal-frac", action="append", default=[], metavar="LABEL=FRAC",
                   help="Override the flux-scale systematic for a map (e.g. 0.05).")
    g.add_argument("--no-header-rms", action="store_true",
                   help="Ignore RMS/NOISE keywords and always measure the noise.")
    g.add_argument("--astrometry-rms", action="append", default=[],
                   metavar="LABEL=ARCSEC",
                   help="Measured absolute astrometric RMS, used to pick the reference.")
    g.add_argument("--reference-label",
                   help="Explicitly nominate the astrometric reference map.")
    g.add_argument("--show-error-provenance", action="store_true",
                   help="Print how every noise and calibration value was obtained.")

    g = p.add_argument_group("source and cosmology")
    g.add_argument("--redshift", type=float, default=DEFAULT_REDSHIFT)
    g.add_argument("--h0", type=float, default=70.0, help="H0 in km/s/Mpc.")
    g.add_argument("--om0", type=float, default=0.3)
    g.add_argument(
        "--detection-map", type=str, default=None,
        help=(
            "Label of the band used to seed source segmentation. Default: "
            "whichever band has the best peak signal-to-noise (not "
            "necessarily the lowest frequency) -- a shallow low-frequency "
            "map otherwise builds a mask that is too tight and shorts its "
            "own integrated flux."
        ),
    )
    g.add_argument("--source-sigma", type=float, default=DEFAULT_SOURCE_SIGMA,
                   help="Seed threshold for hysteresis segmentation.")
    g.add_argument("--source-grow-sigma", type=float, default=DEFAULT_GROW_SIGMA,
                   help="Growth threshold; forced below the seed threshold.")
    g.add_argument("--no-fill-holes", action="store_true",
                   help="Do not fill interior holes in the source mask.")
    g.add_argument("--no-background-subtraction", action="store_true",
                   help="Do not subtract the measured annulus zero level.")

    g = p.add_argument_group("spectral-index maps")
    g.add_argument("--spectral-snr", type=float, default=DEFAULT_MIN_SNR_MAP)
    g.add_argument("--spectral-min-bands", type=int, default=DEFAULT_MIN_FIT_BANDS)
    g.add_argument("--mc-alpha-maps", type=int, default=0, metavar="N",
                   help="Monte-Carlo realisations for unbiased alpha error maps.")
    g.add_argument("--no-curvature-map", action="store_true")

    g = p.add_argument_group("spectral-ageing models")
    g.add_argument("--no-synchrofit", action="store_true",
                   help="Disable all synchrofit model fitting.")
    g.add_argument("--no-ensemble", action="store_true",
                   help="Fit only the single selected model instead of all six.")
    g.add_argument("--sed-model", choices=list(MODEL_MENU), default="Ensemble")
    g.add_argument("--injection-model", choices=list(MODEL_MENU), default="Ensemble")
    g.add_argument("--spectral-map-model", choices=list(MODEL_MENU), default="Power-law")
    g.add_argument("--inject-range", type=float, nargs=2,
                   default=list(DEFAULT_INJECT_RANGE), metavar=("S_MIN", "S_MAX"),
                   help="Prior on the injection ENERGY index s (default 2.01 3.00).")
    g.add_argument("--break-range", type=float, nargs=2,
                   default=list(DEFAULT_BREAK_RANGE),
                   metavar=("LOG_MIN", "LOG_MAX"),
                   help="Prior on log10(nu_break/Hz).")
    g.add_argument("--model-n-breaks", type=int, default=31)
    g.add_argument("--model-n-injects", type=int, default=31)
    g.add_argument("--model-n-remnants", type=int, default=21)
    g.add_argument("--model-iterations", type=int, default=3)
    g.add_argument(
        "--ensemble-models", nargs="+", default=None,
        choices=list(SPECTRAL_AGE_MODELS),
        help=(
            "Models fitted in the ageing ensemble (default: all six). The two "
            "Tribble variants dominate the runtime and barely respond to a "
            "coarser parameter grid; drop them here when the data cannot "
            "distinguish them anyway."
        ),
    )
    g.add_argument("--no-scan-injection", action="store_true",
                   help="Skip the profile-likelihood scan in the injection index.")
    g.add_argument("--scan-models", nargs="+", default=["JP", "CI"],
                   help="Models used for the injection-index profile scan.")
    g.add_argument("--n-scan", type=int, default=9)
    g.add_argument("--posterior-grid", action="store_true",
                   help="Force a single wide grid so posterior marginals span the prior.")
    g.add_argument("--require-robust-model-injection", action="store_true",
                   help="Fail unless an identifiable model-based injection index exists.")

    g = p.add_argument_group("equipartition and energetics")
    g.add_argument("--no-equipartition", action="store_true",
                   help="Skip all pysynch calculations.")
    g.add_argument("--gamma-min", type=float, default=DEFAULT_GAMMA_MIN)
    g.add_argument("--gamma-max", type=float, default=DEFAULT_GAMMA_MAX)
    g.add_argument("--kappa", type=float, default=DEFAULT_KAPPA,
                   help=("Proton-to-electron energy ratio. zeta = 1 + kappa. "
                         "0 = electron-positron (default); 100 = classical."))
    g.add_argument("--zeta", type=float, default=None,
                   help="Set zeta = u_B/u_particles directly (overrides --kappa).")
    g.add_argument("--filling-factor", type=float, default=DEFAULT_FILLING_FACTOR)
    g.add_argument("--alpha-inj", type=float,
                   help="Override the injection index (signed, e.g. -0.7).")
    g.add_argument("--alpha-inj-err", type=float)
    g.add_argument("--no-aged-spectrum", action="store_true",
                   help="Always normalise with an unbroken power law.")
    g.add_argument(
        "--analysis-at-sub-equipartition", action="store_true",
        help=(
            "Use B = F x B_eq (F from --b-field-fraction) as the magnetic "
            "field for the WHOLE analysis -- BRATS model fitting and the "
            "resolved ageing maps -- instead of only reporting the energy "
            "budget at it. Spectral ages scale strongly with the assumed "
            "field, so this changes every age in the run, not just the "
            "quoted energetics."
        ),
    )
    g.add_argument(
        "--b-field-fraction", type=float, default=0.4, metavar="F",
        help=("Also report the full energy budget at a fixed "
              "sub-equipartition field B = F x B_eq (default 0.4, as "
              "commonly indicated by inverse-Compton measurements). u_B, "
              "u_e, pressure and the spectral age are all recomputed at "
              "that field with the observed flux held fixed. Set 0 to "
              "disable."),
    )
    g.add_argument("--self-consistent-iterations", type=int, default=6)
    g.add_argument(
        "--self-consistent-damping", type=float, default=0.5, metavar="W",
        help=("Under-relaxation for the alpha_inj <-> B loop (default 0.5). "
              "Each new field is combined with the previous one in log space "
              "as B = B_prev^(1-W) B_new^W. The loop can otherwise oscillate "
              "indefinitely, because alpha_inj is fitted on a discrete grid "
              "and successive iterations snap between neighbouring grid "
              "points. W = 1 disables damping and reproduces the old "
              "behaviour; smaller values damp harder but converge slower."),
    )
    g.add_argument("--beq-mc", type=int, default=DEFAULT_BEQ_MC,
                   help="Monte-Carlo samples for the field error budget (0 disables).")
    g.add_argument("--alpha-bootstrap", type=int, default=DEFAULT_ALPHA_BOOTSTRAP)

    g = p.add_argument_group("fitting engine (synchrofit / BRATS)")
    g.add_argument(
        "--engine", choices=["synchrofit", "brats", "both"], default="synchrofit",
        help=("Which spectral-ageing backend to use. 'both' runs each "
              "independently over the same maps and cross-compares them, "
              "which is the most robust option."),
    )
    g.add_argument("--brats-path", help="Full path to the BRATS executable.")
    g.add_argument(
        "--brats-container", default="auto",
        choices=["auto", "none", "docker", "podman", "apptainer", "singularity"],
        help=("Run BRATS inside a container. 'auto' (default) uses a native "
              "BRATS if one is found and otherwise falls back to whichever "
              "container runtime is working. On macOS this is normally the "
              "only way to run BRATS, because PGPLOT and FUNTOOLS have no "
              "maintained build there; a Dockerfile is written into the "
              "workspace for you."),
    )
    g.add_argument(
        "--brats-image", default=None,
        help=("Container image holding BRATS (default: brats:latest, or the "
              "BRATS_IMAGE environment variable). Build it from the "
              "Dockerfile written into the BRATS workspace."),
    )
    g.add_argument(
        "--brats-render", dest="brats_render", action="store_true", default=True,
        help=("Paint the BRATS per-region results back onto the sky and write "
              "FITS maps and figures with reference contours (default)."),
    )
    g.add_argument(
        "--no-brats-render", dest="brats_render", action="store_false",
        help="Keep the raw BRATS exports only; do not build maps or figures.",
    )
    g.add_argument(
        "--open-brats", action="store_true",
        help=("When everything else has finished and all products are "
              "written, open BRATS in a NEW terminal window sitting in the "
              "prepared workspace, so you can carry on interactively. If the "
              "scripted engine already ran, the session is pointed at the "
              ".brats backup it saved, so its fits come back with a single "
              "fullimport instead of being recomputed. Works with any "
              "--engine: with synchrofit alone the workspace is built at that "
              "point using the injection index and field this run measured."),
    )
    g.add_argument(
        "--brats-interactive", action="store_true",
        help=("Prepare a complete BRATS workspace and open BRATS in a NEW "
              "terminal window instead of scripting it, so you can drive its "
              "interactive PGPLOT session yourself."),
    )
    g.add_argument(
        "--brats-models", nargs="+",
        default=["JP", "KP", "JP_Tribble", "CI", "CI-OFF"],
        help="Models to fit with BRATS (KP_Tribble is unsupported by BRATS).",
    )
    g.add_argument("--brats-no-maps", action="store_true",
                   help="Skip BRATS resolved (map) fitting; integrated only.")
    g.add_argument("--brats-no-integrated", action="store_true",
                   help="Skip BRATS integrated fitting; maps only.")
    g.add_argument("--brats-findinject", action="store_true",
                   help="Run BRATS findinject chi-square minimisation (SLOW).")
    g.add_argument("--brats-inject-min", type=float, default=0.5)
    g.add_argument("--brats-inject-max", type=float, default=1.0)
    g.add_argument("--brats-inject-intervals", type=int, default=10)
    g.add_argument("--brats-max-myears", type=float, default=50.0)
    g.add_argument("--brats-levels", type=int, default=3,
                   help="BRATS age search depth (3 testing, 5 for final results).")
    g.add_argument("--brats-ageres", type=int, default=10)
    g.add_argument("--brats-onsource", type=float, default=3.0,
                   help="On-source noise multiplier used by BRATS.")
    g.add_argument(
        "--brats-searcharea", type=float, default=None,
        help=("Maximum number of growth steps an adaptive region may take "
              "before BRATS gives up on it (BRATS default 1). This MUST be "
              "raised whenever --brats-signaltonoise is raised: the growth "
              "loop is `while (S/N < target) { if (steps >= searcharea) "
              "break; grow; }`, so at 1 a region cannot grow at all and the "
              "S/N target collapses into a per-pixel cut. Left unset, it is "
              "scaled to three beam areas when a target above 1 is asked "
              "for."),
    )
    g.add_argument("--brats-signaltonoise", type=float, default=1.0)
    g.add_argument("--brats-source-region", type=Path,
                   help="Your own DS9 source region instead of the automatic box.")
    g.add_argument("--brats-background-region", type=Path,
                   help="Your own DS9 background region instead of the automatic box.")
    g.add_argument("--brats-target-name", default="PYSYNCH_SOURCE")
    g.add_argument("--brats-timeout", type=float, default=None,
                   help="Seconds before a scripted BRATS run is abandoned.")
    g.add_argument("--brats-keep-own-noise", action="store_true",
                   help="Let BRATS derive its own noise instead of using ours.")

    g = p.add_argument_group("figures and contour overlays")
    g.add_argument(
        "--contour-label", default=None,
        help=("Label of the map whose total-intensity contours are drawn over "
              "every derived map (spectral index, ages, errors). Defaults to "
              "the astrometric reference map."),
    )
    g.add_argument(
        "--contour-sigma", type=float, default=5.0,
        help=("Contour level in units of the reference map RMS (default 5). "
              "With --contour-max-levels 1 this draws a SINGLE 5-sigma "
              "contour, which is the cleanest way to show where the source "
              "is without the ladder obscuring the map underneath."),
    )
    g.add_argument(
        "--contour-factor", type=float, default=2.0,
        help="Ratio between successive contour levels (default 2).",
    )
    g.add_argument(
        "--contour-levels-sigma", type=float, nargs="+", default=None,
        metavar="SIGMA",
        help=(
            "Explicit contour ladder, as multiples of the reference map's "
            "rms, e.g. --contour-levels-sigma 20 40 80 160 320 640 720. "
            "Overrides --contour-sigma / --contour-factor / "
            "--contour-max-levels / --contour-drop-lowest for the DRAWN "
            "levels; the contour mask stays anchored on --contour-sigma. "
            "Levels above the map peak are dropped."
        ),
    )
    g.add_argument(
        "--contour-drop-lowest", type=int, default=2,
        help=(
            "Discard this many levels from the bottom of the contour ladder. "
            "The lowest level is the detection isophote and traces almost "
            "exactly the same curve as the segmentation boundary, so on a "
            "derived map it reads as that boundary stamped over the data. "
            "--contour-drop-lowest 1 removes it; the ladder stays anchored "
            "on --contour-sigma and the level count is made up from the top."
        ),
    )
    g.add_argument(
        "--contour-max-levels", type=int, default=5,
        help=("Number of contour levels (default 5). Levels are the standard "
              "radio ladder from --contour-sigma upward, doubling each time: "
              "5, 10, 20, 40, 80 sigma. Set 1 for a single contour."),
    )
    g.add_argument(
        "--no-contours", action="store_true",
        help="Do not overlay reference contours on the derived maps.",
    )
    g.add_argument(
        "--contour-annotate", action="store_true",
        help=("Stamp the contour provenance inside the map axes. Off by "
              "default: in a paper that belongs in the figure caption, and a "
              "text box over the map is a common referee objection. The "
              "levels are always recorded in the report and the JSON."),
    )
    g.add_argument(
        "--no-spectral-maps", action="store_true",
        help=("Skip the resolved spectral-index, curvature and ageing maps "
              "entirely, and produce only the integrated products. Use this "
              "for a band set that includes a low-resolution survey map "
              "(TGSS at 25-37 arcsec, say): such a band is valuable for the "
              "integrated SED, where only total flux matters, but it forces "
              "the common beam so coarse that a per-pixel spectral index is "
              "dominated by its PSF rather than by the source."),
    )

    g = p.add_argument_group("resolved ageing maps")
    g.add_argument("--resolved-ageing", action="store_true",
                   help="Adaptively bin the source and fit ageing models per region.")
    g.add_argument("--resolved-snr", type=float, default=10.0,
                   help="Target signal-to-noise per region in the worst band.")
    g.add_argument("--min-beams-per-region", type=float, default=1.0)
    g.add_argument("--template-nodes", type=int, default=60)

    g = p.add_argument_group("run control")
    g.add_argument("--thorough", action="store_true",
                   help="Raise all Monte-Carlo counts and enable B-marginalised "
                        "Tribble fits and posterior grids.")
    g.add_argument("--seed", type=int, default=DEFAULT_SEED)
    g.add_argument("--output-dir", type=Path, default=Path("PySynch_Products"))
    g.add_argument("--keep-stale", action="store_true",
                   help="Do not delete products from a previous run.")
    g.add_argument("--version", action="version",
                   version=f"PySynch {__version__}")
    return p


PRODUCT_FILENAMES = {
    "input_FITS_header_summary.json", "common_beam_noise_summary.json",
    "integrated_SED.csv", "PySynch_results.json", "PySynch_report.txt",
    "source_properties.txt", "source_mask.fits", "depth_map.fits",
    "spectral_index.fits", "spectral_index_error_stat.fits",
    "spectral_index_error_sys.fits", "spectral_index_error_total.fits",
    "spectral_index_chi2.fits", "spectral_index_nbands.fits",
    "spectral_curvature.fits", "spectral_curvature_error.fits",
    "spectral_index_error_mc.fits",
    "resolved_age.fits", "resolved_age_error.fits",
    "resolved_break_frequency.fits", "resolved_Beq.fits",
    "resolved_pressure.fits", "resolved_chi2.fits",
    "radio_SED.png", "radio_SED_ensemble.png",
    "injection_index_scan.png", "model_comparison.png",
    "volume_emissivity_vs_rest_frequency.png", "energy_budget.png",
    "spectral_index.png", "spectral_index_error_stat.png",
    "spectral_index_error_total.png", "spectral_curvature.png",
    "spectral_index_chi2.png", "spectral_index_nbands.png",
    "source_segmentation_QA.png", "resolved_age.png",
    "resolved_break_frequency.png", "resolved_Beq.png",
    "resolved_pressure.png", "resolved_regions.csv",
    "engine_comparison.json", "BRATS_summary.txt",
}


def clean_products(outdir: Path) -> None:
    """Remove products of a previous run so stale files cannot be mistaken for new."""
    for name in PRODUCT_FILENAMES:
        path = outdir / name
        if path.is_file():
            path.unlink()
    for pattern in ("*_commonbeam.png", "component_*"):
        for path in outdir.glob(pattern):
            if path.is_file():
                path.unlink()
            elif path.is_dir():
                shutil.rmtree(path, ignore_errors=True)
    common = outdir / "common_beam_FITS"
    if common.exists():
        shutil.rmtree(common, ignore_errors=True)


def _safe_header_summary(path: Path) -> Dict[str, Any]:
    with fits.open(path, memmap=False) as hdul:
        data = np.squeeze(np.asarray(hdul[0].data, dtype=float))
        header = hdul[0].header.copy()
    if data.ndim != 2:
        raise RuntimeError(
            f"{path.name}: expected a 2-D image after squeezing; got {data.shape}."
        )
    wcs = WCS(header).celestial
    beam = beam_from_header(header)
    freq = frequency_from_header(header)
    return {
        "path": path.resolve(),
        "shape": tuple(data.shape),
        "frequency_GHz": freq / 1e9,
        "frequency_Hz": freq,
        "beam_major_arcsec": beam.major.to_value(u.arcsec),
        "beam_minor_arcsec": beam.minor.to_value(u.arcsec),
        "beam_pa_deg": beam.pa.to_value(u.deg),
        "bunit": str(header.get("BUNIT", "")).strip(),
        "object": str(header.get("OBJECT", "")).strip(),
        "pixel_scale_arcsec": pixel_scales_arcsec(wcs),
        "date_obs": str(header.get("DATE-OBS", "")),
        "telescope": str(header.get("TELESCOP", "")),
        "instrument": str(header.get("INSTRUME", "")),
    }


def _choose_fits_macos() -> List[Path]:
    applescript = (
        'set chosenFiles to choose file with prompt '
        '"Select cropped radio FITS images" with multiple selections allowed\n'
        'set outputText to ""\n'
        'repeat with f in chosenFiles\n'
        'set outputText to outputText & (POSIX path of f) & linefeed\n'
        'end repeat\n'
        'return outputText'
    )
    proc = subprocess.run(
        ["osascript", "-e", applescript], capture_output=True, text=True,
        timeout=600, check=False,
    )
    if proc.returncode != 0:
        err = (proc.stderr or "").strip()
        if "User canceled" in err:
            raise SystemExit(0)
        raise RuntimeError(f"macOS file chooser failed ({proc.returncode}): {err}")
    paths = [
        Path(x.strip()).expanduser().resolve()
        for x in (proc.stdout or "").splitlines() if x.strip()
    ]
    if not paths:
        raise RuntimeError("No FITS files were selected.")
    return paths


def _choose_fits_linux() -> List[Path]:
    if shutil.which("zenity"):
        proc = subprocess.run(
            ["zenity", "--file-selection", "--multiple", "--separator=\n",
             "--title=Select cropped radio FITS images"],
            capture_output=True, text=True, check=False,
        )
        if proc.returncode == 0:
            paths = [
                Path(x.strip()).expanduser().resolve()
                for x in proc.stdout.splitlines() if x.strip()
            ]
            if paths:
                return paths
    if shutil.which("kdialog"):
        proc = subprocess.run(
            ["kdialog", "--getopenfilename", str(Path.home()), "*.fits *.fit *.fts"],
            capture_output=True, text=True, check=False,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            return [Path(proc.stdout.strip()).expanduser().resolve()]
    raise RuntimeError("No graphical file chooser (zenity/kdialog) is available.")


def collect_inputs_interactively() -> Dict[str, Any]:
    """Fallback wizard when no --image arguments are supplied."""
    if sys.platform == "darwin":
        paths = _choose_fits_macos()
    elif sys.platform.startswith("linux"):
        paths = _choose_fits_linux()
    else:
        raise RuntimeError(
            f"Interactive FITS selection is not implemented for {sys.platform!r}. "
            "Use --image LABEL=FITS."
        )

    summaries, errors = [], []
    for path in paths:
        try:
            summaries.append(_safe_header_summary(path))
        except Exception as exc:                               # noqa: BLE001
            errors.append(f"{path.name}: {type(exc).__name__}: {exc}")
    if errors:
        raise RuntimeError("FITS validation failed:\n  " + "\n  ".join(errors))
    summaries.sort(key=lambda s: s["frequency_Hz"])

    print("\nSelected FITS images")
    print(f"{'File':<34}{'Freq (GHz)':>12}{'Beam (arcsec)':>24}{'BUNIT':>12}")
    print("-" * 84)
    for s in summaries:
        print(
            f"{Path(s['path']).name[:34]:<34}{s['frequency_GHz']:>12.6f}"
            f"{s['beam_major_arcsec']:>11.3f} x{s['beam_minor_arcsec']:>8.3f}"
            f"{str(s['bunit'])[:12]:>12}"
        )

    labels = [Path(s["path"]).stem for s in summaries]
    print("\nAstrometric reference map:")
    for i, lab in enumerate(labels, 1):
        print(f"  {i}. {lab}")
    raw = input(f"Select reference [1-{len(labels)}, default 1]: ").strip()
    try:
        ref_index = int(raw) - 1 if raw else 0
    except ValueError as exc:
        raise RuntimeError("Invalid reference selection.") from exc
    if not 0 <= ref_index < len(labels):
        raise RuntimeError("Invalid reference selection.")

    def ask_float(prompt: str, default: float) -> float:
        value = input(f"{prompt} [{default}]: ").strip()
        return float(value) if value else float(default)

    redshift = ask_float("Source redshift", DEFAULT_REDSHIFT)
    source_sigma = ask_float("Detection threshold (sigma)", DEFAULT_SOURCE_SIGMA)
    kappa = ask_float("Proton:electron energy ratio kappa", DEFAULT_KAPPA)
    resolved = input("Compute resolved ageing maps? [y/N]: ").strip().lower() in {"y", "yes"}
    raw_out = input(
        f"Output directory [{Path.cwd() / 'PySynch_Products'}]: "
    ).strip()

    return {
        "image_specs": [(Path(s["path"]).stem, Path(s["path"])) for s in summaries],
        "reference_label": labels[ref_index],
        "redshift": redshift,
        "source_sigma": source_sigma,
        "kappa": kappa,
        "resolved_ageing": resolved,
        "output_dir": (
            Path(raw_out).expanduser().resolve() if raw_out
            else (Path.cwd() / "PySynch_Products").resolve()
        ),
        "header_summaries": summaries,
    }


# ===========================================================================
# PART 18.  MAIN PIPELINE
# ===========================================================================

def _banner(text: str) -> None:
    print("\n" + text)
    print("-" * min(len(text), 78))


def main(argv: Optional[Sequence[str]] = None) -> int:
    t_start = time.time()
    args = build_parser().parse_args(argv)

    # -- inputs ------------------------------------------------------------
    if args.image:
        image_specs = parse_images(args.image)
        reference_label = args.reference_label
        redshift = float(args.redshift)
        source_sigma = float(args.source_sigma)
        kappa = float(args.kappa)
        resolved_requested = bool(args.resolved_ageing)
        outdir = Path(args.output_dir).expanduser().resolve()
    else:
        wizard = collect_inputs_interactively()
        image_specs = wizard["image_specs"]
        reference_label = wizard["reference_label"]
        redshift = float(wizard["redshift"])
        source_sigma = float(wizard["source_sigma"])
        kappa = float(wizard["kappa"])
        resolved_requested = bool(wizard["resolved_ageing"])
        outdir = Path(wizard["output_dir"])

    if len(image_specs) < 2:
        raise SystemExit(
            "At least two frequency maps are required; "
            f"{len(image_specs)} supplied."
        )

    zeta = float(args.zeta) if args.zeta is not None else 1.0 + kappa
    if zeta <= 0:
        raise SystemExit("zeta must be positive.")
    kappa_effective = zeta - 1.0

    cosmology = FlatLambdaCDM(H0=float(args.h0), Om0=float(args.om0), Tcmb0=T_CMB0)
    validate_pysynch_cosmology(cosmology)

    use_equipartition = not args.no_equipartition
    if use_equipartition:
        require_pysynch()
    # --engine brats means BRATS is the ageing backend; synchrofit is then
    # only used if the user explicitly asked for both engines.
    use_synchrofit = (
        HAVE_SYNCHROFIT
        and not args.no_synchrofit
        and args.engine in {"synchrofit", "both"}
    )

    mc_samples = int(args.beq_mc)
    if args.thorough and mc_samples > 0:
        mc_samples = max(mc_samples, DEFAULT_THOROUGH_BEQ_MC)

    cfg: Dict[str, Any] = {
        "n_breaks": int(args.model_n_breaks),
        "n_injects": int(args.model_n_injects),
        "n_remnants": int(args.model_n_remnants),
        "iterations": int(args.model_iterations),
        "inject_range": tuple(args.inject_range),
        "break_range": tuple(args.break_range),
        "scan_injection": not args.no_scan_injection,
        "scan_models": tuple(args.scan_models),
        "n_scan": int(args.n_scan),
        "ensemble_models": (
            tuple(args.ensemble_models) if args.ensemble_models else None
        ),
        "harvest_posterior": bool(args.posterior_grid or args.thorough),
        "thorough": bool(args.thorough),
        "use_synchrofit": use_synchrofit,
        "use_aged_spectrum": not args.no_aged_spectrum,
        "kappa": kappa_effective,
        "gmin": float(args.gamma_min),
        "gmax": float(args.gamma_max),
        "filling_factor": float(args.filling_factor),
        "mc_samples": mc_samples,
        "seed": int(args.seed),
        "cosmology": cosmology,
        "min_beams_per_region": float(args.min_beams_per_region),
        "template_nodes": int(args.template_nodes),
        "source_sigma": float(source_sigma),
        "brats_onsource": float(args.brats_onsource),
        "brats_searcharea": float(args.brats_searcharea or 1.0),
        "brats_signaltonoise": float(args.brats_signaltonoise),
        "brats_max_myears": float(args.brats_max_myears),
        "brats_levels": int(args.brats_levels),
        "brats_ageres": int(args.brats_ageres),
        "brats_inject_min": float(args.brats_inject_min),
        "brats_inject_max": float(args.brats_inject_max),
        "brats_inject_intervals": int(args.brats_inject_intervals),
        "brats_findinject": bool(args.brats_findinject),
        "brats_source_region": args.brats_source_region,
        "brats_background_region": args.brats_background_region,
        "brats_target_name": str(args.brats_target_name),
        "brats_set_noise": not args.brats_keep_own_noise,
    }

    outdir.mkdir(parents=True, exist_ok=True)

    # A band set too coarse for spectral-index maps is equally too coarse for
    # resolved ageing, which is fitted on the same pixels and the same beam.
    resolved_requested = resolved_requested and not args.no_spectral_maps
    if not args.keep_stale:
        clean_products(outdir)

    rms_overrides = parse_label_values(args.rms, float)
    cal_overrides = parse_label_values(args.cal_frac, float)

    _banner("1. Loading and characterising input maps")
    maps = [
        load_radio_map(
            label_=label_, path=path,
            rms_jybeam=rms_overrides.get(label_),
            cal_frac=cal_overrides.get(label_),
            trust_header_rms=not args.no_header_rms,
        )
        for label_, path in image_specs
    ]
    maps.sort(key=lambda m: m.freq_hz)
    for m in maps:
        print(
            f"  {m.label:<18} {m.freq_hz/1e9:9.4f} GHz  "
            f"beam {m.beam.major.to_value(u.arcsec):6.2f} x "
            f"{m.beam.minor.to_value(u.arcsec):6.2f} arcsec  "
            f"rms {m.rms_jybeam*1e6:9.2f} uJy/beam  cal {m.cal_frac:.0%}"
        )
        if args.show_error_provenance:
            print(f"      rms  : {m.rms_method}")
            print(f"      cal  : {m.cal_method}")

    audit = [{
        "label": m.label, "path": str(m.path),
        "frequency_Hz": m.freq_hz, "frequency_GHz": m.freq_hz / 1e9,
        "shape": list(m.data.shape),
        "beam_major_arcsec": m.beam.major.to_value(u.arcsec),
        "beam_minor_arcsec": m.beam.minor.to_value(u.arcsec),
        "beam_pa_deg": m.beam.pa.to_value(u.deg),
        "pixel_scale_arcsec": list(pixel_scales_arcsec(m.wcs)),
        "BUNIT": str(m.header.get("BUNIT", "")),
        "OBJECT": str(m.header.get("OBJECT", "")),
        "TELESCOP": str(m.header.get("TELESCOP", "")),
        "INSTRUME": str(m.header.get("INSTRUME", "")),
        "DATE-OBS": str(m.header.get("DATE-OBS", "")),
        "rms_Jy_beam": m.rms_jybeam, "rms_method": m.rms_method,
        "rms_area_fraction": m.rms_area_fraction,
        "cal_frac": m.cal_frac, "cal_method": m.cal_method,
    } for m in maps]
    (outdir / "input_FITS_header_summary.json").write_text(
        json.dumps(json_safe_copy(audit), indent=2)
    )

    # -- common grid and beam ----------------------------------------------
    _banner("2. Common grid and common restoring beam")
    astrometry = parse_label_values(args.astrometry_rms, float)
    reference = choose_reference_map(maps, reference_label, astrometry)
    common_beam = make_common_beam(maps)
    target_wcs, target_shape = build_common_target(reference, maps)
    print(
        f"  Reference    : {reference.label}\n"
        f"  Common beam  : {common_beam.major.to_value(u.arcsec):.4f} x "
        f"{common_beam.minor.to_value(u.arcsec):.4f} arcsec, "
        f"PA {common_beam.pa.to_value(u.deg):.2f} deg\n"
        f"  Common grid  : {target_shape[1]} x {target_shape[0]} pixels"
    )

    common_dir = outdir / "common_beam_FITS"
    common_dir.mkdir(exist_ok=True)
    common_maps: List[RadioMap] = []
    for m in maps:
        data, _, header = regrid_and_common_beam(
            m, target_wcs, target_shape, common_beam
        )
        path = common_dir / f"{m.label}_commonbeam.fits"
        write_fits_map(path, data, header)
        # The noise MUST be re-measured after convolution: smoothing
        # correlates neighbouring pixels and changes the noise per beam.
        rms, rms_method, area_fraction = estimate_rms_jybeam(
            data, header, trust_header=False
        )
        common_maps.append(RadioMap(
            label=m.label, path=path, data=data, header=header,
            wcs=target_wcs, beam=common_beam, freq_hz=m.freq_hz,
            rms_jybeam=rms, rms_method="post-common-beam; " + rms_method,
            cal_frac=m.cal_frac, cal_method=m.cal_method,
            rms_area_fraction=area_fraction,
        ))
        save_common_beam_png(
            outdir / f"{m.label}_commonbeam.png", data, m.label,
            m.freq_hz, target_wcs,
        )
        print(
            f"  {m.label:<18} post-convolution rms "
            f"{rms*1e6:9.2f} uJy/beam"
        )

    (outdir / "common_beam_noise_summary.json").write_text(json.dumps(
        json_safe_copy({
            m.label: {
                "frequency_Hz": m.freq_hz,
                "rms_Jy_beam": m.rms_jybeam, "rms_method": m.rms_method,
                "cal_frac": m.cal_frac, "cal_method": m.cal_method,
            } for m in common_maps
        }), indent=2,
    ))

    # -- segmentation -------------------------------------------------------
    _banner("3. Source segmentation")
    # The detection map sets the aperture EVERY band is photometered through
    # (`hysteresis_source_mask` below, then `coverage`-clipped the same way
    # for all bands). Picking it by lowest frequency alone is a common
    # convention -- steep-spectrum emission is most extended there -- but it
    # silently assumes that band is also deep. When it is not, the mask is
    # built from the noisiest data available and comes out tighter than the
    # source's true extent, which then quietly shorts EVERY band's integrated
    # flux, worst for the shallow band itself.
    #
    # Concretely: on this source the lowest-frequency map (144 MHz) has
    # peak/rms ~ 25, thirty times worse than the deepest map (peak/rms ~ 700
    # at 1284 MHz here). A 5-sigma seed on the 144 MHz map alone recovers
    # 17% less flux at 144 MHz than the same photometry through a mask seeded
    # on the deepest band -- large enough to be the dominant term in why a
    # low-frequency point can come out looking suppressed.
    #
    # So: seed the mask on whichever band has the best PEAK SIGNAL-TO-NOISE,
    # not whichever has the lowest frequency. --detection-map overrides this
    # with an explicit label when a specific choice is wanted instead.
    if getattr(args, "detection_map", None):
        by_label = {m.label: m for m in common_maps}
        detection_map = by_label.get(args.detection_map)
        if detection_map is None:
            raise SystemExit(
                f"--detection-map '{args.detection_map}' is not one of "
                f"{sorted(by_label)}."
            )
    else:
        detection_map = max(
            common_maps,
            key=lambda m: float(np.nanmax(m.data)) / max(m.rms_jybeam, 1e-30),
        )
    detection_snr = detection_map.data / detection_map.rms_jybeam
    coverage = all_band_coverage(common_maps)

    high_sigma = float(source_sigma)
    grow_sigma = float(args.source_grow_sigma)
    if not np.isfinite(high_sigma) or high_sigma <= 0:
        raise SystemExit(f"Invalid --source-sigma {high_sigma}.")
    low_sigma = min(grow_sigma, 0.8 * high_sigma)
    if low_sigma >= high_sigma or low_sigma <= 0:
        low_sigma = 0.5 * high_sigma

    source_mask = hysteresis_source_mask(
        detection_snr, high_sigma=high_sigma, low_sigma=low_sigma,
        fill_holes=not args.no_fill_holes,
    ) & coverage
    if not np.any(source_mask):
        raise SystemExit(
            f"No emission survived the {high_sigma:g}-sigma seed / "
            f"{low_sigma:g}-sigma growth segmentation inside the common "
            "footprint of all bands."
        )
    print(
        f"  Detection map: {detection_map.label} "
        f"({detection_map.freq_hz/1e9:.3f} GHz)\n"
        f"  Thresholds   : seed {high_sigma:g} sigma, grow {low_sigma:g} sigma\n"
        f"  Mask         : {int(np.count_nonzero(source_mask))} pixels "
        f"({np.count_nonzero(source_mask)/beam_area_pixels(common_beam, target_wcs):.1f} beams)"
    )

    # -- BRATS adaptive-region growth cap ------------------------------------
    #
    # BRATS grows each region until it reaches --brats-signaltonoise, but
    # --brats-searcharea caps the number of growth steps (adaptiveregions.h:
    # `while (S/N < target) { if (steps >= maxarea) break; grow; }`). BRATS
    # defaults that cap to 1, which is correct only when the target is also
    # 1. Raise the target and leave the cap alone and no region can grow, so
    # the target silently degenerates into a per-pixel threshold and only the
    # brightest pixels survive -- 6.6 per cent of the source in a real run
    # here. Scale the cap with the beam unless it was given explicitly.
    if args.brats_searcharea is None and float(args.brats_signaltonoise) > 1.0:
        cfg["brats_searcharea"] = float(max(
            1.0, round(3.0 * beam_area_pixels(common_beam, target_wcs))
        ))
        print(
            f"  NOTE: --brats-searcharea was not set while "
            f"--brats-signaltonoise is {args.brats_signaltonoise:g}. Raising "
            f"searcharea to {cfg['brats_searcharea']:g} (three beam areas) so "
            "regions can grow to the requested signal-to-noise; at the BRATS "
            "default of 1 they cannot grow at all. Pass --brats-searcharea "
            "to override."
        )

    # -- reference image for contour overlays --------------------------------
    #
    # Every derived map from here on -- spectral index, curvature, ages,
    # errors, and everything BRATS produces -- is a quantity with no
    # morphology of its own. Drawn on its own it is impossible to tell which
    # structure a given value belongs to, so the total intensity of one
    # chosen band is contoured over all of them. The astrometric reference is
    # the natural default: it is the map every other band was aligned to, so
    # its contours are the ones the pixel grid is actually registered against.
    contour_map: Optional[RadioMap] = None
    contour_levels_jy: List[float] = []
    if not args.no_contours:
        wanted = args.contour_label or reference.label
        by_label = {m.label: m for m in common_maps}
        contour_map = by_label.get(wanted)
        if contour_map is None:
            print(
                f"  NOTE: --contour-label '{wanted}' is not one of "
                f"{sorted(by_label)}; falling back to the detection map "
                f"'{detection_map.label}'."
            )
            contour_map = detection_map
        contour_levels_jy = reference_contour_levels(
            contour_map.data, contour_map.rms_jybeam,
            start_sigma=float(args.contour_sigma),
            factor=float(args.contour_factor),
            max_levels=int(args.contour_max_levels),
            drop_lowest=int(args.contour_drop_lowest),
            explicit_sigma=args.contour_levels_sigma,
        )
        if contour_levels_jy:
            print(
                f"  Contours     : {contour_map.label} "
                f"({contour_map.freq_hz/1e9:.3f} GHz), "
                f"{len(contour_levels_jy)} levels from "
                f"{contour_levels_jy[0]/contour_map.rms_jybeam:g} sigma "
                f"({contour_levels_jy[0]*1e3:.3f} mJy/beam) "
                f"x {args.contour_factor:g}"
                + (
                    f"  [ladder anchored at {args.contour_sigma:g} sigma; "
                    f"lowest {args.contour_drop_lowest} level(s) not drawn]"
                    if args.contour_drop_lowest else ""
                )
            )
        else:
            print(
                f"  NOTE: no contour levels could be built from "
                f"{contour_map.label}: its peak does not reach "
                f"{args.contour_sigma:g} sigma."
            )
            contour_map = None

    # Contours are confined to the segmented source, grown by a couple of
    # pixels so the outermost level closes cleanly just outside the mask
    # rather than being cut off along it.
    contour_mask = source_mask
    if contour_map is not None and contour_levels_jy:
        # Grow by roughly one beam radius so the lowest contour closes just
        # outside the segmentation instead of being sliced along it.
        _beam_pix = math.sqrt(max(beam_area_pixels(common_beam, target_wcs), 1.0))
        contour_mask = source_contour_mask(
            contour_map.data,
            # Anchored on --contour-sigma, NOT on the lowest DRAWN level.
            # These differ once --contour-drop-lowest is used, and the mask
            # must keep following the detection isophote: at a higher level
            # the source breaks into disconnected pieces (hotspot, lobe,
            # tail) and "the component containing the source" would keep
            # only one of them.
            float(args.contour_sigma) * float(contour_map.rms_jybeam),
            source_mask,
            # HALF a beam, not a full one. A full beam was chosen to let the
            # lowest contour close rather than be sliced along the mask, but
            # it is also the distance the contour region reaches towards the
            # neighbours: on this field it extended 6 pixels (12 arcsec) east,
            # far enough to pick up a companion and draw it as open arcs in
            # otherwise blank sky.
            #
            # Component selection alone does not stop this. The post-
            # convolution RMS is ~2.4x lower than a plain MAD estimate, so the
            # 5 sigma level sits low enough that the source and its companion
            # merge into ONE component -- and "keep the component touching the
            # source" then keeps the companion too. Bounding the region
            # geometrically is what actually holds.
            grow_pixels=int(max(2, round(0.5 * _beam_pix))),
        )

    #: keyword bundle handed to every derived-map figure
    contour_kw: Dict[str, Any] = {}
    if contour_map is not None and contour_levels_jy:
        contour_kw = {
            "reference_image": contour_map.data,
            "reference_levels": contour_levels_jy,
            "reference_mask": contour_mask,
            # One weight for every figure in the run. The default ladder
            # tapers (heavy lowest level, fine above it) to keep a crowded
            # geometric ladder readable, but the BRATS maps are drawn at a
            # single heavy weight and two engines' maps of the same source
            # must be comparable by eye.
            "reference_color": "black",
            "reference_linewidth": 1.9,
            "reference_label": None if not args.contour_annotate else (
                f"{contour_map.label} "
                f"({contour_map.freq_hz/1e9:.3f} GHz), "
                rf"{args.contour_sigma:g}$\sigma$"
            ),
            "beam": common_beam,
        }

    # alternative thresholds, used to measure the geometric uncertainty
    threshold_masks = []
    for hi, lo in ((high_sigma * 0.6, low_sigma * 0.6),
                   (high_sigma * 1.6, low_sigma * 1.6)):
        if lo >= hi or lo <= 0:
            continue
        try:
            alt = hysteresis_source_mask(
                detection_snr, high_sigma=hi, low_sigma=lo,
                fill_holes=not args.no_fill_holes,
            ) & coverage
            if np.any(alt):
                threshold_masks.append(alt)
        except Exception:                                      # noqa: BLE001
            continue

    save_fits(
        outdir / "source_mask.fits", source_mask.astype(np.uint8),
        product_header(common_maps[0].header, "mask", common_beam),
    )

    # -- integrated photometry ---------------------------------------------
    _banner("4. Integrated photometry")
    sed_rows: List[Dict[str, Any]] = []
    sed_nu, sed_flux, sed_err = [], [], []
    for m in common_maps:
        background = measure_background_offset(m, source_mask)
        phot = integrated_flux_for_mask(
            m, source_mask, background=background,
            subtract_background=not args.no_background_subtraction,
        )
        sed_nu.append(m.freq_hz)
        sed_flux.append(phot["flux_Jy"])
        sed_err.append(phot["flux_err_Jy"])
        row = {
            "label": m.label, "frequency_Hz": m.freq_hz,
            "frequency_GHz": m.freq_hz / 1e9, **phot,
            "rms_Jy_beam": m.rms_jybeam, "cal_frac": m.cal_frac,
            "background_method": background["method"],
        }
        sed_rows.append(row)
        print(
            f"  {m.label:<18} S = {phot['flux_Jy']*1e3:10.4f} +/- "
            f"{phot['flux_err_Jy']*1e3:8.4f} mJy   "
            f"(thermal {phot['flux_err_thermal_Jy']*1e3:.4f}, "
            f"zero-level {phot['flux_err_zerolevel_Jy']*1e3:.4f}, "
            f"cal {phot['flux_err_cal_Jy']*1e3:.4f})"
        )

    sed_nu = np.asarray(sed_nu, dtype=float)
    sed_flux = np.asarray(sed_flux, dtype=float)
    sed_err = np.asarray(sed_err, dtype=float)
    detected = np.asarray(
        [flux_is_detection(f, e, 3.0) for f, e in zip(sed_flux, sed_err)],
        dtype=bool,
    )
    if not np.any(detected):
        raise SystemExit("The integrated SED contains no significant detection.")

    labels = [m.label for m in common_maps]
    observed = characterise_observed_spectrum(
        sed_nu, sed_flux, sed_err, detected, labels=labels,
        bootstrap_n=int(args.alpha_bootstrap), seed=int(args.seed),
    )
    if np.isfinite(observed.get("alpha_all", np.nan)):
        print(
            f"  Observed alpha (all bands) = {observed['alpha_all']:+.4f} "
            f"+/- {observed['alpha_all_err']:.4f}"
        )
    if "curvature" in observed:
        print(
            f"  Spectral curvature         = {observed['curvature']:+.4f} "
            f"+/- {observed['curvature_err']:.4f} "
            f"({observed['curvature_significance']:.1f} sigma)"
        )

    # -- geometry and volume ------------------------------------------------
    _banner("5. Geometry and source volume")
    geometry = build_geometry(
        source_mask, common_beam, target_wcs, redshift,
        threshold_masks=threshold_masks,
        filling_factor=float(args.filling_factor), cosmology=cosmology,
    )
    axes = principal_axes(source_mask, target_wcs)
    raytrace = integrated_volume_raytrace(
        source_mask, target_wcs, common_beam, axes=axes, deconv="quadrature"
    )
    geometry["volume_realisations_arcsec3"]["raytrace_quadrature"] = (
        raytrace["volume_arcsec3"]
    )
    geometry["raytrace_volume_arcsec3"] = raytrace["volume_arcsec3"]
    geometry["raytrace_volume_m3"] = arcsec3_to_m3(
        raytrace["volume_arcsec3"], redshift, cosmology
    ) * float(args.filling_factor)
    depth_map = raytrace["depth_map_arcsec"]
    save_fits(
        outdir / "depth_map.fits", depth_map.astype(np.float32),
        product_header(common_maps[0].header, "arcsec", common_beam),
    )
    volume_m3 = float(geometry["adopted_volume_m3"])
    print(
        f"  Model        : {geometry['geometry']} "
        f"(axis ratio {geometry['axis_ratio']:.2f})\n"
        f"  LAS / LLS    : {geometry['largest_angular_size_arcsec']:.2f} arcsec "
        f"/ {geometry['largest_linear_size_kpc']:.2f} kpc\n"
        f"  Volume       : {volume_m3:.4e} m^3 "
        f"({geometry['adopted_volume_kpc3']:.4g} kpc^3) "
        f"+/- {geometry['volume_fractional_error']*100:.0f}%"
    )

    # -- self-consistent equipartition + ensemble ---------------------------
    solution: Optional[Dict[str, Any]] = None
    ensemble: Optional[Dict[str, Any]] = None
    norm_index = int(np.where(detected)[0][0])

    if use_equipartition:
        _banner("6. Self-consistent injection index, field and energetics")
        solution = self_consistent_solution(
            sed_nu, sed_flux, sed_err, detected,
            volume_m3=volume_m3,
            volume_ln_sigma=float(geometry["volume_ln_sigma"]),
            redshift=redshift, norm_index=norm_index, cfg=cfg,
            observed=observed,
            alpha_override=args.alpha_inj,
            alpha_override_err=args.alpha_inj_err,
            max_iterations=int(args.self_consistent_iterations),
            damping=float(args.self_consistent_damping),
        )
        ensemble = solution.get("ensemble")
        head = solution["equipartition"]["headline"]
        print(
            f"  alpha_inj    = {solution['alpha_inj_signed']:+.4f} "
            f"+/- {solution['alpha_inj_err']:.4f} "
            f"(alpha_positive {solution['alpha_inj_positive']:.4f}) "
            f"[{solution['alpha_source']}]\n"
            f"  Spectrum     = {solution['electron_spectrum']}\n"
            f"  B_eq         = {head['B_eq_uG']:.4f} uG\n"
            f"  B_min-energy = {head['B_min_energy_uG']:.4f} uG\n"
            f"  u_min        = {head['u_min_J_m3']:.4e} J m^-3\n"
            f"  E_min        = {head['E_min_J']:.4e} J\n"
            f"  p_min        = {head['p_min_Pa']:.4e} Pa\n"
            f"  Converged    = {solution['converged']} "
            f"in {solution['n_iterations']} iterations"
        )
    elif use_synchrofit:
        _banner("6. Spectral-ageing ensemble (no equipartition requested)")
        ensemble = fit_spectral_ageing_ensemble(
            sed_nu[detected], sed_flux[detected], sed_err[detected],
            bfield_t=None, redshift=redshift, cfg=cfg, observed=observed,
        )

    if args.require_robust_model_injection:
        robust = (ensemble or {}).get("robust_injection_index", {})
        if robust.get("confidence") not in {"model-identified", "degeneracy-limited"}:
            raise SystemExit(
                "--require-robust-model-injection: no identifiable model-based "
                "injection index could be established "
                f"(confidence = {robust.get('confidence')!r}). "
                "More frequency coverage is needed."
            )

    alpha_final = (
        solution["alpha_inj_signed"] if solution
        else float(observed.get("alpha_all", -0.7))
    )

    # -- spectral index maps -------------------------------------------------
    mc_alpha = 0
    if args.no_spectral_maps:
        _banner("7. Resolved maps SKIPPED (--no-spectral-maps)")
        print(
            "  Resolved spectral-index, curvature and ageing maps are not "
            "produced for this band set.\n"
            f"  The common beam here is "
            f"{common_beam.major.to_value(u.arcsec):.2f} arcsec, over which "
            f"the source spans only "
            f"{np.count_nonzero(source_mask)/beam_area_pixels(common_beam, target_wcs):.1f} "
            "independent beams; a per-pixel spectral index would be set by "
            "the coarsest map's PSF rather than by the source. The "
            "integrated SED and energetics below use every band and are "
            "unaffected."
        )
        alpha_products = None
        alpha_summary = None
    else:
        _banner("7. Resolved spectral-index, curvature and error maps")
        mc_alpha = int(args.mc_alpha_maps)
        if args.thorough and mc_alpha == 0:
            mc_alpha = 200
        alpha_products = spectral_index_maps(
            common_maps, source_mask,
            min_snr=float(args.spectral_snr),
            min_bands=int(args.spectral_min_bands),
            compute_curvature=not args.no_curvature_map,
            mc_realisations=mc_alpha, seed=int(args.seed),
        )
        alpha_summary = spectral_index_map_summary(
            alpha_products, common_maps, common_beam, target_wcs
        )
        print(
            f"  Fitted pixels: {alpha_summary['n_pixels_fitted']} "
            f"({alpha_summary['n_independent_beams']:.0f} independent beams)"
        )
        if "alpha_median" in alpha_summary:
            print(
                f"  alpha        : median {alpha_summary['alpha_median']:+.4f}, "
                f"range {alpha_summary['alpha_min']:+.3f} to "
                f"{alpha_summary['alpha_max']:+.3f}\n"
                f"  median errors: stat "
                f"{alpha_summary.get('median_statistical_error', float('nan')):.4f}, "
                f"cal-sys "
                f"{alpha_summary.get('median_calibration_systematic_error', float('nan')):.4f}"
            )

        fits_targets = {
            "spectral_index.fits": ("alpha", "spectral_index"),
            "spectral_index_error_stat.fits": ("alpha_err_stat", "spectral_index_error"),
            "spectral_index_error_sys.fits": ("alpha_err_sys", "spectral_index_error"),
            "spectral_index_error_total.fits": ("alpha_err_total", "spectral_index_error"),
            "spectral_index_chi2.fits": ("chi2_reduced", "chi2_per_dof"),
            "spectral_index_nbands.fits": ("nbands", "count"),
            "spectral_curvature.fits": ("curvature", "curvature"),
            "spectral_curvature_error.fits": ("curvature_err", "curvature"),
            "spectral_index_error_mc.fits": ("alpha_err_mc", "spectral_index_error"),
        }
        for filename, (key, bunit) in fits_targets.items():
            arr = alpha_products.get(key)
            if arr is None:
                continue
            save_fits(
                outdir / filename, np.asarray(arr, dtype=np.float32),
                product_header(common_maps[0].header, bunit, common_beam),
            )

        save_png_map(
            outdir / "spectral_index.png", alpha_products["alpha"],
            r"$\alpha$   ($S_\nu \propto \nu^{\alpha}$)", cmap="coolwarm",
            vmin=-2.2, vmax=0.2, wcs=target_wcs, **contour_kw,
        )
        for key, filename, label in (
            ("alpha_err_stat", "spectral_index_error_stat.png",
             r"$\sigma_\alpha$ (statistical)"),
            ("alpha_err_total", "spectral_index_error_total.png",
             r"$\sigma_\alpha$ (total)"),
        ):
            arr = alpha_products.get(key)
            if arr is None:
                continue
            finite = arr[np.isfinite(arr)]
            vmax = float(np.nanpercentile(finite, 98)) if finite.size else 0.5
            save_png_map(
                outdir / filename, arr, label, cmap="viridis",
                vmin=0.0, vmax=max(vmax, 0.02), wcs=target_wcs, **contour_kw,
            )
        if alpha_products.get("curvature") is not None:
            save_png_map(
                outdir / "spectral_curvature.png", alpha_products["curvature"],
                r"$d^2\ln S/d(\ln\nu)^2$", cmap="PuOr",
                diverging_zero=True, wcs=target_wcs, **contour_kw,
            )
        if alpha_products.get("chi2_reduced") is not None:
            save_png_map(
                outdir / "spectral_index_chi2.png", alpha_products["chi2_reduced"],
                r"$\chi^2/{\rm dof}$", cmap="magma", vmin=0.0, vmax=5.0,
                wcs=target_wcs, **contour_kw,
            )
        save_png_map(
            outdir / "spectral_index_nbands.png",
            alpha_products["nbands"].astype(float),
            "Number of bands used", cmap="cividis",
            vmin=0, vmax=float(len(common_maps)), wcs=target_wcs, **contour_kw,
        )

    # -- resolved ageing ----------------------------------------------------
    resolved: Optional[Dict[str, Any]] = None
    regions_for_plot: Optional[List[Region]] = None
    if resolved_requested:
        _banner("8. Resolved spectral-ageing maps")
        if not use_equipartition or solution is None:
            print("  Skipped: resolved ageing maps require the pysynch engine.")
        else:
            b_field = solution["equipartition"]["direct"]["equipartition"]["B_T"]
            if getattr(args, "analysis_at_sub_equipartition", False):
                _frac = float(args.b_field_fraction)
                if _frac > 0:
                    b_field *= _frac
                    print(
                        f"  Field        : {b_field/TESLA_PER_MICROGAUSS:.4f} uG "
                        f"({_frac:g} x B_eq, sub-equipartition)"
                    )
            resolved = resolved_ageing_maps(
                common_maps, source_mask, common_beam, target_wcs,
                alpha_inj=solution["alpha_inj_signed"],
                b_field_t=b_field, redshift=redshift, cfg=cfg,
                depth_map_arcsec=depth_map,
                energy_scaling=solution["equipartition"].get("ue_scaling"),
                target_snr=float(args.resolved_snr),
            )
            if resolved.get("status") != "ok":
                print(f"  Not produced: {resolved.get('reason')}")
            else:
                summary = resolved["summary"]
                print(
                    f"  Regions      : {summary['n_regions']} "
                    f"(target S/N {summary['target_snr']:g})\n"
                    f"  Ages         : "
                    f"{summary.get('age_min_Myr', float('nan')):.3g} - "
                    f"{summary.get('age_max_Myr', float('nan')):.3g} Myr "
                    f"(median {summary.get('age_median_Myr', float('nan')):.3g})"
                )
                rmaps = resolved["maps"]
                for filename, key, bunit in (
                    ("resolved_age.fits", "age_Myr", "Myr"),
                    ("resolved_age_error.fits", "age_err_Myr", "Myr"),
                    ("resolved_break_frequency.fits", "log10_break_Hz", "log10(Hz)"),
                    ("resolved_Beq.fits", "B_eq_uG", "microGauss"),
                    ("resolved_pressure.fits", "p_min_Pa", "Pa"),
                    ("resolved_chi2.fits", "chi2_reduced", "chi2_per_dof"),
                ):
                    save_fits(
                        outdir / filename,
                        np.asarray(rmaps[key], dtype=np.float32),
                        product_header(common_maps[0].header, bunit, common_beam),
                    )
                save_png_map(
                    outdir / "resolved_age.png", rmaps["age_Myr"],
                    r"Spectral age $\tau$ (Myr)", cmap="plasma",
                    wcs=target_wcs, **contour_kw,
                )
                save_png_map(
                    outdir / "resolved_break_frequency.png",
                    rmaps["log10_break_Hz"],
                    r"$\log_{10}(\nu_{\rm break}/{\rm Hz})$", cmap="viridis",
                    wcs=target_wcs, **contour_kw,
                )
                save_png_map(
                    outdir / "resolved_Beq.png", rmaps["B_eq_uG"],
                    r"$B_{\rm eq}$ ($\mu$G)", cmap="cividis", wcs=target_wcs,
                    **contour_kw,
                )
                save_png_map(
                    outdir / "resolved_pressure.png", rmaps["p_min_Pa"],
                    r"$p_{\rm min}$ (Pa)", cmap="magma", wcs=target_wcs,
                    **contour_kw,
                )
                if resolved["regions"]:
                    with (outdir / "resolved_regions.csv").open(
                        "w", newline=""
                    ) as fh:
                        flat = []
                        for rec in resolved["regions"]:
                            row = {
                                k: v for k, v in rec.items()
                                if not isinstance(v, (list, dict))
                            }
                            for i, (f, e) in enumerate(zip(
                                rec["flux_Jy"], rec["flux_err_Jy"]
                            )):
                                row[f"flux_{labels[i]}_Jy"] = f
                                row[f"flux_err_{labels[i]}_Jy"] = e
                            flat.append(row)
                        writer = csv.DictWriter(fh, fieldnames=list(flat[0]))
                        writer.writeheader()
                        writer.writerows(flat)
                regions_for_plot = resolved.get("region_objects")

    # -- BRATS engine ---------------------------------------------------------
    brats_result: Optional[Dict[str, Any]] = None
    engine_comparison: Optional[Dict[str, Any]] = None
    runner: Optional[BratsRunner] = None
    best_model_record: Optional[Dict[str, Any]] = None
    if args.engine in {"brats", "both"}:
        _banner("9. BRATS engine")
        runner = BratsRunner(
            args.brats_path, container=args.brats_container,
            image=args.brats_image,
        )
        if not runner.available:
            print(
                "  BRATS not found. Searched --brats-path, the BRATS "
                "environment variable, PATH, a source checkout next to the "
                "data, the usual install locations, and the container "
                "runtimes (docker, podman, apptainer, singularity).\n"
                "  The workspace will still be prepared, together with a "
                "Dockerfile: build it once with\n"
                "      docker build -t brats:latest -f Dockerfile .\n"
                "  from the workspace directory, then re-run with "
                "--brats-container docker."
            )
        else:
            print(f"  BRATS        : {runner.describe()}  [{runner.mode}]")
            print(f"  Version      : {runner.version or 'unknown'}")

        # BRATS needs a magnetic field and an injection index up front. Use
        # the self-consistent values when we have them, otherwise fall back
        # to BRATS' own default field and say so.
        if solution is not None:
            b_for_brats = float(
                solution["equipartition"]["direct"]["equipartition"]["B_T"]
            )
            b_source = "PySynch equipartition (self-consistent)"
            if getattr(args, "analysis_at_sub_equipartition", False):
                _frac = float(args.b_field_fraction)
                if _frac > 0:
                    b_for_brats *= _frac
                    b_source = (
                        f"{_frac:g} x PySynch equipartition "
                        "(sub-equipartition, applied to the whole analysis)"
                    )
        else:
            b_for_brats = 1.0e-9
            b_source = "BRATS default 1e-9 T (no equipartition solution available)"
        print(
            f"  B field      : {b_for_brats/TESLA_PER_MICROGAUSS:.4f} uG  "
            f"[{b_source}]"
        )
        alpha_supplied = float(-alpha_final)
        print(
            f"  alpha_inj    : {alpha_supplied:.4f} (positive convention, as "
            "BRATS expects)"
        )
        # An injection index can never be steeper than the integrated
        # spectrum it produced: ageing only steepens. When the ensemble is
        # degenerate -- a break railed at the edge of its prior, which is
        # what happens when the bands do not bracket the break -- the fitted
        # "injection index" collapses onto the observed index and violates
        # this. Passing it to BRATS would bias every age it fits.
        try:
            _use = detected & np.isfinite(sed_nu) & (sed_flux > 0)
            if int(np.count_nonzero(_use)) >= 2:
                _obs = -float(np.polyfit(
                    np.log(sed_nu[_use]), np.log(sed_flux[_use]), 1
                )[0])
                print(f"  alpha_obs    : {_obs:.4f} (integrated, positive)")
                if alpha_supplied >= _obs - 0.02:
                    print(
                        "  WARNING: the supplied injection index is not "
                        "flatter than the observed integrated index. An aged "
                        "spectrum can only be STEEPER than the injected one, "
                        "so this value is unphysical -- it is what a "
                        "degenerate ensemble fit returns when the break is "
                        "unconstrained.\n"
                        "           Use --brats-findinject so BRATS measures "
                        "the injection index from the resolved data instead."
                        if not args.brats_findinject else
                        "  NOTE: the supplied injection index is not flatter "
                        "than the observed integrated index and is therefore "
                        "unphysical. --brats-findinject is on, so BRATS will "
                        "measure its own value and use that for the fits."
                    )
        except Exception:                                      # noqa: BLE001
            pass

        unsupported = [
            m for m in args.brats_models
            if not BRATS_MODELS.get(m, {}).get("maps")
            and m not in {"CI", "CI-OFF"}
        ]
        for name in unsupported:
            reason = BRATS_MODELS.get(name, {}).get(
                "unavailable_reason", "not supported by BRATS"
            )
            print(f"  NOTE: BRATS cannot fit {name}. {reason}")

        try:
            brats_result = run_brats_analysis(
                outdir=outdir,
                common_maps=common_maps,
                source_mask=source_mask,
                redshift=redshift,
                alpha_inj_signed=float(alpha_final),
                b_field_t=b_for_brats,
                sed={
                    "frequency_Hz": sed_nu,
                    "flux_Jy": sed_flux,
                    "flux_err_Jy": sed_err,
                    "detected": detected,
                },
                cfg=cfg,
                runner=runner,
                models=list(args.brats_models),
                do_maps=not args.brats_no_maps,
                do_integrated=not args.brats_no_integrated,
                find_injection=bool(args.brats_findinject),
                interactive=bool(args.brats_interactive) or not runner.available,
                launch=not args.open_brats,
                timeout=args.brats_timeout,
                common_beam=common_beam,
                beam_area_pixels_value=beam_area_pixels(common_beam, target_wcs),
            )
            brats_result["b_field_source"] = b_source
            print(f"  Workspace    : {brats_result['workspace']}")
            for entry in brats_result.get("scripts", []):
                print(f"  Command file : {entry['file']}")
            if brats_result.get("mode") == "interactive":
                launch = brats_result.get("interactive_launch", {})
                if launch.get("deferred"):
                    print(
                        "  Workspace prepared. BRATS will be opened at the "
                        "end of the run (--open-brats)."
                    )
                elif launch.get("launched"):
                    print(
                        f"  BRATS opened in a new window via {launch['method']}."
                    )
                else:
                    print("  Could not open a terminal window automatically"
                          + (f": {launch.get('error')}" if launch.get("error") else "")
                          + "\n  Run it yourself with:\n"
                          f"      {launch.get('manual_command')}")
            else:
                print(
                    f"  Outputs      : {brats_result.get('n_output_maps', 0)} FITS maps, "
                    f"{brats_result.get('n_output_tables', 0)} data tables"
                )
                measured = brats_result.get(
                    "injection_index_measured_by_brats"
                ) or {}
                if measured:
                    print(
                        "  alpha_inj used for the BRATS fits (measured by "
                        "BRATS itself):"
                    )
                    for _m, _v in sorted(measured.items()):
                        print(
                            f"    {_m:<12} {_v:.4f}   "
                            f"(supplied was "
                            f"{brats_result.get('injection_index_supplied', float('nan')):.4f})"
                        )
                elif brats_result.get("injection_index_supplied") is not None:
                    print(
                        "  NOTE: findinject produced no usable minimum, so "
                        "the supplied injection index was used for the fits."
                    )

                problems = brats_result.get("problems") or []
                if problems:
                    print("  BRATS reported errors:")
                    for problem in problems:
                        print(f"    [{problem.get('stage')}] "
                              f"{problem['message']}")
                        for line in textwrap.wrap(
                            problem["meaning"], width=66,
                            initial_indent="        ",
                            subsequent_indent="        ",
                        ):
                            print(line)
                if not brats_result.get("succeeded"):
                    print(
                        "  WARNING: the scripted BRATS run did not complete "
                        "cleanly. Inspect brats_*.log in the workspace; the "
                        "command files are there and can be pasted into an "
                        "interactive BRATS session to see where it stopped."
                    )
        except BratsUnavailable as exc:
            print(f"  {exc}")
        except Exception as exc:                               # noqa: BLE001
            print(f"  BRATS stage failed: {type(exc).__name__}: {exc}")
            brats_result = {"error": f"{type(exc).__name__}: {exc}"}

        # -- BRATS results -> FITS maps and figures --------------------------
        if (brats_result and args.brats_render
                and brats_result.get("mode") == "scripted"
                and brats_result.get("runs")):
            try:
                rendered = render_brats_products(
                    outdir=outdir,
                    brats_result=brats_result,
                    common_maps=common_maps,
                    source_mask=source_mask,
                    target_wcs=target_wcs,
                    common_beam=common_beam,
                    reference_map=contour_map,
                    contour_start_sigma=float(args.contour_sigma),
                    contour_factor=float(args.contour_factor),
                    contour_max_levels=int(args.contour_max_levels),
                    contour_drop_lowest=int(args.contour_drop_lowest),
                    contour_levels_sigma=args.contour_levels_sigma,
                    contour_mask=contour_mask,
                )
                brats_result["rendered_products"] = rendered
                print(
                    f"  Maps written : {rendered['n_written']} FITS, "
                    f"{rendered['n_figures']} figures"
                    + (f", {rendered.get('n_pgplot_figures', 0)} PGPLOT PNGs"
                       if rendered.get('n_pgplot_figures') else "")
                    + f" -> {outdir / 'brats_maps'}"
                )
                for entry in rendered.get("skipped", []):
                    print(f"  NOTE: {entry.get('reason')}")
                # The painted maps are only trustworthy if they landed on the
                # source. A low overlap means the region array and the image
                # do not correspond, and the maps must not be used.
                for name, diag in (rendered.get("diagnostics") or {}).items():
                    frac = diag.get("fraction_inside_source_mask")
                    if frac is not None and frac < 0.5:
                        print(
                            f"  WARNING: only {frac:.0%} of the painted "
                            f"{name} pixels fall inside the source mask. "
                            "The BRATS region array may not correspond to "
                            "these images -- check brats_maps/ before use."
                        )
            except Exception as exc:                           # noqa: BLE001
                print(
                    f"  BRATS map rendering failed: {type(exc).__name__}: {exc}"
                )
                brats_result["rendered_products"] = {
                    "error": f"{type(exc).__name__}: {exc}"
                }

        if brats_result:
            (outdir / "BRATS_summary.txt").write_text(
                _format_brats_summary(brats_result)
            )

        if args.engine == "both":
            engine_comparison = compare_engines(
                ensemble, brats_result,
                beam_area_pixels_value=beam_area_pixels(common_beam, target_wcs),
            )
            (outdir / "engine_comparison.json").write_text(
                json.dumps(json_safe_copy(engine_comparison), indent=2)
            )

    # -- plots and tables ----------------------------------------------------
    _banner("10. Writing publication products")
    save_segmentation_png(
        outdir / "source_segmentation_QA.png", detection_map, source_mask,
        regions=regions_for_plot,
    )

    model_curves: List[Dict[str, Any]] = []
    pos = sed_nu[sed_nu > 0]
    if pos.size >= 2:
        grid = np.logspace(
            math.log10(float(pos.min()) * 0.7),
            math.log10(float(pos.max()) * 1.5), 300,
        )
        try:
            pl = weighted_logpoly(
                sed_nu[detected], sed_flux[detected], sed_err[detected], degree=1
            )
            model_curves.append({
                "nu_Hz": grid,
                "flux_Jy": pl["flux_at_pivot_Jy"] * (grid / pl["pivot_Hz"]) ** pl["alpha"],
                "label": (
                    fr"Power law $\alpha={pl['alpha']:.3f}\pm{pl['alpha_err']:.3f}$"
                ),
                "color": "tab:red", "ls": "--", "lw": 1.5,
            })
        except Exception:                                      # noqa: BLE001
            pass
        if ensemble:
            b_used = (
                solution["equipartition"]["direct"]["equipartition"]["B_T"]
                if solution else None
            )
            best = (ensemble.get("free_s_consensus", {}).get("best_model")
                    or (ensemble.get("restricted_consensus", {}) or {}).get("best_model"))
            for curve in build_ensemble_curves(ensemble, grid, redshift, b_used):
                if best and curve["label"].startswith(best):
                    curve["lw"] = 2.4
                    model_curves.append(curve)

    sed_products = save_science_sed(
        outdir / "radio_SED.png", sed_nu, sed_flux, sed_err, detected,
        redshift, alpha_final, model_curves=model_curves,
        luminosity_output=outdir / "luminosity_vs_rest_frequency.png",
        alpha_err=(solution.get("alpha_inj_err") if solution else None),
    )
    if ensemble and ensemble.get("models"):
        save_ensemble_sed_plot(
            outdir / "radio_SED_ensemble.png", sed_nu, sed_flux, sed_err,
            detected, ensemble, redshift,
            solution["equipartition"]["direct"]["equipartition"]["B_T"]
            if solution else None,
        )
        save_model_comparison_plot(outdir / "model_comparison.png", ensemble)

        # -- the AICc-preferred model on its own ----------------------------
        try:
            best = save_best_model_sed(
                outdir / "radio_SED_best_model.png",
                sed_nu, sed_flux, sed_err, detected, ensemble, redshift,
                solution["equipartition"]["direct"]["equipartition"]["B_T"]
                if solution else None,
                force_model=(
                    args.sed_model if args.sed_model != "Ensemble" else None
                ),
            )
        except Exception as exc:                               # noqa: BLE001
            print(f"  Best-model SED skipped: {type(exc).__name__}: {exc}")
            best = None
        if best:
            best_model_record = best
            print(
                f"  Best model   : {best['best_model']} "
                f"(Akaike weight {best['akaike_weight']:.2f}, "
                f"Delta {best['criterion']} to runner-up "
                f"{best['delta_criterion_to_second']:.1f})"
            )
            for line in textwrap.wrap(best["verdict"], width=68,
                                      initial_indent="                 ",
                                      subsequent_indent="                 "):
                print(line)
            (outdir / "model_ranking.csv").write_text(
                "model,criterion,criterion_value,delta_criterion,"
                "akaike_weight,injection_index_s,"
                "log10_break_frequency_Hz\n"
                + "\n".join(
                    f"{r['model']},{r['criterion']},"
                    f"{r['criterion_value']:.4f},{r['delta_criterion']:.4f},"
                    f"{r['akaike_weight']:.4f},"
                    f"{r['injection_index_s']},{r['log10_break_frequency_Hz']}"
                    for r in best["ranking"]
                ) + "\n"
            )
        if ensemble.get("injection_index_scans"):
            save_injection_scan_plot(
                outdir / "injection_index_scan.png",
                ensemble["injection_index_scans"],
            )

    emissivity = None
    if solution:
        save_energy_budget_png(
            outdir / "energy_budget.png", solution["equipartition"]
        )

        # -- fixed sub-equipartition budget ---------------------------------
        if float(args.b_field_fraction) > 0:
            try:
                eqd = solution["equipartition"]["direct"]["equipartition"]
                brk = None
                robust = (ensemble or {}).get("robust_injection_index", {})
                logb = robust.get("log10_break_frequency_Hz")
                if logb is None:
                    for rec in ((ensemble or {}).get("models") or {}).values():
                        if rec.get("fit_success"):
                            logb = rec.get("log10_break_frequency_Hz")
                            break
                if logb is not None and np.isfinite(logb):
                    brk = 10.0 ** float(logb)
                sub = sub_equipartition_budget(
                    fraction=float(args.b_field_fraction),
                    b_eq_t=float(eqd["B_T"]),
                    volume_m3=float(
                        solution["equipartition"]["headline"]["volume_m3"]
                    ),
                    alpha_inj=float(solution["alpha_inj_signed"]),
                    redshift=redshift,
                    norm_freq_hz=float(solution["normalisation_frequency_Hz"]),
                    norm_flux_jy=float(solution["normalisation_flux_Jy"]),
                    zeta=float(eqd["zeta"]),
                    gmin=float(args.gamma_min), gmax=float(args.gamma_max),
                    break_frequency_hz=brk,
                )
                # Whether this field DROVE the ageing maps and BRATS, or is
                # only a what-if budget. The report cannot tell otherwise, and
                # it previously labelled the budget "field used for the
                # analysis" on runs whose maps were computed at full B_eq.
                sub["applied_to_analysis"] = bool(
                    getattr(args, "analysis_at_sub_equipartition", False)
                )
                solution["sub_equipartition"] = sub
                print(
                    f"  B = {sub['fraction_of_B_eq']:g} B_eq : "
                    f"{sub['B_uG']:.4f} uG  |  "
                    f"u_tot = {sub['total_energy_density_J_m3']:.4g} J/m^3  |  "
                    f"p = {sub['pressure_Pa']:.4g} Pa"
                    + (f"  |  age = {sub['spectral_age_Myr']:.3g} Myr"
                       if sub.get("spectral_age_Myr") is not None else "")
                )
                if sub["inverse_compton_dominated"]:
                    print(
                        "    NOTE: this field is below B_CMB/sqrt(3) = "
                        f"{sub['B_CMB_over_sqrt3_uG']:.4f} uG "
                        f"(B_CMB = {sub['B_CMB_uG']:.4f} uG), which is where "
                        "the spectral age peaks. Inverse-Compton losses "
                        "dominate below it, so the age is SHORTER here, not "
                        "longer."
                    )
            except Exception as exc:                           # noqa: BLE001
                print(
                    "  Sub-equipartition budget skipped: "
                    f"{type(exc).__name__}: {exc}"
                )
        try:
            emissivity = save_volume_emissivity(
                outdir / "volume_emissivity_vs_rest_frequency.png",
                sed_nu, sed_flux, sed_err, detected, redshift,
                volume_m3, alpha_final,
                ensemble=ensemble,
                b_field_t=(
                    solution["equipartition"]["direct"]["equipartition"]["B_T"]
                    if solution else None
                ),
                alpha_err=(solution.get("alpha_inj_err") if solution else None),
                volume_ln_sigma=(geometry or {}).get(
                    "volume_fractional_error"
                ),
            )
        except Exception as exc:                               # noqa: BLE001
            print(f"  Volume-emissivity plot skipped: {exc}")

    lnu = rest_lnu_from_flux(sed_flux, redshift, alpha=alpha_final)
    elnu = rest_lnu_from_flux(sed_err, redshift, alpha=alpha_final)
    for row, rest_nu, l_, el_ in zip(
        sed_rows, sed_nu * (1.0 + redshift), lnu, elnu
    ):
        row["detected"] = bool(
            flux_is_detection(row["flux_Jy"], row["flux_err_Jy"], 3.0)
        )
        row["rest_frequency_Hz"] = float(rest_nu)
        row["rest_luminosity_W_Hz"] = float(l_)
        row["rest_luminosity_err_W_Hz"] = float(el_)
        row["volume_emissivity_W_m3_Hz"] = (
            float(l_ / volume_m3) if volume_m3 > 0 else float("nan")
        )
    with (outdir / "integrated_SED.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(sed_rows[0]))
        writer.writeheader()
        writer.writerows(sed_rows)

    save_source_properties_text(
        outdir / "source_properties.txt",
        str(common_maps[0].header.get("OBJECT", "RADIO_SOURCE")),
        geometry, solution, observed, ensemble,
        {"kappa": kappa_effective},
    )

    # -- machine-readable results -------------------------------------------
    result = {
        "pysynch_pipeline_version": __version__,
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "runtime_seconds": float(time.time() - t_start),
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy": np.__version__,
            "astropy": __import__("astropy").__version__,
            "pysynch_available": HAVE_PYSYNCH,
            "pysynch_import_error": (
                str(PYSYNCH_IMPORT_ERROR) if PYSYNCH_IMPORT_ERROR else None
            ),
            "synchrofit_available": HAVE_SYNCHROFIT,
            "synchrofit_flavour": SYNCHROFIT.flavour,
            "synchrofit_version": SYNCHROFIT.version,
            "synchrofit_import_error": (
                str(SYNCHROFIT.import_error) if not HAVE_SYNCHROFIT else None
            ),
        },
        "configuration": {
            "redshift": redshift,
            "cosmology": {
                "H0_km_s_Mpc": float(args.h0), "Om0": float(args.om0),
                "Tcmb0_K": T_CMB0,
            },
            "B_CMB_T": b_cmb_tesla(redshift),
            "B_CMB_uG": b_cmb_tesla(redshift) / TESLA_PER_MICROGAUSS,
            "reference_label": reference.label,
            "common_beam_arcsec": {
                "major": common_beam.major.to_value(u.arcsec),
                "minor": common_beam.minor.to_value(u.arcsec),
                "pa_deg": common_beam.pa.to_value(u.deg),
            },
            "source_seed_sigma": high_sigma,
            "source_grow_sigma": low_sigma,
            "spectral_snr": float(args.spectral_snr),
            "spectral_min_bands": int(args.spectral_min_bands),
            "engine": args.engine,
            "brats_models_requested": list(args.brats_models),
            "brats_interactive": bool(args.brats_interactive),
            "gamma_min": cfg["gmin"], "gamma_max": cfg["gmax"],
            "kappa_proton_electron_energy_ratio": kappa_effective,
            "zeta_uB_over_uparticles": zeta,
            "filling_factor": cfg["filling_factor"],
            "inject_range_prior_s": list(cfg["inject_range"]),
            "break_range_prior_log10Hz": list(cfg["break_range"]),
            "monte_carlo_samples": cfg["mc_samples"],
            "alpha_map_mc_realisations": mc_alpha,
            "seed": cfg["seed"],
            "thorough": cfg["thorough"],
            "background_subtracted": not args.no_background_subtraction,
            "aged_spectrum_allowed": cfg["use_aged_spectrum"],
            "alpha_convention": "S_nu ~ nu^alpha, alpha negative",
        },
        "input_maps": audit,
        "integrated_sed": sed_rows,
        "sed_products": sed_products,
        "volume_emissivity": emissivity,
        "observed_spectrum": observed,
        "geometry": geometry,
        "geometry_physical": geometry_physical_summary(geometry, redshift),
        "spectral_ageing_ensemble": ensemble,
        "self_consistent_solution": (
            {k: v for k, v in solution.items() if k != "ensemble"}
            if solution else None
        ),
        "spectral_index_map_summary": alpha_summary,
        "best_model": best_model_record,
        "brats": brats_result,
        "engine_comparison": engine_comparison,
        "resolved_ageing": (
            {"summary": resolved["summary"], "regions": resolved["regions"]}
            if resolved and resolved.get("status") == "ok" else
            (resolved if resolved else None)
        ),
        "output_files": sorted(
            p.name for p in outdir.iterdir() if p.is_file()
        ),
    }
    (outdir / "PySynch_results.json").write_text(
        json.dumps(json_safe_copy(result), indent=2)
    )

    # -- text report ---------------------------------------------------------

    # BRATS may be a native binary or a container runtime; name whichever it
    # actually was, so the report says how the numbers were produced.
    if brats_result and (brats_result.get("executable")
                         or brats_result.get("runtime")):
        _brats_where = (
            brats_result.get("executable")
            or brats_result.get("runtime")
            or "unavailable"
        )
        _brats_report_line = (
            f"{_brats_where} "
            f"(v{brats_result.get('version') or '?'}), "
            f"mode={brats_result.get('mode')}"
        )
    elif args.engine in {"brats", "both"}:
        _brats_report_line = "requested but not found"
    else:
        _brats_report_line = "not requested"

    report: List[str] = [
        f"PySynch v{__version__} -- analysis report",
        "=" * 72,
        f"Generated        : {time.strftime('%Y-%m-%d %H:%M:%S %Z')}",
        f"Runtime          : {time.time() - t_start:.1f} s",
        f"Output directory : {outdir}",
        "",
        "BACKENDS",
        "-" * 72,
        f"  pysynch     : {'available' if HAVE_PYSYNCH else 'NOT AVAILABLE'}"
        + ("" if HAVE_PYSYNCH else f"  ({PYSYNCH_IMPORT_ERROR})"),
        f"  synchrofit  : {'available' if HAVE_SYNCHROFIT else 'NOT AVAILABLE'}"
        f"  [{SYNCHROFIT.flavour}]"
        + ("" if HAVE_SYNCHROFIT else f"  ({SYNCHROFIT.import_error})"),
        f"  BRATS       : {_brats_report_line}",
        f"  Engine      : {args.engine}",
        "",
        "INPUT",
        "-" * 72,
        f"  Frequencies      : {len(common_maps)} "
        f"({', '.join(f'{m.freq_hz/1e9:.3f}' for m in common_maps)} GHz)",
        f"  Reference map    : {reference.label}",
        f"  Common beam      : "
        f"{common_beam.major.to_value(u.arcsec):.4f} x "
        f"{common_beam.minor.to_value(u.arcsec):.4f} arcsec, "
        f"PA {common_beam.pa.to_value(u.deg):.2f} deg",
        f"  Redshift         : {redshift:.6g}",
        f"  B_CMB            : {b_cmb_tesla(redshift)/TESLA_PER_MICROGAUSS:.4f} uG",
        "",
    ]
    if solution:
        head = solution["equipartition"]["headline"]
        report += [
            "HEADLINE RESULTS",
            "-" * 72,
            f"  alpha_inj (positive)  : {solution['alpha_inj_positive']:.4f} "
            f"+/- {solution['alpha_inj_err']:.4f}",
            f"  Injection index s     : "
            f"{alpha_signed_to_s(solution['alpha_inj_signed']):.4f}",
            f"  B_equipartition       : {head['B_eq_uG']:.5g} uG",
            f"  B_minimum-energy      : {head['B_min_energy_uG']:.5g} uG",
            f"  Volume                : {head['volume_m3']:.5e} m^3 "
            f"({head['volume_kpc3']:.5g} kpc^3)",
            f"  u_min                 : {head['u_min_J_m3']:.5e} J m^-3",
            f"  E_min                 : {head['E_min_J']:.5e} J",
            f"  p_min                 : {head['p_min_Pa']:.5e} Pa",
            "",
        ]
    report += ["PRODUCTS", "-" * 72]
    report += [f"  {p.name}" for p in sorted(outdir.iterdir()) if p.is_file()]
    report += [
        "",
        "NOTES",
        "-" * 72,
        "  Measured SED points and their errors come directly from the",
        "  regridded common-beam FITS maps; synchrofit is used only for the",
        "  ageing-model fits and never to generate a data point.",
        "  Non-detections are kept as measured (including negative values) and",
        "  excluded from fits rather than clipped to a positive floor.",
        "  See source_properties.txt for the full numerical summary and",
        "  PySynch_results.json for every intermediate quantity.",
    ]
    (outdir / "PySynch_report.txt").write_text("\n".join(report) + "\n")

    _banner("Complete")
    print(f"  Output directory : {outdir}")
    print(f"  Files written    : {sum(1 for p in outdir.iterdir() if p.is_file())}")
    print(f"  Runtime          : {time.time() - t_start:.1f} s")
    if solution:
        head = solution["equipartition"]["headline"]
        print(
            f"  alpha_inj = {solution['alpha_inj_positive']:.4f} +/- "
            f"{solution['alpha_inj_err']:.4f}  |  "
            f"B_eq = {head['B_eq_uG']:.4g} uG  |  "
            f"B_me = {head['B_min_energy_uG']:.4g} uG  |  "
            f"u_min = {head['u_min_J_m3']:.4g} J/m^3"
        )

    # -- hand over to an interactive BRATS session ---------------------------
    #
    # Deliberately the last thing that happens. Every product is already on
    # disk and the summary has been printed, so the pipeline's own output is
    # complete and unaffected by whatever happens to the window -- and the
    # user gets dropped into BRATS with this run's results, injection index
    # and magnetic field already in place.
    if args.open_brats:
        _banner("Opening BRATS")
        # Reuse the runner the BRATS stage already resolved when there was
        # one; with --engine synchrofit there is not, so locate BRATS now.
        session_runner = runner or BratsRunner(
            args.brats_path, container=args.brats_container,
            image=args.brats_image,
        )
        if solution is not None:
            b_for_session = float(
                solution["equipartition"]["direct"]["equipartition"]["B_T"]
            )
        else:
            b_for_session = 1.0e-9
        try:
            session = open_brats_session(
                outdir=outdir,
                runner=session_runner,
                brats_result=brats_result,
                common_maps=common_maps,
                source_mask=source_mask,
                redshift=redshift,
                alpha_inj_signed=float(alpha_final),
                b_field_t=b_for_session,
                sed={
                    "frequency_Hz": sed_nu,
                    "flux_Jy": sed_flux,
                    "flux_err_Jy": sed_err,
                    "detected": detected,
                },
                cfg=cfg,
                models=list(args.brats_models),
                common_beam=common_beam,
            )
            _report_brats_session(session, session_runner)
        except Exception as exc:                               # noqa: BLE001
            # The window is a convenience; the science is already written.
            print(f"  Could not open BRATS: {type(exc).__name__}: {exc}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
