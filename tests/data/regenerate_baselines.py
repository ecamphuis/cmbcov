"""
Regenerate the golden files pinned by tests/test_end_to_end_baseline.py and
tests/test_acc_coupling.py: baseline_covariance.npy, baseline_lbins.npy and
baseline_acc_coupling.npz.

WHEN it is legitimate to run this script
-----------------------------------------
Only after a *deliberate* numerical change to the package (a fix to an
approximation, a change of algorithm, a precision improvement) that is
expected to move these golden files by a known, understood amount. Golden
files are a safety net for *unintentional* drift; regenerating them erases
that net for whatever changed, so:

  1. Measure the shift BEFORE running this script (see the recipe in
     tests/test_end_to_end_baseline.py / tests/test_acc_coupling.py for how
     to reproduce the pipeline outputs, and diff them against the *existing*
     baseline_covariance.npy / baseline_acc_coupling.npz).
  2. Understand *why* the shift has the size it has -- ideally down to which
     code path produced it (see e.g. the ablation approach used to attribute
     the mask-alm shift below).
  3. Record the measured shift (max and median relative change, and any
     other useful summary such as a Frobenius-norm-relative or diagonal-only
     figure) in the commit message that updates these files. A regenerated
     golden with no recorded shift is indistinguishable from one that papered
     over a bug.

Do NOT run this script to "make the tests pass" without following the above.

This script does NOT regenerate baseline_mask.fits or baseline_cls.dat --
those are inputs, not outputs of the pipeline, and are not touched here.

Example of a deliberate change and its recorded shift:
mask.py's MaskWlm.compute_spherical_harmonics switched from a single adjoint
synthesis (ducc0_map2alm, ~5e-3 relative alm error) to a converged
least-squares solve (ducc0.sht.pseudo_analysis). Measured shift before
regenerating: covariance median relative change ~0.53% (diagonal median
~0.51%, Frobenius-norm-relative ~0.78%), ACC coupling median relative change
~1.3% (diagonal median ~0.58%, global L2-relative ~0.39%). Elementwise max
relative changes are much larger (up to ~23% for the covariance, ~55% for
ACC) but concentrated in specific near-cancellation off-diagonal entries --
confirmed by comparing against the diagonal-only and whole-matrix-norm
figures above, and by an ablation showing the entire shift is attributable to
the mask-alm change (the W^2_l / MASTER-Msq-kernel change introduced in the
same commit measured *zero* effect on either golden, because neither the
"nka" covariance approximation nor the HEALPix ACC coupling path used at
lmax=48/dmax=2 route through the Msq kernel for this configuration).
"""

import glob
import os
import shutil
import tempfile

import numpy as np

from cmbcov import CovarianceMatrixGenerator
from cmbcov.approximations.acc import precompute_acc_kernels

DATA = os.path.dirname(os.path.abspath(__file__))

# ACC golden parameters -- must match tests/test_acc_coupling.py exactly.
ACC_ELL = 16
ACC_DMAX = 2
ACC_ELLP_RANGE = [16, 17]  # = coupling_ellprange(ACC_ELL, ACC_DMAX)
ACC_NSIDE = 16


def regenerate_end_to_end_baseline() -> None:
    """Recreate baseline_covariance.npy and baseline_lbins.npy."""
    out = tempfile.mkdtemp()
    try:
        template = open(os.path.join(DATA, "baseline_params.yml")).read()
        params = os.path.join(out, "params.yml")
        with open(params, "w") as handle:
            handle.write(
                template.replace("PLACEHOLDER_OUT", out).replace(
                    "PLACEHOLDER_DATA", os.path.abspath(DATA)
                )
            )
        CovarianceMatrixGenerator(params).run_full_analysis()
        version = sorted(glob.glob(os.path.join(out, "v*")))[-1]
        covariance = np.loadtxt(os.path.join(version, "covariance_matrix.dat"))
        bandpowers = np.loadtxt(os.path.join(version, "lbins.dat"))
    finally:
        shutil.rmtree(out, ignore_errors=True)

    np.save(os.path.join(DATA, "baseline_covariance.npy"), covariance)
    np.save(os.path.join(DATA, "baseline_lbins.npy"), bandpowers)
    print(
        f"Wrote baseline_covariance.npy {covariance.shape} and baseline_lbins.npy "
        f"{bandpowers.shape}"
    )


def regenerate_acc_coupling_baseline() -> None:
    """Recreate baseline_acc_coupling.npz."""
    coupling = precompute_acc_kernels(
        "baseline_mask.fits",
        None,
        centralell=ACC_ELL,
        dmax=ACC_DMAX,
        mask_path=os.path.abspath(DATA),
        nside=ACC_NSIDE,
        grid="healpix",
        dryrun=True,
    )
    assert sorted(coupling) == ACC_ELLP_RANGE

    flat = {}
    for ellp, inner in coupling.items():
        for stokes_key, array in inner.items():
            flat[f"{ellp}|{stokes_key}"] = array

    np.savez(os.path.join(DATA, "baseline_acc_coupling.npz"), **flat)
    print(f"Wrote baseline_acc_coupling.npz with {len(flat)} arrays")


if __name__ == "__main__":
    regenerate_end_to_end_baseline()
    regenerate_acc_coupling_baseline()
