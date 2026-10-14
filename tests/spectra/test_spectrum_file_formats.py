"""
``read_spectrum_file`` returns ``(lmax, nspec)`` whatever the format.

The ``.fits`` branch went through ``healpy.read_cl``, which gives
``(nspec, nell)``, and returned that sliced and transposed: the multi-spectrum
case came back as ``(nspec, lmax)`` -- the transpose of what the ``.txt``
branch returns and of what every caller indexes (``beams_from_file[:, i]``,
``cmb_from_file.shape[1] == 4``, ``spectra_from_file.T[index]`` for the
removed ``StokesIndices`` enum)
-- and its "extends to lmax" check tested the number of spectra, so a
four-spectrum file was rejected for any ``lmax > 4``.
"""

import os
import warnings

import numpy as np
import pytest

healpy = pytest.importorskip("healpy")

from cmbcov.utils.array_utils import (  # noqa: E402
    read_spectrum_file,
)

NELL = 40


def _spectra(n_spectra):
    ell = np.arange(NELL, dtype=float)
    return np.array([(i + 1) * (ell + 1) ** -(i + 1) for i in range(n_spectra)])


def _write_pair(tmp_path, spectra):
    txt = str(tmp_path / "spectra.dat")
    table = np.column_stack([np.arange(NELL, dtype=float), spectra.T])
    np.savetxt(txt, table)
    fits = str(tmp_path / "spectra.fits")
    healpy.write_cl(fits, list(spectra), overwrite=True, dtype=np.float64)
    return txt, fits


@pytest.mark.parametrize("n_spectra", [1, 4, 9])
@pytest.mark.parametrize("lmax", [10, NELL])
def test_fits_and_text_round_trip_to_the_same_array(tmp_path, n_spectra, lmax):
    spectra = _spectra(n_spectra)
    txt, fits = _write_pair(tmp_path, spectra)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from_fits = read_spectrum_file(fits, lmax)
    from_txt = read_spectrum_file(txt, lmax)

    assert from_txt.shape == (lmax, n_spectra)
    assert from_fits.shape == from_txt.shape
    np.testing.assert_allclose(from_fits, from_txt, rtol=1e-6, atol=0.0)
    np.testing.assert_allclose(from_fits, spectra.T[:lmax], rtol=1e-6, atol=0.0)


def test_fits_too_short_is_reported_on_the_multipoles_not_the_spectra(tmp_path):
    spectra = _spectra(4)
    _, fits = _write_pair(tmp_path, spectra)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(ValueError, match=f"ends at ell = {NELL - 1}"):
            read_spectrum_file(fits, NELL + 1)


def test_fits_reading_warns_once_per_process(tmp_path):
    import cmbcov.utils.array_utils as array_utils

    _, fits = _write_pair(tmp_path, _spectra(2))
    del array_utils._WARNED_FITS_READ[:]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        read_spectrum_file(fits, 10)
        read_spectrum_file(fits, 10)
    assert len([w for w in caught if "Reading .fits" in str(w.message)]) == 1
    assert os.path.exists(fits)
