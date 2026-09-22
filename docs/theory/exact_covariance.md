# The exact pseudo-Cℓ covariance

The package computes the exact covariance of pseudo power spectra on a
masked sky row by row, following the algorithm of Camphuis et al. 2022
([arXiv:2204.13721](https://arxiv.org/abs/2204.13721), Sect. 3 and Fig. 2),
with every spherical-harmonic integral evaluated exactly on a
Gauss-Legendre grid. It is too slow for production covariances, which cost
of order $\ell_{\max}^5$, but a handful of rows is cheap, and those rows are
the reference against which [ACC](acc.md) and its
[B-mode extension](bmode_kernels.md) are validated. The only approximation
is the band-limit $L_w$ of the mask.

## 1. What it computes

For a Gaussian sky with spectrum $C_L$ and a mask $W$, the pseudo-coefficients
are $\tilde a = K a$ with $K = A W S$ ($S$ synthesis, $A$ analysis, $W$
pointwise multiplication). Their correlation matrix is
$\langle\tilde a\tilde a^\dagger\rangle = K\thinspace\mathrm{diag}(C_L)\thinspace K^\dagger$,
and the covariance of the temperature pseudo-spectrum is the paper's Eq. 7,

```math
\tilde\Sigma_{\ell\ell'} = \frac{2}{(2\ell+1)(2\ell'+1)}\sum_{mm'}\left\vert\langle\tilde a_{\ell m}\thinspace\tilde a_{\ell' m'}^{\ast}\rangle\right\vert^2 .
```

One column $(\ell', m')$ of the correlation matrix is obtained by applying the
operators right to left to the unit vector $e_{\ell' m'}$:

1. $I_{LM} = (K^\dagger e_{\ell' m'})\mathstrut_{LM}$: synthesise the single
   harmonic $Y_{\ell' m'}$, multiply by $W$, analyse.
2. $x_{LM} = C_L I_{LM}$ (Eq. 10).
3. $c_{\ell m} = (K x)\mathstrut_{\ell m} = \langle\tilde a_{\ell m}\tilde a_{\ell' m'}^{\ast}\rangle$ for every $(\ell, m)$ at once (Eq. 11).

A row $\ell'$ sums $\vert c\vert^2$ over its $2\ell' + 1$ columns; for a real mask
and real spectra the columns $\pm m'$ contribute equally, so only
$m' \ge 0$ is computed by default (`use_symmetry=True`).

**Polarisation.** With fields $T, E, B$, $K = \mathrm{blockdiag}(K_0, K_2)$,
where $K_2$ acts on $(E, B)$ through the spin-2 transforms, and $C$ is the
$3\times3$ matrix of TT, EE, BB, TE, TB and EB at each $L$. A column is now
$u = K e_{Z, \ell' m'}$ for a unit vector in field $Z$, then
$v^X = \sum_{Z'} C^{XZ'} u^{Z'}$, then $K v$. Wick's theorem gives every block

```math
\mathrm{Cov}\big(\tilde C_{\ell}^{XY}, \tilde C_{\ell'}^{ZW}\big) = \frac{1}{(2\ell+1)(2\ell'+1)}\sum_{mm'}\Big[R^{XZ}\thinspace\overline{R^{YW}} + R^{XW}\thinspace\overline{R^{YZ}}\Big]
```

from the correlators $R^{XZ}$ of the three columns $Z = T, E, B$ of one
$(\ell', m')$, so all blocks among the six spectra come out of one pass over
the row. Only the columns the requested spectra need are computed: T alone
for TT, T and E for TT, TE and EE, all three as soon as B appears. The spin-2
convention is HEALPix's, so signs match `healpy.anafast`.

Entry points, in `cmbcov.exact`:

| function | returns |
|---|---|
| `exact_covariance_row` | one TT row $\tilde\Sigma_{\ell\ell'}$ for $\ell = 0 \ldots \ell_{\max}$ |
| `exact_covariance` | TT matrix, all rows or the ones listed in `rows` |
| `exact_covariance_row_pol` | one row of every block among the requested `spectra` |
| `exact_covariance_pol` | the same for several rows, as full block matrices |

The returned arrays run over $\ell = 0 \ldots \ell_{\max}$ inclusive.

## 2. The Gauss-Legendre grid and the band-limited mask

HEALPix has no exact quadrature: `map2alm` approaches the pixel
least-squares solution, not the integral, and never reaches machine
precision. On a Gauss-Legendre grid of band-limit $L_g$, with $L_g + 1$
rings at the Gauss-Legendre nodes in $\cos\theta$ and $2L_g + 2$ points per
ring, the analysis is the exact inverse and the exact adjoint of the
synthesis. So once the mask enters as a set of harmonic coefficients
truncated at $L_w$, every step above is exact: $K^\dagger = K$, and no
adjoint correction is needed.

The integrand $W\thinspace Y_{\ell m}\thinspace Y_{\ell' m'}^{\ast}$ is a polynomial of degree
$L_w + 2\ell_{\max}$ in $\cos\theta$, so exactness needs

```math
L_g \ge \ell_{\max} + \left\lceil\frac{L_w - 1}{2}\right\rceil
```

(`gl_minimal_lmax`, applied to the internal band-limit of Sect. 3). The error
is of order $10^{-2}$ one step below this grid and of order $10^{-15}$ at it.

The **mask band-limit** $L_w$ is the one approximation. A pixel mask is
converted once to its coefficients up to $L_w$, by default
$3\thinspace n_{\mathrm{side}} - 1$; a coefficient array can be passed instead.
The result is the covariance of the exact-integral pseudo-spectrum estimator
for the band-limited mask. Apodised masks converge quickly in $L_w$;
hard-edged masks and point-source holes converge slowly and need a larger
$L_w$.

**The HEALPix path.** `exact_covariance_row` and `exact_covariance` also
accept `grid="healpix"`, which is their default and is TT only. It computes
the covariance of the pixelised estimator that `healpy.anafast` applies to a
HEALPix map times the pixel mask, with iterated `map2alm` and its exact
adjoint in step 1. This answers a different question from the GL path; the
two differ by about $10^{-5}$ relative on the diagonal band and up to
$10^{-3}$ far off the diagonal on a mask at $n_{\mathrm{side}} = 32$, and
converge as $n_{\mathrm{side}}^{-2}$. The polarised functions default to
`grid="gl"` and require it for any spectrum other than TT. Use the GL grid
when comparing with ACC kernels, which are computed on it.

## 3. The internal band-limit

A row $\ell'$ sums $C_L$ over the mask's coupling range around $\ell'$, so it
needs the spectrum **above** the largest multipole it reports. Every
function therefore takes `lmax_int`: the spectra, the harmonic arrays and the
grid are carried to $\ell_{\mathrm{int}}$, and only $\ell \le \ell_{\max}$ is
returned, exactly as if the matrix had been computed to $\ell_{\mathrm{int}}$
and sliced. The default `lmax_int=None` means no margin and is biased low
near $\ell_{\max}$.

The margin needed is a property of the mask. `mask_coupling_width` returns
the smallest $\Delta$ such that the multipoles $L \le \Delta$ hold 99% of the
mask's coupling weight $\sum_L (2L+1) W_L$, and a warning names both numbers
when $\ell_{\mathrm{int}} - \ell_{\max}$ is smaller. Two measurements show why
it matters:

- On an apodised survey footprint covering about 4% of the sky, at
  $n_{\mathrm{side}} = 512$ with $L_w = 512$, the diagonal element of row 500
  computed with a margin of 0 is 0.316 of the value with a margin of 200; with
  a margin of 120 it is 0.99957.
- Against 20000 simulated skies masked by an apodised cap at
  $n_{\mathrm{side}} = 32$, pseudo-spectra to $\ell_{\max} = 12$, the diagonal
  computed on the HEALPix path with $\ell_{\mathrm{int}} = 36$ matches the sample covariance to
  0.4 to 1.5% at every multipole, where the simulations' own 1σ is 1.0% per
  element. Without a margin it falls to 0.49 of the sample value at
  $\ell = 12$.

## 4. Cost

Each column costs a fixed number of spherical-harmonic transforms on the
grid, about $\ell_{\mathrm{int}}^3$ operations each, and a row needs
$\ell' + 1$ columns, so a row costs of order $\ell'^4$ and the full matrix of
order $\ell_{\max}^5$. The GL path carries each complex column as two real
transforms and synthesises the single harmonic of step 1 with one azimuthal
order only.

Measured on a 14-core laptop, TT, on the 4% footprint above with
$L_w = 384$: one row at $\ell' = 500$ takes of order 10 s and the full matrix
to $\ell_{\max} = 500$ about half an hour; one row at $\ell' = 1000$ takes
about 2 minutes, and the full matrix to $\ell_{\max} = 1000$ about 16 hours.
A polarised row computes up to three columns per $(\ell', m')$, one per
field $Z$, with spin-2 transforms for E and B.
Rows are independent: on the GL grid, `exact_covariance(..., nprocs=N)`
spreads them over processes, although on a single machine ducc0's own threads are usually the
better lever. [Term selection](term_selection.md) can skip the orders $m'$
that the mask does not overlap.

## 5. Validation

- **Full sky.** With $W = 1$ the pseudo-spectra are the true ones, and all 21
  polarised blocks reproduce the analytic full-sky covariance to
  $2\times10^{-15}$.
- **Symmetry.** Rows are computed independently, yet element $(\ell, \ell')$
  of block XY×ZW from row $\ell'$ equals element $(\ell', \ell)$ of block
  ZW×XY from row $\ell$ to $3\times10^{-13}$, with no symmetrisation.
- **Simulations.** Against 10000 `healpy` skies at $n_{\mathrm{side}} = 32$,
  with $\ell_{\max} = 60$ and $C^{BB} = 0$, seven polarised blocks agree within
  2.8σ of the Monte Carlo error, with band-averaged ratios between 0.98 and
  1.02 and no sign bias. This includes the BB block, which is pure E-to-B
  leakage through the mask.
- **ACC kernels.** Evaluating the kernel form of the covariance with kernels
  built at the true $(\ell, \ell')$ reproduces the exact rows to machine
  precision for all 36 ordered blocks among the six spectra
  ([B-mode kernels](bmode_kernels.md), Sect. 4).

## 6. Using it as a reference

To validate an ACC covariance on your mask:

1. Use the same band-limited mask for both, on the GL grid: pass the mask
   and `lw` the ACC kernels were built with.
2. Choose a few rows: one at or near $\ell_\ast$, where ACC should agree to
   the accuracy of its normalisation, and one near each end of the multipole
   range you will use, where the translation error of ACC is largest.
3. Give `lmax_int` a margin of at least `mask_coupling_width` above the
   largest row.
4. Compare at the pseudo-Cℓ level, before any PolSpice transform, $D_\ell$
   scaling or binning (a run with `polspice_postprocess` off returns the
   pseudo-Cℓ covariance, and `save_raw_blocks` keeps the unbinned blocks),
   and quote errors in units of
   $\sqrt{\mathrm{Cov}\mathstrut_{aa}\mathrm{Cov}\mathstrut_{bb}}$ for the
   off-diagonal and cross blocks, where plain relative errors are
   meaningless near zero crossings. ACC blocks are indexed
   $\ell = 0 \ldots \ell_{\max} - 1$, the exact rows $0 \ldots \ell_{\max}$.
