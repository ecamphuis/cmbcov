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
order $\ell_{\max}^5$.

On the GL grid the TT columns of a row are computed together. Every
transform is a Legendre sum per azimuthal order $M$ followed by an FFT per
ring, so a chunk of columns goes through the Legendre sums as matrix
products with a table of $\lambda_{LM}(\theta)$ and through one batched FFT.
Step 1 needs no transform: the masked single harmonic has the ring profile
$W_{M-m'}(\theta)\thinspace \lambda_{\ell' m'}(\theta)$ in order $M$, so its analysis
is one more matrix product. Only the orders that can be non-zero are
computed, $|M - m'| \le L_w$ after step 1 and $\le 2L_w$ after step 3. The
number of columns per chunk, and whether the Legendre table is kept for all
rows, follow from `max_memory_gb` (default 2 GiB); the result depends on
it only through rounding, and agrees with the column-by-column computation
to a few $10^{-15}$ of the largest entry.

Measured on a 14-core laptop shared with other jobs (load 10-15), TT,
mask band-limit $L_w = 384$, a two-patch mask, one row at
$\ell' = \ell_{\max}$ with the default budget, batched against column by
column: 3.7-10 s against 25 s at $\ell_{\max} = 500$, and 28-36 s against
130 s at 1000; at 1500 and 2000 (estimated from sampled columns) 88-90 ms
and 112-179 ms per column against 245 and 521 ms. So the batched row is
faster at every size, roughly 2.5-4 times (the ranges reflect the load).
The rows are identical for any number of threads.
On a fixed grid every column costs about the same, so a full matrix,
$\sum_{\ell'} (\ell' + 1)$ columns, costs roughly $\ell_{\max}/2$ times its
last row.
A polarised row computes up to three columns per $(\ell', m')$, one per
field $Z$, with spin-2 transforms for E and B, and they are batched the
same way. In the basis $(E, iB)$ on the coefficients and $(Q, iU)$ on the
rings, the spin-2 transform of order $M$ is the real block

```math
\begin{pmatrix} \lambda^+_{LM}(\theta) & \lambda^-_{LM}(\theta) \\ \lambda^-_{LM}(\theta) & \lambda^+_{LM}(\theta) \end{pmatrix}
```

of two spin-weighted Legendre functions (the convention of `ducc0` and
HEALPix), so every spin-2 Legendre sum is again a matrix product. The B
column needs no step-1 work of its own (its coefficients are those of the
E column with E and $iB$ exchanged), and the Wick contraction of the six
spectra is done on the whole chunk at once. A polarised $m'$ carries up to
nine ring profiles (three columns, each on T, Q and U) against one for
TT, and the Legendre tables are up to three times larger, so at the same
`max_memory_gb` a chunk holds up to nine times fewer $m'$. The result
agrees with the column-by-column computation to about $10^{-15}$ of the
largest entry of the row.

Measured on the same laptop, all six spectra, $L_w = 384$, a two-patch
mask, one row at $\ell' = \ell_{\max}$ with the default budget, batched
against column by column: 59-122 s against 309 s at $\ell_{\max} = 500$,
and 591-749 s against 2241 s at 1000; at 1500 and 2000 (estimated from
sampled columns) 1.4 and 1.6-2.0 s per column against 4.4 and 5.7 s. As for
TT, the batched row is faster at every size, roughly 2.5-4 times, with no
need to raise `max_memory_gb`. From $\ell_{\max} \approx 500$ the tables
no longer fit in half of 2 GiB; the Legendre sums of a chunk are then
ducc0 transforms on the chunk's own bands rather than products with a
table, which gives the same row to rounding. The rows are identical for any
number of threads.

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
