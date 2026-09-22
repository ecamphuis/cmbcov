# The Analytical Covariance Coupling (ACC) method

ACC computes the pseudo-Cℓ covariance of a masked sky from the exact
covariance coupling kernel, evaluated once per diagonal at a single central
multipole $\ell_\ast$ and translated along that diagonal to every other
multipole (Camphuis et al. 2022, [arXiv:2204.13721](https://arxiv.org/abs/2204.13721), Sect. 5).
The expensive part, the kernels, depends only on the mask and is computed
once; each covariance for new spectra is then a set of small matrix
products. This page describes what is computed, which approximations are
made, and how far they can be trusted.

Related pages: [exact covariance](exact_covariance.md) (the reference ACC is
checked against), [B-mode kernels](bmode_kernels.md) (blocks with a B
observable), [term selection](term_selection.md) (a faster kernel
precompute), [ACC precomputation](../acc_precomputation.md) (the practical
workflow and its parameters), [approximations](../approximations.md)
(ACC against NKA and INKA).

## 1. The exact covariance in kernel form

A mask $W$ turns the true harmonic coefficients $a$ into pseudo-coefficients
$\tilde a = K a$. For temperature, $K = A_0 W S_0$ with $S_0$ the spin-0
synthesis and $A_0$ the analysis; for polarisation $K_2 = A_2 W S_2$ acts on
$(E, B)$. Write $u$ for the response to a single unit mode:

```math
u_{\ell m, LM}^{T} = \big(K e_{\ell m}^{T}\big)\mathstrut_{LM} = I_{\ell m, LM}[W],
\qquad
\big(K e_{\ell m}^{E}\big)\mathstrut_{LM} = \big(u_{\ell m, LM}^{D},\ u_{\ell m, LM}^{L}\big)\ \text{on}\ (E, B),
```

where $I_{\ell m, LM}[W]$ are the mode-coupling integrals of the paper's
Eq. 2. The three families are the **coupling fields**: T ($u^T$, the spin-0
response), D ($u^D$, the direct spin-2 response, E in to E out) and L
($u^L$, the E-to-B leakage of a single mode). A **channel** $ab$ pairs field
$a$ on the $\ell$ leg with field $b$ on the $\ell'$ leg:

```math
X_{mm'}^{ab}(L) = \sum_{M} u_{\ell m, LM}^{a}\thinspace\overline{u_{\ell' m', LM}^{b}},
\qquad ab \in \lbrace TT, DD, LL, TD, DT, TL, LT, DL, LD \rbrace .
```

The channel names use the letters T, D and L so that they are never confused
with spectra: TT, EE, BB, TE, TB and EB always denote power spectra or
covariance blocks, never channels.

The **covariance coupling kernel** of two channels is

```math
\Theta_{\ell\ell'}^{ab\times cd}(L_1, L_2) = \mathrm{Re}\sum_{mm'} X_{mm'}^{ab}(L_1)\thinspace\overline{X_{mm'}^{cd}(L_2)} .
```

For temperature the exact covariance of the paper's Eq. 7,
$\tilde\Sigma_{\ell\ell'} = \frac{2}{n}\sum_{mm'} \vert\langle \tilde a_{\ell m}\tilde a_{\ell' m'}^{\ast}\rangle\vert^2$
with $n = (2\ell+1)(2\ell'+1)$, becomes a contraction of this kernel with the
spectrum (Eq. 20):

```math
\tilde\Sigma_{\ell\ell'}^{TT\times TT} = \frac{2}{n}\sum_{L_1 L_2} C_{L_1}^{TT}\thinspace\Theta_{\ell\ell'}^{TT\times TT}(L_1, L_2)\thinspace C_{L_2}^{TT}
\equiv \frac{2}{n}\thinspace C^{TT}\cdot\Theta_{\ell\ell'}^{TT\times TT}\cdot C^{TT} .
```

The kernel keeps $X$ complex: taking the real part of $X$ before the product
is only correct for a mask with a mirror symmetry in azimuth. Every block
between spectra $XY$ and $ZW$ follows from Wick's theorem,

```math
\mathrm{Cov}\big(\tilde C_{\ell}^{XY}, \tilde C_{\ell'}^{ZW}\big)
= \frac{1}{n}\sum_{mm'}\Big[R^{XZ}\thinspace\overline{R^{YW}} + R^{XW}\thinspace\overline{R^{YZ}}\Big]\mathstrut_{\ell m, \ell' m'},
\qquad
R_{\ell m, \ell' m'}^{XZ} = \sum_{Y Y'}\sum_{LM} C_{L}^{YY'}\thinspace v_{\ell m, LM}^{XY}\thinspace\overline{v_{\ell' m', LM}^{ZY'}},
```

with $v^{XY} = (K e^X)^Y$ the columns of $K$. On the Gauss-Legendre grid $K$
is Hermitian, and this $R$ is the complex conjugate of
$\langle\tilde a^X\tilde a^{Z\ast}\rangle$; the covariance is real, so the
conjugation changes nothing. Inserting the columns turns every block into a
sum of terms $C^{a}\cdot\Theta^{c_1\times c_2}\cdot C^{b}$. For the T/E blocks
with $C^{BB} = 0$:

| block | $n\thinspace\mathrm{Cov}$ |
|---|---|
| TT×TT | $2\thinspace TT\cdot\Theta^{TT\times TT}\cdot TT$ |
| TT×EE | $2\thinspace TE\cdot\Theta^{TD\times TD}\cdot TE$ |
| TT×TE | $2\thinspace TT\cdot\Theta^{TT\times TD}\cdot TE$ |
| EE×EE | $2\thinspace EE\cdot\Theta^{DD\times DD}\cdot EE$ |
| EE×TE | $2\thinspace EE\cdot\Theta^{DD\times DT}\cdot TE$ |
| TE×TE | $TT\cdot\Theta^{TT\times DD}\cdot EE + TE\cdot\Theta^{TD\times DT}\cdot TE$ |

where $TT$ inside a product stands for $C^{TT}$, and so on. Two points matter
in practice. The second term of TE×TE uses the $TD\times DT$ kernel, not
$TD\times TD$: the two coincide on the diagonal $\ell = \ell'$ and differ
off it, so the cache stores both channels. And with $C^{BB} \ne 0$ the
EE×EE, TE×TE, TE×EE and EE×TE blocks gain further terms, described in
[B-mode kernels](bmode_kernels.md).

Computing $\Theta_{\ell\ell'}$ at every $(\ell, \ell')$ is as expensive as
the exact covariance itself, which scales as $\ell_{\max}^5$
([exact covariance](exact_covariance.md)). ACC avoids that.

## 2. Normalisation (Eqs. 22 and 23)

The paper's Eq. 3 defines the symmetric coupling operator of a function
with power spectrum $A_L$,

```math
\Xi_{\ell\ell'}^{ss'}[A] = \sum_{L} \frac{2L+1}{4\pi}\thinspace A_L
\begin{pmatrix}\ell & \ell' & L\\ s & -s & 0\end{pmatrix}
\begin{pmatrix}\ell & \ell' & L\\ s' & -s' & 0\end{pmatrix},
```

so that the MASTER matrix is $(2\ell'+1)\thinspace\Xi$. Below, $\Xi[W^2]$ is
this operator for the power spectrum of the squared mask $W^2$, and the four
channels used are $\Xi^{00}$, $\Xi^{20}$,
$\Xi^{EE\to EE} = \frac{1}{2}(\Xi^{22} + \Xi^{2,-2})$ and
$\Xi^{EE\to BB} = \frac{1}{2}(\Xi^{22} - \Xi^{2,-2})$.

Completeness of the spherical harmonics gives the sum rule of Eq. 22:

```math
\sum_{L_1 L_2}\Theta_{\ell\ell'}^{TT\times TT}(L_1, L_2) = n\thinspace\Xi_{\ell\ell'}^{00}[W^2] .
```

ACC therefore splits the kernel into an amplitude, carried by $\Xi$, and a
unit-sum shape $\bar\Theta = \Theta / \sum\Theta$. Eq. 23 is

```math
\tilde\Sigma_{\ell\ell'} \approx 2\thinspace\Xi_{\ell\ell'}^{00}[W^2]\thinspace C\cdot\bar\Theta\cdot C ,
```

which is an identity wherever $\bar\Theta$ is the kernel of that very pair.
Its purpose is the next step: the shape is reused elsewhere, while $\Xi$ is
always evaluated at the $(\ell, \ell')$ being computed, so the growth of the
amplitude with $\ell$ is carried exactly by a cheap MASTER-type kernel.

## 3. The central multipole and the translation along diagonals (Eq. 33)

The kernel shape depends mainly on the offset $\Delta = \ell' - \ell$ and on
the mask, and only slowly on $\ell$ itself once $\ell$ is large compared with
the mask's own harmonic width. ACC computes the kernels once, at
$(\ell_\ast, \ell_\ast + \Delta)$ for each diagonal, and translates them
(Eq. 33):

```math
\bar\Theta_{\ell, \ell+\Delta}(L_1, L_2) \approx \bar\Theta_{\ell_\ast, \ell_\ast+\Delta}(L_1 - s, L_2 - s),
\qquad s = \ell - \ell_\ast .
```

The approximation is exact at $\ell = \ell_\ast$ and its error grows with
$\vert\ell - \ell_\ast\vert$ (Sect. 6). In the code, a kernel is an
$S\times S$ matrix with $S = 2\thinspace n_{\mathrm{side}}$ of the ACC working
resolution, and its index $i$ stands for the multipole
$L = \min(\ell, \ell') - \ell_\ast + i$. Rows that fall below $L = 0$ are
dropped. At the top, the spectra must extend past the reported $\ell_{\max}$
to

```math
\ell_{\mathrm{int}} = \ell_{\max} + \max(0,\thinspace S - 1 - \ell_\ast)
```

so that no reported element loses kernel weight (`acc_window_pad`,
`acc_internal_lmax`); the covariance is still returned up to $\ell_{\max}$.

Because an ACC element depends on $(\ell, \ell')$ only through
$\min(\ell, \ell')$ and $\Delta$, every assembled block is symmetric under
$\ell \leftrightarrow \ell'$. That is right for the blocks of one spectrum
with itself (TT×TT, EE×EE, TE×TE). A block between two different spectra,
such as TT×EE, is not symmetric in the exact covariance, and ACC returns one
value for both elements. The best a symmetric approximation can do is their
mean, so ACC carries half the exact asymmetry as an error; it grows with
$\vert\ell - \ell'\vert$ and was smaller than the other error terms on the
masks tested.

The choice of $\ell_\ast$ is a trade-off. A larger $\ell_\ast$ makes the
translation more accurate, because the kernel shape converges as $\ell$
grows, but the kernels then need a higher working resolution and cost more.
For an apodised survey mask covering about 4% of the sky, $\ell_\ast$ around
200 to 250 is a good choice (Sect. 7). `CovarianceConfig` warns when
$\ell_\ast \le 150$.

## 4. dmax

ACC computes the $\mathrm{dmax}$ diagonals $\vert\ell - \ell'\vert < \mathrm{dmax}$
and sets every other element to zero. Each diagonal $\Delta$ needs its own
kernels at $(\ell_\ast, \ell_\ast + \Delta)$, so the precompute builds the
pairs $\ell' = \ell_\ast, \ldots, \ell_\ast + \mathrm{dmax} - 1$
(`coupling_ellprange`), and its cost is proportional to $\mathrm{dmax}$.

The right $\mathrm{dmax}$ is set by how fast the true covariance decays away
from the diagonal, which depends on the mask: a smaller or more structured
footprint couples more multipoles. On the 4% apodised footprint of Sect. 7,
the exact ratio $\tilde\Sigma_{\ell'+d, \ell'}/\tilde\Sigma_{\ell'\ell'}$ is
0.84 to 0.94 at $d = 1$, about 0.04 at $d = 10$ and about $10^{-3}$ at
$d = 50$, stable over $20 \le \ell' \le 400$; $\mathrm{dmax} = 20$ keeps
every element above about 1% of the diagonal. For another mask, compute one
exact row (`exact_covariance_row`) and read off where it drops below your
target.

## 5. The T/E blocks and the spin-weight ladder

For spin 2 the completeness relation holds for the full spin-2 field, that
is for D and L together, not for D alone. The identities that follow are

```math
\sum_{L_1 L_2}\Big[\Theta^{TT\times DD} + \Theta^{TT\times LL}\Big] = n\thinspace\Xi^{20}[W^2],
\qquad
\sum_{L_1 L_2}\Big[\Theta^{DD\times DD} + 2\thinspace\Theta^{DD\times LL} + \Theta^{LL\times LL}\Big] = n\thinspace\Xi^{EE\to EE}[W^2] .
```

A single T/E kernel such as $\Theta^{TT\times DD}$ therefore has no exact
$\Xi$ of its own. Its natural normalisation follows a spin-weight rule: each
letter D or L of the kernel pair weighs $\frac{1}{2}$, T weighs 0, and the
pair's total weight $w$ picks

```math
\Xi^{(w)} = \Xi^{00}\left(\frac{\Xi^{20}}{\Xi^{00}}\right)^{w},
```

which lands on an existing channel at the integer rungs: $\Xi^{00}$ at
$w = 0$, $\Xi^{20}$ at $w = 1$ and $\Xi^{EE\to EE}$ at $w = 2$. A run whose
observables are all T/E (TT, EE, TE) applies Eq. 23 to each Wick contraction
with the channel of this ladder (`Cov.norm_Xi`):

| Wick contraction (kernel pair) | $w$ | $\Xi$ used | comment |
|---|---|---|---|
| $TT\times TT$ | 0 | $\Xi^{00}$ | Eq. 22, exact |
| $TT\times DD$, $TD\times TD$, $TD\times DT$ | 1 | $\Xi^{20}$ | exact up to E-to-B leakage |
| $DD\times DD$ | 2 | $\Xi^{EE\to EE}$ | exact up to E-to-B leakage |
| $TT\times TD$ | $\frac{1}{2}$ | $\Xi^{20}$ | nearest rung; no channel exists |
| $DD\times TD$, $DD\times DT$ | $\frac{3}{2}$ | $\Xi^{EE\to EE}$ | nearest rung; no channel exists |

"Up to E-to-B leakage" is the leading error of these blocks at $\ell_\ast$,
described next. A run with any B observable instead normalises every
expanded Wick term separately, T/E blocks included, with a rule that is
exact at $\ell_\ast$ ([B-mode kernels](bmode_kernels.md), Sect. 5).

## 6. What limits the accuracy

**Translation error (Eq. 33).** This is zero at $\ell_\ast$ and grows with
$\vert\ell - \ell_\ast\vert$, faster below $\ell_\ast$ than above it. It is the
only error of TT×TT, whose normalisation is exact, so TT×TT at a given
$\ell$ is the floor for every other block. No cheap proxy for it is known:
the run's error budget (`error_budget.txt`) reports only
$\vert\ell - \ell_\ast\vert$ at the band edges for it. Measure it with exact
rows (Sect. 7).

**E-to-B leakage in the T/E normalisation.** Dividing $\Theta^{TT\times DD}$
by its own sum and multiplying by $\Xi^{20}$ over-counts it by
$1 + \lambda$, with

```math
\lambda = \frac{\sum\Theta^{TT\times LL}}{\sum\Theta^{TT\times DD}}
```

evaluated at $(\ell_\ast, \ell_\ast)$. The over-count comes from the missing
$\Theta^{TT\times LL}$ part of the identity above. A T/E block with $n_E$ polarised legs
among its four spectrum letters is biased high by about
$\frac{1}{2}n_E\lambda$ at $\ell_\ast$: $2\lambda$ for EE×EE, $\lambda$ for
TE×TE and TT×EE. The leakage falls with $\ell$, roughly as
$\langle L(L+1)\rangle_{W^2}/\ell^2$ times an order-one, mask-dependent
factor, where $\langle L(L+1)\rangle_{W^2}$ is the power-weighted mean of
$L(L+1)$ over the squared mask's spectrum (`mask_spectral_moment`). Sharp
edges and point-source holes raise it. On the 4% footprint of Sect. 7,
$\lambda = 4.9\times10^{-3}$ at $\ell_\ast = 250$ and
$1.4\times10^{-3}$ at $\ell_\ast = 500$, so a T/E-only run at
$\ell_\ast = 250$ has Cov(EE,EE) about 1% high at $\ell_\ast$. The run's
error budget quotes $\lambda$ from the kernels in use when the cache holds
the $TT\times LL$ kernel.

**Half-integer rungs.** TT×TE and EE×TE have no exact channel, and the
nearest rung leaves a small residual of either sign on top of the leakage.

**Mask band-limit.** The kernels are built from the mask band-limited at
$L_w$ (default $3\thinspace n_{\mathrm{side}} - 1$ of the ACC resolution), while
$\Xi$ is computed from the mask at the run's resolution. The two agree when
the mask has negligible power above $L_w$. That is the case for the apodised
footprint of Sect. 7 at $L_w = 512$; hard edges and point-source holes need a
larger $L_w$.

**Symmetry.** Blocks between two different spectra carry half their exact
$\ell\leftrightarrow\ell'$ asymmetry (Sect. 3).

**B modes.** Blocks with a B observable are dominated by leaked E power and
have their own limits, larger than the ones above
([B-mode kernels](bmode_kernels.md), Sect. 8).

## 7. When to trust it

Measured on an apodised survey footprint covering about 4% of the sky, at
HEALPix $n_{\mathrm{side}} = 512$ and with $\langle L(L+1)\rangle_{W^2} = 284$,
with kernels on the Gauss-Legendre grid at $n_{\mathrm{side}} = 256$, mask band-limit
$L_w = 512$, $\mathrm{dmax} = 20$, against exact rows computed with 200
multipoles of spectrum margin above each row:

- **TT only**, Planck 2018 lensed TT plus 10 μK-arcmin white noise with a
  low-multipole knee at $\ell = 200$, rows $\ell' = 250$ to 400 and the 20
  diagonals: with $\ell_\ast = 250$ ACC is exact at $\ell_\ast$ to
  $5\times10^{-5}$, and every element is within 0.3% of the exact one, the
  worst being the diagonal at $\ell' = 400$. With $\ell_\ast = 75$, rows
  $\ell' = 78$ to 150 are 1 to 2% off.
- **Far below** $\ell_\ast$ the translation dominates: at $\ell = 100$ with
  $\ell_\ast = 250$, the diagonal of TT×TT is 5.6% high.
- **T/E blocks**, Planck 2018 lensed spectra without noise, rows at
  $\ell' = 150$ to 400, error measured as the largest
  $\vert\mathrm{ACC} - \mathrm{exact}\vert/\sqrt{\mathrm{Cov}\mathstrut_{aa}\mathrm{Cov}\mathstrut_{bb}}$
  within 19 multipoles of each row, i.e. in units of the diagonal: with the
  per-term normalisation of a run with a B observable, every T/E block is
  within 0.43% for $\ell' \ge 200$ and within 0.47% at $\ell' = 150$, except
  TT×TT at 1.5%, the translation floor. With the T/E-only normalisation of
  Sect. 5 the worst errors are 4 to 8 times larger, for example EE×EE 2.5%
  against 0.34%.

Rules of thumb that follow:

- Put $\ell_\ast$ where the covariance matters most, and not far below the
  bins you care about: the error grows faster below $\ell_\ast$ than above.
- Treat multipoles well below $\ell_\ast$ as approximate at the few-percent
  level.
- On a new mask, check. Compute a few exact rows with
  `exact_covariance_row` (TT) or `exact_covariance_row_pol` (all spectra) on
  the Gauss-Legendre grid, at one $\ell'$ near $\ell_\ast$ and one near each
  end of your multipole range, with an internal band-limit `lmax_int` that
  leaves a margin above the rows ([exact covariance](exact_covariance.md)).
  Compare them with the corresponding rows of the ACC pseudo-Cℓ
  covariance, before any PolSpice transform or binning. A few rows cost
  seconds to minutes, against hours for the full exact matrix.
