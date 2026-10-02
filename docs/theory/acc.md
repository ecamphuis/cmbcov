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

The sum rule is exact only when $\Theta$ and $\Xi$ come from the same
band-limited mask and $\Theta$ is summed over its full support. In practice
$\Xi$ (`Cov.Xi`) is built from the mask at its own band limit, the kernels
from the mask truncated at `acc_precompute.lw` and stored for
$L_1, L_2 < 2\thinspace n_{\mathrm{side}}$. The error budget written beside every
ACC covariance (`error_budget.txt`) therefore evaluates the sum rule at
$(\ell_\ast, \ell_\ast + d)$ for every diagonal $d < d_{\max}$ and reports
$\max_d \left|\sum\Theta^{TT\times TT}/(n\thinspace\Xi^{00}) - 1\right|$,
warning above $10^{-2}$. It is the share of the mask's weight that the kernel
shape lacks and $\Xi$ carries, so it bounds the error on an element rather
than equalling it. A value already large at $d = 0$ means the mask has power
beyond `lw`; one that grows with $d$ means the kernel range cuts the tail of
$\Theta$ (raise `nside`). It is a check, not a correction: nothing enters
the covariance.

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

An ACC element depends on $(\ell, \ell')$ only through
$\min(\ell, \ell')$, $\Delta$ and the orientation of its kernel. A block of
one spectrum with itself (TT×TT, EE×EE, TE×TE) is symmetric under
$\ell \leftrightarrow \ell'$, exactly and in ACC. A block between two
different spectra, such as TT×EE, is not:
$\mathrm{Cov}(\tilde C_{\ell}^{a}, \tilde C_{\ell'}^{b}) \ne \mathrm{Cov}(\tilde C_{\ell'}^{a}, \tilde C_{\ell}^{b})$,
and the kernel of the block $(a, b)$ at $(\ell_\ast, \ell_\ast+\Delta)$ is
the right one only with $a$ on the lower multipole. A T/E-only run therefore
computes the upper triangle ($\ell \le \ell'$) of the block $(a, b)$ from
the Wick contractions of $(a, b)$, and the lower triangle from those of
$(b, a)$, since
$\mathrm{Cov}(\tilde C_{\ell+\Delta}^{a}, \tilde C_{\ell}^{b}) = \mathrm{Cov}(\tilde C_{\ell}^{b}, \tilde C_{\ell+\Delta}^{a})$:
the lower triangle of $(a, b)$ is the transposed upper triangle of $(b, a)$.
An element then depends on its two spectra and on which of its multipoles is
the smaller, never on the order in which the run lists its spectra. Blocks
between two TT spectra or two EE spectra have the same contractions in both
orientations and are unchanged, so T-only and E-only runs are too, bit for
bit. The other blocks cost up to twice the kernel products; over a whole
T/E run the assembly does 37%, 54% and 61% more of them with 1, 2 and 3
frequencies. Runs with a B observable follow the same rule for every
block, and their kernel-pair sets include both orientations of every block
([B-mode kernels](bmode_kernels.md), Sect. 6).

That cost is won back by computing the blocks of a run together. On
diagonal $\Delta$ the band of one contraction is, for every
$m = \min(\ell, \ell')$ at once, the row-wise product of
$W^{a}\thinspace\bar\Theta$ with $W^{b}$, where row $m$ of the window
matrix $W^{a}$ holds the $S$ multipoles of $C^{a}$ the translated kernel
sees. The matrix product $W^{a}\thinspace\bar\Theta$, about
$n_\ell S^2$ operations, is nearly all of the cost; the row-wise product
with $W^{b}$ is $n_\ell S$. Many blocks of a run need the same product, and
many more the same left factor $W^{a}\thinspace\bar\Theta$ with a
different right spectrum. `Cov.compute_covariance_matrix` therefore hands
the strategy all its blocks (`compute_covariance_terms`); ACC groups the
blocks that share left factors into batches bounded in memory (512 MiB)
and, diagonal by diagonal, computes every left factor of a batch once and
every distinct product once. Each product is evaluated by exactly the
operations a block alone would use, and each block sums its terms in the
same order, so the result is bit-identical; products are shared by the
content of the spectra, so equal arrays under different keys (TE and ET of
one frequency pair, in a survey data model) share them too. On the survey
footprint above this takes the matrix products per diagonal from 635 to
109 with 3 frequencies (392 before the orientation fix) and from 136 to 43
with 2 (88 before); the assembly of the 3-frequency run went from 514 s
to 127 s (341 s before the fix). One frequency gains nothing: its 11 products per
diagonal (8 before) have 11 different left factors. The cost that remains
is set by the distinct left factors of the run, not by the number of
blocks.

Up to cmbcov 0.2.0 both triangles took the upper value, and which element of
a block had the right kernel depended on the order of the spectra: in a
two-frequency run, Cov(TE 90×90, EE 90×150) put TE on the lower multipole
and Cov(EE 90×90, TE 90×150) put EE there. In a multi-frequency T/E run the
CMB then no longer cancelled in the differences between frequency pairs,
whose true variance is set by noise and foregrounds alone, and the matrix was
not positive definite. On a 4% apodised survey footprint at $\ell_{\max} = 3500$
($\ell_\ast = 250$, $\mathrm{dmax} = 100$, bins of 50 from $\ell = 200$) the
binned correlation matrix had 20 negative eigenvalues down to
$-1.3\times10^{-4}$ with 2 frequencies, and 47 down to $-1.8\times10^{-4}$
with 3; with both orientations it has none, its lowest eigenvalues being
$+7.6\times10^{-9}$ and $+1.5\times10^{-10}$. With the same spectra at every
frequency the two maps are one map and the exact covariance vanishes on the
frequency differences; the raw matrix restricted to them (2 frequencies,
$\ell_{\max} = 800$) had eigenvalues of $\pm3\times10^{-2}$ in correlation
units with the old assembly, and has $3\times10^{-15}$ with the new one.
Against NKA, which has no orientation to choose (3 frequencies, same run),
the new matrix's variance in NKA's 50 weakest directions is 1.000 to 1.023
times NKA's, and its lowest eigenvalue $+1.48\times10^{-10}$ against NKA's
$+1.48\times10^{-10}$; the old one ranged from $-24$ to $+27$ times NKA's.
The normalisation of Sect. 5 played no part: with the old orientation,
replacing the per-contraction $\Xi$ by the field-factorised ladder
$\Xi^{00}(\Xi^{20}/\Xi^{00})^{w}$ moved the lowest eigenvalue by 2% and did
not remove a single negative one.

The orientation does not remove the asymmetry error itself. The Eq. 23
amplitude of every element comes from the symmetric $\Xi_{\ell\ell'}$, and
the exact asymmetry is mostly one of amplitude, so the two orientations give
nearly the same value: on the rippled cap of the test suite at
$\ell_\ast = 8$, where the exact asymmetry of TT×TE, TT×EE and TE×EE at
$\ell_\ast$ is 0.7 to 5% of the element for $\Delta = 1$ to 5, the two
orientations differ by less than 9% of it, with either sign. ACC thus
carries nearly the whole exact asymmetry on the lower triangle (the upper
one is exact at $\ell_\ast$ up to the normalisation of Sect. 5). It grows
with $\vert\ell - \ell'\vert$ and was smaller than the other error terms on
the masks tested; on that survey footprint the two orientation rules
differ by at most $1.3\times10^{-4}$ in correlation and not at all in
$\sigma$.

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
(`coupling_ellprange`), and its cost is proportional to $\mathrm{dmax}$. The
recompute step that follows (spectra, noise, beams, binning) also grows with
$\mathrm{dmax}$, since it touches $\mathrm{dmax}$ diagonals, but stays cheap:
a few seconds at $\mathrm{dmax} = 20$ to $100$ on a 66-bandpower T/E run.

For binned output, the right $\mathrm{dmax}$ is set by the bin width $w$, not
by how far the true correlation reaches: $\mathrm{dmax} \ge w$ for correct
bandpower error bars, $\mathrm{dmax} \ge 2w$ for correct correlations between
adjacent bandpowers. A bandpower's variance sums every multipole pair inside
one bin of width $w$, i.e. separations $0$ to $w - 1$; a value of
$\mathrm{dmax}$ below $w$ drops the pairs at separations $\mathrm{dmax}$ to
$w - 1$ and biases every $\sigma$ low. The covariance of two adjacent
bandpowers sums pairs straddling the two bins, separations $1$ to $2w - 1$,
and the number of such pairs keeps growing with separation up to $w$, so the
many small terms at intermediate separation matter as much as the few large
ones near the diagonal -- hence the stricter $\mathrm{dmax} \ge 2w$.

Measured on a 4% survey footprint, bins of width $w = 50$, one frequency
T/E, $\ell_{\max} = 3500$, against 500 simulations: at $\mathrm{dmax} = 20$
every $\sigma$ is 0.7% low, uniformly across multipole and TT/TE/EE, and the
diagonal has converged by $\mathrm{dmax} = 50$; the mean adjacent-bandpower
correlation is 0.010 at $\mathrm{dmax} = 20$ rising to 0.023 at
$\mathrm{dmax} = 60$. At $\mathrm{dmax} = 100 = 2w$ it reaches 0.024 and
stops rising -- the diagonal itself has not moved since $\mathrm{dmax} = 60$
(a $1.4\times10^{-4}$ change), so the truncation is by then fully accounted
for. The remaining gap to the simulations' $0.030 \pm 0.003$, i.e. 0.006,
is $1.8\sigma$: consistent with Monte Carlo noise from 500 realisations, not
with further truncation.

On this footprint with three frequencies, the smallest eigenvalue of the
correlation matrix was $-1.6\times10^{-5}$ at $\mathrm{dmax} = 20$ and
$-1.8\times10^{-4}$ at $\mathrm{dmax} = 100$ up to cmbcov 0.2.0. Neither
came from $\mathrm{dmax}$: both were the orientation of the lower triangle
of the blocks between different spectra (Sect. 3), and with each element in
its own orientation the smallest eigenvalue is $+1.5\times10^{-10}$ at both
$\mathrm{dmax} = 20$ and $100$, as in NKA. A larger $\mathrm{dmax}$ still
brings in more ACC-approximated off-diagonal elements, so check the
eigenvalues of whatever $\mathrm{dmax}$ you settle on.

For an *unbinned*, per-multipole covariance there is no bin width to size
$\mathrm{dmax}$ against, so instead it is set by how fast the true
covariance decays away from the diagonal, which depends on the mask -- a
smaller or more structured footprint couples more multipoles. On the 4%
apodised footprint of Sect. 7, the exact ratio
$\tilde\Sigma_{\ell'+d, \ell'}/\tilde\Sigma_{\ell'\ell'}$ is 0.84 to 0.94 at
$d = 1$, about 0.04 at $d = 10$ and about $10^{-3}$ at $d = 50$, stable over
$20 \le \ell' \le 400$; $\mathrm{dmax} = 20$ keeps every element above about
1% of the diagonal. For another mask, compute one exact row
(`exact_covariance_row`) and read off where it drops below your target.

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

**Symmetry.** Blocks between two different spectra carry nearly their whole
exact $\ell\leftrightarrow\ell'$ asymmetry on the lower triangle, since the
Eq. 23 amplitude comes from the symmetric $\Xi$ (Sect. 3).

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
