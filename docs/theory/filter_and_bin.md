# Filter-and-bin map-making

Some maps are made by filtering each scan of the time stream and binning
it. The filter removes large scales along the scan direction, so the map is
the sky seen through a filter before the analysis mask is applied, and the
pseudo-spectrum covariance computed from the biased spectra and the
unfiltered kernels is wrong. The optional `map_filter` block corrects the
T and E blocks (TT, TE, EE and their cross-blocks) with one factor per pair
of multipoles. Blocks with a B leg are left uncorrected (Sect. 4).

## 1. What is corrected

With a scan along iso-latitude rings and a Fourier high-pass along each ring
(azimuthal order $M$ of the sky mode cut at $m_c = l_x \sin\theta$), the
map-making operator followed by the analysis mask is $K_0 = A_0 W F S_0$ on
T and $K_2 = A_2 W F S_2$ on $(E, B)$, with $S$ the synthesis, $A$ the
analysis, $W$ the mask and $F$ the ring filter, applied to the T map and to
the Q and U maps separately. The filtered sum rule of the mask is

$$\Xi^F(\ell, \ell') = \frac{1}{n n'} \sum_{m m'} \left| (K K^\dagger)_{\ell m, \ell' m'} \right|^2,
\qquad
\rho(\ell, \ell') = \frac{\Xi^F(\ell, \ell')}{\Xi(\ell, \ell')},$$

where $n = 2\ell + 1$, $n' = 2\ell' + 1$, and $\Xi$ is the same sum for the
unfiltered mask. $2\thinspace\Xi^F$ is the covariance of the filtered
estimator for a white spectrum, so $\rho$ is what the filter does to it,
including the dependence on the mask shape and on $\ell - \ell'$ that a
function of the transfer function alone cannot carry.

For polarisation there is one such ratio per sum-rule channel, the same
channels that normalise the unfiltered blocks (`Cov._compute_norm_xi`):

| channel | blocks | $2\thinspace\Xi^{F,ch}$ is the exact filtered block |
|---|---|---|
| `00` | TT x TT | TT x TT with only $C^{TT} = 1$ |
| `20` | TT x EE, TT x TE, TT x ET, TE x TE, TE x ET, ET x TE, ET x ET, and transposes | TT x EE with only $C^{TE} = 1$ |
| `EE` | EE x EE, EE x TE, EE x ET, and transposes | EE x EE with only $C^{EE} = 1$ |

Each raw T/E pseudo-spectrum block $\tilde\Sigma^{s_1 s_2}$ (computed from
the biased spectra, which contain the transfer functions) is replaced by

```math
\tilde\Sigma^{F, s_1 s_2}(\ell_1, \ell_2) = \frac{\rho^{ch}(\ell_1, \ell_2)}{F^{L, s_1}_{\ell_1} \thinspace F^{R, s_2}_{\ell_2}} \thinspace \tilde\Sigma^{s_1 s_2}(\ell_1, \ell_2),
```

with $ch$ the block's channel and $F^{L, s_1}$, $F^{R, s_2}$ the `fl` of the
left leg's spectrum ($s_1$, e.g. TT) for its frequency pair and of the right
leg's spectrum ($s_2$, e.g. EE) for its frequency pair (1 without `fl`; ET
uses the TE `fl`). The factor multiplies the raw block, before the PolSpice
transform, $D_\ell$ scaling, debiasing and binning, for NKA, INKA and ACC
alike. With `spectrum_aware: true` the TT x TT blocks take their own
spectrum-aware factor instead, with no `fl` division (Sect. 2, last
subsection). The raw-block cache (`save_raw_blocks`) keeps the uncorrected block,
so changing the filter never invalidates it.

**The TE transfer function** should be a smooth positive reference, such as
$\sqrt{F^{TT} F^{EE}}$ or the ratio of filtered to unfiltered mean
pseudo-TE for a positive reference $C^{TE} = \sqrt{C^{TT} C^{EE}}$. It must
not be the ratio of the mean pseudo-TE spectra of the actual, sign-changing
$C^{TE}$: that ratio agrees with the reference to about 1% except within a
few multipoles of a zero of the pseudo-TE spectrum, where it jumps (to 0.75
and 0.47 times the reference in the research measurements) and puts a
spurious feature into TE x TE (+0.7% instead of 0.2%). The two smooth
choices agree to 0.2% at $\ell \ge 100$ there.

## 2. How $\rho$ is obtained

$\rho$ has no closed form. It is estimated with random probes: a real
unit-variance field $z$ supported on a single degree $\ell'$ (the right-hand
field of the block) is pushed through the filtered and the unfiltered
operator of the channel, and the ratio of the two power spectra
$\sum_m \lvert y_{\ell m} \rvert^2$ gives $\rho^{ch}(\ell, \ell')$ for every
$\ell$ at once, for the $\lvert \ell - \ell' \rvert \le$ `dmax` near the
diagonal:

- `00`: $z$ a spin-0 field, $y = K_0 K_0^\dagger z$;
- `20`: $z$ an E mode ($B = 0$), $u = [K_2^\dagger (z, 0)]_E$ (the B output
  is dropped), $y = K_0 u$ with $u$ read as a spin-0 field;
- `EE`: $z$ an E mode, $u$ as for `20`, $y = [K_2 (u, 0)]_E$.

With the complete basis of the degree (used when $2\ell' + 1$ does not
exceed `nprobe`) these sums are exactly $n n'\thinspace \Xi^{F,ch}$; the test
suite checks them against exact white-spectrum rows to round-off. The `20`
probe has T on the left and E on the right; its transpose agreed with it to
0.02% at $\ell' = 160$, and the interpolated $\rho$ is symmetric, so one
matrix serves a block and its transpose. Only the channels a run's blocks
need are computed: a spin-2 transform costs two to three spin-0 ones, and
the three channels together cost four to six times the TT probing (3.8 times
in a single-thread timing at $\ell' = 200$, six times in the research runs). The probes are
run at nodes $\ell'$ equal to `node_min`, `node_min * node_ratio`, ..., plus
dense nodes near the cut (subsection "Near the cut" below), and $\rho$ is
interpolated between them (it is smooth in $\ell$). The error of a node with
random probes falls as $1/\sqrt{N_{\mathrm{probe}}\thinspace \ell'}$. The
result is cached in the kernel directory under the mask digest, the
channels, the nodes with their probe counts and band-limits, and all filter
parameters.

### Spectrum-aware TT x TT (`spectrum_aware: true`)

$\rho$ is a white-spectrum quantity. A sharp latitude-dependent cut also
moves power between multipoles: it chops each low $\ell$ mode in $\theta$
where $l_x \sin\theta$ crosses its $M$, and the step puts some of that
mode's power at high $\ell$. For a steep spectrum (the beam-damped TT
signal) this adds variance and neighbour correlation at high $\ell$ that no
white ratio can carry. With `spectrum_aware: true` every TT x TT block is
instead probed with the run's own spectra. For the block of the spectra
$ab$ and $cd$ (frequency pairs), the exact filtered covariance is

```math
\Sigma^F(\ell, \ell') = \frac{1}{n n'} \sum_{m m'} \left[ R^{ac} \overline{R^{bd}} + R^{ad} \overline{R^{bc}} \right]_{\ell m, \ell' m'}, \qquad R^{pq} = K^F C^{pq}_{pre} K^{F\dagger},
```

with $C_{\mathrm{pre}}^{pq}$ the TT spectrum of the sky before the filter. The
unfiltered $\Sigma$ has the same form with $K$ and $C_{\mathrm{used}}^{pq}$, the
spectrum the covariance is computed with (signal times beams, pixel window
and `fl`, plus noise). The same probes $z$ go through both sides, and

```math
g_C(\ell, \ell') = \frac{\sum_{\mathrm{probes}} \sum_{\mathrm{terms}} \sum_m \mathrm{Re}\left[ (R^{p_1}_F z)_{\ell m} \overline{(R^{p_2}_F z)_{\ell m}} \right]}{\text{the same with } K \text{ and } C_{used}}
```

is the ratio of the two blocks. The block is multiplied by $g_C$
(interpolated between the nodes like $\rho$) and **not divided by `fl`**:
the transfer function is inside the ratio. With the complete basis of a
degree this reproduces the exact filtered row from the exact unfiltered one
to round-off (below $2 \times 10^{-15}$ in the test suite, including two
frequencies with different spectra). The other T/E blocks keep their white channel, and
`00` is then not probed.

- $C_{pre}$: the signal (CMB plus foregrounds) times $B_1 B_2$ and the
  pixel window, without `fl`, plus the noise before the filter, taken as
  `nl / fl` where `fl` exceeds 0.01 and `nl` itself below that. This
  assumes the noise is filtered like the signal (its transfer function
  equals `fl`). In the survey test (Sect. 3) the noise transfer function
  changed the white-ratio result by under 0.3% (median); it was not
  re-measured for the spectrum-aware ratio. Below the cut (`fl` near 0) the
  pre-filter noise is not recoverable, so it is effectively neglected
  there; the signal dominates those multipoles.
- The probes carry every sky multipole up to the node's internal band-limit
  (node + `dmax` + the mask coupling width), so low $\ell$ power scattered
  to the node is included. If the spectra are shorter (NKA and INKA load
  them to `lmax` only), the internal band-limit is capped at their length
  for both operators.
- **Grid.** The sharp cut is not band-limited, and on a GL grid the
  scattered power above the grid aliases back. On the minimal exact grid
  this overstated it: the excess of the mean filtered signal spectrum over
  the white one at $\ell$ = 3400-3500 was 6.7% on the minimal grid against
  4.2% on a grid twice as fine and 4.2% in the HEALPix nside-2048
  simulations. The spectrum-aware probes therefore use a grid of at least
  twice the internal band-limit.
- **Cost and caching.** One frequency costs what the white `00` channel
  costs on the same grid (seven spin-0 transforms per probe). Each further
  frequency pair adds two transforms per operator. The ratio depends on
  the spectra, so it is cached under a digest of both spectrum sets and the
  Wick terms: **changing the spectra probes again**, while the white
  channels do not depend on them. That fits the fixed-mask workflow less
  well: a spectrum change costs a re-probe, which is affordable at the
  survey scale below.

### Near the cut

Between the cut and twice the cut, $\rho$ rises from nearly zero to a few
tenths and the spectrum-aware ratio peaks (near $1.25\thinspace l_x$ for a
sharp cut). A survey-like test (the SPT-3G-like mask at nside 512, sharp
and smooth cuts at $l_x = 300$, INKA, 13200 paired simulations, bins of 50)
traced the v0.6.0 errors there to four sources:

- **Interpolation.** PCHIP across geometric nodes a factor 1.25 apart (298,
  372, 465 at this cut) missed the rise: the white TT variance of the bin
  350 to 399 was 2.3% low and the spectrum-aware table 5% high at
  $\ell' = 400$ from this alone. The error is set by the node spacing in
  units of $l_x$, so it does not shrink at larger scales.
- **Probe noise.** With 32 random probes a node near the cut is off by 2
  to 3%, the same for every $d$ (one seed put -3.3% into the bin 350 to
  399).
- **The probes' band-limit.** A sharp cut scatters power into the nodes at
  the cut from far multipoles, which the internal band-limit
  ($\ell'$ + `dmax` + the mask coupling width) leaves out: the white ratio
  of the node at $l_x$ was 11% low (mean over $\lvert d\rvert \le 10$),
  0.6% at $1.07\thinspace l_x$ and 0.3% at $1.13\thinspace l_x$.
- **The white ratio itself.** It cannot carry the power a sharp cut
  scatters across multipoles: with exact tables, white TT is 1.6% below
  the spectrum-aware TT in the bin 350 to 399, 0.85% in 400 to 449 and 0.2
  to 0.5% up to 800. `spectrum_aware: true` removes this for TT; EE and TE
  have no spectrum-aware path.

What the defaults do now (all three settings apply to the white channels
and to the spectrum-aware probes):

- **Dense nodes.** Nodes every $l_x / 15$ multipoles (`near_cut_step`;
  20 at $l_x = 300$, a ratio of about 1.05) from $0.95\thinspace l_x$ to
  $2\thinspace l_x$, merged with the geometric ones (a geometric node
  closer than half a step to a dense one is dropped). Measured with exact
  tables at the dense nodes, this brought the interpolation error below
  0.1% from the bin 350 to 399 up.
- **More probes near the cut.** The nodes from $0.95\thinspace l_x$ to
  below $2\thinspace l_x$ use 128 random probes (`nprobe_near_cut`, against
  32 elsewhere), or with `nprobe_near_cut: null` the complete basis of
  their degree ($2\ell' + 1$ probes), which makes their ratio exact. The nodes below $0.95\thinspace l_x$ keep `nprobe`: there $F_\ell$
  is near zero and their bins are not usable anyway.
- **Band-limit floor.** The nodes from $0.95\thinspace l_x$ to below
  $1.3\thinspace l_x$ carry every multipole up to at least
  $3.5\thinspace l_x$ (not a parameter). In a band-limit scan at the
  survey-like scale (the same probes at every band-limit, relative to
  $5\thinspace l_x$, mean over $\lvert d\rvert \le 10$), the white ratio of
  the node at $l_x$ with the sharp cut was 10.9% low at the default
  band-limit ($1.57\thinspace l_x$), 3.6% at $2.5\thinspace l_x$, 4.6% at
  $3\thinspace l_x$ (the grid changes with the band-limit, so the approach
  is not monotonic) and 0.8% at $3.5\thinspace l_x$; the nodes from
  $1.07\thinspace l_x$ to $1.33\thinspace l_x$ were within 0.04% at
  $3.5\thinspace l_x$ (0.13 to 0.56% low at the default). The sharp cut's
  white ratio at $l_x$ does not converge strictly (it still moved by 1%
  between $3.5$ and $5\thinspace l_x$). With the smooth cut every node had
  converged to $10^{-5}$ at $2.5\thinspace l_x$ (0.1% low at the default).
  The spectrum-aware ratio is less sensitive: 1.6% low at the node at the
  cut with the default band-limit and 0.2% at $3\thinspace l_x$ (against
  $3.33\thinspace l_x$, where the spectra of that run ended); the floor is
  capped at the spectra's length like the default band-limit.
- **Warnings.** For every bin, T/E spectrum and frequency pair the run
  takes the lowest `fl` over the bin's multipoles (from `lmin`) and logs
  one warning naming the bins below 0.3 ("accuracy degraded") and a
  stronger one naming those below 0.1 ("cut these bins"). The numbers are
  not changed.

**Measured with the defaults.** Package runs with these defaults in the
same survey-like test (paired ratio of model to simulations, filtered over
unfiltered, per bin of 50; Monte Carlo error 1.1 to 1.4% per bin; TT
divided by the INKA baseline error of that test, EE and TE raw), sharp cut:

| bin | 350 to 399 | 400 to 449 | 450 to 499 | 500 to 549 | 550 to 599 |
|---|---|---|---|---|---|
| TT, white | 0.997 | 0.978 | 1.009 | 0.981 | 0.993 |
| TT, `spectrum_aware: true` | 1.011 | 0.986 | 1.012 | 0.982 | 0.995 |
| EE | 1.003 | 0.986 | 1.003 | 0.995 | 0.997 |
| TE | 0.975 | 0.980 | 1.009 | 0.993 | 0.991 |
| v0.6.0, TT white | 0.939 | 0.974 | 1.013 | 0.987 | 0.999 |
| v0.6.0, TT spectrum-aware | 0.967 | 1.007 | 1.015 | 0.986 | 1.001 |
| v0.6.0, EE | 0.957 | 0.981 | 1.006 | 1.002 | 1.003 |
| v0.6.0, TE | 0.924 | 0.975 | 1.012 | 1.000 | 0.997 |

Smooth cut, $h = \exp[-(l_x \sin\theta / \lvert M \rvert)^6]$, same test:

| bin | 350 to 399 | 400 to 449 | 450 to 499 | 500 to 549 | 550 to 599 |
|---|---|---|---|---|---|
| TT, white | 1.011 | 0.990 | 1.005 | 0.984 | 0.998 |
| TT, `spectrum_aware: true` | 1.011 | 0.990 | 1.006 | 0.986 | 0.999 |
| EE | 1.007 | 0.981 | 0.996 | 1.013 | 1.005 |
| TE | 0.990 | 0.980 | 1.002 | 0.991 | 0.988 |
| v0.6.0, TT white | 1.006 | 0.967 | 1.005 | 0.985 | 1.002 |
| v0.6.0, TT spectrum-aware | 0.981 | 0.982 | 1.009 | 0.991 | 1.004 |
| v0.6.0, EE | 0.985 | 0.955 | 0.997 | 1.013 | 1.010 |
| v0.6.0, TE | 0.980 | 0.956 | 1.002 | 0.992 | 0.992 |

So every T/E block is within 3% of the simulations from
$1.17\thinspace l_x$ (the bin 350 to 399) up with the sharp cut, and within
2.1% with the smooth cut (the v0.6.0 defaults reached 3% only from
$1.5\thinspace l_x$ there); 1% per bin cannot be demonstrated with these
simulations. The bin that contains the
cut (300 to 349, where $F_\ell$ goes from 0.03 to 0.34) is **not usable**:
its ratio was 1.35 (white TT), 0.90 (spectrum-aware TT), 1.00 (EE) and 1.05
(TE) with the defaults, and anywhere from 0.46 to 1.35 across the variants
tried, for both cuts; the run warns to cut it. Without simulations, on the same raw
blocks, the defaults agree with exact tables at dense nodes to 0.2% (white)
and 0.06% (spectrum-aware) from 350 to 599 with the sharp cut, and to 0.6%
with the smooth cut (in the bin 350 to 399, where the package's average of
the two sides of each node differs most from keeping them apart).

**Cost.** The complete basis is expensive at the survey scale. On the
survey configuration (the nside-2048 mask, mask band-limit 875, `dmax` 99,
$l_x = 300$, 8 threads on a shared machine) the 16 near-cut nodes cost
about 2.6 hours for the three white channels and 1.3 hours for the
spectrum-aware TT (timed per probe and extrapolated), against 33 and 12 to
17 minutes for the whole v0.6.0 probing. In single-thread CPU time
(measured on the nside-512 version, whose probing cost is the same) they
took 11 hours and 4.2 hours. `nprobe_near_cut: 128` costs about 24 and 13
minutes there (6.4 and 4.7 times less CPU) and moved the binned variances
of the test by at most 0.4% from 350 up (one seed of probes).

### The low-pass (`lowpass`)

A time-stream filter often removes small scales along the scan as well as
large ones (SPT-3G: a high-pass at $l_x = 300$ and a low-pass at 6144).
The optional `lowpass` sub-block adds that. The low-pass is the same kind of
ring filter as the high-pass: it multiplies the azimuthal mode $M$ of a ring
at colatitude $\theta$ by a real $h_{\mathrm{low}}(M, \theta)$ with its cut at
$m_c = l_x^{\mathrm{low}} \sin\theta$, the same convention as the high-pass, and the filter of the
run is the product $h = h_{\mathrm{high}}\thinspace h_{\mathrm{low}}$, with
$l_x^{\mathrm{low}} > l_x$. The two shapes are

```math
h_{\mathrm{low}} = \exp\left[ -\left( \frac{\lvert M \rvert}{l_x^{\mathrm{low}} \sin\theta} \right)^{p} \right]
\qquad \text{(exp, } p = 6 \text{ by default)}
```

```math
h_{\mathrm{low}} = \begin{cases} 1 & \lvert M \rvert \le l_x^{\mathrm{low}} \sin\theta \\ 0 & \text{otherwise} \end{cases}
\qquad \text{(sharp)}
```

Nothing else in the method changes. $K$ contains the combined $h$, so the
probes, the channels, $\rho$ and the correction $\rho / (F^L F^R)$ are those of
Sects. 1 and 2, and `fl` must be the transfer function of the band (the
maps' own, as it already has to be for the high-pass alone; nothing checks that it includes the low-pass). A mode of degree
$\ell$ with its wave vector at an angle $\alpha$ to the scan has $\lvert M \rvert = \ell \lvert\cos\alpha\rvert \sin\theta$ at
colatitude $\theta$, so the sharp low-pass removes it where
$\ell \lvert\cos\alpha\rvert > l_x^{\mathrm{low}}$: nothing below
$\ell = l_x^{\mathrm{low}}$, then a fraction $1 - (2/\pi) \arcsin(l_x^{\mathrm{low}}/\ell)$
that rises as a square root of $\ell - l_x^{\mathrm{low}}$. The exponential shape starts to bite
from about $0.5\thinspace l_x^{\mathrm{low}}$ ($p = 6$). The transfer
function falls inside the band of the low-pass, and the run's warnings for bins
with a low `fl` (Sect. 2) apply there as at the high-pass cut.

**Nodes.** What the low-pass changes is the shape of $\rho(\ell, \ell')$
through its rolloff. At toy scale (below) the geometric nodes, a factor 1.25
apart, left the interpolated $\rho(\ell, \ell)$ up to 4.3% off for the
exponential low-pass and 35% off for the sharp one. With a low-pass the
default nodes therefore also include, from $0.6\thinspace l_x^{\mathrm{low}}$ to
$2\thinspace l_x^{\mathrm{low}}$, a node every $l_x^{\mathrm{low}}/15$,
and for the sharp shape every $l_x^{\mathrm{low}}/50$ from $0.95$ to
$1.15\thinspace l_x^{\mathrm{low}}$, where its square-root onset takes
$\rho$ from 0.92 to 0.52 over a tenth of $l_x^{\mathrm{low}}$. These nodes use the
`nprobe_near_cut` probes (128 by default): few modes survive in the rolloff
and the probe noise of a node there is about six times that of the high-pass
alone with the same number of probes. With `near_cut_step: 0` none of this is
added, and a rolloff inside the covariance ($0.6\thinspace l_x^{\mathrm{low}} < \ell_{\max}$) is then interpolated from the geometric
nodes alone (the run logs a warning). A low-pass above $\ell_{\max}/0.6$, such as the SPT-3G one at 6144
with $\ell_{\max} = 3500$, adds no node.

**Toy-scale validation.** Same toy as Sect. 3 (GL grid 360, $\ell \le 320$, mask
band-limit 64, sharp high-pass at 30), with a low-pass at 200 so that its
rolloff is inside the band, $F^{TT}$ falling from 0.74 at $\ell = 160$ to 0.31
at 300 (exponential) and from 0.88 to 0.40 (sharp). Corrected block against the
exact filtered covariance, diagonal error in per cent, for $\rho$ probed by
the package at the nodes of its plan with the complete basis at each node (the
largest error over $\lvert \ell - \ell' \rvert \le 12$ is within 1.2 percentage
points of the diagonal one in every cell of the two low-pass variants and six
blocks measured):

| $\ell'$ | 100 | 160 | 200 | 240 | 280 | 300 |
|---|---|---|---|---|---|---|
| $F^{TT}$, exponential | 0.80 | 0.74 | 0.56 | 0.42 | 0.34 | 0.31 |
| TT x TT, no correction | -23.8 | -8.9 | -7.4 | -26.2 | -39.2 | -43.7 |
| TT x TT, $\rho$ of the high-pass alone | +2.8 | +51 | +175 | +302 | +401 | +452 |
| TT x TT, $\rho$ with the low-pass | -0.3 | -0.4 | +0.2 | +0.3 | -0.3 | +0.1 |
| EE x EE, $\rho$ with the low-pass | -0.3 | -0.3 | 0.0 | -1.2 | -3.5 | -3.7 |
| TE x TE, $\rho$ with the low-pass | 0.0 | -0.2 | +0.4 | +0.1 | -0.6 | +0.2 |
| TT x EE, $\rho$ with the low-pass | -0.8 | -0.4 | +0.5 | -0.3 | -3.7 | -11.2 |

(The high-pass alone row ignores the low-pass in $\rho$ but not in $F$, which
the data carry. The sharp low-pass gives the same picture, with TT x TT within
0.8% from 240 up and 3% at the onset, $\ell' = 200$.) Every T/E block is within
1.2% of the exact filtered block at $\ell' \le 240$ ($F^{TT} \ge 0.42$) for the exponential
low-pass, and for the sharp one except at its onset ($\ell' = 200$, up to 3.9% in
TT x TE). The errors are 3 to 6% in the blocks with an E leg at $F^{TT} = 0.34$
($\ell' = 280$) and up to 11% (TT x EE) at 0.31, with exact $\rho$ as well: the
model, not the nodes (and at 300 the toy's $C^{TE}$ crosses zero at 310 and its
$\ell_{\max}$ is 320). With exact white rows instead of probed $\rho$ the
errors are the same to 0.1% up to 280. The nodes: the interpolated $\rho(\ell, \ell)$
differs from the dense reference (a node every 5 multipoles, complete basis)
by at most 0.43% for the exponential low-pass (0.08% above $\ell = 150$) and 1.1%
in the onset of the sharp one (0.7% just below it, 0.1% above 250). With the geometric
nodes alone the block errors were 2.3 to 2.9% at $\ell' = 200$ and 4 to 7% at 280
(exponential) and 3 to 11% at 200 (sharp). Random probes at the plan's counts (128 in the rolloff, 32
elsewhere) add the noise of Sect. 2: 0.1 to 0.4% per node below $0.85\thinspace l_x^{\mathrm{low}}$ and about 1% (0.1 to 2.1%) through the rolloff, against 2.4 to 2.8% with 32 probes at the node 238; the complete basis (`nprobe_near_cut: null`) removes it. The same toy with the high-pass
alone reproduces Sect. 3: at $\ell' \ge 100$ (3.3 $l_x$) every block is within 0.7% of the exact
filtered one with exact $\rho$, within 0.7% with probed $\rho$ (complete basis at each
node) and within 1.4% with 32 random probes per node; at $2\thinspace l_x$ 0.6 to 2.2%
(exact $\rho$) and 0.8 to 2.4% (probed).

**Limit of the sharp shape.** At its kink the ratio of the sharp low-pass
has off-diagonal structure ($\rho(\ell, \ell')$ at fixed $\ell'$ near
$l_x^{\mathrm{low}}$ changes by about 10% over a few $\lvert\ell - \ell'\rvert$) that the
interpolation in the mean multipole and the distance cannot carry, so
blocks within a few multipoles of $l_x^{\mathrm{low}}$ are about 3 to 4% off even with the fine
nodes (3.9% at most in the toy). The run logs a warning when the sharp
low-pass has $l_x^{\mathrm{low}} \le \ell_{\max}$. The exponential shape, which pipelines use, has no
such limit.

**Survey-scale check.** On the survey mask of Sect. 3 (nside 2048, mask
band-limit 875, `dmax` 99, sharp high-pass at 300) with the SPT-3G low-pass
(6144, exponential, $p = 6$), $\rho$ probed with the default 32 random probes
at the nodes 2000, 2800 and 3499 (four seeds; the probes of a seed are the same
with and without the low-pass) falls by 0.19%, 1.39% and 5.07% at $d = 0$
(2.76% at 3150, one seed), the same in the three channels to 0.01% and
growing as $(\ell / l_x^{\mathrm{low}})^6$; averaged over $\lvert d \rvert \le 99$ the
shifts are 0.12%, 0.90% and 3.29%. The seed-to-seed spread of $\rho$ is 0.07 to
0.10% at $d = 0$ (0.13 to 0.17% rms over $d$), and 0.001 to 0.03% for the ratio
of the two. Leaving the low-pass out of $\rho$ (but not out of $F$, which the
data carry) would therefore leave the corrected variance 5.3% high at
$\ell = 3499$, 1.4% at 2800 and 0.2% at 2000. The default geometric nodes
interpolate the shift: PCHIP through the nodes 2000, 2800 and 3499 predicted
$\rho$ at the extra node 3150 to 0.05% at $d = 0$ (0.4% at $d = 50$, the size of the probe
noise) and the shift to 0.16%. The complete basis was not run: at the top node it
is 7001 probes of 18 s each (3 channels, filtered and unfiltered, measured on
a shared machine), about 35 hours.

## 3. Assumptions and validity

- **Scan frame.** The filter is diagonal in the azimuthal order around the
  map's polar axis. The mask must therefore be given in the coordinate system
  whose polar axis is the scan axis, so that the iso-latitude rings of the
  map are the scan lines. Nothing checks this.
- **Toy-scale validation.** The correction was validated against exact
  filtered covariances on toy patches and caps, at low resolution, not on a
  survey-scale mask. For TT the factor reproduced the exact filtered
  covariance to about 0.3% for $\ell \gtrsim 4 m_c$, and to a few per cent
  near $2 m_c$. For the T/E blocks (GL grid 360, $\ell \le 320$, mask
  band-limit 64, $l_x = 30$, a sharp and a smooth cut), every block was
  within 0.7% of the exact filtered block with the exact $\rho$ and 0.5% with
  the probed and interpolated one at $\ell \ge 3.3\thinspace l_x$, and 2-3%
  off at $2\thinspace l_x$. That toy error near the cut (the kernel *shape*
  changing) falls fast with the cut measured in mask coupling widths, as
  $l_x^{-2}$ to $l_x^{-3}$ between $l_x$ = 30 and 90 on a toy patch; the
  toy cuts sit three coupling widths from $\ell = 0$, a survey cut at
  $l_x = 300$ about thirty, where it is below 0.1% from
  $1.33\thinspace l_x$ (extrapolated). On a survey-sized footprint the
  usable range near the cut is set by the sources of the subsection "Near
  the cut" instead. The factor also assumes $F_L$ is flat across the mask's coupling
  width: on a toy cap with a coupling width of 17 multipoles it was 0.8% low
  at five times the cut and 0.25% low at seven times, shrinking as the
  multipole grows relative to that width.
- **Survey-scale validation (SPT-3G-like).** 500 paired simulations on
  the nside-2048 border-apodised mask (dec $-41$ to $-71$ deg, 4% of the
  sky), lmax 3500, bins of 50, one frequency with beam, pixel window and
  white noise, filtered ring by ring with the sharp cut at $l_x = 300$ and
  with the smooth cut $\exp(-(300\sin\theta/|M|)^6)$; ACC kernels with
  centralell 250, dmax 100; `map_filter` with the default nodes (22, from
  32 to 3499) and 32 probes (probing: about 33 minutes per filter for the
  three channels). Measured as (analytic filtered / analytic unfiltered)
  over (simulated filtered / simulated unfiltered variance), which cancels
  the 1-2% baseline error of the unfiltered pipeline:
  - today's $C \to F C$ underestimates the variance by 30-35% at
    $\ell$ = 400 to 900, 14-17% at 900-2000 and 6-9% at 2000-3500, in every
    T/E block, for both filters;
  - the correction is within the Monte Carlo noise (range means 0.98-1.02,
    errors 0.005-0.02) from $\ell \approx 600$ (2 $l_x$) up for the smooth
    cut on all blocks and for the sharp cut on EE, TE and TT x EE;
  - at $\ell$ = 400 to 600 (1.3 to 2 $l_x$) the corrected variance came
    out 2-7% high (TT +2%, EE +4-6%, TE +7%, 0.6-2.3 sigma). **A later
    test with 13200 simulations did not reproduce this**: with the same
    mask at nside 512, the sharp cut and the v0.6.0 defaults, the paired
    ratio averaged over 450 to 700 was 0.995 to 0.998 on TT, EE and TE
    (error about 0.5%), and per bin of 50 from 450 to 649 between 0.986
    and 1.012 (errors 1.1 to 1.2%); the smooth cut gave 0.991 to 0.998. The excess was most likely Monte Carlo noise
    (500 simulations, errors about 3%), with the simulated bandpowers of
    that comparison also binned one multipole off the analytic ones;
  - below $\ell \approx 350$ (F below 0.2, zero below 300) the factor
    $\rho/F^2$ is meaningless and the matrix is not positive definite:
    cut those bins (the run now warns about them, subsection "Near the
    cut");
  - **the sharp cut fails on TT at $\ell$ = 2000 to 3500**: the variance is
    5.5 $\pm$ 0.6% low and a neighbour-bin correlation of +0.035 is
    missed. The sharp latitude-dependent cut scatters signal at low $\ell$ to
    high $\ell$ (the signal-only transfer function sits above the
    isotropic value), which a white-spectrum $\rho$ cannot describe. The
    smooth cut does not show it;
  - **`spectrum_aware: true` fixes it** (Sect. 2): sharp-cut TT at
    2000-3500 goes from 0.945 $\pm$ 0.006 to 0.996 $\pm$ 0.006, and the
    model's neighbour-bin correlation change from +0.001 to +0.033 (sims
    +0.035 $\pm$ 0.005). TT at 900-2000 goes from 0.995 to 0.999 $\pm$
    0.007; at 400-600 and 600-900 it is unchanged within 1% (1.028 and
    0.977, errors 0.035 and 0.019). The smooth cut is unchanged (1.003 at
    2000-3500), and so are all EE, TE and TT x EE blocks (bit for bit, as
    they keep the white channels). On the minimal GL grid the same probing
    was 5% high (1.051 $\pm$ 0.006, correlation +0.070): the grid aliasing
    described in Sect. 2. Against the signal-only sims, sharp TT at
    2000-3500 goes from 0.932 to 1.008 $\pm$ 0.007. The EE and TE
    neighbour correlations keep a small miss (TE +0.016 $\pm$ 0.004 in the
    signal-only sims, +0.007 $\pm$ 0.003 in the total ones, against +0.0005
    modelled): those blocks still use the white channels. Probing: 745 s (sharp) and 1048 s (smooth) for 22
    nodes and 32 probes with 8 threads on a shared machine, against about
    33 minutes for the three white channels;
  - the noise transfer function does not matter: replacing $F^N$ by the
    signal $F$ in the noise spectrum changes the diagonal by under 0.3%
    (median), 1.8% at most. Strictly the factor wants the noise spectrum
    scaled by the same `fl` as the signal; a delivered noise $F^N N$ comes
    out scaled by $(F^N/F)^2$.
- **Positive definiteness.** An elementwise factor does not preserve it. In
  the regime where the correction is trusted the effect is at round-off;
  near or below the cut the corrected matrix can acquire small negative
  eigenvalues, which the conditioning report shows.
- **Fourier high-pass only.** Filters that are not diagonal in $M$ (a
  polynomial subtracted along a partial ring, point-source masking during
  filtering) are not described.
- **One filter.** All maps are assumed filtered the same way. Cross-spectra
  between maps with different filters (where TE x ET differs from TE x TE)
  were not studied.
- **Transfer function.** `fl` is used twice: the spectra are multiplied by it
  and the factor divides it out again, so $\rho$ is the only new information.
  Without `fl` the factor is $\rho$ alone and the run warns.

## 4. Why the B blocks stay uncorrected

Any block with a B leg (BB x BB, TB, EB, and their cross-blocks) is left
unchanged, and the run emits one `UserWarning` naming those blocks. No
correction is safer here than a wrong one:

- The filter makes E to B leakage of its own. The filtered pseudo-BB mean
  carries more leaked E than the unfiltered one (the ratio of the leaked-E
  to the true-B part of the mean rises from 0.29 to 0.41 at $\ell = 220$
  with the sharp cut), so the pieces of BB x BB (signal, leak, cross term)
  are scaled by different amounts and no single factor describes the block.
- A single $\rho^{EE}$ fails. On the same toy patch it left BB x BB 1.4-19%
  off at $\ell' \ge 100$, and with the sharp cut at $\ell' = 220$ the
  corrected block was 19-22% low, worse than the 17% of the uncorrected one
  (there the filtered block is 1.16 times the unfiltered one although the
  filter removes power). TB x TB and EB x EB were 1-7% off.
- A sharp latitude-dependent cut makes that leakage non-local in multipole,
  which no white-spectrum channel can capture. Paired probing with the
  fiducial spectra does capture it, but it is not implemented.

The parameters are in [`parameters.md`](../parameters.md), `map_filter` block.
