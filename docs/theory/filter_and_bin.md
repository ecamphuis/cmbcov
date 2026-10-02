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
alike. The raw-block cache (`save_raw_blocks`) keeps the uncorrected block,
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
run at nodes $\ell'$ equal to `node_min`, `node_min * node_ratio`, ... and $\rho$
is interpolated between them (it is smooth in $\ell$). The error of a node
falls as $1/\sqrt{N_{\mathrm{probe}}\thinspace \ell'}$. The result is cached
in the kernel directory under the mask digest, the channels and all filter
parameters.

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
  off at $2\thinspace l_x$. Below about three times the cut the kernel shape
  itself changes and the correction is only approximate. Treat results as
  accurate above about three times the cut, and do not rely on them below
  it. The factor also assumes $F_L$ is flat across the mask's coupling
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
    $\ell = 400$-900, 14-17% at 900-2000 and 6-9% at 2000-3500, in every
    T/E block, for both filters;
  - the correction is within the Monte Carlo noise (range means 0.98-1.02,
    errors 0.005-0.02) from $\ell \approx 600$ (2 $l_x$) up for the smooth
    cut on all blocks and for the sharp cut on EE, TE and TT x EE;
  - at $\ell = 400$-600 (1.3-2 $l_x$) the corrected variance is 2-7% high
    (TT +2%, EE +4-6%, TE +7%, 0.6-2.3 sigma);
  - below $\ell \approx 350$ (F below 0.2, zero below 300) the factor
    $\rho/F^2$ is meaningless and the matrix is not positive definite:
    cut those bins;
  - **the sharp cut fails on TT at $\ell = 2000$-3500**: the variance is
    5.5 $\pm$ 0.6% low and a neighbour-bin correlation of +0.035 is
    missed. The sharp latitude-dependent cut scatters signal at low $\ell$ to
    high $\ell$ (the signal-only transfer function sits above the
    isotropic value), which a white-spectrum $\rho$ cannot describe. The
    smooth cut does not show it. The remedy would be probing with the
    fiducial spectrum, as for the B blocks;
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
