# Chapter 6: Nonequilibrium Excess Carriers in Semiconductors

!Figure 6.18  
**Figure 6.18** | Steady-state excess hole concentration versus distance from a semiconductor surface.

Assume that excess carriers are being generated at a constant rate throughout the entire semiconductor material. We showed that, in steady state, the generation rate is equal to the recombination rate for the case of a homogeneous, infinite semiconductor. Using this argument, the recombination rates at the surface and in the bulk material must be equal. Since \(\tau_{p0} < \tau_{pb}\), then the excess minority carrier concentration at the surface is smaller than the excess minority carrier concentration in the bulk region, or \(\delta p_s < \delta p_b\). Figure 6.18 shows an example of the excess carrier concentration plotted as a function of distance from the semiconductor surface.

## Example 6.9

**Objective:** Determine the steady-state excess carrier concentration as a function of distance from the surface of a semiconductor.

Consider Figure 6.18, in which the surface is at \(x = 0\). Assume that in the n-type semiconductor \(\delta p_b = 10^{14} \, \text{cm}^{-3}\) and \(\tau_p = 10^{-6} \, \text{s}\) in the bulk, and \(\tau_{p0} = 10^{-7} \, \text{s}\) at the surface. Assume zero applied electric field and let \(D_p = 10 \, \text{cm}^2/\text{s}\).

### Solution

From Equations (6.106) and (6.107), we have

\[
\frac{\delta p_b}{\tau_{p0}} = \frac{\delta p_f}{\tau_{p0b}}
\]

so that

\[
\delta p_f = \delta p_b \left( \frac{\tau_{p0b}}{\tau_{p0}} \right) = (10^{14}) \left( \frac{10^{-7}}{10^{-6}} \right) = 10^{13} \, \text{cm}^{-3}
\]

From Equation (6.56), we can write

\[
D_p \frac{d^2 (\delta p)}{dx^2} + g' - \frac{\delta p}{\tau_{p0}} = 0 \tag{6.108}
\]

The generation rate can be determined from the steady-state conditions in the bulk, or

\[
g' = \frac{\delta p_b}{\tau_p} = \frac{10^{14}}{10^{-6}} = 10^{20} \, \text{cm}^{-3} \cdot \text{s}^{-1}
\]