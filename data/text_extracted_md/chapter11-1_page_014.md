# Chapter 11: Metal–Oxide–Semiconductor Field-Effect Transistor: Additional Concepts

## Table 11.1 Summary of constant-field device scaling

| Device and circuit parameters | Scaling factor (k < 1) |
|-------------------------------|------------------------|
| **Scaled parameters**         |                        |
| Device dimensions (L, t<sub>ox</sub>, W, x) | k |
| Doping concentration (N<sub>a</sub>, N<sub>d</sub>) | 1/k |
| Voltages                      | k                      |
| **Effect on device parameters** |                      |
| Electric field                | 1                      |
| Carrier velocity              | 1                      |
| Depletion widths              | k                      |
| Capacitance (C = εA/l)        | k                      |
| Drift current                 | 1                      |
| **Effect on circuit parameters** |                    |
| Device density                | 1/k<sup>2</sup>        |
| Power density                 | 1                      |
| Power dissipation per device (P = IV) | k<sup>2</sup> |
| Circuit delay time (≈ CV/I)   | k                      |
| Power–delay product (Pτ)      | k<sup>2</sup>          |

Source: Taur and Ning [23].

Since the channel length is being reduced, the depletion widths also need to be reduced. If the substrate doping concentration is increased by the factor (1/k), then the depletion width is reduced by approximately the factor k since V<sub>B</sub> is reduced by k.

The drain current per channel width, for the transistor biased in the saturation region, can be written as

\[
\frac{I_D}{W} = \frac{\mu_n \epsilon_{ox}}{2t_{ox}L} (V_G - V_T)^2 \rightarrow \frac{\mu_n \epsilon_{ox}}{2t_{ox}(kL)} (kV_G - V_T)^2 = \text{constant} \quad (11.22)
\]

The drift current per channel width remains essentially a constant, so if the channel width is reduced by k, then the drain current is also reduced by k. The area of the device, A ≈ WL, is then reduced by k<sup>2</sup> and the power, P = IV, is also reduced by k<sup>2</sup>. The power density in the chip remains unchanged.

Table 11.1 summarizes the device scaling and the effect on circuit parameters. Keep in mind that the width and length of interconnect lines are also assumed to be reduced by the same scaling factor.

### 11.2.2 Threshold Voltage—First Approximation

In constant-field scaling, the device voltages are reduced by the scaling factor k. It would seem appropriate that the threshold voltage should also be scaled by the same factor. The threshold voltage, for a uniformly doped substrate, can be written as

\[
V_T = V_{FB} + 2\phi_p + \frac{\sqrt{2\epsilon_s N_a (2\phi_p)}}{C_{ox}} \quad (11.23)
\]

The first two terms in Equation (11.23) are functions of material parameters that do not scale and are only very slight functions of doping concentration. The last term is approximately proportional to \(\sqrt{k}\), so the threshold voltage does not scale directly with the scaling factor k.