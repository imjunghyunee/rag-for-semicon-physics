# Chapter 11: Metal–Oxide–Semiconductor Field-Effect Transistor: Additional Concepts

The effective inversion charge mobility is a strong function of temperature because of lattice scattering. As the temperature is reduced, the mobility increases.

## Example 11.2

**Objective:** Calculate the effective electric field at threshold for a given semiconductor doping concentration.

Consider a p-type silicon substrate at \( T = 300 \, \text{K} \) doped to \( N_a = 3 \times 10^{16} \, \text{cm}^{-3} \).

### Solution

From the results of Chapter 10, we can calculate

\[
\phi_{fp} = V_t \ln \left( \frac{N_a}{n_i} \right) = (0.0259) \ln \left( \frac{3 \times 10^{16}}{1.5 \times 10^{10}} \right) = 0.376 \, \text{V}
\]

and

\[
x_{T} = \left[ \frac{4 \epsilon_s \phi_{fp}}{e N_a} \right]^{1/2} = \left[ \frac{4(11.7) \times 8.85 \times 10^{-14} \times 0.376}{(1.6 \times 10^{-19}) \times 3 \times 10^{16}} \right]^{1/2}
\]

which is \( x_T = 0.18 \, \mu\text{m} \). Then

\[
|Q_{SD}(\text{max})| = e N_a x_T = 8.64 \times 10^{-8} \, \text{C/cm}^2
\]

At the threshold inversion point, we may assume that \( Q'_I = 0 \), so the effective electric field from Equation (11.14) is found as

\[
E_{\text{eff}} = \frac{1}{\epsilon_i} |Q_{SD}(\text{max})| = \frac{8.64 \times 10^{-8}}{(11.7) \times 8.85 \times 10^{-14}} = 8.34 \times 10^4 \, \text{V/cm}
\]

### Comment

We can see, from Figure 11.10, that this value of effective transverse electric field at the surface is sufficient for the effective inversion charge mobility to be significantly less than the bulk semiconductor value.

### Exercise Problem

**Ex 11.2** Determine (using Figure 11.10) the effective inversion layer electron mobility for a surface electric field of \( E_{\text{eff}} = 2 \times 10^5 \, \text{V/cm} \).

----

The effective mobility is a function of gate voltage through the inversion charge density in Equation (11.14). As the gate voltage increases, the carrier mobility decreases even further.

## 11.1.4 Velocity Saturation

In the analysis of the long-channel MOSFET, we assume the mobility to be constant, which means that the drift velocity increases without limit as the electric field increases. In this ideal case, the carrier velocity increases until the ideal current is attained. However, we have seen that the carrier velocity saturates with increasing electric field. Velocity saturation will become more prominent in shorter-channel devices since the corresponding horizontal electric field is generally larger.