# Chapter 11: Metal–Oxide–Semiconductor Field-Effect Transistor: Additional Concepts

## Solution

We can find

\[
C_{\text{ox}} = \frac{\varepsilon_{\text{ox}}}{t_{\text{ox}}} = \frac{(3.9)(8.85 \times 10^{-14})}{200 \times 10^{-8}} = 1.726 \times 10^{-7} \, \text{F/cm}^2
\]

\[
\phi_p = V_t \ln \left( \frac{N_a}{n_i} \right) = (0.0259) \ln \left( \frac{3 \times 10^{16}}{1.5 \times 10^{10}} \right) = 0.3758 \, \text{V}
\]

and

\[
x_{\text{tr}} = \left[ \frac{4 \varepsilon_s \phi_p}{e N_a} \right]^{1/2} = \left[ \frac{(4)(11.7)(8.85 \times 10^{-14})(0.3758)}{(1.6 \times 10^{-19})(3 \times 10^{16})} \right]^{1/2}
\]

\[
= 0.18 \times 10^{-4} \, \text{cm} = 0.18 \, \mu\text{m}
\]

The shift in threshold voltage is now found as

\[
\Delta V_T = \frac{-e N_a x_{\text{tr}}}{C_{\text{ox}}} \left[ \frac{t_i}{L} \left( \sqrt{1 + \frac{2 x_{\text{tr}}}{t_i}} - 1 \right) \right]
\]

\[
= \frac{-(1.6 \times 10^{-19})(3 \times 10^{16})(0.18 \times 10^{-4})}{1.726 \times 10^{-7}} \left[ \frac{0.3}{1.0} \left( \sqrt{1 + \frac{2(0.18)}{0.3}} - 1 \right) \right]
\]

or

\[
\Delta V_T = -0.0726 \, \text{V}
\]

## Comment

If the design value of the threshold voltage of an n-channel MOSFET is to be \( V_T = 0.35 \, \text{V} \), for example, a shift of \( \Delta V_T = -0.0726 \, \text{V} \) due to short-channel effects is significant and needs to be taken into account.

## Exercise Problem

**Ex 11.3** Repeat Example 11.3 if the device parameters are \( N_a = 10^{16} \, \text{cm}^{-3} \), \( L = 0.75 \, \mu\text{m} \), \( r_j = 0.25 \, \mu\text{m} \), and \( t_{\text{ox}} = 12 \, \text{nm} = 120 \, \text{Å} \).

The effect of short channels becomes more pronounced as the channel length is reduced further.

The shift in threshold voltage with channel length for an n-channel MOSFET is shown in Figure 11.16. As the substrate doping increases, the initial threshold voltage increases, as we have seen in the previous chapter, and the short-channel threshold shift also becomes larger. The short-channel effects on threshold voltage do not become significant until the channel length becomes less than approximately 2 \(\mu\text{m}\). The threshold voltage shift also becomes smaller as the diffusion depth \( r_j \) becomes smaller so that very shallow junctions reduce the threshold voltage dependence on channel length.

Equation (11.29) is derived assuming that the source, channel, and drain space charge widths are all equal. If we now apply a drain voltage, the space charge width at the drain terminal widens, which makes \( L' \) smaller, and the amount of bulk charge controlled by the gate voltage decreases. This effect makes the threshold voltage a function of drain voltage. As the drain voltage increases, the threshold voltage of an n-channel MOSFET decreases. The threshold voltage versus channel length is