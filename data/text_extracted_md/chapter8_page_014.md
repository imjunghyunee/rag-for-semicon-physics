## 8.1 pn Junction Current

which may be rewritten as

\[
J_s = en_i^2 \left( \frac{1}{N_A} \sqrt{\frac{D_n}{\tau_{n0}}} + \frac{1}{N_D} \sqrt{\frac{D_p}{\tau_{p0}}} \right)
\]

Then

\[
J_s = (1.6 \times 10^{-19})(1.5 \times 10^{10})^2 \left( \frac{1}{10^{16}} \sqrt{\frac{25}{5 \times 10^{-7}}} + \frac{1}{10^{16}} \sqrt{\frac{10}{5 \times 10^{-7}}} \right)
\]

or

\[
J_s = 4.16 \times 10^{-11} \, \text{A/cm}^2
\]

### Comment

The ideal reverse-biased saturation current density is very small. If the pn junction cross-sectional area were \( A = 10^{-4} \, \text{cm}^2 \), for example, then the ideal reverse-biased diode current would be \( I_s = 4.15 \times 10^{-15} \, \text{A} \).

### EXERCISE PROBLEM

**Ex 8.2** Consider a GaAs pn junction diode at \( T = 300 \, \text{K} \). The parameters of the device are \( N_D = 2 \times 10^{16} \, \text{cm}^{-3}, N_A = 8 \times 10^{15} \, \text{cm}^{-3}, D_n = 210 \, \text{cm}^2/\text{s}, D_p = 8 \, \text{cm}^2/\text{s}, \tau_{n0} = 10^{-7} \, \text{s}, \) and \( \tau_{p0} = 5 \times 10^{-8} \, \text{s} \). Determine the ideal reverse-saturation current density.

\((\mu \text{m}/y = 0.1 \, \text{X} \, 0^{\circ} \text{C} \, \text{E} = f' \, \text{SUY})\)

If the forward-bias voltage in Equation (8.27) is positive by more than a few \( kT/eV \), then the \((-1)\) term in Equation (8.27) becomes negligible. Figure 8.9 shows the forward-bias current–voltage characteristic when the current is plotted on a log scale. Ideally, this plot yields a straight line when \( V_a \) is greater than a few \( kT/eV \). The forward-bias current is an exponential function of the forward-bias voltage.

!Figure 8.9

**Figure 8.9** | Ideal \( I-V \) characteristic of a pn junction diode with the current plotted on a log scale.