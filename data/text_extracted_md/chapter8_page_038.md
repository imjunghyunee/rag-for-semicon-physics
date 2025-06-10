```markdown
### 8.3 Small-Signal Model of the pn Junction

#### EXERCISE PROBLEM

**Ex 8.7** A silicon pn junction diode at \( T = 300 \, \text{K} \) has the following parameters: \( N_d = 8 \times 10^{16} \, \text{cm}^{-3}, N_a = 2 \times 10^{16} \, \text{cm}^{-3}, D_n = 25 \, \text{cm}^2/\text{s}, D_p = 10 \, \text{cm}^2/\text{s}, \tau_{n0} = 5 \times 10^{-7} \, \text{s}, \) and \( \tau_{p0} = 10^{-7} \, \text{s} \). The cross-sectional area is \( A = 10^{-3} \, \text{cm}^2 \). Determine the diffusion resistance and diffusion capacitance if the diode is forward biased at (a) \( V_a = 0.550 \, \text{V} \) and (b) \( V_a = 0.610 \, \text{V} \).

The diffusion capacitance tends to dominate the capacitance terms in a forward-biased pn junction. The small-signal diffusion resistance can be fairly small if the diode current is a fairly large value. As the diode current decreases, the diffusion resistance increases. We will consider the impedance of forward-biased pn junctions again when we discuss bipolar transistors.

### 8.3.3 Equivalent Circuit

The small-signal equivalent circuit of the forward-biased pn junction is derived from Equation (8.103). This circuit is shown in Figure 8.22a. We need to add the junction capacitance, which will be in parallel with the diffusion resistance and diffusion capacitance. The last element we add, to complete the equivalent circuit, is a series resistance. The neutral n and p regions have finite resistances so the actual pn junction will include a series resistance. The complete equivalent circuit is given in Figure 8.22b.

The voltage across the actual junction is \( V_a \) and the total voltage applied to the pn diode is given by \( V_{\text{app}} \). The junction voltage \( V_a \) is the voltage in the ideal current–voltage expression. We can write the expression

\[
V_{\text{app}} = V_a + I_r
\]

(8.106)

Figure 8.23 is a plot of the current–voltage characteristic from Equation (8.106) showing the effect of the series resistance. A larger applied voltage is required to

!Figure 8.22

**Figure 8.22** (a) Small-signal equivalent circuit of ideal forward-biased pn junction diode; (b) complete small-signal equivalent circuit of pn junction.
```