```markdown
The minority carrier concentration becomes a linear function of distance. The minority carrier hole diffusion current density is given by

\[
J_p = -eD_p \frac{d\Delta p(x)}{dx}
\]

so that in the short n region, we have

\[
J_p(x) = \frac{eD_p \Delta p_0}{W_n} \left[ \exp \left( \frac{eV_A}{kT} \right) - 1 \right]
\]

(8.34)

The minority carrier hole diffusion current density now contains the length \( W_n \) in the denominator, rather than the diffusion length \( L_p \). The diffusion current density is larger for a short diode than for a long diode since \( W_n \ll L_p \). In addition, since the minority carrier concentration is approximately a linear function of distance through the n region, the minority carrier diffusion current density is a constant. This constant current implies that there is no recombination of minority carriers in the short region.

----

### TEST YOUR UNDERSTANDING

**TYU 8.1**  
The doping concentrations in a GaAs pn junction diode at \( T = 300 \, \text{K} \) are \( N_d = 5 \times 10^{15} \, \text{cm}^{-3} \) and \( N_a = 5 \times 10^{16} \, \text{cm}^{-3} \). The minority carrier concentration at either space charge edge is to be no larger than 10 percent of the respective majority carrier concentration. Calculate the maximum forward-bias voltage that may be applied to this junction and still meet the required specifications.  
[\( \Delta L901 = (\text{exu}^9 \Delta) \text{suV} \)]

**TYU 8.2**  
A silicon pn junction at \( T = 300 \, \text{K} \) has the following parameters: \( N_d = 5 \times 10^{16} \, \text{cm}^{-3} \), \( N_a = 1 \times 10^{16} \, \text{cm}^{-3} \), \( D_n = 25 \, \text{cm}^2/\text{s} \), \( D_p = 10 \, \text{cm}^2/\text{s} \), \( \tau_n = 5 \times 10^{-7} \, \text{s} \), and \( \tau_p = 1 \times 10^{-7} \, \text{s} \). The cross-sectional area is \( A = 10^{-2} \, \text{cm}^2 \) and the forward-bias voltage is \( V_A = 0.625 \, \text{V} \). Calculate the (a) minority electron diffusion current at the space charge edge, (b) minority hole diffusion current at the space charge edge, and (c) total current in the pn junction diode.  
[\( \nuU \tau T (q) \nuU 601 (q) \nuU 1510 (q) \nuU \)]

**TYU 8.3**  
Consider the silicon pn junction diode described in TYU 8.2. The p region is long and the n region is short with \( W_n = 2 \, \mu\text{m} \). (a) Calculate the electron and hole currents in the depletion region. (b) Why has the hole current increased compared to that found in TYU 8.2?  
[\( \text{paseau} \nuU \tau T (q) \nuU 1510 (q) \nuU 175 (q) \nuU 175 = q \nuU 1510 = \nu (q) \nuU \)]

----

## 8.2 | GENERATION–RECOMBINATION CURRENTS AND HIGH-INJECTION LEVELS

In the derivation of the ideal current–voltage relationship, we assumed low injection and neglected any effects occurring within the space charge region. High-level injection and other current components generated within the space charge region cause
```