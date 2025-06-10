```markdown
# CHAPTER 8 The pn Junction Diode

## EXAMPLE 8.3

**Objective:** Design a pn junction diode to produce particular electron and hole current densities at a given forward-bias voltage.

Consider a silicon pn junction diode at \( T = 300 \, \text{K} \). Design the diode such that \( J_n = 20 \, \text{A/cm}^2 \) and \( J_p = 5 \, \text{A/cm}^2 \) at \( V_a = 0.65 \, \text{V} \). Assume the remaining semiconductor parameters are as given in Example 8.2.

### Solution

The electron diffusion current density is given by Equation (8.24) as

\[
J_n = e D_n \frac{n_{po}}{L_n} \left[ \exp \left( \frac{eV_a}{kT} \right) - 1 \right] = e \sqrt{\frac{D_n}{\tau_{po}}} \frac{n_i^2}{N_a} \left[ \exp \left( \frac{eV_a}{kT} \right) - 1 \right]
\]

Substituting the numbers, we have

\[
20 = (1.6 \times 10^{-19}) \sqrt{\frac{25}{5 \times 10^{-7}}} \frac{(1.5 \times 10^{10})^2}{N_a} \left[ \exp \left( \frac{0.65}{0.0259} \right) - 1 \right]
\]

which yields

\[
N_a = 1.01 \times 10^{15} \, \text{cm}^{-3}
\]

The hole diffusion current density is given by Equation (8.22) as

\[
J_p = e D_p \frac{p_{no}}{L_p} \left[ \exp \left( \frac{eV_a}{kT} \right) - 1 \right] = e \sqrt{\frac{D_p}{\tau_{no}}} \frac{n_i^2}{N_d} \left[ \exp \left( \frac{eV_a}{kT} \right) - 1 \right]
\]

Substituting the numbers, we have

\[
5 = (1.6 \times 10^{-19}) \sqrt{\frac{10}{5 \times 10^{-7}}} \frac{(1.5 \times 10^{10})^2}{N_d} \left[ \exp \left( \frac{0.65}{0.0259} \right) - 1 \right]
\]

which yields

\[
N_d = 2.55 \times 10^{15} \, \text{cm}^{-3}
\]

### Comment

The relative magnitude of the electron and hole current densities through a diode can be varied by changing the doping concentrations in the device.

### EXERCISE PROBLEM

**Ex 8.3** Using the parameters given in Ex 8.2 for the GaAs diode, determine the electron and hole current densities at the space charge edges, and determine the total current density in the diode for a forward-bias voltage of \( V_a = 1.05 \, \text{V} \).

## 8.1.6 Summary of Physics

We have been considering the case of a forward-bias voltage being applied to a pn junction. The forward-bias voltage lowers the potential barrier so that electrons and holes are injected across the space charge region. The injected carriers become minority carriers which then diffuse from the junction and recombine with majority carriers.

We calculated the minority carrier diffusion current densities at the edge of the space charge region. We can reconsider Equations (8.14) and (8.15) and determine...
```