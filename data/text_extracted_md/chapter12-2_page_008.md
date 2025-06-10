# Chapter 12: The Bipolar Transistor

## Table 12.3 | Summary of Limiting Factors

### Emitter Injection Efficiency

\[
\gamma \approx \frac{1}{1 + \frac{N_B}{N_E} \cdot \frac{D_E}{D_B} \cdot \frac{x_B}{x_E}} \quad (x_B \ll L_B), (x_E \ll L_E)
\]

### Base Transport Factor

\[
\alpha_T \approx \frac{1}{1 + \frac{1}{2} \left( \frac{x_B}{L_B} \right)} \quad (x_B \ll L_B)
\]

### Recombination Factor

\[
\delta = \frac{1}{1 + \frac{J_{BO}}{J_O} \exp \left( \frac{-eV_{BE}}{2kT} \right)}
\]

### Common-Base Current Gain

\[
\alpha = \gamma \alpha_T \delta \approx \frac{1}{1 + \frac{N_B}{N_E} \cdot \frac{D_E}{D_B} \cdot \frac{x_B}{x_E} + \frac{1}{2} \left( \frac{x_B}{L_B} \right) + \frac{J_{BO}}{J_O} \exp \left( \frac{-eV_{BE}}{2kT} \right)}
\]

### Common-Emitter Current Gain

\[
\beta = \frac{\alpha}{1 - \alpha} \approx \frac{1}{\frac{N_B}{N_E} \cdot \frac{D_E}{D_B} \cdot \frac{x_B}{x_E} + \frac{1}{2} \left( \frac{x_B}{L_B} \right) + \frac{J_{BO}}{J_O} \exp \left( \frac{-eV_{BE}}{2kT} \right)}
\]

----

## Design Example 12.4

**Objective:** Design the ratio of emitter doping to base doping in order to achieve an emitter injection efficiency factor of \(\gamma = 0.9967\).

- Consider an npn bipolar transistor. Assume, for simplicity, that \(D_E = D_B, L_E = L_B\), and \(x_E = x_B\).

### Solution

Equation (12.35b) reduces to

\[
\gamma = \frac{1}{1 + \frac{p_{BO}}{n_{BO}}} = \frac{1}{1 + n_i^2 / N_E}
\]

so

\[
\gamma = \frac{1}{1 + \frac{N_B}{N_E}} = 0.9967
\]

Then

\[
\frac{N_B}{N_E} = 0.00331 \quad \text{or} \quad \frac{N_E}{N_B} = 302
\]

### Comment

The emitter doping concentration must be much larger than the base doping concentration to achieve a high emitter injection efficiency.