# Transistor Currents and Low-Frequency Common-Base Current Gain

The emitter injection efficiency, from Equation (12.32), then becomes

\[
\gamma = \frac{1}{1 + \frac{p_{E0} D_E L_B}{n_{B0} D_B L_E} \cdot \frac{\tanh(x_B/L_B)}{\tanh(x_E/L_E)}}
\]

(12.35a)

If we assume that all the parameters in Equation (12.35a) except \(p_{E0}\) and \(n_{B0}\) are fixed, then in order for \(\gamma \approx 1\), we must have \(p_{E0} \ll n_{B0}\). We can write

\[
p_{E0} = \frac{n_i^2}{N_E} \quad \text{and} \quad n_{B0} = \frac{n_i^2}{N_B}
\]

where \(N_E\) and \(N_B\) are the impurity doping concentrations in the emitter and base, respectively. Then the condition that \(p_{E0} \ll n_{B0}\) implies that \(N_E \gg N_B\). For the emitter injection efficiency to be close to unity, the emitter doping must be large compared to the base doping. This condition means that many more electrons from the n-type emitter than holes from the p-type base will be injected across the B–E space charge region. If both \(x_B \ll L_B\) and \(x_E \ll L_E\), then the emitter injection efficiency can be written as

\[
\gamma \approx \frac{1}{1 + \frac{N_B}{N_E} \cdot \frac{D_E}{D_B} \cdot \frac{x_B}{x_E}}
\]

(12.35b)

## Objective

Calculate the emitter injection efficiency.

Assume the following transistor parameters: \(N_B = 10^{15} \, \text{cm}^{-3}\), \(N_E = 10^{17} \, \text{cm}^{-3}\), \(D_E = 10 \, \text{cm}^2/\text{s}\), \(D_B = 20 \, \text{cm}^2/\text{s}\), \(x_B = 0.80 \, \mu\text{m}\), and \(x_E = 0.60 \, \mu\text{m}\).

## Solution

From Equation (12.35b), we find

\[
\gamma = \frac{1}{1 + \left(\frac{N_B}{N_E} \cdot \frac{D_E}{D_B} \cdot \frac{x_B}{x_E}\right)} = \frac{1}{1 + \left(\frac{10^{15}}{10^{17}} \cdot \frac{10}{20} \cdot \frac{0.80}{0.60}\right)} = 0.9934
\]

## Comment

This simple example shows a typical magnitude of the emitter injection efficiency.

## EXERCISE PROBLEM

**Ex 12.1** Repeat Example 12.1 if the base and emitter doping concentrations are \(N_B = 5 \times 10^{15} \, \text{cm}^{-3}\) and \(N_E = 10^{18} \, \text{cm}^{-3}\), respectively.  
(\(J_{S6} = 0 = \lambda \cdot SV\))

## Base Transport Factor

The next term to consider is the base transport factor, given by Equation (12.31b) as \(\alpha_T = J_{C}/J_{E}\). From the definitions of the current directions shown in Figure 12.19, we can write

\[
J_{C} = (-e) D_B \left. \frac{d[n_B(x)]}{dx} \right|_{x=x_B}
\]

(12.36a)