# 9.1 The Schottky Barrier Diode

!Comparison of forward-bias I-V characteristics between a Schottky diode and a pn junction diode.

**Figure 9.10** Comparison of forward-bias I-V characteristics between a Schottky diode and a pn junction diode.

Recall that the reverse-biased current in a silicon pn junction diode is dominated by the generation current. A typical generation current density is approximately \(10^{-7} \, \text{A/cm}^2\), which is still two to three orders of magnitude less than the reverse-saturation current density of the Schottky barrier diode. A generation current also exists in the reverse-biased Schottky barrier diode; however, the generation current is negligible compared with the \(J_{0}\) value.

Since \(J_{0T} \gg J_{0}\), the forward-bias characteristics of the two types of diodes will also be different. Figure 9.10 shows typical I-V characteristics of a Schottky barrier diode and a pn junction diode. The effective turn-on voltage of the Schottky diode is less than that of the pn junction diode.

## Objective

Calculate the forward-bias voltage required to induce a forward-bias current density of \(10 \, \text{A/cm}^2\) in a Schottky barrier diode and a pn junction diode.

Consider diodes with the parameters given in Example 9.5. We can assume that the pn junction diode will be sufficiently forward biased so that the ideal diffusion current will dominate. Let \(T = 300 \, \text{K}\).

### Solution

For the Schottky barrier diode, we have

\[
J = J_{0T} \left[ \exp \left( \frac{eV}{kT} \right) - 1 \right]
\]

Neglecting the \((-1)\) term, we can solve for the forward-bias voltage. We find

\[
V_{s} = \left( \frac{kT}{e} \right) \ln \left( \frac{J}{J_{0T}} \right) = V_{t} \ln \left( \frac{J}{J_{0T}} \right) = (0.0259) \ln \left( \frac{10}{5.98 \times 10^{-3}} \right) = 0.312 \, \text{V}
\]

For the pn junction diode, we have

\[
V_{d} = V_{t} \ln \left( \frac{J}{J_{0}} \right) = (0.0259) \ln \left( \frac{10}{3.66 \times 10^{-11}} \right) = 0.682 \, \text{V}
\]