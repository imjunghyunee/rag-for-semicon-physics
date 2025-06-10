# 12.3.2 Derivation of Transistor Current Components and Current Gain Factors

We now wish to determine the various transistor current components and each of the gain factors in terms of the electrical and geometrical parameters of the transistor. The results of these derivations show how the various parameters in the transistor influence the electrical properties of the device and point the way to the design of a "good" bipolar transistor.

## Emitter Injection Efficiency Factor

Consider, initially, the emitter injection efficiency factor. We have from Equation (12.31a)

\[
\gamma = \left( \frac{J_{nE}}{J_{nE} + J_{pE}} \right) = \frac{1}{\left( 1 + \frac{J_{pE}}{J_{nE}} \right)}
\]

(12.32)

We derived the minority carrier distribution functions for the forward-active mode in Section 12.2.1. Noting that \( J_{pE} \), as defined in Figure 12.19, is in the negative \( x \) direction, we can write the current densities as

\[
J_{pE} = -eD_{E} \left. \frac{d\delta p_{E}(x')}{dx'} \right|_{x'=0}
\]

(12.33a)

and

\[
J_{nE} = -eD_{B} \left. \frac{d\delta n_{B}(x)}{dx} \right|_{x=0}
\]

(12.33b)

where \(\delta p_{E}(x')\) and \(\delta n_{B}(x)\) are given by Equations (12.21) and (12.15), respectively.

Taking the appropriate derivatives of \(\delta p_{E}(x')\) and \(\delta n_{B}(x)\), we obtain

\[
J_{pE} = eD_{pE} \frac{p_{E0}}{L_{E}} \left[ \exp \left( \frac{eV_{BE}}{kT} \right) - 1 \right] \cdot \frac{1}{\tanh (x_{E}/L_{E})}
\]

(12.34a)

and

\[
J_{nE} = eD_{B} \frac{n_{B0}}{L_{B}} \left[ \frac{1}{\sinh (x_{B}/L_{B})} + \frac{\exp \left( \frac{eV_{BE}}{kT} \right) - 1}{\tanh (x_{B}/L_{B})} \right]
\]

(12.34b)

Positive \( J_{pE} \) and \( J_{nE} \) values imply that the currents are in the directions shown in Figure 12.19. If we assume that the B–E junction is biased sufficiently far in the forward bias so that \( V_{BE} \gg kT/e \), then

\[
\exp \left( \frac{eV_{BE}}{kT} \right) \gg 1
\]

and also

\[
\frac{\exp \left( eV_{BE}/kT \right)}{\tanh (x_{B}/L_{B})} \gg \frac{1}{\sinh (x_{B}/L_{B})}
\]