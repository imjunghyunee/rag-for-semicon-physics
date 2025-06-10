# 12.3 Transistor Currents and Low-Frequency Common-Base Current Gain

## Recombination Factor

The recombination factor is given by Equation (12.31c). We can write

\[
\delta = \frac{J_{xe} + J_{pe}}{J_{xe} + J_R + J_{pe}} = \frac{J_{xe}}{J_{xe} + J_R} = \frac{1}{1 + J_R/J_{xe}}
\]

(12.40)

We have assumed in Equation (12.40) that \( J_{pe} \ll J_{xe} \). The recombination current density, due to the recombination in a forward-biased pn junction, was discussed in Chapter 8 and can be written as

\[
J_R = \frac{e x_{BE} n_i}{2 \tau_o} \exp \left( \frac{eV_{BE}}{2kT} \right) = J_o \exp \left( \frac{eV_{BE}}{2kT} \right)
\]

(12.41)

where \( x_{BE} \) is the B–E space charge width.

The current \( J_{xe} \) from Equation (12.34b) can be approximated as

\[
J_{xe} = J_{xo} \exp \left( \frac{eV_{BE}}{kT} \right)
\]

(12.42)

where

\[
J_{xo} = \frac{e D_B n_{i0}}{L_B \tanh (x_{B}/L_B)}
\]

(12.43)

The recombination factor, from Equation (12.40), can then be written as

\[
\delta = \frac{1}{1 + \frac{J_o}{J_{xo}} \exp \left( -\frac{eV_{BE}}{2kT} \right)}
\]

(12.44)

The recombination factor is a function of the B–E voltage. As \( V_{BE} \) increases, the recombination current becomes less dominant and the recombination factor approaches unity.

----

## Objective: Calculate the recombination factor.

Assume the following transistor parameters: \( x_{BE} = 0.10 \, \mu m, \, \tau_o = 10^{-7} s, \, N_B = 5 \times 10^{15} \, cm^{-3}, \, D_B = 20 \, cm^2/s, \, L_B = 10 \, \mu m, \) and \( x_B = 0.80 \, \mu m \). Assume \( V_{BE} = 0.50 \, V \).

### Solution

From Equation (12.41), we find

\[
J_o = \frac{e x_{BE} n_i}{2 \tau_o} = \frac{(1.6 \times 10^{-19})(0.10 \times 10^{-4})(1.5 \times 10^{10})}{2(10^{-7})} = 1.2 \times 10^{-7} \, A/cm^2
\]

and from Equation (12.43), we find

\[
J_{xo} = \frac{e D_B n_{i0}}{L_B \tanh (x_B/L_B)} = \frac{e D_B (n_i^2/N_B)}{L_B \tanh (x_B/L_B)}
\]

\[
= \frac{(1.6 \times 10^{-19})(20)((1.5 \times 10^{10})^2/5 \times 10^{15})}{(10 \times 10^{-4})\tanh(0.80/10.0)} = 1.804 \times 10^{-9} \, A/cm^2
\]