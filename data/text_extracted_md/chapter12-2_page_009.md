## 12.3 Transistor Currents and Low-Frequency Common-Base Current Gain

### EXERCISE PROBLEM

**Ex 12.4**  
Assume that transistor parameters are the same as described in Example 12.4. In addition, let \( N_B = 6 \times 10^{18} \, \text{cm}^{-3} \). Determine the base doping concentration such that the emitter injection efficiency is \( \gamma = 0.9950 \).

----

### Objective

Design the base width required to achieve a base transport factor of \( \alpha_T = 0.9967 \).

Consider a pnp bipolar transistor. Assume that \( D_B = 10 \, \text{cm}^2/\text{s} \) and \( \tau_{B0} = 10^{-7} \text{s} \).

**DESIGN EXAMPLE 12.5**

#### Solution

The base transport factor applies to both pnp and npn transistors and is given by

\[
\alpha_T = \frac{1}{\cosh(x_B/L_B)} = 0.9967
\]

Then

\[
x_B/L_B = 0.0814
\]

We have

\[
L_B = \sqrt{D_B \tau_{B0}} = \sqrt{(10)(10^{-7})} = 10^{-3} \, \text{cm}
\]

so that the base width must then be

\[
x_B = 0.814 \times 10^{-4} \, \text{cm} = 0.814 \, \mu\text{m}
\]

#### Comment

If the base width is less than approximately 0.8 \(\mu\text{m}\), then the required base transport factor will be achieved. In most cases, the base transport factor will not be the limiting factor in the bipolar transistor current gain.

----

### EXERCISE PROBLEM

**Ex 12.5**  
Assume that transistor parameters are the same as described in Example 12.5. Determine the minimum base width \( x_B \) such that the base transport factor is \( \alpha_T = 0.9980 \).

----

### Objective

Determine the forward-biased B–E voltage required to achieve a recombination factor equal to \( \delta = 0.9967 \).

Consider an npn bipolar transistor at \( T = 300 \, \text{K} \). Assume that \( J_{s0} = 10^{-8} \, \text{A/cm}^2 \) and that \( J_0 = 10^{-11} \, \text{A/cm}^2 \).

**DESIGN EXAMPLE 12.6**

#### Solution

The recombination factor, from Equation (12.44), is

\[
\delta = \frac{1}{1 + \frac{J_{s0}}{J_0} \exp\left(\frac{-eV_{BE}}{2kT}\right)}
\]