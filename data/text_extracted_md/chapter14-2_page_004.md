## 14.3 Photodetectors

Assume that a photon flux \(\Phi_0\) is incident on the \(p^+\) region. If we assume that the \(p^+\) region width \(W_p\) is very thin, then the photon flux, as a function of distance, in the intrinsic region is \(\Phi(x) = \Phi_0 e^{-\alpha x}\), where \(\alpha\) is the photon absorption coefficient. This nonlinear photon absorption is shown in Figure 14.19b. The photocurrent density generated in the intrinsic region can be found as

\[
J_L = e \int_0^W G_L \, dx = e \int_0^W \Phi_0 e^{-\alpha x} \, dx = e \Phi_0 (1 - e^{-\alpha W})
\]

(14.43)

This equation assumes that there is no electron–hole recombination within the space charge region and also that each photon absorbed creates one electron–hole pair.

### Objective

**Calculate the photocurrent density in a PIN photodiode.**

**EXAMPLE 14.6**

Consider a silicon PIN diode with an intrinsic region width of \(W = 20 \, \mu m\). Assume that the photon flux is \(10^7 \, \text{cm}^{-2}\text{s}^{-1}\) and the absorption coefficient is \(\alpha = 10^3 \, \text{cm}^{-1}\).

#### Solution

The generation rate of electron–hole pairs at the front edge of the intrinsic region is

\[
G_{L1} = \alpha \Phi_0 = (10^3)(10^7) = 10^{10} \, \text{cm}^{-3}\text{s}^{-1}
\]

and the generation rate at the back edge of the intrinsic region is

\[
G_{L2} = \alpha \Phi_0 e^{-\alpha W} = (10^3)(10^7) \exp[-(10^3)(20 \times 10^{-4})]
\]

\[
= 0.135 \times 10^9 \, \text{cm}^{-3}\text{s}^{-1}
\]

The generation rate is obviously not uniform throughout the intrinsic region. The photocurrent density is then

\[
J_L = e \Phi_0 (1 - e^{-\alpha W})
\]

\[
= (1.6 \times 10^{-19})(10^7)[1 - \exp[-(10^3)(20 \times 10^{-4})]]
\]

\[
= 13.8 \, \text{mA/cm}^2
\]

#### Comment

The prompt photocurrent density of a PIN photodiode will be larger than that of a regular photodiode since the space charge region is larger in a PIN photodiode.

#### EXERCISE PROBLEM

**Ex 14.6** Repeat Example 14.6 for photon absorption coefficients of (a) \(\alpha = 10^2 \, \text{cm}^{-1}\) and (b) \(\alpha = 10^4 \, \text{cm}^{-1}\).

In most situations, we will not have a long diode; thus, the steady-state photocurrent described by Equation (14.42) will not apply for most photodiodes.

### 14.3.4 Avalanche Photodiode

The avalanche photodiode is similar to the pn or PIN photodiode except that the bias applied to the avalanche photodiode is sufficiently large to cause impact ionization.