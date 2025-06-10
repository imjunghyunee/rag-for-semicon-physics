# Fundamentals of the Metal–Oxide–Semiconductor Field-Effect Transistor

The parameter \( k' = \mu_n C_{\text{ox}} \) is called the **process conduction parameter** for the n-channel MOSFET and has units of A/V\(^2\). The parameter \( K_n = (W/L) \mu_n C_{\text{ox}}/2L = (k'/2) \cdot (W/L) \) is called the **conduction parameter** for the n-channel MOSFET and also has units of A/V\(^2\).

When the transistor is biased in the saturation region, the ideal current–voltage relation is given by

\[
I_D = \frac{W \mu_n C_{\text{ox}}}{2L} (V_{\text{GS}} - V_T)^2 \tag{10.45a}
\]

which can be written as

\[
I_D = \frac{k'}{2} \cdot \frac{W}{L} \cdot (V_{\text{GS}} - V_T)^2 \tag{10.45b}
\]

or

\[
I_D = K_n (V_{\text{GS}} - V_T)^2 \tag{10.45c}
\]

In general, for a given technology, the process conduction parameter, \( k' \), is a constant. From Equations (10.44b) and (10.45b), then, we see that the design of a MOSFET, in terms of current capability, is determined by the width-to-length parameter.

The operation of a p-channel device is the same as that of the n-channel device, except the charge carrier is the hole and the conventional current direction and voltage polarities are reversed.

## *10.3.3 Current–Voltage Relationship—Mathematical Derivation

In the previous section, we qualitatively discussed the current–voltage characteristics. In this section, we derive the mathematical relation between the drain current, the gate-to-source voltage, and the drain-to-source voltage. Figure 10.43 shows the geometry of the device that we use in this derivation.

In this analysis, we make the following assumptions:

1. The current in the channel is due to drift rather than diffusion.
2. There is no current through the gate oxide.
3. A gradual channel approximation is used in which \(\partial E_y/\partial y \gg \partial E_x/\partial x\). This approximation means that \(E_x\) is essentially a constant.
4. Any fixed oxide charge is an equivalent charge density at the oxide–semiconductor interface.
5. The carrier mobility in the channel is constant.

We start the analysis with Ohm’s law, which can be written as

\[
J_x = \sigma E_x \tag{10.46}
\]

where \(\sigma\) is the channel conductivity and \(E_x\) is the electric field along the channel created by the drain-to-source voltage. The channel conductivity is given by \(\sigma = e \mu_n n(y)\), where \(\mu_n\) is the electron mobility and \(n(y)\) is the electron concentration in the inversion layer.