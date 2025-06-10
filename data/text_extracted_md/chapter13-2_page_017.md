# 13.5 High Electron Mobility Transistor

## Objective

Determine the two-dimensional electron concentration for an N–AlGaAs–intrinsic GaAs heterojunction.

Consider an N-Al\(_{0.3}\)Ga\(_{0.7}\)As layer doped to 10\(^1^8\) cm\(^{-3}\) and having a thickness of 500 Å. Assume an undoped spacer layer of 20 Å. Let \(\phi_b = 0.85\) V and \(\Delta E_c/q = 0.22\) V. The relative dielectric constant of Al\(_{0.3}\)Ga\(_{0.7}\)As is \(\epsilon_r = 12.2\).

## Solution

The parameter \(V_{2i}\) is found as

\[
V_{2i} = \frac{qN_d d^2}{2\epsilon \epsilon_0} = \frac{(1.6 \times 10^{-19})(10^{18})(500 \times 10^{-8})^2}{2(12.2)(8.85 \times 10^{-14})} = 1.85 \, \text{V}
\]

Then the threshold voltage is

\[
V_{th} = \phi_b - \frac{\Delta E_c}{q} - V_{2i} = 0.85 - 0.22 - 1.85 = -1.22 \, \text{V}
\]

The channel electron concentration for \(V_g = 0\) is found from Equation (13.68) to be

\[
n_s = \frac{(12.2)(8.85 \times 10^{-14})}{(1.6 \times 10^{-19})(500 + 20 + 80) \times 10^{-8} [-(-1.22)]} = 1.37 \times 10^{12} \, \text{cm}^{-2}
\]

## Comment

The threshold voltage \(V_{th}\) is negative, making this device a depletion mode MODFET; applying a negative gate voltage will turn off the device. A value of \(n_s \approx 10^{12} \, \text{cm}^{-2}\) is a typical channel concentration.

----

The current–voltage characteristics of the MODFET can be found using the charge control model and the gradual channel approximation. The channel carrier concentration can be written as

\[
n_s(x) = \frac{q\epsilon_r \epsilon_0}{qd + \Delta d} [V_g - V_{off} - V(x)]
\]

where \(V(x)\) is the potential along the channel due to the drain-to-source voltage. The drain current is

\[
I_D = qn_s v(E)W
\]

where \(v(E)\) is the carrier drift velocity and \(W\) is the channel width. This analysis is very similar to that for the pn JFET in Section 13.2.2.

If we assume a constant mobility, then for low \(V_{DS}\) values, we have

\[
I_D = \frac{\epsilon_r \epsilon_0 \mu W}{2L(d + \Delta d)} [2(V_g - V_{off}) V_{DS} - V_{DS}^2]
\]

The form of this equation is the same as that for the pn JFET or MESFET operating in the nonsaturation region. If \(V_{DS}\) increases so that the carriers reach the saturation velocity, then

\[
I_{D(sat)} = \frac{\epsilon_r \epsilon_0 v_s W}{d + \Delta d} (V_g - V_{off} - V_0)_{v_{sat}}
\]

where \(v_{sat}\) is the saturation velocity and \(V_0 = E_s L\) with \(E_s\) being the electric field in the channel that produces the saturation velocity.