# 13.2 The Device Characteristics

Since the electron mobility in GaAs is much larger than the hole mobility, we will concentrate our discussion on n-channel GaAs MESFETs or JFETs. The definition of internal pinch-off voltage, given by Equation (13.3), also applies to these devices. In considering the enhancement mode JFET, the term threshold voltage is commonly used in place of pinch-off voltage. For this reason, we shall use the term threshold voltage in our discussion of MESFETs.

For the n-channel MESFET, the threshold voltage is defined from Equation (13.4) as

\[
V_{bi} - V_T = V_{\phi 0} \quad \text{or} \quad V_T = V_{bi} - V_{\phi 0}
\]

(13.42)

For an n-channel depletion mode JFET, \( V_T < 0 \), and for the enhancement mode device, \( V_T > 0 \). We can see from Equation (13.42) that \( V_{bi} > V_{\phi 0} \) for an enhancement mode n-channel JFET.

----

**Objective:** Determine the channel thickness of a GaAs MESFET to achieve a specified threshold voltage.

**Design Example 13.5**

Consider an n-channel GaAs MESFET at \( T = 300 \, \text{K} \) with a gold Schottky barrier contact. Assume the barrier height is \( \phi_b = 0.89 \, \text{V} \). The n-channel doping is \( N_d = 2 \times 10^{15} \, \text{cm}^{-3} \). Design the channel thickness such that \( V_T = +0.25 \, \text{V} \).

### Solution

We find that

\[
\phi_b = V_t \ln \left( \frac{N_c}{N_d} \right) = (0.0259) \ln \left( \frac{4.7 \times 10^{17}}{2 \times 10^{15}} \right) = 0.141 \, \text{V}
\]

The built-in potential barrier is then

\[
V_{bi} = \phi_{bn} - \phi_b = 0.89 - 0.141 = 0.749 \, \text{V}
\]

The threshold voltage, from Equation (13.42), is

\[
V_T = V_{bi} - V_{\phi 0}
\]

or

\[
V_{\phi 0} = V_{bi} - V_T = 0.749 - 0.25 = 0.499 \, \text{V}
\]

Now

\[
V_{\phi 0} = \frac{ea^2 N_d}{2 \varepsilon_e}
\]

or

\[
0.499 = \frac{a^2 (1.6 \times 10^{-19}) (2 \times 10^{15})}{2 (13.1) (8.85 \times 10^{-14})}
\]

The channel thickness is then

\[
a = 0.601 \, \mu\text{m}
\]