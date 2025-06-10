# 10.1 The Two-Terminal MOS Structure

!Charge distribution in a MOS capacitor at flat band.

**Figure 10.18** Charge distribution in a MOS capacitor at flat band.

where \( C_{\text{ox}} \) is the oxide capacitance per unit area. Substituting Equation (10.22) into Equation (10.23), we have

\[
V_{\text{ox}} = -\frac{Q'_{\text{ox}}}{C_{\text{ox}}}
\]

(10.24)

In the flat-band condition, the surface potential is zero, or \( \phi_s = 0 \). Then from Equation (10.21), we have

\[
V_G = V_{FB} = \phi_{ms} - \frac{Q'_{\text{ox}}}{C_{\text{ox}}}
\]

(10.25)

Equation (10.25) is the flat-band voltage for this MOS device.

----

## Objective

Calculate the flat-band voltage for a MOS capacitor with a p-type semiconductor substrate.

**Example 10.3**

Consider a MOS capacitor with a p-type silicon substrate doped to \( N_a = 10^{16} \, \text{cm}^{-3} \), a silicon dioxide insulator with a thickness of \( t_{\text{ox}} = 20 \, \text{nm} = 200 \, \text{Å} \), and an n\(^+\) polysilicon gate. Assume that \( Q'_{\text{ox}} = 5 \times 10^{10} \) electronic charges per cm\(^2\).

### Solution

The work function difference, from Figure 10.16, is \( \phi_{ms} = -1.1 \, \text{V} \). The oxide capacitance is found to be

\[
C_{\text{ox}} = \frac{\varepsilon_{\text{ox}}}{t_{\text{ox}}} = \frac{(3.9)(8.85 \times 10^{-14})}{200 \times 10^{-8}} = 1.726 \times 10^{-7} \, \text{F/cm}^2
\]

The equivalent oxide charge density is

\[
Q'_{\text{ox}} = (5 \times 10^{10})(1.6 \times 10^{-19}) = 8 \times 10^{-9} \, \text{C/cm}^2
\]

The flat-band voltage is then determined to be

\[
V_{FB} = \phi_{ms} - \frac{Q'_{\text{ox}}}{C_{\text{ox}}} = -1.1 - \frac{8 \times 10^{-9}}{1.726 \times 10^{-7}} = -1.15 \, \text{V}
\]

----

1 Although we will, in general, use the primed notation for capacitance per unit area or charge per unit area, we will omit, for convenience, the prime on the oxide capacitance per unit area parameter.