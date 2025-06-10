# 10.3 The Basic MOSFET Operation

!Figure 10.46

**Figure 10.46:** (a) Potentials at a point \( x \) along the channel. (b) Energy-band diagram through the MOS structure at the point \( x \).

The energy-band diagram through the MOS structure at point \( x \) is shown in Figure 10.46b. The Fermi level in the p-type semiconductor is \( E_{Fp} \) and the Fermi level in the metal is \( E_{Fm} \). We have

\[
E_{Fp} - E_{Fm} = e(V_{GS} - V_x)
\]

(10.55)

Considering the potential barriers, we can write

\[
V_{GS} - V_x = (\phi_{b'} + V_{ox}) - \left( x' + \frac{E_c}{2e} - \phi_b + \phi_p \right)
\]

(10.56)

which can also be written as

\[
V_{GS} - V_x = V_{ox} + 2\phi_b + \phi_{ms}
\]

(10.57)

where \( \phi_{ms} \) is the metal–semiconductor work function difference, and \( \phi_b = 2\phi_p \) for the inversion condition.

The electric field in the oxide is

\[
E_{ox} = \frac{V_{ox}}{t_{ox}}
\]

(10.58)

Combining Equations (10.54), (10.57), and (10.58), we find that

\[
-\epsilon_{ox} E_{ox} = \frac{\epsilon_{ox}}{t_{ox}} [(V_{GS} - V_x) - (\phi_{ms} + 2\phi_b)]
\]

\[
= Q_i' + Q_i'' + Q_{D_s}(\text{max})
\]

(10.59)

The inversion charge density, \( Q_i'' \), from Equation (10.59) can be substituted into Equation (10.49) and we obtain

\[
I_x = -W \mu_n C_{ox} \frac{dV_x}{dx} [(V_{GS} - V_T) - V_x]
\]

(10.60)

where \( E_x = -dV_x/dx \) and \( V_T \) is the threshold voltage defined by Equation (10.31b).

We can now integrate Equation (10.60) over the length of the channel. We have

\[
\int_0{L} I_x \, dx = -W \mu_n C_{ox} \int_{V_{GS}}^{V_{(L)}} [(V_{GS} - V_T) - V_x] \, dV_x
\]

(10.61)