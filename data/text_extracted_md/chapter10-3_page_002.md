# Fundamentals of the Metal–Oxide–Semiconductor Field-Effect Transistor

!Charge distribution in the n-channel enhancement mode MOSFET for \( V_{GS} > V_T \).

**Figure 10.44** Charge distribution in the n-channel enhancement mode MOSFET for \( V_{GS} > V_T \).

!Geometry for applying Gauss’s law.

**Figure 10.45** Geometry for applying Gauss’s law.

where the integral is over a closed surface. \( Q_T \) is the total charge enclosed by the surface, and \( E_n \) is the outward directed normal component of the electric field crossing the surface \( S \). Gauss’s law is applied to the surface defined in Figure 10.45. Since the surface must be enclosed, we must take into account the two end surfaces in the x-y plane. However, there is no z-component of the electric field so these two end surfaces do not contribute to the integral of Equation (10.51).

Now consider the surfaces labeled 1 and 2 in Figure 10.45. From the gradual channel approximation, we assume that \( E_n \) is essentially a constant along the channel length. This assumption means that \( E_n \) into surface 2 is the same as \( E_n \) out of surface 1. Since the integral in Equation (10.51) involves the outward component of the E-field, the contributions of surfaces 1 and 2 cancel each other. Surface 3 is in the neutral p region, so the electric field is zero at this surface.

Surface 4 is the only surface that contributes to Equation (10.51). Taking into account the direction of the electric field in the oxide, Equation (10.51) becomes

\[
\int_S \varepsilon_n E_n \, dS = -\varepsilon_{ox} E_{ox} W \, dx = Q_T \tag{10.52}
\]

where \( \varepsilon_{ox} \) is the permittivity of the oxide. The total charge enclosed is

\[
Q_T = [Q'_i + Q'_d + Q'_{Si(\text{max})}] W \, dx \tag{10.53}
\]

Combining Equations (10.52) and (10.53), we have

\[
-\varepsilon_{ox} E_{ox} = Q'_i + Q'_d + Q'_{Si(\text{max})} \tag{10.54}
\]

We now need an expression for \( E_{ox} \). Figure 10.46a shows the oxide and channel. We assume that the source is at ground potential. The voltage \( V_x \) is the potential in the channel at a point \( x \) along the channel length. The potential difference across the oxide at \( x \) is a function of \( V_{GS}, V_x \), and the metal–semiconductor work function difference.