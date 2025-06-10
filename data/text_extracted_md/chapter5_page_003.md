# Chapter 5: Carrier Transport Phenomena

## Table 5.1: Typical mobility values at \( T = 300 \, K \) and low doping concentrations

| Material          | \(\mu_n\) (cm\(^2\)/V·s) | \(\mu_p\) (cm\(^2\)/V·s) |
|-------------------|--------------------------|--------------------------|
| Silicon           | 1350                     | 480                      |
| Gallium arsenide  | 8500                     | 400                      |
| Germanium         | 3900                     | 1900                     |

The unit of mobility is usually expressed in terms of cm\(^2\)/V·s.

By combining Equations (5.2) and (5.4), we may write the drift current density due to holes as

\[
J_{\text{drift}} = (ep)v_{\text{dp}} = e\mu_p pE
\]

(5.5)

The drift current due to holes is in the same direction as the applied electric field. The same discussion of drift applies to electrons. We may write

\[
J_{\text{drift}} = pv_{\text{dn}} = (-en)v_{\text{dn}}
\]

(5.6)

where \( J_{\text{drift}} \) is the drift current density due to electrons and \( v_{\text{dn}} \) is the average drift velocity of electrons. The net charge density of electrons is negative.

The average drift velocity of an electron is also proportional to the electric field for small fields. However, since the electron is negatively charged, the net motion of the electron is opposite to the electric field direction. We can then write

\[
v_{\text{dn}} = -\mu_n E
\]

(5.7)

where \(\mu_n\) is the electron mobility and is a positive quantity. Equation (5.6) may now be written as

\[
J_{\text{drift}} = (-en)(-\mu_n E) = e\mu_n nE
\]

(5.8)

The conventional drift current due to electrons is also in the same direction as the applied electric field even though the electron movement is in the opposite direction.

Electron and hole mobilities are functions of temperature and doping concentrations, as we will see in the next section. Table 5.1 shows some typical mobility values at \( T = 300 \, K \) for low doping concentrations.

Since both electrons and holes contribute to the drift current, the total drift current density is the sum of the individual electron and hole drift current densities, so we may write

\[
J_{\text{drift}} = e(\mu_n n + \mu_p p)E
\]

(5.9)

----

### Example 5.1

**Objective:** Calculate the drift current density in a semiconductor for a given electric field.

Consider a gallium arsenide sample at \( T = 300 \, K \) with doping concentrations of \( N_d = 0 \) and \( N_a = 10^{16} \, \text{cm}^{-3} \). Assume complete ionization and assume electron and hole mobilities given in Table 5.1. Calculate the drift current density if the applied electric field is \( E = 10 \, \text{V/cm} \).