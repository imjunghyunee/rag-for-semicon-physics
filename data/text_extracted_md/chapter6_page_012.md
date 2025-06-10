# 6.3 Ambipolar Transport

where

\[
D' = \frac{\mu_n n D_n + \mu_p D_p}{\mu_n n + \mu_p p}
\]
(6.40)

and

\[
\mu' = \frac{\mu_n \mu_p (p - n)}{\mu_n n + \mu_p p}
\]
(6.41)

Equation (6.39) is called the **ambipolar transport equation** and describes the behavior of the excess electrons and holes in time and space. The parameter \( D' \) is called the **ambipolar diffusion coefficient** and \( \mu' \) is called the **ambipolar mobility**.

The Einstein relation relates the mobility and diffusion coefficient by

\[
\frac{\mu_n}{D_n} = \frac{\mu_p}{D_p} = \frac{e}{kT}
\]
(6.42)

Using these relations, the ambipolar diffusion coefficient may be written in the form

\[
D' = \frac{D_n D_p (n + p)}{D_n n + D_p p}
\]
(6.43)

The ambipolar diffusion coefficient, \( D' \), and the ambipolar mobility, \( \mu' \), are functions of the electron and hole concentrations, \( n \) and \( p \), respectively. Since both \( n \) and \( p \) contain the excess carrier concentration \( \delta n \), the coefficient in the ambipolar transport equation are not constants. The ambipolar transport equation, given by Equation (6.39), then, is a nonlinear differential equation.

## 6.3.2 Limits of Extrinsic Doping and Low Injection

The ambipolar transport equation may be simplified and linearized by considering an extrinsic semiconductor and by considering low-level injection. The ambipolar diffusion coefficient, from Equation (6.43), may be written as

\[
D' = \frac{D_n D_p [(n_0 + \delta n) + (p_0 + \delta n)]}{D_n (n_0 + \delta n) + D_p (p_0 + \delta n)}
\]
(6.44)

where \( n_0 \) and \( p_0 \) are the thermal-equilibrium electron and hole concentrations, respectively, and \( \delta n \) is the excess carrier concentration. If we consider a p-type semiconductor, we can assume that \( p_0 \gg n_0 \). The condition of low-level injection, or just low injection, means that the excess carrier concentration is much smaller than the thermal-equilibrium majority carrier concentration. For the p-type semiconductor, then, low injection implies that \( \delta n \ll p_0 \). Assuming that \( n_0 \ll p_0 \) and \( \delta n \ll p_0 \), and assuming that \( D_n \) and \( D_p \) are on the same order of magnitude, the ambipolar diffusion coefficient from Equation (6.44) reduces to

\[
D' = D_n
\]
(6.45)