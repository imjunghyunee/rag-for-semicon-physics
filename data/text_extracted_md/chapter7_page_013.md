```markdown
**Comment**  
The space charge width has increased from 0.951 μm at zero bias to 2.83 μm at a reverse bias of 5 V.

## EXERCISE PROBLEM

**Ex 7.3**  
(a) A silicon pn junction at \( T = 300 \, K \) has doping concentrations of \( N_d = 5 \times 10^{15} \, \text{cm}^{-3} \) and \( N_a = 5 \times 10^{16} \, \text{cm}^{-3} \). A reverse-biased voltage of \( V_R = 4 \, V \) is applied.  
Determine \( V_{bi}, x_n, x_p, \) and \( W \). (b) Repeat part (a) for \( V_R = 8 \, V \).

The magnitude of the electric field in the depletion region increases with an applied reverse-biased voltage. The electric field is still given by Equations (7.14) and (7.16) and is still a linear function of distance through the space charge region. Since \( x_n \) and \( x_p \) increase with reverse-biased voltage, the magnitude of the electric field also increases. The maximum electric field still occurs at the metallurgical junction.

The maximum electric field at the metallurgical junction, from Equations (7.14) and (7.16), is

\[
E_{\text{max}} = -\frac{eN_dx_n}{\varepsilon_s} = -\frac{eN_ax_p}{\varepsilon_s}
\]

(7.35)

If we use either Equation (7.28) or (7.29) in conjunction with the total potential barrier, \( V_{bi} + V_R \), then

\[
E_{\text{max}} = -\left\{ \frac{2e(V_{bi} + V_R)}{\varepsilon_s} \left( \frac{N_aN_d}{N_a + N_d} \right) \right\}^{1/2}
\]

(7.36)

We can show that the maximum electric field in the pn junction can also be written as

\[
E_{\text{max}} = -\frac{2(V_{bi} + V_R)}{W}
\]

(7.37)

where \( W \) is the total space charge width.

----

**Objective:** Design a pn junction to meet maximum electric field and voltage specifications.

Consider a silicon pn junction at \( T = 300 \, K \) with a p-type doping concentration of \( N_a = 2 \times 10^{17} \, \text{cm}^{-3} \). Determine the n-type doping concentration such that the maximum electric field is \( |E_{\text{max}}| = 2.5 \times 10^6 \, \text{V/cm} \) at a reverse-biased voltage of \( V_R = 25 \, V \).

**Solution**  
The maximum electric field is given by Equation (7.36). Neglecting \( V_{bi} \) compared to \( V_R \), we can write

\[
|E_{\text{max}}| = \left\{ \frac{2eV_R}{\varepsilon_s} \left( \frac{N_aN_d}{N_a + N_d} \right) \right\}^{1/2}
\]

----

**DESIGN EXAMPLE 7.4**
```