# 8.1 pn Junction Current

The forward-bias current–voltage relation is given by Equation (8.27). This relation includes \( J_s \) as well as the \(\exp(eV/kT)\) factor, making the forward-bias current–voltage relation a function of temperature also. As temperature increases, less forward-bias voltage is required to obtain the same diode current. If the voltage is held constant, the diode current will increase as temperature increases. The change in forward-bias current with temperature is less sensitive than the reverse-saturation current.

## Objective

Determine the change in the forward-bias voltage on a pn junction with a change in temperature to maintain a constant diode current.

Consider a silicon pn junction initially biased at 0.60 V at \( T = 300 \, K \). Assume the temperature increases to \( T = 310 \, K \). Calculate the change in the forward-bias voltage required to maintain a constant current through the junction.

### Solution

The forward-bias current can be written as follows:

\[
J \approx \exp\left(-\frac{E_g}{kT}\right) \exp\left(\frac{eV_a}{kT}\right)
\]

If the temperature changes, we may take the ratio of the diode currents at the two temperatures. This ratio is

\[
\frac{J_2}{J_1} = \frac{\exp\left(-\frac{E_g}{kT_2}\right) \exp\left(\frac{eV_{a2}}{kT_2}\right)}{\exp\left(-\frac{E_g}{kT_1}\right) \exp\left(\frac{eV_{a1}}{kT_1}\right)}
\]

If current is to be held constant, then \( J_1 = J_2 \), and we must have

\[
\frac{E_g - eV_{a2}}{kT_2} = \frac{E_g - eV_{a1}}{kT_1}
\]

For \( T_1 = 300 \, K, \, T_2 = 310 \, K, \, E_g = 1.12 \, eV, \) and \( V_{a1} = 0.60 \, V \). We then find

\[
\frac{1.12 - V_{a2}}{310} = \frac{1.12 - 0.60}{300}
\]

which yields

\[
V_{a2} = 0.5827 \, V
\]

### Comment

The change in the forward-bias voltage is \(-17.3 \, mV\) for a \(10^\circ C\) temperature change.

### EXERCISE PROBLEM

**Ex 8.5** Repeat Example 8.5 for a GaAs pn junction diode biased at \( V_a = 1.050 \, V \) for \( T = 300 \, K \).

----

## 8.1.8 The “Short” Diode

We assumed in the previous analysis that both p and n regions were long compared with the minority carrier diffusion lengths. In many pn junction structures, one region may, in fact, be short compared with the minority carrier diffusion length. Figure 8.11 shows one such example: the length \( W_n \) is assumed to be much smaller than the minority carrier hole diffusion length, \( L_p \).