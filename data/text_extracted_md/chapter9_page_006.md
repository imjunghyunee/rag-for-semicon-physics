# CHAPTER 9 Metal–Semiconductor and Semiconductor Heterojunctions

**Comment**  
The values of space charge width and electric field are very similar to those obtained for a pn junction.

## EXERCISE PROBLEM

**Ex 9.1** Consider an ideal tungsten-to-n-type GaAs junction. Assume the GaAs is doped to a concentration of \( N_d = 5 \times 10^{15} \, \text{cm}^{-3} \). Determine the theoretical barrier height, the built-in potential barrier, and maximum electric field for the case of zero applied bias.

A junction capacitance can also be determined in the same way as we do for the pn junction. We have that

\[
C' = eN_d \frac{dx_n}{dV_R} = \left[ \frac{2 \, \varepsilon e \, N_d}{(V_{bi} + V_R)} \right]^{1/2}
\]

(9.8)

where \( C' \) is the capacitance per unit area. If we square the reciprocal of Equation (9.8), we obtain

\[
\left( \frac{1}{C'} \right)^2 = \frac{2(V_{bi} + V_R)}{\varepsilon e N_d}
\]

(9.9)

We can use Equation (9.9) to obtain, to a first approximation, the built-in potential barrier \( V_{bi} \), and the slope of the curve from Equation (9.9) to yield the semiconductor doping \( N_d \). We can calculate the potential \( \phi_n \), and then determine the Schottky barrier \( \phi_{bn} \) from Equation (9.2).

## EXAMPLE 9.2

**Objective:** To calculate the semiconductor doping concentration and Schottky barrier height from the silicon diode experimental capacitance data shown in Figure 9.3. Assume \( T = 300 \, \text{K} \).

**Solution**  
The intercept of the tungsten–silicon curve is approximately at \( V_{bi} = 0.40 \, \text{V} \). From Equation (9.9), we can write

\[
\frac{d(1/C')^2}{dV_R} = \frac{\Delta(1/C')^2}{\Delta V_R} = \frac{2}{\varepsilon e N_d}
\]

Then, from the figure, we have

\[
\frac{\Delta(1/C')^2}{\Delta V_R} = 4.4 \times 10^{13}
\]

so that

\[
N_d = \frac{2}{(1.6 \times 10^{-19})(11.7)(8.85 \times 10^{-14})(4.4 \times 10^{13})} = 2.7 \times 10^{17} \, \text{cm}^{-3}
\]

We can calculate

\[
\phi_n = \frac{kT}{e} \ln \left( \frac{N_d}{N_i} \right) = (0.0259) \ln \left( \frac{2.8 \times 10^{19}}{2.7 \times 10^{17}} \right) = 0.12 \, \text{V}
\]

so that

\[
\phi_{bn} = V_{bi} + \phi_n = 0.40 + 0.12 = 0.52 \, \text{V}
\]

where \( \phi_{bn} \) is the actual Schottky barrier height.