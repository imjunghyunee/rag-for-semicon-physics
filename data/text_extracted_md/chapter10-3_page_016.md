# Fundamentals of the Metal–Oxide–Semiconductor Field-Effect Transistor

!Figure 10.57  
**Figure 10.57 | Small-signal equivalent circuit including Miller capacitance.**

Combining Equations (10.87) and (10.88) to eliminate the voltage variable \( V_b \), we can determine the input current as

\[
I_i = j\omega \left[ C_{gpr} + C_{gd} \left( 1 + \frac{g_m R_L}{1 + j\omega R_L C_{gd}} \right) \right] V_x
\]

(10.89)

Normally, \(\omega R_L C_{gd}\) is much less than unity; therefore, we may neglect the \((j\omega R_L C_{gd})\) term in the denominator. Equation (10.89) then simplifies to

\[
I_i = j\omega [C_{gpr} + C_{gd}(1 + g_m R_L)] V_x
\]

(10.90)

Figure 10.57 shows the equivalent circuit with the equivalent input impedance described by Equation (10.90). The parameter \( C_M \) is the Miller capacitance and is given by

\[
C_M = C_{gd}(1 + g_m R_L)
\]

(10.91)

The serious effect of the drain overlap capacitance now becomes apparent. When the transistor is operating in the saturation region, \( C_{gd} \) essentially becomes zero, but \( C_{gde} \) is a constant. This parasitic capacitance is multiplied by the gain of the transistor and can become a significant factor in the input impedance.

The cutoff frequency \( f_T \) is defined to be the frequency at which the magnitude of the current gain of the device is unity, or when the magnitude of the input current \( I_i \) is equal to the ideal load current \( I_d \). From Figure 10.57, we can see that

\[
I_i = j\omega (C_{gpr} + C_M) V_x
\]

(10.92)

and the ideal load current is

\[
I_d = g_m V_x
\]

(10.93)

The magnitude of the current gain is then

\[
\left| \frac{I_d}{I_i} \right| = \frac{g_m}{2\pi f (C_{gpr} + C_M)}
\]

(10.94)

Setting the magnitude of the current gain equal to unity at the cutoff frequency, we find

\[
f_T = \frac{g_m}{2\pi (C_{gpr} + C_M)} = \frac{g_m}{2\pi C_G}
\]

(10.95)

where \( C_G \) is the equivalent input gate capacitance.