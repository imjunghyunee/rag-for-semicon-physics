# Fundamentals of the Metal–Oxide–Semiconductor Field-Effect Transistor

We are assuming a constant mobility \( \mu_n \). For the n-channel device, the drain current enters the drain terminal and is a constant along the entire channel length. Letting \( I_D = -I_n \), Equation (10.61) becomes

\[
I_D = \frac{W \mu_n C_{ox}}{2L} \left[ 2(V_{GS} - V_T)V_{DS} - V_{DS}^2 \right] \tag{10.62}
\]

Equation (10.62) is valid for \( V_{GS} \geq V_T \) and for \( 0 \leq V_{DS} \leq V_{DS}(sat) \).

Equation (10.62) can also be written as

\[
I_D = \frac{k_i}{2} \cdot \frac{W}{L} \left[ 2(V_{GS} - V_T)V_{DS} - V_{DS}^2 \right] = K_n [2(V_{GS} - V_T) V_{DS} - V_{DS}^2] \tag{10.63}
\]

where \( k_i \) is the process conduction parameter and \( K_n \) is the conduction parameter. These parameters are described and defined in Equations (10.44b) and (10.44c).

Figure 10.47 shows plots of Equation (10.62) as a function of \( V_{DS} \) for several values of \( V_{GS} \). We can find the value of \( V_{DS} \) at the peak current value from \( \partial I_D / \partial V_{DS} = 0 \). Then, using Equation (10.62), the peak current occurs when

\[
V_{DS} = V_{GS} - V_T \tag{10.64}
\]

This value of \( V_{DS} \) is just \( V_{DS}(sat) \), the point at which saturation occurs. For \( V_{DS} > V_{DS}(sat) \), the ideal drain current is a constant and is equal to

\[
I_D(sat) = \frac{W \mu_n C_{ox}}{2L} \left[ 2(V_{GS} - V_T)V_{DS}(sat) - V_{DS}(sat)^2 \right] \tag{10.65}
\]

Using Equation (10.64) for \( V_{DS}(sat) \), Equation (10.65) becomes

\[
I_D(sat) = \frac{W \mu_n C_{ox}}{2L} (V_{GS} - V_T)^2 \tag{10.66}
\]

Equation (10.66) can also be written as

\[
I_D = \frac{k_i}{2} \cdot \frac{W}{L} \cdot (V_{GS} - V_T)^2 = K_n (V_{GS} - V_T)^2 \tag{10.67}
\]

!Figure 10.47

**Figure 10.47** | Plots of \( I_D \) versus \( V_{DS} \) from Equation (10.62).