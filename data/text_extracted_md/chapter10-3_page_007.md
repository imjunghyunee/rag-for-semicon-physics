## 10.3 The Basic MOSFET Operation

which yields

\[
\mu_n = 773 \, \text{cm}^2/\text{V-s}
\]

We can then determine

\[
V_T = 0.625 \, \text{V}
\]

### Comment

The mobility of carriers in the inversion layer is less than that in the bulk semiconductor due to the surface scattering effect. We will discuss this effect in the next chapter.

### Exercise Problem

**Ex 10.8** An n-channel silicon MOSFET has the following parameters: \( W = 6 \, \mu\text{m}, L = 1.5 \, \mu\text{m}, \) and \( t_{ox} = 8 \, \text{nm} = 80 \, \text{Å}. \) When the transistor is biased in the saturation region, the drain current is \( I_D(sat) = 0.132 \, \text{mA} \) at \( V_{GS} = 1.0 \, \text{V} \) and \( I_D(sat) = 0.295 \, \text{mA} \) at \( V_{GS} = 1.25 \, \text{V}. \) Determine the electron mobility and the threshold voltage.

The current–voltage relationship of a p-channel device can be obtained by the same type of analysis. Figure 10.49 shows a p-channel enhancement mode MOSFET. The voltage polarities and current direction are the reverse of those in the n-channel device. We may note the change in the subscript notation for this device. For the current direction shown in the figure, the \( I–V \) relation for the p-channel MOSFET biased in the nonsaturation region is

\[
I_D = \frac{W \mu_p C_{ox}}{2L} \left[ 2(V_{SG} + V_T)V_{SD} - V_{SD}^2 \right]
\]

(10.70)

Equation (10.70) is valid for \( 0 \leq V_{SD} \leq V_{SD}(sat). \)

!Figure 10.49

**Figure 10.49** | Cross section and bias configuration for a p-channel enhancement mode MOSFET.