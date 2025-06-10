# 11.1 Nonideal Effects

!Figure 11.7

**Figure 11.7** Current–voltage characteristics of a MOSFET showing short-channel effects. (From See [22].)

Since \( I_D \) is now a function of \( V_{DS} \), the output resistance is no longer infinite. The drain current in the saturation region can be written as

\[
I_D = \frac{k'}{2} \cdot \frac{W}{L} \cdot \left[(V_{GS} - V_T)^2 (1 + \lambda V_{DS})\right]
\]

(11.12)

where \(\lambda\) is the **channel length modulation parameter**.

The output resistance is given by

\[
r_o = \left( \frac{\partial I_D}{\partial V_{DS}} \right)^{-1} = \left[ \frac{k'}{2} \cdot \frac{W}{L} \cdot (V_{GS} - V_T)^2 \cdot \lambda \right]^{-1}
\]

(11.13a)

Since \(\lambda\) is normally small, Equation (11.13a) can be written as

\[
r_o \approx \frac{1}{\lambda I_D}
\]

(11.13b)

Figure 11.7 shows some typical \( I_D \) versus \( V_{DS} \) curves with positive slopes in the saturation region due to channel length modulation. As the MOSFET dimensions become smaller, the change in the channel length \(\Delta L\) becomes a larger fraction of the original length \(L\), and the channel length modulation becomes more severe.

----

## Objective

Determine the increase in drain current due to short channel modulation.

Consider an n-channel MOSFET with a substrate doping concentration of \( N_a = 2 \times 10^{16} \, \text{cm}^{-3} \), a threshold voltage of \( V_T = 0.4 \, \text{V} \), and a channel length of \( L = 1 \, \mu\text{m} \). The device is biased at \( V_{GS} = 1 \, \text{V} \) and \( V_{DS} = 2.5 \, \text{V} \). Determine the ratio of actual drain current compared to the ideal value.

### Solution

We find

\[
\phi_p = V_T \ln \left( \frac{N_a}{n_i} \right) = (0.0259) \ln \left( \frac{2 \times 10^{16}}{1.5 \times 10^{10}} \right) = 0.3653 \, \text{V}
\]

\[
V_{DS(\text{sat})} = V_{GS} - V_T = 1.0 - 0.4 = 0.6 \, \text{V}
\]