!Figure 13.10  
**Figure 13.10** | Geometries of simplified (a) n-channel and (b) p-channel pn JFETs.

At pinchoff, \( h = a \) and the total potential across the p*n junction is called the **internal pinchoff voltage**, denoted by \( V_{po} \). We now have

\[
a = \left[ \frac{2 \varepsilon_s V_{po}}{e N_d} \right]^{1/2}
\]

(13.2)

or

\[
V_{po} = \frac{ea^2 N_d}{2 \varepsilon_s}
\]

(13.3)

Note that the internal pinchoff voltage is defined as a positive quantity.

The internal pinchoff voltage \( V_{po} \) is not the gate-to-source voltage to achieve pinchoff. The gate-to-source voltage that must be applied to achieve pinchoff is described as the **pinchoff voltage** and is also variously called the **turn-off voltage** or **threshold voltage**. The pinchoff voltage is denoted by \( V_p \) and is defined from Equations (13.1) and (13.2) as

\[
V_p - V_s = V_{po} \quad \text{or} \quad V_p = V_{th} - V_{po}
\]

(13.4)

The gate-to-source voltage to achieve pinchoff in an n-channel depletion mode JFET is negative; thus, \( V_{po} > V_{gs} \).

----

### Objective:

Calculate the internal pinchoff voltage and pinchoff voltage of an n-channel JFET.

**Example 13.1**

Assume that the p*n junction of a uniformly doped silicon n-channel JFET at \( T = 300 \, K \) has doping concentrations of \( N_d = 10^{18} \, \text{cm}^{-3} \) and \( N_a = 10^{16} \, \text{cm}^{-3} \). Assume that the metallurgical channel thickness, \( a \), is 0.75 μm = 0.75 × 10\(^{-4}\) cm.

#### Solution

The internal pinchoff voltage is given by Equation (13.3), so we have

\[
V_{po} = \frac{ea^2 N_d}{2 \varepsilon_s} = \frac{(1.6 \times 10^{-19})(0.75 \times 10^{-4})^2(10^{18})}{2(11.7)(8.85 \times 10^{-14})} = 4.35 \, V
\]