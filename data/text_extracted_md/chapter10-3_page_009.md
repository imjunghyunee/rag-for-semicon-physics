## 10.3 The Basic MOSFET Operation

If we consider an n-channel MOSFET operating in the nonsaturation region, then using Equation (10.62), we have

\[
g_{m} = \frac{\partial I_D}{\partial V_{GS}} = \frac{W \mu_n C_{ox}}{L} \cdot V_{DS}
\]

(10.76)

The transconductance increases linearly with \( V_{DS} \) but is independent of \( V_{GS} \) in the nonsaturation region.

The I–V characteristics of an n-channel MOSFET in the saturation region are given by Equation (10.66). The transconductance in this region of operation is given by

\[
g_{m} = \frac{\partial I_D(\text{sat})}{\partial V_{GS}} = \frac{W \mu_n C_{ox}}{L} \cdot (V_{GS} - V_T)
\]

(10.77)

In the saturation region, the transconductance is a linear function of \( V_{GS} \) and is independent of \( V_{DS} \).

The transconductance is a function of the geometry of the device as well as of carrier mobility and threshold voltage. The transconductance increases as the width of the device increases, and it also increases as the channel length and oxide thickness decrease. In the design of MOSFET circuits, the size of the transistor, in particular the channel width \( W \), is an important engineering design parameter.

### 10.3.5 Substrate Bias Effects

In all of our analyses so far, the substrate, or body, has been connected to the source and held at ground potential. In MOSFET circuits, the source and body may not be at the same potential. Figure 10.50a shows an n-channel MOSFET and the associated double-subscripted voltage variables. The source-to-substrate pn junction must always be zero or reverse biased, so \( V_{SB} \) must always be greater than or equal to zero.

!Figure 10.50

**Figure 10.50** (a) Applied voltages on an n-channel MOSFET. (b) Energy-band diagram at inversion point when \( V_{SB} = 0 \). (c) Energy-band diagram at inversion point when \( V_{SB} > 0 \) is applied.