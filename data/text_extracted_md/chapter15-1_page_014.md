# Power Bipolar Transistors

!Figure 15.18  
**Figure 15.18** | An integrated circuit implementation of the npn Darlington pair configuration.

The overall common-emitter current gain is then

\[
\frac{i_c}{i_b} = \beta_A \beta_B + \beta_A + \beta_B
\]

(15.8)

Thus, if the gain of each individual transistor is \(\beta_A = \beta_B = 15\), then the overall gain of the Darlington pair is \(\frac{i_c}{i_b} = 255\). This overall gain is then substantially larger than that of the individual device. A diode may be incorporated as shown in Figure 15.17 to aid in turning off the transistor \(Q_A\). A reverse current out of the base of \(Q_B\) through the diode will pull charge out of the base of this transistor and turn the device off faster than when no diode is used.

The Darlington pair shown in Figure 15.17 is typically used in the output stage of a power amplifier when an npn bipolar transistor is required. A pnp Darlington pair may also be used to increase the effective current gain of a power pnp device.

The integrated circuit configuration of the npn Darlington pair may be as shown in Figure 15.18. The silicon dioxide that is shown completely penetrates through the p-type base region so that the base regions of the two transistors are isolated.

----

## TEST YOUR UNDERSTANDING

**TYU 15.1**  
Consider the vertical power silicon BJT shown in Figure 15.10. Assume that a reverse-biased voltage of 200 V is applied to the base–collector junction. Calculate the space charge width that extends into the (a) collector region and (b) base region.  
(\( \epsilon_r = 11.7 \), \( q = 1.6 \times 10^{-19} \, \text{C} \), \( \epsilon_0 = 8.85 \times 10^{-12} \, \text{F/m} \))

**TYU 15.2**  
For the emitter–follower circuit in Figure 15.19, the parameters are \( V_{CC} = 10 \, \text{V} \) and \( R_E = 200 \, \Omega \). The transistor current gain is \(\beta = 150\), and the current and voltage limitations are \( I_{C,\text{max}} = 200 \, \text{mA} \) and \( V_{CE,\text{sat}} = 5 \, \text{V} \). Determine the minimum transistor power rating such that the transistor Q-point is always inside the safe operating area.  
(\( \lambda = 0 \), \( \epsilon = 11.7 \), \( q = 1.6 \times 10^{-19} \, \text{C} \))

!Figure 15.19  
**Figure 15.19** | Figure for Exercise TYU 15.2.