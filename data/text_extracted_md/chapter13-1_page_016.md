# Chapter 13: The Junction Field-Effect Transistor

or

\[
I_D(\text{max}) = (0.522) \left\{ 1 - \frac{3}{4.335} \sqrt{0.814} \left[ 1 - \frac{2}{3} \sqrt{\frac{0.814}{4.335}} \right] \right\} = 0.313 \, \text{mA}
\]

**Comment**  
The maximum current through the JFET is less than the pinchoff current \(I_{p1}\).

## Exercise Problem

**Ex 13.3** Consider an n-channel silicon pn JFET with parameters \(N_d = 10^{18} \, \text{cm}^{-3}\), \(N_g = 10^{16} \, \text{cm}^{-2}\), \(a = 0.40 \, \mu\text{m}\), \(L = 5 \, \mu\text{m}\), \(W = 50 \, \mu\text{m}\), and \(\mu_n = 900 \, \text{cm}^2/\text{V}\cdot\text{s}\). Calculate the pinchoff current \(I_{p1}\) and the maximum drain current \(I_{D(\text{sat})}\) for \(V_{GS} = 0\).

The maximum saturation current calculated in this example is considerably less than that shown in Figure 13.12 because of the big difference in the width-to-length ratios. Once the pinchoff voltage of JFET has been designed, the channel width \(W\) is the primary design variable for determining the current capability of a device.

**Summary** Equations (13.29) and (13.35) are rather cumbersome to use in any hand calculations. We may show that, in the saturation region, the drain current is given to a good approximation by Equation (13.14), stated at the beginning of this section as

\[
I_D = I_{DSS} \left( 1 - \frac{V_{GS}}{V_p} \right)^2
\]

The current \(I_{DSS}\) is the maximum drain current and is the same as \(I_D(\text{max})\) in Equation (13.36). The parameter \(V_{GS}\) is the gate-to-source voltage and \(V_p\) is the pinchoff voltage.

!Figure 13.12

**Figure 13.12** Ideal current-voltage characteristics of a silicon n-channel JFET with \(a = 1.5 \, \mu\text{m}\), \(W/L = 170\), and \(N_g = 2.5 \times 10^{15} \, \text{cm}^{-2}\).  
*(From Yang [22].)*

| \(V_{DS}\) (V) | \(I_D\) (mA) |
|----------------|-------------|
| 0              | 0           |
| 2              | 8           |
| 4              | 16          |
| 6              | 24          |
| 8              | 28          |

- **Nonsaturation region**: \(V_{DS} < (V_{p0} - V_{gs}) + V_{GS}\)
- **Saturation region**: \(V_{DS} \geq (V_{p0} - V_{gs}) + V_{GS}\)

- \(V_{GS} = 0, -1, -2, -3, -4 \, \text{V}\)