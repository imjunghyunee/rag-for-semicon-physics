# Fundamentals of the Metal–Oxide–Semiconductor Field-Effect Transistor

The figure qualitatively indicates the relative charge density, which is essentially constant along the entire channel length for this case. The corresponding \( I_D \) versus \( V_{DS} \) curve is shown in the figure.

Figure 10.39b shows the situation when the \( V_{DS} \) value increases. As the drain voltage increases, the voltage drop across the oxide near the drain terminal decreases, which means that the induced inversion charge density near the drain also decreases. The incremental conductance of the channel at the drain decreases, which then means that the slope of the \( I_D \) versus \( V_{DS} \) curve will decrease. This effect is shown in the \( I_D \) versus \( V_{DS} \) curve in the figure.

When \( V_{DS} \) increases to the point where the potential drop across the oxide at the drain terminal is equal to \( V_T \), the induced inversion charge density is zero at the drain terminal. This effect is schematically shown in Figure 10.39c. At this point, the incremental conductance at the drain is zero, which means that the slope of the \( I_D \) versus \( V_{DS} \) curve is zero. We can write

\[
V_{GS} - V_{DS(sat)} = V_T \tag{10.43a}
\]

or

\[
V_{DS(sat)} = V_{GS} - V_T \tag{10.43b}
\]

where \( V_{DS(sat)} \) is the drain-to-source voltage producing zero inversion charge density at the drain terminal.

When \( V_{DS} \) becomes larger than the \( V_{DS(sat)} \) value, the point in the channel at which the inversion charge is just zero moves toward the source terminal. In this case, electrons enter the channel at the source, travel through the channel toward the drain, and then, at the point where the charge goes to zero, the electrons are injected into the space charge region where they are swept by the E-field to the drain contact. If we assume that the change in channel length \( \Delta L \) is small compared to the original length \( L \), then the drain current will be a constant for \( V_{DS} > V_{DS(sat)} \). The region of the \( I_D \) versus \( V_{DS} \) characteristic is referred to as the saturation region. Figure 10.39d shows this region of operation.

When \( V_{GS} \) changes, the \( I_D \) versus \( V_{DS} \) curve will change. We saw that, if \( V_{GS} \) increases, the initial slope of \( I_D \) versus \( V_{DS} \) increases. We can also note from Equation (10.43b) that the value of \( V_{DS(sat)} \) is a function of \( V_{GS} \). We can generate the family of curves for this n-channel enhancement mode MOSFET as shown in Figure 10.40.

Figure 10.41 shows an n-channel depletion mode MOSFET. If the n-channel region is actually an induced electron inversion layer created by the metal–semiconductor work function difference and fixed charge in the oxide, the current–voltage characteristics are exactly the same as we have discussed, except that \( V_T \) is a negative quantity. We may also consider the case when the n-channel region is actually an n-type semiconductor region. In this type of device, a negative gate voltage will induce a space charge region under the oxide, reducing the thickness of the n-channel region. The reduced thickness decreases the channel conductance, which reduces the drain current. A positive gate voltage will create an electron accumulation layer, which increases the drain current. One basic requirement for this device is that the channel