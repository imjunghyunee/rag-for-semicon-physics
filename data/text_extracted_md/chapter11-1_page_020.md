# CHAPTER 11 Metal–Oxide–Semiconductor Field-Effect Transistor: Additional Concepts

If we neglect short-channel effects, the gate-controlled bulk charge can be written as

\[
Q_B = Q_{B0} + \Delta Q_B
\]

(11.31)

where \( Q_B \) is the total bulk charge, \( Q_{B0} \) is the ideal bulk charge, and \( \Delta Q_B \) is the additional bulk charge at the ends of the channel width. For a uniformly doped p-type semiconductor biased at the threshold inversion point, we may write

\[
|Q_{B0}| = eN_sWLx_{tr}
\]

(11.32)

and

\[
\Delta Q_B = eN_sLx_{tr}(\xi x_{tr})
\]

(11.33)

where \( \xi \) is a fitting parameter that accounts for the lateral space charge width. The lateral space charge width may not be the same as the vertical width \( x_{tr} \) due to the thicker field oxide at the ends, and/or due to the nonuniform semiconductor doping created by an ion implantation. If the ends were a semicircle, then \( \xi = \pi/2 \).

We may now write

\[
|Q_B| = |Q_{B0}| + |\Delta Q_B| = eN_sWLx_{tr} + eN_sLx_{tr}(\xi x_{tr})
\]

\[
= eN_sWLx_{tr} \left( 1 + \frac{\xi x_{tr}}{W} \right)
\]

(11.34)

The effect of the end space charge regions becomes significant as the width \( W \) decreases and the factor \( (\xi x_{tr}) \) becomes a significant fraction of the width \( W \).

The change in threshold voltage due to the additional space charge is

\[
\Delta V_T = \frac{eN_sx_{tr}}{C_{ox}} \left( \frac{\xi x_{tr}}{W} \right)
\]

(11.35)

The shift in threshold voltage due to a narrow channel is in the positive direction for the n-channel MOSFET. As the width \( W \) becomes smaller, the shift in threshold voltage becomes larger.

## EXAMPLE 11.4

**Objective:** Design the channel width that will limit the threshold voltage shift because of narrow channel effects to a specified value.

Consider a silicon n-channel MOSFET with \( N_s = 3 \times 10^{16} \text{ cm}^{-3} \) and \( t_{ox} = 20 \text{ nm} = 200 \text{ Å} \). Let \( \xi = \pi/2 \). Assume that the threshold voltage shift is to be limited to \( \Delta V_T = 0.2 \text{ V} \).

**Solution**

We find

\[
C_{ox} = 1.726 \times 10^{-7} \text{ F/cm}^2 \quad \text{and} \quad x_{tr} = 0.18 \text{ μm}
\]

From Equation (11.35), we can express the channel width as

\[
W = \frac{eN_s(\xi x_{tr}^2)}{C_{ox}(\Delta V_T)} = \frac{(1.6 \times 10^{-19})(3 \times 10^{16})(\frac{7}{2})(0.18 \times 10^{-4})^2}{(1.726 \times 10^{-7})(0.2)}
\]

\[
= 7.08 \times 10^{-5} \text{ cm}
\]