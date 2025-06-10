# Fundamentals of the Metal–Oxide–Semiconductor Field-Effect Transistor

in charge densities are shown in the figure. The total capacitance of the series combination is

\[
\frac{1}{C'(depl)} = \frac{1}{C_{ox}} + \frac{1}{C_{SD}}
\]
(10.36a)

or

\[
C'(depl) = \frac{C_{ox} C_{SD}}{C_{ox} + C_{SD}}
\]
(10.36b)

Since \( C_{ox} = \frac{\varepsilon_{ox}}{t_{ox}} \) and \( C_{SD} = \frac{\varepsilon_{s}}{x_d} \), Equation (10.36b) can be written as

\[
C'(depl) = \frac{C_{ox}}{1 + \frac{C_{ox}}{C_{SD}}} = \frac{\varepsilon_{ox}}{t_{ox} + \left( \frac{\varepsilon_{ox}}{\varepsilon_{s}} \right) x_d}
\]
(10.37)

As the space charge width increases, the total capacitance \( C'(depl) \) decreases.

We had defined the threshold inversion point to be the condition when the maximum depletion width is reached, but there is essentially zero inversion charge density. This condition will yield a minimum capacitance \( C'_{min} \), which is given by

\[
C'_{min} = \frac{\varepsilon_{ox}}{t_{ox} + \left( \frac{\varepsilon_{ox}}{\varepsilon_{s}} \right) x_{dT}}
\]
(10.38)

Figure 10.25a shows the energy-band diagram of this MOS device for the inversion condition. In the ideal case, a small incremental change in the voltage across the MOS capacitor will cause a differential change in the inversion layer charge density. The space charge width does not change. If the inversion charge can respond to the change in capacitor voltage as indicated in Figure 10.25b, then the capacitance is again just the oxide capacitance, or

\[
C'(inv) = C_{ox} = \frac{\varepsilon_{ox}}{t_{ox}}
\]
(10.39)

!Figure 10.25

**Figure 10.25** | (a) Energy-band diagram through a MOS capacitor for the inversion mode. (b) Differential charge distribution at inversion for a low-frequency differential change in gate voltage.