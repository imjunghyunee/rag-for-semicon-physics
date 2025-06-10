Equation (10.62) is the ideal current–voltage relationship of the n-channel MOSFET in the nonsaturation region for \(0 \leq V_{DS} \leq V_{DS(sat)}\), and Equation (10.66) is the ideal current–voltage relationship of the n-channel MOSFET in the saturation region for \(V_{DS} \geq V_{DS(sat)}\). These I–V expressions were explicitly derived for an n-channel enhancement mode device. However, these same equations apply to an n-channel depletion mode MOSFET in which the threshold voltage \(V_T\) is a negative quantity.

----

**Objective:** Design the width of a MOSFET such that a specified current is induced for a given applied bias.

**Example 10.7**

Consider an ideal n-channel MOSFET with parameters \(L = 1.25 \, \mu m\), \(\mu_n = 650 \, \text{cm}^2/\text{V} \cdot \text{s}\), \(C_{ox} = 6.9 \times 10^{-8} \, \text{F/cm}^2\), and \(V_T = 0.65 \, \text{V}\). Design the channel width \(W\) such that \(I_D(sat) = 4 \, \text{mA}\) for \(V_{GS} = 5 \, \text{V}\).

**Solution**

For the transition biased in the saturation region, we have, from Equation (10.66),

\[
I_D(sat) = \frac{W \mu_n C_{ox}}{2L} (V_{GS} - V_T)^2
\]

or

\[
4 \times 10^{-3} = \frac{W(650)(6.9 \times 10^{-8})}{2(1.25 \times 10^{-4})} \cdot (5 - 0.65)^2 = 3.39 \, W
\]

Then

\[
W = 11.8 \, \mu m
\]

**Comment**

The current capability of a MOSFET is directly proportional to the channel width \(W\). The current handling capability can be increased by increasing \(W\).

----

**Exercise Problem**

**Ex 10.7** The parameters of an n-channel silicon MOSFET are \(\mu_n = 650 \, \text{cm}^2/\text{V} \cdot \text{s}\), \(t_{ox} = 8 \, \text{nm} = 80 \, \text{Å}\), \(W/L = 12\), and \(V_T = 0.40 \, \text{V}\). If the transistor is biased in the saturation region, find the drain current for (a) \(V_{GS} = 0.8 \, \text{V}\), (b) \(V_{GS} = 1.2 \, \text{V}\), and (c) \(V_{GS} = 1.6 \, \text{V}\).

----

We can use the I–V relations to experimentally determine the mobility and threshold voltage parameters. From Equation (10.62), we can write, for very small values of \(V_{DS}\),

\[
I_D = \frac{W \mu_n C_{ox}}{L} (V_{GS} - V_T)V_{DS} \tag{10.68}
\]

Figure 10.48a shows a plot of Equation (10.68) as a function of \(V_{GS}\) for constant \(V_{DS}\). A straight line is fitted through the points. The deviation from the straight line at low values of \(V_{GS}\) is due to subthreshold conduction and the deviation at higher values of \(V_{GS}\).