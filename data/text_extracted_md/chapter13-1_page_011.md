## Objective

Design the channel doping concentration and metallurgical channel thickness to achieve a given pinch-off voltage.

Consider a silicon p-channel pn JFET at \( T = 300 \, \text{K} \). Assume that the gate doping concentration is \( N_g = 10^{18} \, \text{cm}^{-3} \). Determine the channel doping concentration and channel thickness so that the pinch-off voltage is \( V_p = 2.25 \, \text{V} \).

### Solution

There is not a unique solution to this design problem. We will pick a channel doping concentration of \( N_c = 2 \times 10^{16} \, \text{cm}^{-3} \) and determine the channel thickness. The built-in potential barrier is

\[
V_{bi} = V_t \ln \left( \frac{N_g N_c}{n_i^2} \right) = (0.0259) \ln \left( \frac{2 \times 10^{16} \times 10^{18}}{(1.5 \times 10^{10})^2} \right) = 0.832 \, \text{V}
\]

From Equation (13.8), the internal pinch-off voltage must be

\[
V_{po} = V_{bi} + V_p = 0.832 + 2.25 = 3.08 \, \text{V}
\]

and from Equation (13.6), the channel thickness can be determined as

\[
a = \left[ \frac{2 \varepsilon_s V_{po}}{e N_c} \right]^{1/2} = \left[ \frac{2(11.7 \times 8.85 \times 10^{-14})(3.08)}{(1.6 \times 10^{-19})(2 \times 10^{16})} \right]^{1/2} = 0.446 \, \mu\text{m}
\]

### Comment

If the channel doping concentration chosen were larger, the required channel thickness would decrease; a very small value of channel thickness would be difficult to fabricate within reasonable tolerance limits.

### EXERCISE PROBLEM

**Ex 13.2**  
The n\(^+\)p junction of a uniformly doped silicon p-channel JFET at \( T = 300 \, \text{K} \) has doping concentrations of \( N_g = 10^{18} \, \text{cm}^{-3} \) and \( N_c = 10^{16} \, \text{cm}^{-3} \). The metallurgical channel thickness is \( a = 0.40 \, \mu\text{m} \). Determine the internal pinch-off voltage and the pinch-off voltage of the JFET.  
(\( \Lambda \, \text{C}\lambda \, \Theta = \text{A} \, \Lambda \, \Theta \, \text{C} \, \text{I} = \text{A}^4 \, \text{S}\text{t}\text{U}\text{V} \))

Also, we will see later that if the channel doping concentration were smaller the current capability of the device would decrease. There are definite tradeoffs to be considered in any design problem.

We have determined the pinch-off voltage for both n-channel and p-channel JFETs when the drain-to-source voltage is zero. Now consider the case when both gate and drain voltages are applied. The depletion region width will vary with distance through the channel. Figure 13.11 shows the simplified geometry for an n-channel device. The depletion width \( h_t \) at the source end is a function of \( V_{bi} \) and \( V_{GS} \) but is not a function of drain voltage. The depletion width at the drain terminal is given by

\[
h_z = \left[ \frac{2 \varepsilon_s (V_{bi} + V_{DS} - V_{GS})}{e N_c} \right]^{1/2}
\]

(13.9)

Again, we must keep in mind that \( V_{GS} \) is a negative quantity for the n-channel device.