# 10.2 Capacitance–Voltage Characteristics

!Figure 10.26

**Figure 10.26** | Ideal low-frequency capacitance versus gate voltage of a MOS capacitor with a p-type substrate. Individual capacitance components are also shown.

Figure 10.26 shows the ideal capacitance versus gate voltage, or C–V, characteristics of the MOS capacitor with a p-type substrate. The three dashed segments correspond to the three components \( C_{ox}, C_{SD}, \) and \( C'_{inv} \). The solid curve is the ideal net capacitance of the MOS capacitor. Moderate inversion, which is indicated in the figure, is the transition region between the point when only the space charge density changes with gate voltage and when only the inversion charge density changes with gate voltage.

The point on the curve that corresponds to the flat-band condition is of interest. The flat-band condition occurs between the accumulation and depletion conditions. The capacitance at flat band is given by

\[
C_{FB} = \frac{\varepsilon_{ox}}{t_{ox}} + \left( \frac{\varepsilon_{0}}{\varepsilon_s} \right) \sqrt{\frac{kT}{q} \left( \frac{\varepsilon_s}{qN_a} \right)}
\]

(10.40)

We may note that the flat-band capacitance is a function of oxide thickness as well as semiconductor doping. The general location of this point on the C–V plot is shown in Figure 10.26.

----

## Objective

Calculate \( C_{ox}, C'_{inv}, \) and \( C_{FB} \) for a MOS capacitor.

Consider a p-type silicon substrate at \( T = 300 \, K \) doped to \( N_a = 10^{16} \, \text{cm}^{-3} \).

The oxide is silicon dioxide with a thickness of \( t_{ox} = 18 \, \text{nm} = 180 \, \text{Å} \), and the gate is aluminum.

### Solution

The oxide capacitance is

\[
C_{ox} = \frac{\varepsilon_{ox}}{t_{ox}} = \frac{(3.9)(8.85 \times 10^{-14})}{180 \times 10^{-8}} = 1.9175 \times 10^{-7} \, \text{F/cm}^2
\]

To find the minimum capacitance, we need to calculate

\[
\phi_b = V_t \ln \left( \frac{N_a}{n_i} \right) = (0.0259) \ln \left( \frac{10^{16}}{1.5 \times 10^{10}} \right) = 0.3473 \, \text{V}
\]