# 11.5 Radiation and Hot-Electron Effects

holes and electrons recombine in the oxide. Hence, the amount of charge reaching the Si–SiO₂ interface and being trapped is less than for a large positive gate voltage, where essentially all radiation-generated holes reach the interface without recombining with electrons. If the fraction of generated holes that become trapped is relatively constant, then the voltage shift becomes independent of positive gate bias, as shown in the figure. For negative applied gate voltages, the radiation-induced holes move toward the gate terminal. There can be positive charge trapping in the oxide near the gate, but the effect of this trapped charge on the threshold voltage is small.

## Objective

**Calculate the threshold voltage shift due to radiation-induced oxide charge trapping.**

Consider a MOS device with an oxide thickness of \( t_{\text{ox}} = 25 \, \text{nm} = 250 \, \text{Å} \). Assume that a pulse of ionizing radiation creates \( 10^8 \) electron–hole pairs per cm³ in the oxide. Also assume that the electrons are swept out through the gate terminal with zero recombination, and that 20 percent of the generated holes are trapped at the oxide–semiconductor interface.

### Solution

The areal density of holes generated in the oxide is

\[
N_s = (10^8)(250 \times 10^{-8}) = 2.5 \times 10^{12} \, \text{cm}^{-2}
\]

The equivalent trapped surface charge is then

\[
Q_i = (2.5 \times 10^{12})(0.2)(1.6 \times 10^{-19}) = 8 \times 10^{-6} \, \text{C/cm}^2
\]

We find

\[
C_{\text{ox}} = \frac{\varepsilon_{\text{ox}}}{t_{\text{ox}}} = \frac{(3.9)(8.85 \times 10^{-14})}{250 \times 10^{-8}} = 1.381 \times 10^{-7} \, \text{F/cm}^2
\]

The threshold voltage shift is then

\[
\Delta V_T = \frac{-Q_i}{C_{\text{ox}}} = \frac{-8 \times 10^{-6}}{1.381 \times 10^{-7}} = -0.579 \, \text{V}
\]

### Comment

As we have seen previously, a positive fixed oxide charge shifts the threshold voltage in the negative voltage direction. The ionizing radiation may shift an enhancement-mode device into the depletion mode.

### EXERCISE PROBLEM

**Ex 11.7** Repeat Example 11.7 for a MOS device with an oxide thickness of (a) \( t_{\text{ox}} = 12 \, \text{nm} = 120 \, \text{Å} \) and (b) \( t_{\text{ox}} = 8 \, \text{nm} = 80 \, \text{Å} \). (c) What can be said about the shift in threshold voltage as the oxide thickness decreases?

----

One failure mechanism, therefore, caused by the radiation-induced oxide charge in an n-channel MOSFET in an integrated circuit is a shift from enhancement mode to depletion mode. The device will be turned on rather than off at zero gate voltage; consequently, the circuit function may be disrupted or an excessive power supply current may be generated in the circuit.