# CHAPTER 11 Metal–Oxide–Semiconductor Field-Effect Transistor: Additional Concepts

!Subthreshold current–voltage characteristics

**Figure 11.4** | Subthreshold current–voltage characteristics for several values of substrate voltage (the threshold voltage is indicated on each curve).  
(From Schroder [17].)

Measurement of the slope of these curves has been used to experimentally determine the oxide–semiconductor interface state density.

If a MOSFET is biased at or even slightly below the threshold voltage, the drain current is not zero. The subthreshold current may add significantly to power dissipation in a large-scale integrated circuit in which hundreds or thousands of MOSFETs are used. The circuit design must include the subthreshold current or ensure that the MOSFET is biased sufficiently below the threshold voltage in the “off” state.

## 11.1.2 Channel Length Modulation

We assumed in the derivation of the ideal current–voltage relationship that the channel length \( L \) was a constant. However, when the MOSFET is biased in the saturation region, the depletion region at the drain terminal extends laterally into the channel, reducing the effective channel length. Since the depletion region width is bias dependent, the effective channel length is also bias dependent and is modulated by the drain-to-source voltage. This channel length modulation effect is shown in Figure 11.15 for an n-channel MOSFET.

The depletion width extending into the p-region of a p-n junction under zero bias can be written as

\[
x_p = \sqrt{\frac{2 \varepsilon \phi_{ir}}{e N_a}}
\]

(11.2)