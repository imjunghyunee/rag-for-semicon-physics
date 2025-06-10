# 12.4 Nonideal Effects

In all previous discussions, we have considered a transistor with uniformly doped regions, low injection, constant emitter and base widths, an ideal constant energy bandgap, uniform current densities, and junctions that are not in breakdown. If any of these ideal conditions is not present, then the transistor properties will deviate from the ideal characteristics we have derived.

## 12.4.1 Base Width Modulation

We have implicitly assumed that the neutral base width \( x_B \) is constant. This base width, however, is a function of the B–C voltage, since the width of the space charge region extending into the base region varies with B–C voltage. As the B–C reverse-biased voltage increases, the B–C space charge region width increases, which reduces \( x_B \). A change in the neutral base width will change the collector current as can be observed in Figure 12.21. A reduction in base width will cause the gradient in the minority carrier concentration to increase, which in turn causes an increase in the diffusion current. This effect is known as **base width modulation**; it is also called the **Early effect**.

The Early effect can be seen in the current–voltage characteristics shown in Figure 12.22. In most cases, a constant base current is equivalent to a constant B–E voltage. Ideally the collector current is independent of the B–C voltage so that the slope of the curves would be zero; thus, the output conductance of the transistor would be zero. However, the base width modulation, or Early effect, produces a nonzero slope and gives rise to a finite output conductance. If the collector current characteristics are extrapolated to zero collector current, the curves intersect the voltage axis at a point that is defined as the Early voltage. The Early voltage is considered to be a positive value. It is a common parameter given in transistor specifications; typical values of Early voltage are in the 100- to 300-V range.

!Figure 12.21

**Figure 12.21** The change in the base width and the change in the minority carrier gradient as the B–C space charge width changes.