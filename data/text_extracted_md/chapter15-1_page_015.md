# 15.5 Power MOSFETs

The basic operation of the power MOSFET is the same as that of any MOSFET. However, the current handling capability of these devices is usually in the ampere range, and the drain-to-source blocking voltage may be in the range of 50 to 100 volts or even higher. One big advantage that a power MOSFET has over a bipolar power device is that the control signal is applied to the gate whose input impedance is extremely large. Even during switching between on and off states, the gate current is small, so that relatively large currents can be switched with very small control currents.

## 15.5.1 Power Transistor Structures

Large currents can be obtained in a MOSFET with a very large channel width. To achieve a large channel width device with good characteristics, power MOSFETs are fabricated with a repetitive pattern of small cells operating in parallel. To achieve a large blocking voltage, a vertical structure is used. There are two basic power MOSFET structures. The first is called a DMOS device and is shown in Figure 15.20. The DMOS device uses a double diffusion process: The p-base or the p-substrate region and the n⁺ source contact are diffused through a common window defined by the edge of the gate. The p-base region is diffused deeper than the n⁺ source, and the difference in the lateral diffusion distance between the p-base and the n⁺ source defines the surface channel length.

Electrons enter the source terminal and flow laterally through the inversion layer under the gate to the n-drift region. The electrons then flow vertically through the n-drift region to the drain terminal. The conventional current direction is from the drain to the source. The n-drift region must be moderately doped so that the drain breakdown voltage is sufficiently large. However, the thickness of the n-drift region should also be as thin as possible to minimize drain resistance.

The second power MOSFET structure, shown in Figure 15.21, is a VMOS structure. The vertical channel or VMOS power device is a nonplanar structure that requires a different type of fabrication process. In this case, a p-base or p⁺-substrate diffusion

!Figure 15.20  
**Figure 15.20** | Cross section of a double-diffused MOS (DMOS) transistor.

!Figure 15.21  
**Figure 15.21** | Cross section of a vertical channel MOS (VMOS) transistor.