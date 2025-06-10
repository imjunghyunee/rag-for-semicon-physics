# 10.4 Frequency Limitations

!Inherent resistances and capacitances in the n-channel MOSFET structure.

**Figure 10.52** Inherent resistances and capacitances in the n-channel MOSFET structure.

Within the transistor structure, along with elements that represent the basic device equations, is shown in Figure 10.52. One simplifying assumption we will make in the equivalent circuit is that the source and substrate are both tied to ground potential.

Two of the capacitances connected to the gate are inherent in the device. These capacitances are \( C_{gs} \) and \( C_{gb} \) which represent the interaction between the gate and the channel charge near the source and drain terminals, respectively. The remaining two gate capacitances, \( C_{gd} \) and \( C_{db} \), are parasitic or overlap capacitances. In real devices, the gate oxide will overlap the source and drain contacts because of tolerance or fabrication factors. As we will see, the drain overlap capacitance—\( C_{db} \) in particular—will lower the frequency response of the device. The parameter \( C_{ds} \) is the drain-to-substrate pn junction capacitance, and \( r_s \) and \( r_d \) are the series resistances associated with the source and drain terminals. The small-signal channel current is controlled by the internal gate-to-source voltage through the transconductance.

The small-signal equivalent circuit for the n-channel common-source MOSFET is shown in Figure 10.53. The voltage \( V_{gs}' \) is the internal gate-to-source voltage that

!Small-signal equivalent circuit of a common-source n-channel MOSFET.

**Figure 10.53** Small-signal equivalent circuit of a common-source n-channel MOSFET.