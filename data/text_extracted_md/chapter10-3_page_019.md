## 10.5 The CMOS Technology

must always be at a higher potential than the p well; therefore, this pn junction will always be reverse biased.

With ion implantation now being extensively used for threshold voltage control, both the n-well CMOS process and twin-well CMOS process can be used. The n-well CMOS process, shown in Figure 10.58b, starts with an optimized p-type substrate that is used to form the n-channel MOSFETs. (The n-channel MOSFETs, in general, have superior characteristics, so this starting point should yield excellent n-channel devices.) The n well is then added, in which the p-channel devices are fabricated. The n-well doping can be controlled by ion implantation.

The twin-well CMOS process, shown in Figure 10.58c, allows both the p-well and n-well regions to be optimally doped to control the threshold voltage and transconductance of each transistor. The twin-well process allows a higher packing density because of self-aligned channel stops.

One major problem in CMOS circuits has been latch-up. **Latch-up** refers to a high-current, low-voltage condition that may occur in a four-layer pnpn structure. Figure 10.59a shows the circuit of a CMOS inverter and Figure 10.59b shows a simplified integrated circuit layout of the inverter circuit. In the CMOS layout, p⁺ source to n substrate to p well to n⁺ source forms such a four-layer structure.

The equivalent circuit of this four-layer structure is shown in Figure 10.60. The silicon-controlled rectifier action involves the interaction of the parasitic pnp and npn bipolar transistors. Bipolar transistors are discussed in Chapter 12. The npn transistor corresponds to the vertical n⁺-source to p-well to n-substrate structure and the pnp transistor corresponds to the lateral p-well to n-substrate to p⁺-source structure. Under normal CMOS operation, both parasitic bipolar transistors are cut off. However, under certain conditions, avalanche breakdown may occur in the p-well to n-substrate junction, driving both bipolar transistors into saturation. This high-current, low-voltage condition—latch-up—can sustain itself by positive feedback. The condition can prevent the CMOS circuit from operating and can also cause permanent damage and burnout of the circuit.

Latch-up can be prevented if the product \( \beta_n \beta_p \) is less than unity at all times, where \( \beta_n \) and \( \beta_p \) are the common-emitter current gains of the npn and pnp parasitic transistors.

!CMOS Inverter Circuit

**Figure 10.59** (a) CMOS inverter circuit. (b) Simplified integrated circuit cross section of CMOS inverter.