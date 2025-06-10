# CHAPTER 15 Semiconductor Microwave and Power Devices

!Figure 15.28

**Figure 15.28** (a) Cross section of vertical MOSFET showing parasitic BJT and distributed resistance; (b) equivalent circuit of MOSFET and parasitic BJT with distributed parameters.

The BJT should be cutoff at all times, which means the source-to-body voltage (emitter-to-base voltage) should be as close to zero as possible. We see from the geometries of Figures 15.20 and 15.21 that the source ohmic contact also goes across the p-type body region so that this junction voltage is zero during steady-state operation of the transistor. However, the BJT may be turned on during high-speed switching of the MOSFET.

Figure 15.28b shows that the base and the collector of the parasitic BJT are connected by the gate-to-drain capacitance. A parasitic or distributed resistance also connects the base to the emitter of the BJT. When the MOSFET is being turned off, the drain-to-source voltage increases and induces a current in the gate-to-drain capacitance in the direction from the parasitic collector terminal to the parasitic base terminal. This induced current may be large enough to induce a voltage in the parasitic resistance that is sufficient to forward bias the base–emitter junction and therefore turn the BJT on. The turned on BJT may then induce a large drain current that can cause burnout of the MOSFET. This breakdown mechanism is known as **snapback breakdown** and has been discussed briefly in Section 11.4.1.