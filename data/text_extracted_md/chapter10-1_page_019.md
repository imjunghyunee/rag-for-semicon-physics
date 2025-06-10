# 10.1 The Two-Terminal MOS Structure

!Energy-band diagram through the MOS structure with a positive applied gate bias.

**Figure 10.20** | Energy-band diagram through the MOS structure with a positive applied gate bias.

and is the magnitude of the maximum space charge density per unit area of the depletion region.

The energy-band diagram of the MOS system with an applied positive gate voltage is shown in Figure 10.20. As we mentioned, an applied gate voltage will change the voltage across the oxide and will change the surface potential. We had from Equation (10.20) that

\[
V_G = \Delta V_{ox} + \Delta \phi_s = V_{ox} + \phi_s + \phi_{ms}
\]

At threshold, we can define \( V_G = V_{TN} \), where \( V_{TN} \) is the threshold voltage that creates the electron inversion layer charge. The surface potential is \( \phi_s = 2\phi_f \) at threshold, so Equation (10.20) can be written as

\[
V_{TN} = V_{oxT} + 2\phi_f + \phi_{ms} \tag{10.28}
\]

where \( V_{oxT} \) is the voltage across the oxide at this threshold inversion point.

The voltage \( V_{oxT} \) can be related to the charge on the metal and to the oxide capacitance by

\[
V_{oxT} = \frac{Q'_m}{C_{ox}} \tag{10.29}
\]

where again \( C_{ox} \) is the oxide capacitance per unit area. Using Equation (10.26), we can write

\[
V_{oxT} = \frac{Q'_m}{C_{ox}} = \frac{1}{C_{ox}} \left( |Q'_s(\text{max})| - Q'_i \right) \tag{10.30}
\]

Finally, the threshold voltage can be written as

\[
V_{TN} = \frac{|Q'_s(\text{max})|}{C_{ox}} - \frac{Q'_i}{C_{ox}} + \phi_{ms} + 2\phi_f \tag{10.31a}
\]

or

\[
V_{TN} = \left( \frac{|Q'_s(\text{max})| - Q'_i}{t_{ox}/\epsilon_{ox}} \right) + \phi_{ms} + 2\phi_f \tag{10.31b}
\]