The drain current from Equation (10.84) can now be written as

\[
I_d = \left( \frac{g_m}{1 + g_m r_s'} \right)_{V_{gs} = V_{gs}'} = g_m' V_{gs}'
\]

(10.86)

The source resistance reduces the effective transconductance or transistor gain.

The equivalent circuit of the p-channel MOSFET is exactly the same as that of the n-channel except that all voltage polarities and current directions are reversed. The same capacitances and resistances that are in the n-channel model apply to the p-channel model.

## 10.4.2 Frequency Limitation Factors and Cutoff Frequency

There are two basic frequency limitation factors in the MOSFET. The first factor is the channel transit time. If we assume that carriers are traveling at their saturation drift velocity \( v_{sat} \), then the transit time is \( \tau = L/v_{sat} \) where \( L \) is the channel length. If \( v_{sat} = 10^7 \) cm/s and \( L = 1 \) μm, then \( \tau = 10 \) ps, which translates into a maximum frequency of 100 GHz. This frequency is much larger than the typical maximum frequency response of a MOSFET. The transit time of carriers through the channel is usually not the limiting factor in the frequency responses of MOSFETs.

The second limiting factor is the gate or capacitance charging time. If we neglect \( r_g, r_{ds}, \) and \( C_{db} \), the resulting equivalent small-signal circuit is shown in Figure 10.56 where \( R_L \) is a load resistance.

The input gate impedance in this equivalent circuit is no longer infinite. Summing currents at the input gate node, we have

\[
I_i = j \omega C_{gsT} V_{gs} + j \omega C_{gdT} (V_{gs} - V_d)
\]

(10.87)

where \( I_i \) is the input current. Likewise, summing currents at the output drain node, we have

\[
\frac{V_d}{R_L} + g_m V_{gs} + j \omega C_{gdT} (V_d - V_{gs}) = 0
\]

(10.88)

!Figure 10.56

**Figure 10.56** | High-frequency small-signal equivalent circuit of common-source n-channel MOSFET.