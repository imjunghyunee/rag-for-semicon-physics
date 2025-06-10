controls the channel current. The parameters \( C_{gs} \) and \( C_{gd} \) are the total gate-to-source and total gate-to-drain capacitances. One parameter, \( r_{ds} \), shown in Figure 10.53, is not shown in Figure 10.52. This resistance is associated with the slope \( I_D \) versus \( V_{DS} \). In the ideal MOSFET biased in the saturation region, \( I_D \) is independent of \( V_{DS} \) so that \( r_{ds} \) would be infinite. In short-channel-length devices, in particular, \( r_{ds} \) is finite because of channel length modulation, which we will consider in the next chapter.

A simplified small-signal equivalent circuit valid at low frequency is shown in Figure 10.54. The series resistances, \( r_s \) and \( r_d \), have been neglected, so the drain current is essentially only a function of the gate-to-source voltage through the transconductance. The input gate impedance is infinite in this simplified model.

The source resistance \( r_s \) can have a significant effect on the transistor characteristics. Figure 10.55 shows a simplified, low-frequency equivalent circuit including \( r_s \) but neglecting \( r_{ds} \). The drain current is given by

\[
I_d = g_m V'_{gs}
\]

(10.84)

and the relation between \( V_{gs} \) and \( V'_{gs} \) can be found from

\[
V_{gs} = V'_{gs} + g_m V'_{gs} r_s = (1 + g_m r_s) V'_{gs}
\]

(10.85)

!Figure 10.54

**Figure 10.54** | Simplified, low-frequency small-signal equivalent circuit of a common-source n-channel MOSFET.

!Figure 10.55

**Figure 10.55** | Simplified, low-frequency small-signal equivalent circuit of common-source n-channel MOSFET including source resistance \( r_s \).