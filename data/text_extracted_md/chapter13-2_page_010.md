# Chapter 13: The Junction Field-Effect Transistor

!JFET transconductance versus \( V_{GS} \)

**Figure 13.21** JFET transconductance versus \( V_{GS} \) (a) without and (b) with a source series resistance.

!Small-signal equivalent circuit with capacitance

**Figure 13.22** A small-signal equivalent circuit with capacitance.

The transistor is biased in the saturation region. Figure 13.21 shows a comparison between the theoretical and experimental transconductance values using the parameters from Example 13.4 and letting \( r_s = 2000 \, \Omega \). (A value of \( r_s = 2000 \, \Omega \) may seem excessive, but keep in mind that the active thickness of the semiconductor may be on the order of 1 \(\mu m\) or less; thus, a large series resistance may result if special care is not taken.)

## 13.4.2 Frequency Limitation Factors and Cutoff Frequency

There are two frequency limitation factors in a JFET. The first is the channel transit time. If we assume a channel length of 1 \(\mu m\) and assume carriers are traveling at their saturation velocity, then the transit time is on the order of

\[
\tau_t = \frac{L}{v_s} = \frac{1 \times 10^{-4}}{1 \times 10^7} = 10 \, \text{ps}
\]

(13.61)

The channel transit time is normally not the limiting factor except in very high frequency devices.

The second frequency limitation factor is the capacitance charging time. Figure 13.22 is a simplified equivalent circuit that includes the primary capacitances and ignores the diffusion resistances. The output current will be the short-circuit current. As the frequency of the input-signal voltage \( V_g \) increases, the impedance of \( C_{gd} \) and \( C_{gp} \) decreases so the current through \( C_{gd} \) will increase. For a constant \( g_m V_{gs} \), the \( I_d \) current will then decrease. The output current then becomes a function of frequency.

If the capacitance charging time is the limiting factor, then the cutoff frequency \( f_T \) is defined as the frequency at which the magnitude of the input current \( I_i \) is equal to the magnitude of the ideal output current \( g_m V_{gs} \) of the intrinsic transistor. We have,