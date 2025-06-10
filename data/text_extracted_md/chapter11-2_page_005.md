# 11.4 Additional Electrical Characteristics

Severe as the voltage drop in the substrate approaches 0.6 to 0.7 V. A fraction of the injected electrons diffuses across the parasitic base region into the reverse-biased drain space charge region where they also add to the drain current.

The avalanche breakdown process is a function of not only the electric field but the number of carriers involved. The rate of avalanche breakdown increases as the number of carriers in the drain space charge region increases. We now have a regenerative or positive feedback mechanism. Avalanche breakdown near the drain terminal produces the substrate current, which produces the forward-biased source-substrate pn junction voltage. The forward-biased junction injects carriers that can diffuse back to the drain and increase the avalanche process. The positive feedback produces an unstable system.

The snapback or negative resistance portion of the curve shown in Figure 11.22 can now be explained by using the parasitic bipolar transistor. The potential of the base of the bipolar transistor near the emitter (source) is almost floating, since this voltage is determined primarily by the avalanche-generated substrate current rather than an externally applied voltage.

For the open-base bipolar transistor shown in Figure 11.24, we can write

\[
I_C = \alpha I_E + I_{CBO}
\]

(11.36)

where \( \alpha \) is the common base current gain and \( I_{CBO} \) is the base-collector leakage current. For an open base, \( I_C = I_E \), so Equation (11.36) becomes

\[
I_C = \alpha I_C + I_{CBO}
\]

(11.37)

At breakdown, the current in the B–C junction is multiplied by the multiplication factor \( M \), so we have

\[
I_C = M(\alpha I_C + I_{CBO})
\]

(11.38)

Solving for \( I_C \) we obtain

\[
I_C = \frac{M I_{CBO}}{1 - \alpha M}
\]

(11.39)

Breakdown is defined as the condition that produces \( I_C \to \infty \). For a single reverse-biased pn junction, \( M \to \infty \) at breakdown. However, from Equation (11.39), breakdown is now defined to be the condition when \( \alpha M \to 1 \) or, for the open-base condition, breakdown occurs when \( M \to 1/\alpha \), which is a much lower multiplication factor than for the simple pn junction.

An empirical relation for the multiplication factor is usually written as

\[
M = \frac{1}{1 - (V_{CE}/V_{BD})^m}
\]

(11.40)

where \( m \) is an empirical constant in the range of 3 to 6 and \( V_{BD} \) is the junction breakdown voltage.

The common base current gain factor \( \alpha \) is a strong function of collector current for small values of collector current. This effect will be discussed in Chapter 12 on bipolar transistors. At low currents, the recombination current in the B–E junction is a significant fraction of the total current so that the common base current gain is small.