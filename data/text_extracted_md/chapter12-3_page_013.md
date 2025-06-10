```markdown
### 12.5 Equivalent Circuit Models

!Figure 12.40  
**Figure 12.40** | Hybrid-pi equivalent circuit.

current. These two elements are in parallel with the junction capacitance, which is \( C_{\pi} \). Finally, \( r_{ex} \) is the series resistance between the external emitter terminal and the internal emitter region. This resistance is usually very small and may be on the order of 1 to 2 \(\Omega\).

Figure 12.39b shows the equivalent circuit looking into the collector terminal. The \( r_c \) resistance is the series resistance between the external and internal collector connections and the capacitance \( C_s \) is the junction capacitance of the reverse-biased collector-substrate junction. The dependent current source, \( g_m V_{\pi} \), is the collector current in the transistor, which is controlled by the internal base–emitter voltage. The resistance \( r_o \) is the inverse of the output conductance \( g_o \) and is primarily due to the Early effect.

Finally, Figure 12.39c shows the equivalent circuit of the reverse-biased B'–C' junction. The \( C_{\mu} \) parameter is the reverse-biased junction capacitance and \( r_{\mu} \) is the reverse-biased diffusion resistance. Normally, \( r_{\mu} \) is on the order of megohms and can be neglected. The value of \( C_{\mu} \) is usually much smaller than \( C_{\pi} \), but, because of the feedback effect that leads to the Miller effect and Miller capacitance, \( C_{\mu} \) cannot be ignored in most cases. The Miller capacitance is the equivalent capacitance between B' and E' due to \( C_{\mu} \) and the feedback effect, which includes the gain of the transistor. The Miller effect also reflects \( C_{\mu} \) between the C' and E' terminals at the output. However, the effect on the output characteristics can usually be ignored.

Figure 12.40 shows the complete hybrid-pi equivalent circuit. A computer simulation is usually required for this complete model because of the large number of elements. However, some simplifications can be made in order to gain an appreciation for the frequency effects of the bipolar transistor. The capacitances lead to frequency effects in the transistor, which means that the gain, for example, is a function of the input signal frequency.

----

**Objective:** Determine, to a first approximation, the frequency at which the small-signal current gain decreases to \( 1/\sqrt{2} \) of its low-frequency value.

**Example 12.13**

Consider the simplified hybrid-pi circuit shown in Figure 12.41. We are ignoring \( C_s \), \( C_{\mu} \), \( r_{\mu} \), and the series resistances. We must emphasize that this is a first-order calculation and that \( C_{\pi} \) normally cannot be neglected.
```