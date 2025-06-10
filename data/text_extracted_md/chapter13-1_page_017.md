### 13.2 The Device Characteristics

!Figure 13.13.1

**Figure 13.13.1** Comparison of Equations (13.14) and (13.35) for the \( I_D \) versus \( V_{GS} \) characteristics of a JFET biased in the saturation region.

We may note that, for n-channel depletion mode JFET, both \( V_{GS} \) and \( V_P \) are negative and, for the p-channel depletion mode device, both are positive. Figure 13.13 shows the comparison between Equations (13.14) and (13.35).

#### 13.2.3 Transconductance

The transconductance is the transistor gain of the JFET; it indicates the amount of control the gate voltage has on the drain current. The transconductance is defined as

\[
g_m = \frac{\partial I_D}{\partial V_{GS}}
\]

(13.37)

Using the expressions for the ideal drain current derived in the last section, we can write the expressions for the transconductance.

The drain current for an n-channel depletion mode device in the nonsaturation region is given by Equation (13.29). We can then determine the transconductance of the transistor in the same region as

\[
g_{m} = \frac{\partial I_D}{\partial V_{GS}} = \frac{3I_{D1}}{V_{p0}} \frac{V_{th} - V_{GS}}{V_{p0}} \left[ \sqrt{\frac{V_{DS}}{V_{th} - V_{GS}}} + 1 - 1 \right]
\]

(13.38)

Taking the limit as \( V_{DS} \) becomes small, the transconductance becomes

\[
g_{m} \approx \frac{3I_{D1}}{2V_{p0}} \cdot \frac{V_{DS}}{\sqrt{V_{th}(V_{th} - V_{GS})}}
\]

(13.39)

We can also write Equation (13.39) in terms of the conductance parameter \( G_{0} \) as

\[
g_{m} = \frac{G_{0}}{2} \cdot \frac{V_{DS}}{\sqrt{V_{th}(V_{th} - V_{GS})}}
\]

(13.40)