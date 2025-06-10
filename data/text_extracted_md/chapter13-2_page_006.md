!Figure 13.15  
**Figure 13.15** | Cross section of JFET showing carrier velocity and space charge width saturation effects.

For high-frequency MESFETs, typical channel lengths are on the order of 1 μm. Channel length modulation and other effects become very important in short-channel devices.

### 13.3.2 Velocity Saturation Effects

We have seen that the drift velocity of a carrier in silicon saturates with increasing electric field. This velocity saturation effect implies that the mobility is not a constant. For very short channels, the carriers can easily reach their saturation velocity, which changes the I-V characteristics of the JFET.

Figure 13.15 shows the channel region with an applied drain voltage. As the channel narrows at the drain terminal, the velocity of the carriers increases since the current through the channel is constant. The carriers first saturate at the drain end of the channel. The depletion region will reach a saturation thickness, so we can write

\[
I_{D}(sat) = eN_{d}v_{sat}(a - h_{sat})W
\]

(13.56)

where \( v_{sat} \) is the saturation velocity and \( h_{sat} \) is the saturation depletion width. This saturation effect occurs at a drain voltage smaller than the \( V_{DS}(sat) \) value determined previously. Both \( I_{DS}(sat) \) and \( V_{DS}(sat) \) will be smaller than previously calculated.

Figure 13.16 shows normalized plots of \( I_D \) versus \( V_{DS} \). Figure 13.16a is for the case of a constant mobility and Figure 13.16b is for the case of velocity saturation. Since the I-V characteristics change when velocity saturation occurs, the transconductance will also change—the transconductance will become smaller; hence, the effective gain of the transistor decreases when velocity saturation occurs.

### 13.3.3 Subthreshold and Gate Current Effects

The subthreshold current is the drain current in the JFET that exists when the gate voltage is below the pinchoff or threshold value. The subthreshold conduction is shown in Figure 13.14. When the JFET is biased in the saturation region, the drain current varies quadratically with gate-to-source voltage. When \( V_{GS} \) is below the threshold value, the drain current varies exponentially with gate-to-source voltage. Near threshold, the abrupt depletion approximation does not accurately model the channel region: A more detailed potential profile in the space charge region must be used. However, these calculations are beyond the scope of this chapter.

When the gate voltage is approximately 0.5 to 1.0 V below threshold in an n-channel MESFET, the drain current reaches a minimum value and then slowly increases as the gate voltage decreases. The drain current in this region is the gate leakage current. Figure 13.17 is a plot of the drain current versus \( V_{GS} \) for the three regions.