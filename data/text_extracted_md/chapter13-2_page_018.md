# CHAPTER 13 The Junction Field-Effect Transistor

Various velocity versus electric field models can be used to derive different \( I\text{-}V \) expressions. However, Equations (13.74) and (13.75) yield satisfactory results for most situations. Figure 13.30 shows a comparison between experimental and calculated \( I\text{-}V \) characteristics. As observed in the figure, the current in these heterojunction devices can be quite large. The transconductance of the MODFET is defined as it was for the pn JFET and MESFET. Typical measured values at \( T = 300 \, \text{K} \) are in the range of 250 mS/mm. Higher values have been reported. These transconductance values are significantly larger than for either the pn JFET or the MESFET.

HEMTs may also be fabricated with multiple heterojunction layers. This device type is shown in Figure 13.31. A single heterojunction for an AlGaAs–GaAs interface has a maximum two-dimensional electron sheet density on the order of \( 1 \times 10^{12} \, \text{cm}^{-2} \). This concentration can be increased by fabricating two or more AlGaAs–GaAs interfaces in the same epitaxial layer. The device current capacity is increased, and power performance is improved. The multichannel HEMT behaves as multiple single-channel HEMTs connected in parallel and modulated by the same gate but with slightly different threshold voltages. The maximum transconductance will not scale directly with the number of channels because of the change in threshold voltage with each channel. In addition, the effective channel length increases as the distance between the gate and channel increases.

HEMTs can be used in high-speed logic circuits. They have been used in flip-flop circuits operating at clock frequencies of 5.5 GHz at \( T = 300 \, \text{K} \); the clock frequency can be increased at lower temperatures. Small-signal, high-frequency amplifiers have also been investigated. HEMTs showing low noise and reasonable gains have been operated at 35 GHz. The maximum frequency increases as the channel length decreases. Cutoff frequencies on the order of 100 GHz have been measured with channel lengths of 0.25 \(\mu\text{m}\).

!Figure 13.30

**Figure 13.30** | Current–voltage characteristics of an enhancement mode HEMT, in which solid curves are numerical calculations and dots are measured points.  
*(From Shur [13].)*

!Figure 13.31

**Figure 13.31** | A multilayer HEMT.