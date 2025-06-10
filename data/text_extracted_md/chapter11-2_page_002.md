```markdown
# 11.4 ADDITIONAL ELECTRICAL CHARACTERISTICS

There is a tremendous volume of information on MOSFETs that cannot be included in an introductory text on semiconductor physics and devices. However, two additional topics should be included here: breakdown voltage and threshold adjustment by ion implantation.

## 11.4.1 Breakdown Voltage

Several voltage breakdown mechanisms in the MOSFET must be considered, including voltage breakdown across the oxide as well as the various voltage breakdown mechanisms in the semiconductor junctions.

### Oxide Breakdown

We have assumed that the oxide is a perfect insulator. However, if the electric field in the oxide becomes large enough, breakdown can occur, which can lead to a catastrophic failure. In silicon dioxide, the electric field at breakdown is on the order of \(6 \times 10^6\) V/cm. This breakdown field is larger than that in silicon, but the gate oxides are also quite thin. A gate voltage of approximately 30 V would produce breakdown in an oxide with a thickness of 500 Å. However, a safety margin of a factor of 3 is common, so that the maximum safe gate voltage with \(t_{ox} = 500\) Å would be 10 V. A safety margin is necessary since there may be defects in the oxide that lower the breakdown field. Oxide breakdown is normally not a serious problem except in power devices and ultrathin oxide devices. Other oxide degradation problems are discussed later in this chapter.

### Avalanche Breakdown

Avalanche breakdown may occur by impact ionization in the space charge region near the drain terminal. We have considered avalanche breakdown in pn junctions in Chapter 7. In an ideal planar one-sided pn junction, breakdown is a function primarily of the doping concentration in the low-doped region of the junction. For the MOSFET, the low-doped region corresponds to the semiconductor substrate. If a p-type substrate doping is \(N_a = 3 \times 10^{15}\) cm\(^{-3}\), for example, the pn junction breakdown voltage would be approximately 25 V for a planar junction. However, the n\(^+\) drain contact may be a fairly shallow diffused region with
```

!Figure 11.20

**Figure 11.20** Qualitative variation of threshold voltage (a) with channel length and (b) with channel width.