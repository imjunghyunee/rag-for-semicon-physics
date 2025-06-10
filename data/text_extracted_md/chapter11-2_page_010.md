# 11.4.3 Threshold Adjustment by Ion Implantation

Several factors, such as fixed oxide charge, metal–semiconductor work function difference, oxide thickness, and semiconductor doping, influence the threshold voltage. Although all of these parameters may be fixed in a particular design and fabrication process, the resulting threshold voltage may not be acceptable for all applications. Ion implantation can be used to change and adjust the substrate doping near the oxide–semiconductor surface to provide the desired threshold voltage. In addition, ion implantation is used for more than doping the channel. It is used extensively as a standard part of device fabrication; for example, it is used to form the source and drain regions of the transistor.

To change the doping and thereby change the threshold voltage, a precise, controlled number of either donor or acceptor ions are implanted into the semiconductor near the oxide surface. When an MOS device is biased in either depletion or inversion and when the implanted dopant atoms are within the induced space charge region, then the ionized dopant charge adds to (or subtracts from) the maximum space charge density, which controls the threshold voltage. An implant of acceptor ions into either a p- or n-type substrate will shift the threshold voltage to more positive values, while an implant of donor ions will shift the threshold voltage to more negative values. Ion implantation can be carried out to change a depletion-mode device to enhancement-mode or an enhancement-mode device to depletion-mode, which is an important application of this technology.

As a first approximation, assume that \( D_I \) acceptor atoms per cm\(^2\) are implanted into a p-type substrate directly adjacent to the oxide–semiconductor interface as shown in Figure 11.29a. The shift in threshold voltage due to the implant is

\[
\Delta V_T = + \frac{eD_I}{C_{ox}}
\]

(11.41)

If donor atoms were implanted into the p-type substrate, the space charge density would be reduced; thus, the threshold voltage would shift in the negative voltage direction.

!Figure 11.29

**Figure 11.29** (a) Ion-implanted profile approximated by a delta function. (b) Ion-implanted profile approximated by a step function in which the depth \( x_i \) is less than the space charge width \( x_{dT} \).

| Metal | Oxide | p-type semiconductor |
|-------|-------|-----------------------|
|       |       | \( D_I \) (cm\(^{-2}\)) |
|       |       | \( N_a \)              |
| \( x = 0 \) | \( x = x_{dT} \) |       |

| Metal | Oxide | p-type semiconductor |
|-------|-------|-----------------------|
|       |       | \( N_s \)             |
|       |       | \( N_a \)             |
| \( x = 0 \) | \( x = x_i, x = x_{dT} \) | |
