The effect of short channels on the threshold voltage is discussed further in section 11.3 of this chapter.

### 11.2.3 Generalized Scaling

In constant-field scaling, the applied voltages are scaled with the same scaling factor \( k \) as the device dimensions. However, in actual technology evolution, voltages have not been reduced with the same scaling factor. There has been reluctance, for example, to change standardized power supply levels that have been used in circuits earlier. In addition, other factors that do not scale, such as threshold voltage and subthreshold currents, have made the reduction in applied voltages less desirable. As a consequence, electric fields in MOS devices have tended to increase as device dimensions shrunk.

Consequences of increased electric fields are reduced reliability and increased power density. As the power density increases, the device temperature may increase. Increased temperature may affect the device reliability. As the oxide thickness is reduced and the electric field is increased, gate oxides are closer to breakdown and oxide integrity may be more difficult to maintain. In addition, direct tunneling of carriers through the oxide may be more likely to occur. Increased electric fields may also increase the chances of hot-electron effects, which are discussed later in this chapter. Reducing device dimensions, then, can introduce challenging problems that must be solved.

----

**TEST YOUR UNDERSTANDING**

**TYU 11.3** An NMOS transistor has the following parameters: \( L = 1 \, \mu m, \, W = 10 \, \mu m, \, t_{ox} = 250 \, \text{Å}, \, N_d = 5 \times 10^{15} \, \text{cm}^{-3} \), and applied voltages of 3 V. If the device is to be scaled using constant-field scaling, determine the new device parameters for a scaling factor of \( k = 0.7 \).

----

### 11.3 THRESHOLD VOLTAGE MODIFICATIONS

We have derived the ideal MOSFET relations in the previous chapter, including expressions for threshold voltage and for the current–voltage characteristics. We now consider some of the nonideal effects including channel length modulation. Additional effects on threshold voltage occur as the devices shrink in size. A reduction in channel length increases the transconductance and frequency response of the MOSFET, and a reduction in channel width increases the packing density in an integrated circuit. A reduction in either or both the channel length and channel width can affect the threshold voltage.

#### 11.3.1 Short-Channel Effects

For the ideal MOSFET, we have derived the threshold voltage using the concept of charge neutrality in which the sum of charges in the metal oxide inversion layer and