# Problems

10. Sketch the cross section of a lightly doped drain transistor. What are the advantages and disadvantages of this design?

11. What type of ion should be implanted into a MOSFET to increase the threshold voltage? What type of ion should be implanted into a MOSFET to decrease the threshold voltage?

# PROBLEMS

(Note: In the following problems, assume the semiconductor and oxide in the MOS system are silicon and silicon dioxide, respectively, and assume the temperature is \( T = 300 \, \text{K} \) unless otherwise stated.)

## Section 11.1 Nonideal Effects

**11.1** Assume that the subthreshold current of a MOSFET is given by

\[
I_D = 10^{-15} \exp \left( \frac{V_{GS}}{[2.1]V_T} \right)
\]

over the range \( 0 \leq V_{GS} \leq 1 \) volt and where the factor 2.1 takes into account the effect of interface states. Assume that \( 10^6 \) identical transistors on a chip are all biased at the same \( V_{GS} \) and at \( V_{DD} = 5 \, \text{V} \). (a) Calculate the total current that must be supplied to the chip at \( V_{GS} = 0.5, 0.7, \) and \( 0.9 \, \text{V} \). (b) Calculate the total power dissipated in the chip for the same \( V_{GS} \) values.

**11.2** The subthreshold current in a MOSFET is given by \( I_D = I_0 \exp (V_{GS}/nV_T) \). Determine the change in applied \( V_{GS} \) for a factor of 10 increase in \( I_D \) for (a) \( n = 1, (b) n = 1.5, \) and (c) \( n = 2.1 \).

**11.3** A silicon n-channel MOSFET has a doping concentration of \( N_s = 2 \times 10^{16} \, \text{cm}^{-3} \) and a threshold voltage of \( V_T = 0.40 \, \text{V} \). Determine the change in channel length, \( \Delta L \), for (a) \( V_{GS} = 1.0 \, \text{V}, V_{DS} = 2.0 \, \text{V}; \) (b) \( V_{GS} = 1.0, V_{DS} = 4.0 \, \text{V}; \) (c) \( V_{GS} = 2.0 \, \text{V}, V_{DS} = 2.0 \, \text{V}; \) and (d) \( V_{GS} = 2.0, V_{DS} = 4.0 \).

**11.4** Consider the MOSFET described in Problem 11.3. (a) Determine the minimum channel length so that the incremental change \( \Delta L \) is no more than 10 percent of the original length \( L \) for applied voltages of \( V_{DS} = 2 \, \text{V} \) and \( V_{DS} = 3 \, \text{V} \). (b) Repeat part (a) for \( V_{DS} = 5 \, \text{V} \).

**11.5** A silicon MOSFET has parameters \( N_s = 4 \times 10^{16} \, \text{cm}^{-3}, t_{ox} = 12 \, \text{nm} = 120 \, \text{Å}, Q_i = 4 \times 10^{10} \, \text{cm}^{-2}, \) and \( \phi_{ms} = -0.5 \, \text{V} \). The transistor is biased at \( V_{GS} = 1.25 \, \text{V} \) and \( V_{DS} = 0 \). (a) Calculate \( \Delta L \) for (i) \( \Delta V_S = 1 \, \text{V}, \) (ii) \( \Delta V_S = 2 \, \text{V}, \) and (iii) \( \Delta V_S = 4 \, \text{V} \). (b) Determine the minimum channel length \( L \) such that \( \Delta L/L = 0.12 \) for \( V_{DS} = 1.25 \, \text{V} \) and \( \Delta V_S = 4 \, \text{V} \).

**11.6** The parameters of a silicon MOSFET are \( N_s = 3 \times 10^{16} \, \text{cm}^{-3}, V_T = 0.40 \, \text{V}, k' = 50 \, \mu \text{A/V}^2, L = 0.80 \, \mu \text{m}, \) and \( W = 15 \, \mu \text{m} \). (a) Determine the current \( I_D \) at \( V_{GS} = 1.0 \, \text{V} \) for (i) \( V_{DS} = 2.0 \, \text{V} \) and (ii) \( V_{DS} = 4.0 \, \text{V} \). (b) Defining the output resistance as \( r_o = (\Delta V_{DS}/\Delta I_D) \), determine \( r_o \) for part (a). (c) Repeat parts (a) and (b) for \( V_{GS} = 2.0 \, \text{V} \).

**11.7** Consider an n-channel silicon MOSFET. The parameters are \( k' = 75 \, \mu \text{A/V}^2, W/L = 10, \) and \( V_T = 0.35 \, \text{V} \). The applied drain-to-source voltage is \( V_{DS} = 1.5 \, \text{V}. \) (a) For \( V_{GS} = 0.8 \, \text{V}, \) find (i) the ideal drain current, (ii) the drain current if \( \lambda = 0.02 \, \text{V}^{-1}, \) and (iii) the output resistance for \( \lambda = 0.02 \, \text{V}^{-1}. \) (b) Repeat part (a) for \( V_{GS} = 1.25 \, \text{V} \).