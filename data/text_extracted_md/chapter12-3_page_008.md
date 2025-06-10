# Chapter 12: The Bipolar Transistor

If the B–C junction becomes forward biased, such as in saturation, then we can write the current \( I_R \) as

\[
I_R = I_{CS} \left[ \exp \left( \frac{eV_{BC}}{kT} \right) - 1 \right]
\]

(12.67)

Using Equations (12.66) and (12.67), the collector current from Equation (12.65a) can be written as

\[
I_C = \alpha_F I_{ES} \left[ \exp \left( \frac{eV_{BE}}{kT} \right) - 1 \right] - I_{CS} \left[ \exp \left( \frac{eV_{BC}}{kT} \right) - 1 \right]
\]

(12.68)

We can also write the emitter current as

\[
I_E = \alpha_R I_F - I_F
\]

(12.69)

or

\[
I_E = \alpha_R I_{CS} \left[ \exp \left( \frac{eV_{BC}}{kT} \right) - 1 \right] - I_{ES} \left[ \exp \left( \frac{eV_{BE}}{kT} \right) - 1 \right]
\]

(12.70)

The current \( I_{ES} \) is the reverse-biased B–E junction current and \( \alpha_R \) is the common-base current gain for the inverse-active mode. Equations (12.68) and (12.70) are the classic Ebers–Moll equations.

Figure 12.37 shows the equivalent circuit corresponding to Equations (12.68) and (12.70). The current sources in the equivalent circuit represent current components that depend on voltages across other junctions. The Ebers–Moll model has four parameters: \( \alpha_F, \alpha_R, I_{ES}, \) and \( I_{CS} \). However, only three parameters are independent. The reciprocity relationship states that

\[
\alpha_F I_{ES} = \alpha_R I_{CS}
\]

(12.71)

Since the Ebers–Moll model is valid in each of the four operating modes, we can, for example, use the model for the transistor in saturation. In the saturation mode, both B–E and B–C junctions are forward biased, so that \( V_{BE} > 0 \) and \( V_{BC} > 0 \). The B–E voltage will be a known parameter since we will apply a voltage across this junction. The forward-biased B–C voltage is a result of driving the transistor into saturation and is the unknown to be determined from the Ebers–Moll equations.

!Figure 12.37

**Figure 12.37** | Basic Ebers–Moll equivalent circuit.