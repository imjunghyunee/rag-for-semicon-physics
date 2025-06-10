# 12.5 Equivalent Circuit Models

A detailed study of all possible models is beyond the scope of this chapter. However, we will consider three equivalent circuit models. Each of these follows directly from the work we have done on the pn junction diode and on the bipolar transistor. Computer analysis of electronic circuits is more commonly used than hand calculations, but it is instructive to consider the types of transistor model used in computer codes.

It is useful to divide bipolar transistors into two categories—switching and amplification—defined by their use in electronic circuits. Switching usually involves turning a transistor from its “off” state, or cutoff, to its “on” state, either forward-active or saturation, and then back to its “off” state. Amplification usually involves superimposing sinusoidal signals on dc values so that bias voltages and currents are only perturbed. The **Ebers–Moll model** is used in switching applications; the **hybrid-pi model** is used in amplification applications.

## 12.5.1 Ebers–Moll Model

The Ebers–Moll model, or equivalent circuit, is one of the classic models of the bipolar transistor. This particular model is based on the interacting diode junctions and is applicable in any of the transistor operating modes. Figure 12.36 shows the current directions and voltage polarities used in the Ebers–Moll model. The currents are defined as all entering the terminals so that

\[
I_E + I_B + I_C = 0
\]

(12.64)

The direction of the emitter current is opposite to what we have considered up to this point, but as long as we are consistent in the analysis, the defined direction does not matter.

The collector current can be written in general as

\[
I_C = \alpha_F I_F - I_{CS}
\]

(12.65a)

where \(\alpha_F\) is the common-base current gain in the forward-active mode. In this mode, Equation (12.65a) becomes

\[
I_C = \alpha_F I_F + I_{CS}
\]

(12.65b)

where the current \(I_{CS}\) is the reverse-biased B–C junction current. The current \(I_F\) is given by

\[
I_F = I_{ES} \left[ \exp \left( \frac{V_{BE}}{kT} \right) - 1 \right]
\]

(12.66)

!Figure 12.36

**Figure 12.36** | Current direction and voltage polarity definitions for the Ebers–Moll model.