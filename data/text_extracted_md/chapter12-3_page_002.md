# Chapter 12 The Bipolar Transistor

!Energy-band diagram

**Figure 12.32** | Energy-band diagram of an npn bipolar transistor (a) in thermal equilibrium, and (b) with a reverse-biased B–C voltage before punch-through, \(V_{R1}\), and after punch-through, \(V_{R2}\).

!Geometry of a bipolar transistor

**Figure 12.33** | Geometry of a bipolar transistor to calculate the punch-through voltage.

Figure 12.33 shows the geometry for calculating the punch-through voltage. Assume that \(N_B\) and \(N_C\) are the uniform impurity doping concentrations in the base and collector, respectively. Let \(x_{B0}\) be the metallurgical width of the base and let \(x_{dB}\) be the space charge width extending into the base from the B–C junction. If we neglect the narrow space charge width of a zero-biased or forward-biased B–E junction, then punch-through, assuming the abrupt junction approximation, occurs when \(x_{dB} = x_{B0}\). We can write that

\[
x_{dB} = x_{B0} = \left\{ \frac{2 \varepsilon (V_{bi} + V_{pt})}{e} \cdot \frac{N_C}{N_B} \cdot \frac{1}{N_C + N_B} \right\}^{1/2}
\]

(12.53)

where \(V_{pt}\) is the reverse-biased B–C voltage at punch-through. Neglecting \(V_{bi}\) compared to \(V_{pt}\), we can solve for \(V_{pt}\) as

\[
V_{pt} = \frac{e x_{B0}^2}{2 \varepsilon} \cdot \frac{N_B (N_C + N_B)}{N_C}
\]

(12.54)