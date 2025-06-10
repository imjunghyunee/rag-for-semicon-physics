# Chapter 8: The pn Junction Diode

If we assume that the diode is biased sufficiently far in the forward-bias region, then the \((-1)\) term can be neglected and the incremental conductance becomes

\[
g_d = \frac{dI}{dV} \bigg|_{V=V_o} = \left( \frac{e}{kT} \right) I_o \exp \left( \frac{eV_o}{kT} \right) \approx \frac{I_{DQ}}{V_T}
\]

(8.67)

The small-signal incremental resistance is then the reciprocal function, or

\[
r_d = \frac{V_T}{I_{DQ}}
\]

(8.68)

The incremental resistance decreases as the bias current increases, and is inversely proportional to the slope of the \(I-V\) characteristic as shown in Figure 8.18. The incremental resistance is also known as the **diffusion resistance**.

## 8.3.2 Small-Signal Admittance

In the last chapter, we considered the pn junction capacitance as a function of the reverse-biased voltage. When the pn junction diode is forward-biased, another capacitance becomes a factor in the diode admittance. The small-signal admittance, or impedance, of the pn junction under forward bias is derived using the minority carrier diffusion current relations we have already considered.

**Qualitative Analysis** Before we delve into the mathematical analysis, we can qualitatively understand the physical processes that lead to a diffusion capacitance, which is one component of the junction admittance. Figure 8.19a schematically shows a pn junction forward biased with a dc voltage. A small ac voltage is also applied.

!Figure 8.19

**Figure 8.19** (a) A pn junction with an ac voltage superimposed on a forward-biased dc value; (b) the hole concentration versus time at the space charge edge; (c) the hole concentration versus distance in the n region at three different times.

- **(a)** Schematic of pn junction with ac voltage.
- **(b)** Hole concentration vs. time.
- **(c)** Hole concentration vs. distance.

- \(V_{ac} = \Phi \sin \omega t\)
- \(p_{n0} \exp \left( \frac{V_{ac} + \Phi}{V_T} \right)\)
- \(p_{n0} \exp \left( \frac{V_{ac} - \Phi}{V_T} \right)\)
- \(x = 0\)
- Time: \(t = t_0, t_1, t_2\)