# 15.1 Tunnel Diode

- Present the basic geometry and electrical characteristics of a power bipolar transistor. The limiting current and voltage factors will be analyzed, and the safe operating area of the BJT will be considered.
- Present the basic geometry and electrical characteristics of a power MOSFET. The limiting current and voltage factors will be analyzed, and the safe operating area of the MOSFET will be considered.
- Discuss the operation of a four-layer switching device that is generally referred to as a Thyristor. The operation of several structures will be analyzed.

## 15.1.1 Tunnel Diode

The tunnel diode, also known as the Esaki diode, has been briefly discussed in Section 8.5 of the book. Recall that the device is a pn junction in which both the n and p regions are degenerately doped. With the very high doping concentrations, the space charge region width is very narrow (\(W \approx 0.5 \times 10^{-6} \text{ cm} = 50 \, \text{Å}\)).

The forward-bias current–voltage characteristics are again shown in Figure 15.1a. For small forward-bias voltages (\(V < V_p\)), electrons in the conduction band on the n side are directly opposite empty states in the valence band of the p region (see Figure 8.29). Electrons tunnel through the potential barrier into the empty states producing a tunneling current. For forward-bias voltages in the range \(V_p < V < V_v\), the number of electrons on the n side directly opposite empty states on the p side decreases so that the tunneling current decreases. For \(V > V_v\), the normal diode diffusion currents dominate.

A decrease in current with an increase in voltage produces a region of negative differential resistance in the range \(V_p < V < V_v\). A negative differential resistance phenomenon is necessary for oscillators.

!Tunnel Diode Characteristics

**Figure 15.1** (a) Forward-bias current–voltage characteristics of a tunnel diode.  
(b) Expanded plot of \(I\)–\(V\) characteristics.

### Figure Description

- **(a)** Shows the tunneling current and diffusion current with respect to voltage.
- **(b)** Displays the expanded plot of \(I\)–\(V\) characteristics, highlighting the peak (\(I_p\)) and valley (\(I_v\)) points, as well as the region of negative differential resistance (\(R_{\text{min}}\)).