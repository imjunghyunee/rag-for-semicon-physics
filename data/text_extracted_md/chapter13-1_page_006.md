# The Junction Field-Effect Transistor

## Basic MESFET Operation

The second type of junction field-effect transistor is the MESFET. The gate junction in the pn junction FET is replaced by a Schottky barrier rectifying contact. Although MESFETs can be fabricated in silicon, they are usually associated with gallium arsenide or other compound semiconductor materials. A simplified cross section of a GaAs MESFET is shown in Figure 13.6. A thin epitaxial layer of GaAs is used for the active region; the substrate is a very high resistivity GaAs material referred to as a semi-insulating substrate. GaAs is intentionally doped with chromium, which behaves as a single acceptor close to the center of the energy bandgap, to make it semi-insulating with a resistivity as high as \(10^9 \, \Omega \cdot \text{cm}\). The advantages of these devices include higher electron mobility, hence smaller transit time and faster response; and decreased parasitic capacitance and a simplified fabrication process, resulting from the semi-insulating GaAs substrate.

In the MESFET shown in Figure 13.6, a reverse-biased gate-to-source voltage induces a space charge region under the metal gate that modulates the channel conductance as in the case of the pn JFET. The space charge region will eventually reach the substrate if the applied negative gate voltage is sufficiently large. This condition, again, is known as pinch-off. The device shown in this figure is also a depletion mode device, since a gate voltage must be applied to pinch off the channel.

If we treat the semi-insulating substrate as an intrinsic material, then the energy-band diagram of the substrate–channel–metal structure is as shown in Figure 13.7 for the case of zero bias applied to the gate. Because there is a potential barrier between the channel and substrate and between the channel and metal, the majority carrier electrons are confined to the channel region.

Consider, now, another type of MESFET in which the channel is pinched off even at \(V_{GS} = 0\). Figure 13.8a shows this condition, in which the channel thickness is smaller than the zero-biased space charge width. To open a channel, the depletion region must be reduced: A forward-bias voltage must be applied to the gate–semiconductor.

### Figures

- **Figure 13.5**: Expanded view of the space charge region in the channel for \(V_{DS} > V_{DS(sat)}\).

- **Figure 13.6**: Cross section of an n-channel MESFET with a semi-insulating substrate.

### Equations

- The drain current will be independent of \(V_{DS}\); thus, the device looks like a constant current source.