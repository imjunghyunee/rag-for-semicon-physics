# Chapter 13: The Junction Field-Effect Transistor

The Junction Field-Effect Transistor (JFET) is a separate class of field-effect transistors. The MOSFET has been considered in Chapters 10 and 11. In this chapter, we cover the physics and properties of the JFET. Although we have discussed the MOS and bipolar transistors in previous chapters, the material in this chapter only presumes a knowledge of semiconductor material properties and the characteristics of pn and Schottky barrier junctions.

As with the transistors considered in previous chapters, the JFET, in conjunction with other circuit elements, is capable of voltage gain and signal power gain. Again, the basic transistor action is the control of current at one terminal by the voltage across the other two terminals of the device.

There are two general categories of JFETs. The first is the pn junction FET, or pn JFET, and the second is the MEtal-Semiconductor Field-Effect Transistor, or MESFET. The pn JFET is fabricated with a pn junction and the MESFET is fabricated with a Schottky barrier rectifying junction.

## 13.0 | PREVIEW

In this chapter, we will:

- Present the geometry and discuss the basic operation of the pn JFET and MESFET devices.
- Analyze the modulation of the channel conductance of the JFET by an electric field perpendicular to the channel. The modulating electric field is induced in the space charge region of a reverse-biased pn junction or reverse-biased Schottky barrier junction.
- Derive the ideal current–voltage characteristics of the JFET in terms of the semiconductor material and geometrical properties of the device.
- Consider the transistor gain, or transconductance, of the JFET.
- Discuss a few nonideal effects in JFETs, including channel-length modulation and velocity saturation effects.
- Develop a small-signal equivalent circuit of the JFET that is used to relate small-signal currents and voltages in the device.
- Examine various physical factors affecting the frequency response and limitations of JFETs, and derive an expression for the cutoff frequency.
- Present the geometry and characteristics of a specialized JFET called HEMT.

## 13.1 JFET CONCEPTS

The concept of the field-effect phenomenon was the basis for the first proposed solid-state transistor. Patents filed in the 1920s and 1930s conceived and investigated the transistor shown in Figure 13.1. A voltage applied to the metal plate modulated the conductance of the semiconductor under the metal and controlled the current between the ohmic contacts. Good semiconductor materials and processing technology were not available at that time, so the device was not seriously considered again until the 1950s.

The phenomenon of modulating the conductance of a semiconductor by an electric field applied perpendicular to the surface of a semiconductor is called field effect. This type of transistor has also been called the unipolar transistor, to emphasize that only one type of carrier, the majority carrier, is involved in the operation. We will qualitatively discuss the basic operation of the two types of JFETs in this section, and introduce some of the JFET terminology.

### 13.1.1 Basic pn JFET Operation

The first type of field-effect transistor is the pn junction field-effect transistor, or pn JFET. A simplified cross section of a symmetrical device is shown in Figure 13.2. The n region between the two p regions is known as the channel and, in this n-channel

*Figure 13.1 | Idealization of the Lilienfeld transistor. (From Pierret [10].)*

*Figure 13.2 | Cross section of a symmetrical n-channel pn junction FET.*

Device, majority carrier electrons flow between the source and drain terminals. The source is the terminal from which carriers enter the channel from the external circuit, the drain is the terminal where carriers leave, or are drained from, the device, and the gate is the control terminal. The two gate terminals shown in Figure 13.2 are tied together to form a single gate connection. Since majority carrier electrons are primarily involved in the conduction in this n-channel transistor, the JFET is a majority-carrier device.

A complementary p-channel JFET can also be fabricated in which the p and n regions are reversed from those of the n-channel device. Holes will flow in the p-type channel between source and drain and the source terminal will now be the source of the holes. The current direction and voltage polarities in the p-channel JFET are the reverse of those in the n-channel device. The p-channel JFET is generally a lower frequency device than the n-channel JFET due to the lower hole mobility.

Figure 13.3a shows an n-channel pn JFET with zero volts applied to the gate. If the source is at ground potential, and if a small positive drain voltage is applied, a drain current \( I_D \) is produced between the source and drain terminals. The n channel is essentially a resistance so the \( I_D \) versus \( V_{DS} \) characteristic, for small \( V_{DS} \) values, is approximately linear, as shown in the figure.

When we apply a voltage to the gate of a pn JFET with respect to the source and drain, we alter the channel conductance. If a negative voltage is applied to the gate of the n-channel pn JFET shown in Figure 13.3, the gate-to-channel pn junction becomes reverse biased. The space charge region now widens so the channel region becomes narrower and the resistance of the n channel increases. The slope of the \( I_D \) versus \( V_{DS} \) curve, for small \( V_{DS} \), decreases. These effects are shown in Figure 13.3b. If a larger negative gate voltage is applied, the condition shown in Figure 13.3c can be achieved. The reverse-biased gate-to-channel space charge region has completely filled the channel region. This condition is known as **pinch-off**. The drain current at pinch-off is essentially zero, since the depletion region isolates the source and drain terminals. Figure 13.3c shows the \( I_D \) versus \( V_{DS} \) curve for this case, as well as the other two cases.

The current in the channel is controlled by the gate voltage. The control of the current in one part of the device by a voltage in another part of the device is the basic transistor action. This device is a normally on or **depletion mode** device, which means that a voltage must be applied to the gate terminal to turn the device off.

Now consider the situation in which the gate voltage is held at zero volts, \( V_{GS} = 0 \), and the drain voltage changes. Figure 13.4a is a replica of Figure 13.3a for zero gate voltage and a small drain voltage. As the drain voltage increases (positive), the gate-to-channel pn junction becomes reverse biased near the drain terminal so that the space charge region extends further into the channel. The channel is essentially a resistor, and the effective channel resistance increases as the space charge region widens; therefore, the slope of the \( I_D \) versus \( V_{DS} \) characteristic decreases as shown in Figure 13.4b. The effective channel resistance now varies along the channel length and, since the channel current must be constant, the voltage drop through the channel becomes dependent on position.

- **(a)** \( V_{GS} = 0 \)

  !Diagram (a)

  !Graph (a)

- **(b)** \( V_{GS} = -V_1 \)

  !Diagram (b)

  !Graph (b)

- **(c)** \( V_{GS} = -V_2 \)

**Figure 13.3.1** Gate-to-channel space charge regions and \( I-V \) characteristics for small \( V_{DS} \) values and for (a) zero gate voltage, (b) small reverse-biased gate voltage, and (c) a gate voltage to achieve pinchoff.

If the drain voltage increases further, the condition shown in Figure 13.4c can result. The channel has been pinched off at the drain terminal. Any further increase in drain voltage will not cause an increase in drain current. The \( I-V \) characteristic for this condition is also shown in this figure. The drain voltage at pinchoff is referred to as \( V_{DS}(sat) \). For \( V_{DS} > V_{DS}(sat) \), the transistor is said to be in the saturation region and the drain current, for this ideal case, is independent of \( V_{DS} \). At first glance, we might expect the drain current to go to zero when the channel becomes pinched off at the drain terminal, but we will show why this does not happen.

**Figure 13.4** Gate-to-channel space charge regions and \( I \)–\( V \) characteristics for zero gate voltage and for (a) a small drain voltage, (b) a larger drain voltage, and (c) a drain voltage to achieve pinchoff at the drain terminal.

Figure 13.5 shows an expanded view of the pinchoff region in the channel. The n channel and drain terminal are now separated by a space charge region which has a length \(\Delta L\). The electrons move through the n channel from the source and are injected into the space charge region where, subjected to the E-field force, they are swept through into the drain contact area. If we assume that \(\Delta L \ll L\), then the electric field in the n-channel region remains unchanged from the \(V_{DS}(sat)\) case; the drain current will remain constant as \(V_{DS}\) changes. Once the carriers are in the drain region, the current remains constant.

#### Basic MESFET Operation

The second type of junction field-effect transistor is the MESFET. The gate junction in the pn junction FET is replaced by a Schottky barrier rectifying contact. Although MESFETs can be fabricated in silicon, they are usually associated with gallium arsenide or other compound semiconductor materials. A simplified cross section of a GaAs MESFET is shown in Figure 13.6. A thin epitaxial layer of GaAs is used for the active region; the substrate is a very high resistivity GaAs material referred to as a semi-insulating substrate. GaAs is intentionally doped with chromium, which behaves as a single acceptor close to the center of the energy bandgap, to make it semi-insulating with a resistivity as high as \(10^9 \, \Omega \cdot \text{cm}\). The advantages of these devices include higher electron mobility, hence smaller transit time and faster response; and decreased parasitic capacitance and a simplified fabrication process, resulting from the semi-insulating GaAs substrate.

In the MESFET shown in Figure 13.6, a reverse-biased gate-to-source voltage induces a space charge region under the metal gate that modulates the channel conductance as in the case of the pn JFET. The space charge region will eventually reach the substrate if the applied negative gate voltage is sufficiently large. This condition, again, is known as pinch-off. The device shown in this figure is also a depletion mode device, since a gate voltage must be applied to pinch off the channel.

If we treat the semi-insulating substrate as an intrinsic material, then the energy-band diagram of the substrate–channel–metal structure is as shown in Figure 13.7 for the case of zero bias applied to the gate. Because there is a potential barrier between the channel and substrate and between the channel and metal, the majority carrier electrons are confined to the channel region.

Consider, now, another type of MESFET in which the channel is pinched off even at \(V_{GS} = 0\). Figure 13.8a shows this condition, in which the channel thickness is smaller than the zero-biased space charge width. To open a channel, the depletion region must be reduced: A forward-bias voltage must be applied to the gate–semiconductor.

- **Figure 13.5**: Expanded view of the space charge region in the channel for \(V_{DS} > V_{DS(sat)}\).

- **Figure 13.6**: Cross section of an n-channel MESFET with a semi-insulating substrate.

- The drain current will be independent of \(V_{DS}\); thus, the device looks like a constant current source.

**Figure 13.7** | Idealized energy-band diagram of the substrate–channel–metal in the n-channel MESFET.

**Figure 13.8** | Channel space charge region of an enhancement mode MESFET for (a) \( V_{GS} = 0 \), (b) \( V_{GS} = V_T \), and (c) \( V_{GS} > V_T \).

- **(a)** \( V_{GS} = 0 \)
- **(b)** \( V_{GS} = V_T \)
- **(c)** \( V_{GS} > V_T \)


When a slightly forward-bias voltage is applied, the depletion region just extends through the channel—a condition known as **threshold**, shown in Figure 13.8b. The threshold voltage is the gate-to-source voltage that must be applied to create the pinch-off condition. The threshold voltage for this n-channel MESFET is positive, in contrast to the negative voltage for the n-channel depletion mode device. If a larger forward bias is applied, the channel region opens as shown in Figure 13.8c.

applied forward-bias gate voltage is limited to a few tenths of a volt before there is significant gate current. This device is known as an n-channel enhancement mode MESFET. Enhancement mode p-channel MESFETs and enhancement mode pn junction FETs have also been fabricated. The advantage of enhancement mode MESFETs is that circuits can be designed in which the voltage polarity on the gate and drain is the same. However, the output voltage swing will be quite small with these devices.

## 13.2 THE DEVICE CHARACTERISTICS

To describe the basic electrical characteristics of the JFET, we initially consider a uniformly doped depletion mode pn JFET and then later discuss the enhancement mode device. The pinchoff voltage and drain-to-source saturation voltage are defined and expressions for these parameters derived in terms of geometry and electrical properties. The ideal current–voltage relationship is developed, and then the transconductance, or transistor gain is determined.

Figure 13.9a shows a symmetrical, two-sided pn JFET and Figure 13.9b shows a MESFET with the semi-insulating substrate. One can derive the ideal DC current–voltage relationship for both devices by simply considering the two-sided device to be two JFETs in parallel. We derive the I–V characteristics in terms of \(I_{D1}\) so that the drain current in the two-sided device becomes \(I_{D2} = 2I_{D1}\). We ignore any depletion region at the substrate of the one-sided device in the ideal case.

### 13.2.1 Internal Pinchoff Voltage, Pinchoff Voltage, and Drain-to-Source Saturation Voltage

**n-channel pn JFET** Figure 13.10a shows a simplified one-sided n-channel pn JFET. The metallurgical channel thickness between the p\(^+\) gate region and the substrate is \(a\), and the induced depletion region width for the one-sided p\(^+\)n junction is \(h\). Assume the drain-to-source voltage is zero. If we assume the abrupt depletion approximation, then the space charge width is given by

\[
h = \left[\frac{2\varepsilon_s(V_{bi} - V_{GS})}{eN_d}\right]^{1/2}
\]

(13.1)

where \(V_{GS}\) is the gate-to-source voltage and \(V_{bi}\) is the built-in potential barrier. For a reverse-biased p\(^+\)n junction, \(V_{GS}\) must be a negative voltage.


**Figure 13.9** | Drain currents of (a) a symmetrical, two-sided pn JFET, and (b) a one-sided MESFET.

**Figure 13.10** | Geometries of simplified (a) n-channel and (b) p-channel pn JFETs.

At pinchoff, \( h = a \) and the total potential across the p*n junction is called the **internal pinchoff voltage**, denoted by \( V_{po} \). We now have

\[
a = \left[ \frac{2 \varepsilon_s V_{po}}{e N_d} \right]^{1/2}
\]

(13.2)

or

\[
V_{po} = \frac{ea^2 N_d}{2 \varepsilon_s}
\]

(13.3)

Note that the internal pinchoff voltage is defined as a positive quantity.

The internal pinchoff voltage \( V_{po} \) is not the gate-to-source voltage to achieve pinchoff. The gate-to-source voltage that must be applied to achieve pinchoff is described as the **pinchoff voltage** and is also variously called the **turn-off voltage** or **threshold voltage**. The pinchoff voltage is denoted by \( V_p \) and is defined from Equations (13.1) and (13.2) as

\[
V_p - V_s = V_{po} \quad \text{or} \quad V_p = V_{th} - V_{po}
\]

(13.4)

The gate-to-source voltage to achieve pinchoff in an n-channel depletion mode JFET is negative; thus, \( V_{po} > V_{gs} \).

#### p-channel pn JFET

Figure 13.10b shows a p-channel JFET with the same basic geometry as the n-channel JFET we considered. The induced depletion region for the one-sided n\(^+\)p junction is again denoted by \( h \) and is given by

\[
h = \left[ \frac{2 \varepsilon_s (V_{bi} + V_{GS})}{e N_a} \right]^{1/2} \tag{13.5}
\]

For a reverse-biased n\(^+\)p junction, \( V_{GS} \) must be positive. The internal pinch-off voltage is again defined to be the total pn junction voltage to achieve pinch-off, so that when \( h = a \) we have

\[
a = \left[ \frac{2 \varepsilon_s V_{po}}{e N_a} \right]^{1/2} \tag{13.6}
\]

or

\[
V_{po} = \frac{ea^2 N_a}{2 \varepsilon_s} \tag{13.7}
\]

The internal pinch-off voltage for the p-channel device is also defined to be a positive quantity.

The pinch-off voltage is again defined as the gate-to-source voltage to achieve the pinch-off condition. For the p-channel depletion mode device, we have, from Equation (13.5), at pinch-off

\[
V_{dA} + V_p = V_{po} \quad \text{or} \quad V_p = V_{po} - V_{dA} \tag{13.8}
\]

The pinch-off voltage for a p-channel depletion mode JFET is a positive quantity.

Also, we will see later that if the channel doping concentration were smaller the current capability of the device would decrease. There are definite tradeoffs to be considered in any design problem.

We have determined the pinch-off voltage for both n-channel and p-channel JFETs when the drain-to-source voltage is zero. Now consider the case when both gate and drain voltages are applied. The depletion region width will vary with distance through the channel. Figure 13.11 shows the simplified geometry for an n-channel device. The depletion width \( h_t \) at the source end is a function of \( V_{bi} \) and \( V_{GS} \) but is not a function of drain voltage. The depletion width at the drain terminal is given by

\[
h_z = \left[ \frac{2 \varepsilon_s (V_{bi} + V_{DS} - V_{GS})}{e N_c} \right]^{1/2}
\]

(13.9)

Again, we must keep in mind that \( V_{GS} \) is a negative quantity for the n-channel device.
Pinchoff at the drain terminal occurs when \( h_2 = a \). At this point we reach what is known as the saturation condition; thus, we can write that \( V_{DS} = V_{DS}(sat) \). Then

\[
a = \left[ \frac{2 \varepsilon_s (V_{bi} + V_{DS}(sat) - V_{GS})}{eN_d} \right]^{1/2}
\]

(13.10)

This can be rewritten as

\[
V_{bi} + V_{DS}(sat) - V_{GS} = \frac{ea^2 N_d}{2 \varepsilon_s} = V_{p0}
\]

(13.11)

or

\[
V_{DS}(sat) = V_{p0} - (V_{bi} - V_{GS})
\]

(13.12)

Equation (13.12) gives the drain-to-source voltage to cause pinchoff at the drain terminal. The drain-to-source saturation voltage decreases with increasing reverse-biased gate-to-source voltage. We may note that Equation (13.12) has no meaning if \( |V_{GS}| > |V_t| \).

In a p-channel JFET, the voltage polarities are the reverse of those in the n-channel device. We can show that, in the p-channel JFET at saturation,

\[
V_{SD}(sat) = V_{p0} - (V_{bi} + V_{GS})
\]

(13.13)

where now the source is positive with respect to the drain.

### 13.2.2 Ideal DC Current–Voltage Relationship—Depletion Mode JFET

The derivation of the ideal current–voltage relation of the JFET is somewhat tedious, and the resulting equations are cumbersome in hand calculations. Before we go through this derivation, consider the following expression, which is a good approximation.

to the I–V characteristics when the JFET is biased in the saturation region. This equation is used extensively in JFET applications and is given by

\[
I_D = I_{\text{DSS}} \left( 1 - \frac{V_{GS}}{V_p} \right)^2
\]

(13.14)

where \( I_{\text{DSS}} \) is the saturation current when \( V_{GS} = 0 \). At the end of this section, we compare the approximation given by Equation (13.14) and the ideal current–voltage equation that we have derived.

#### I–V Derivation

The ideal current–voltage relationship of the JFET is derived by starting with Ohm’s law. Consider an n-channel JFET with the geometry shown in Figure 13.11. We are considering half of the two-sided symmetrical geometry. The differential resistance of the channel at a point \( x \) in the channel is

\[
dR = \frac{\rho dx}{A(x)}
\]

(13.15)

where \( \rho \) is the resistivity and \( A(x) \) is the cross-sectional area. If we neglect the minority carrier holes in the n channel, the channel resistivity is

\[
\rho = \frac{1}{e \mu_n N_d}
\]

(13.16)

The cross-sectional area is given by

\[
A(x) = [a - h(x)]W
\]

(13.17)

where \( W \) is the channel width. Equation (13.15) can now be written as

\[
dR = \frac{dx}{e \mu_n N_d [a - h(x)] W}
\]

(13.18)

The differential voltage across a differential length \( dx \) can be written as

\[
dV(x) = I_D dR(x)
\]

(13.19)

where the drain current \( I_D \) is constant through the channel. Substituting Equation (13.18) into Equation (13.19), we have

\[
dV(x) = \frac{I_D \, dx}{e \mu_n N_d W [a - h(x)]}
\]

(13.20a)

or

\[
I_D \, dx = e \mu_n N_d W [a - h(x)] \, dV(x)
\]

(13.20b)

The depletion width \( h(x) \) is given by

\[
h(x) = \left[ \frac{2 \varepsilon_s [V(x) + V_b - V_{GS}]}{e N_d} \right]^{1/2}
\]

(13.21)

where \( V(x) \) is the potential in the channel due to the drain-to-source voltage. Solving for \( V(x) \) in Equation (13.21) and taking the differential, we have

\[
dV(x) = \frac{e N_d h(x) \, dh(x)}{\varepsilon_s}
\]

(13.22)

Then Equation (13.20b) becomes

\[
I_{D1} \, dx = \frac{\mu_n (eN_d)^2 W}{\epsilon_s} \left[ ah(x) \, dh(x) - h(x)^2 \, dh(x) \right]
\]

(13.23)

The drain current \( I_{D1} \) is found by integrating Equation (13.23) along the channel length. Assuming the current and mobility are constant through the channel, we obtain

\[
I_{D1} = \frac{\mu_n (eN_d)^2 W}{\epsilon_s L} \left[ \int_{h_m}^{h_2} ah \, dh - \int_{h_m}^{h_2} h^2 \, dh \right]
\]

(13.24)

or

\[
I_{D1} = \frac{\mu_n (eN_d)^2 W}{\epsilon_s L} \left[ \frac{a}{2} (h_2^2 - h_m^2) - \frac{1}{3} (h_2^3 - h_m^3) \right]
\]

(13.25)

Noting that

\[
h_2^2 = \frac{2 \epsilon_s (V_{DS} + V_{bi} - V_{GS})}{eN_d}
\]

(13.26a)

\[
h_m^2 = \frac{2 \epsilon_s (V_{bi} - V_{GS})}{eN_d}
\]

(13.26b)

and

\[
V_{po} = \frac{ea^2 N_d}{2 \epsilon_s}
\]

(13.26c)

Equation (13.25) can be written as

\[
I_{D1} = \frac{\mu_n (eN_d)^2 Wa^3}{2 \epsilon_s L} \left[ \frac{V_{DS}}{V_{po}} - \frac{2}{3} \left( \frac{V_{DS} + V_{bi} - V_{GS}}{V_{po}} \right)^{3/2} + \frac{2}{3} \left( \frac{V_{bi} - V_{GS}}{V_{po}} \right)^{3/2} \right]
\]

(13.27)

We may define

\[
I_{p1} = \frac{\mu_n (eN_d)^2 Wa^3}{6 \epsilon_s L}
\]

(13.28)

where \( I_{p1} \) is called the pinch-off current. Equation (13.27) becomes

\[
I_{D1} = I_{p1} \left[ 3 \left( \frac{V_{DS}}{V_{po}} \right) - 2 \left( \frac{V_{DS} + V_{bi} - V_{GS}}{V_{po}} \right)^{3/2} + 2 \left( \frac{V_{bi} - V_{GS}}{V_{po}} \right)^{3/2} \right]
\]

(13.29)

Equation (13.29) is valid for \( 0 \leq |V_{GS}| \leq |V_t| \) and \( 0 \leq V_{DS} \leq V_{DS(sat)} \). The pinch-off current \( I_{p1} \) would be the maximum drain current in the JFET if the zero-biased depletion regions could be ignored or if \( V_{GS} \) and \( V_t \) were both zero.

Equation (13.29) is the current–voltage relationship for the one-sided n-channel JFET in the nonsaturation region. For the two-sided symmetrical JFET shown in Figure 13.9a, the total drain current would be \( I_{D2} = 2I_{D1} \).

Equation (13.27) can also be written as

\[
I_{D} = G_{o1} \left\{ V_{DS} - \frac{2}{3} \sqrt{\frac{1}{V_{po}}} \left[ (V_{DS} + V_{bi} - V_{GS})^2 - (V_{bi} - V_{GS})^2 \right] \right\}
\]

(13.30)

where

\[
G_{o1} = \frac{\mu_n (eN_d)^{3/2} W_a^3}{2e_s L V_{po}} = \frac{e \mu_n N_d W_a}{L} = \frac{3I_{p1}}{V_{po}}
\]

(13.31)

The channel conductance is defined as

\[
g_d = \frac{\partial I_D}{\partial V_{DS}} \bigg|_{V_{DS}=0}
\]

(13.32)

Taking the derivative of Equation (13.30) with respect to \(V_{DS}\), we obtain

\[
g_d = \frac{\partial I_D}{\partial V_{DS}} \bigg|_{V_{DS}=0} = G_{o1} \left[ 1 - \left( \frac{V_{bi} - V_{GS}}{V_{po}} \right)^2 \right]
\]

(13.33)

We may note from Equation (13.33) that \(G_{o1}\) would be the conductance of the channel if both \(V_{bi}\) and \(V_{GS}\) were zero. This condition would exist if no space charge regions existed in the channel. We may also note, from Equation (13.33), that the channel conductance is modulated or controlled by the gate voltage. This channel conductance modulation is the basis of the field-effect phenomenon.

We have shown that the drain becomes pinched off, for the n-channel JFET, when

\[
V_{DS} = V_{DS(sat)} = V_{po} - (V_{bi} - V_{GS})
\]

(13.34)

In the saturation region, the saturation drain current is determined by setting \(V_{DS} = V_{DS(sat)}\) in Equation (13.29) so that

\[
I_{D} = I_{D(sat)} = I_{p1} \left[ 1 - \frac{3}{2} \left( \frac{V_{bi} - V_{GS}}{V_{po}} \right) \right] \left[ 1 - \frac{2}{3} \sqrt{\frac{V_{bi} - V_{GS}}{V_{po}}} \right]
\]

(13.35)

The ideal saturation drain current is independent of the drain-to-source voltage. Figure 13.12 shows the ideal current–voltage characteristics of a silicon n-channel JFET.

**Figure 13.12** Ideal current-voltage characteristics of a silicon n-channel JFET with \(a = 1.5 \, \mu\text{m}\), \(W/L = 170\), and \(N_g = 2.5 \times 10^{15} \, \text{cm}^{-2}\).  
*(From Yang [22].)*

| \(V_{DS}\) (V) | \(I_D\) (mA) |
|----------------|-------------|
| 0              | 0           |
| 2              | 8           |
| 4              | 16          |
| 6              | 24          |
| 8              | 28          |

- **Nonsaturation region**: \(V_{DS} < (V_{p0} - V_{gs}) + V_{GS}\)
- **Saturation region**: \(V_{DS} \geq (V_{p0} - V_{gs}) + V_{GS}\)

- \(V_{GS} = 0, -1, -2, -3, -4 \, \text{V}\)

**Figure 13.13.1** Comparison of Equations (13.14) and (13.35) for the \( I_D \) versus \( V_{GS} \) characteristics of a JFET biased in the saturation region.

We may note that, for n-channel depletion mode JFET, both \( V_{GS} \) and \( V_P \) are negative and, for the p-channel depletion mode device, both are positive. Figure 13.13 shows the comparison between Equations (13.14) and (13.35).

### 13.2.3 Transconductance

The transconductance is the transistor gain of the JFET; it indicates the amount of control the gate voltage has on the drain current. The transconductance is defined as

\[
g_m = \frac{\partial I_D}{\partial V_{GS}}
\]

(13.37)

Using the expressions for the ideal drain current derived in the last section, we can write the expressions for the transconductance.

The drain current for an n-channel depletion mode device in the nonsaturation region is given by Equation (13.29). We can then determine the transconductance of the transistor in the same region as

\[
g_{m} = \frac{\partial I_D}{\partial V_{GS}} = \frac{3I_{D1}}{V_{p0}} \frac{V_{th} - V_{GS}}{V_{p0}} \left[ \sqrt{\frac{V_{DS}}{V_{th} - V_{GS}}} + 1 - 1 \right]
\]

(13.38)

Taking the limit as \( V_{DS} \) becomes small, the transconductance becomes

\[
g_{m} \approx \frac{3I_{D1}}{2V_{p0}} \cdot \frac{V_{DS}}{\sqrt{V_{th}(V_{th} - V_{GS})}}
\]

(13.39)

We can also write Equation (13.39) in terms of the conductance parameter \( G_{0} \) as

\[
g_{m} = \frac{G_{0}}{2} \cdot \frac{V_{DS}}{\sqrt{V_{th}(V_{th} - V_{GS})}}
\]

(13.40)

The ideal drain current in the saturation region for the JFET is given by Equation (13.35). The transconductance in the saturation region is then found to be

\[
g_m = \frac{\partial I_D(\text{sat})}{\partial V_{GS}} = \frac{3I_{DSS}}{V_p} \left( 1 - \sqrt{\frac{V_{GS} - V_{GS}}{V_p}} \right) = G_{0} \left( 1 - \sqrt{\frac{V_{GS} - V_{GS}}{V_p}} \right)
\]

(13.41a)

Using the current–voltage approximation given by Equation (13.14), we can also write the transconductance as

\[
g_m = -\frac{2I_{DSS}}{V_p} \left( 1 - \frac{V_{GS}}{V_p} \right)
\]

(13.41b)

Since \( V_p \) is negative for the n-channel JFET, \( g_m \) is positive.

The experimental transconductance may deviate from this ideal expression due to a source series resistance. This effect will be considered later in the discussion of the small signal model of the JFET.

### 13.2.4 The MESFET

So far in our discussion, we have explicitly considered the pn JFET. The MESFET is the same basic device except that the pn junction is replaced by a Schottky barrier rectifying junction. The simplified MESFET geometry is shown in Figure 13.9b. MESFETs are usually fabricated in gallium arsenide. We will neglect any depletion region that may exist between the n channel and the substrate. We have also limited our discussion to depletion mode devices, wherein a gate-to-source voltage is applied to turn the transistor off. Enhancement mode GaAs MESFETs can be fabricated—their basic operation is discussed in Section 13.1.2. We can also consider enhancement mode GaAs pn JFETs.
Since the electron mobility in GaAs is much larger than the hole mobility, we will concentrate our discussion on n-channel GaAs MESFETs or JFETs. The definition of internal pinch-off voltage, given by Equation (13.3), also applies to these devices. In considering the enhancement mode JFET, the term threshold voltage is commonly used in place of pinch-off voltage. For this reason, we shall use the term threshold voltage in our discussion of MESFETs.

For the n-channel MESFET, the threshold voltage is defined from Equation (13.4) as

\[
V_{bi} - V_T = V_{\phi 0} \quad \text{or} \quad V_T = V_{bi} - V_{\phi 0}
\]

(13.42)

For an n-channel depletion mode JFET, \( V_T < 0 \), and for the enhancement mode device, \( V_T > 0 \). We can see from Equation (13.42) that \( V_{bi} > V_{\phi 0} \) for an enhancement mode n-channel JFET.

The design of enhancement mode JFETs implies the use of narrow channel thicknesses and low channel doping concentrations to achieve this condition. The precise control of the channel thickness and doping concentration necessary to achieve internal pinchoff voltages of a few tenths of a volt makes the fabrication of enhancement mode MESFETs difficult.

Ideally, the I-V characteristics of the enhancement mode device are the same as the depletion mode device—the only real difference is the relative values of the internal pinchoff voltage. The current in the saturation region is given by Equation (13.35) as

\[
I_{D1} = I_{D1}(\text{sat}) = I_{p1} \left\{ 1 - 3 \left[ 1 - \frac{(V_{GS} - V_T)}{V_{p0}} \right] + 2 \left[ 1 - \frac{(V_{GS} - V_T)}{V_{p0}} \right]^{\frac{3}{2}} \right\}
\]

The threshold voltage for the n-channel device is defined in Equation (13.42) as \(V_T = V_{bi} - V_{p0}\), so we can also write

\[
V_{bi} = V_T + V_{p0}
\]

(13.43)

Substituting this expression for \(V_{bi}\) into Equation (13.35), we obtain

\[
I_{D1}(\text{sat}) = I_{p1} \left\{ 1 - 3 \left[ 1 - \frac{(V_{GS} - V_T)}{V_{p0}} \right] + 2 \left[ 1 - \frac{(V_{GS} - V_T)}{V_{p0}} \right]^{\frac{3}{2}} \right\}
\]

(13.44)

Equation (13.44) is valid for \(V_{GS} \geq V_T\).

When the transistor first turns on, we have \((V_{GS} - V_T) \ll V_{p0}\). Equation (13.44) can then be expanded into a Taylor series and we obtain

\[
I_{D1}(\text{sat}) \approx I_{p1} \left[ \frac{3}{4} \left( \frac{V_{GS} - V_T}{V_{p0}} \right)^2 \right]
\]

(13.45)

Substituting the expressions for \(I_{p1}\) and \(V_{p0}\), Equation (13.45) becomes

\[
I_{D1}(\text{sat}) = \frac{\mu_n \epsilon_s W}{2aL} (V_{GS} - V_T)^2 \quad \text{for} \quad V_{GS} \geq V_T
\]

(13.46)

We can now write Equation (13.46) as

\[
I_{D1}(\text{sat}) = k_n (V_{GS} - V_T)^2
\]

(13.47)

where

\[
k_n = \frac{\mu_n \epsilon_s W}{2aL}
\]

(13.48)

The factor \(k_n\) is called a conduction parameter. The form of Equation (13.47) is the same as for a MOSFET.

**Figure 13.14** | Experimental and theoretical \(\sqrt{I_D}\) versus \(V_{GS}\) characteristics of an enhancement mode JFET.

The square root of Equation (13.47), or \(\sqrt{I_{D(sat)}}\) versus \(V_{GS}\), is plotted as the ideal dotted curve shown in Figure 13.14. The ideal curve intersects the voltage axis at the threshold voltage, \(V_r\). The solid line shows an experimental plot. Equation (13.46) does not describe the experimental results well near the threshold voltage. The ideal current–voltage relationship is derived assuming an abrupt depletion approximation for the pn junction. However, when the depletion region extends almost through the channel, a more accurate model of the space charge region must be used to more accurately predict the drain current characteristics near threshold. We consider the subthreshold conduction in Section 13.3.3.

## 13.3| NONIDEAL EFFECTS

As with any semiconductor device, there are nonideal effects that will change the ideal device characteristics. In all of the previous discussions, we have considered an ideal transistor with a constant channel length and constant mobility; we have also...

### 13.3.1 Channel Length Modulation

The expression for the drain current is inversely proportional to the channel length \( L \) as given, for example, by Equation (13.27). In deriving the current equations, we have implicitly assumed that the channel length was constant. However, the effective channel length can change. Figure 13.5 shows the space charge region in the channel when the transistor is biased in the saturation region. The neutral n-channel length decreases as \( V_{DS} \) increases; thus, the drain current will increase. The change in the effective channel length and the corresponding change in drain current is called channel length modulation.

The pinchoff current, Equation (13.28), is modified by the channel length modulation and can be written as

\[
I_{p1} = \frac{\mu_n (eN_d)^2 W a^3}{6 \epsilon_s L'}
\]

(13.50)

where

\[
L' \approx L - \frac{1}{2} \Delta L
\]

(13.51)

If we assume the channel depletion region shown in Figure 13.5 extends equally into the channel and drain regions, then as a first approximation, we will include the factor \(\frac{1}{2}\) in the expression for \( L' \).

The drain current can be written as

\[
I_{D1} = I_{D0} \frac{I_{p1}}{I_{p1}} = I_{D0} \left( \frac{L}{L - \frac{1}{2} \Delta L} \right)
\]

(13.52)

where \( I_{D0} \) is the ideal drain current predicted by Equation (13.35). Another form of the current–voltage characteristic in the saturation region is given by

\[
I_{D1} (sat) = I_{D0} (sat)(1 + \lambda V_{DS})
\]

(13.53)

The effective channel length \( L' \) supports the \( V_{DS}(sat) \) voltage, and the space charge region length \( \Delta L \) in the channel supports the drain voltage beyond the saturation value. Neglecting charges in the space charge region due to current flow, the depletion length \( \Delta L \) is then, to a first approximation, given by

\[
\Delta L = \left[ \frac{2 \epsilon_s (V_{DS} - V_{DS}(sat))}{eN_d} \right]^{1/2}
\]

(13.54)

## 13.3 Nonideal Effects

Since the effective channel length changes with \( V_{DS} \), the drain current is now a function of \( V_{DS} \). The small-signal output impedance at the drain terminal can be defined as

\[
r_{ds} = \frac{\partial V_{DS}}{\partial I_D} = \frac{\Delta V_{DS}}{\Delta I_D}
\]

(13.55)

**Figure 13.15** | Cross section of JFET showing carrier velocity and space charge width saturation effects.

For high-frequency MESFETs, typical channel lengths are on the order of 1 μm. Channel length modulation and other effects become very important in short-channel devices.

### 13.3.2 Velocity Saturation Effects

We have seen that the drift velocity of a carrier in silicon saturates with increasing electric field. This velocity saturation effect implies that the mobility is not a constant. For very short channels, the carriers can easily reach their saturation velocity, which changes the I-V characteristics of the JFET.

Figure 13.15 shows the channel region with an applied drain voltage. As the channel narrows at the drain terminal, the velocity of the carriers increases since the current through the channel is constant. The carriers first saturate at the drain end of the channel. The depletion region will reach a saturation thickness, so we can write

\[
I_{D}(sat) = eN_{d}v_{sat}(a - h_{sat})W
\]

(13.56)

where \( v_{sat} \) is the saturation velocity and \( h_{sat} \) is the saturation depletion width. This saturation effect occurs at a drain voltage smaller than the \( V_{DS}(sat) \) value determined previously. Both \( I_{DS}(sat) \) and \( V_{DS}(sat) \) will be smaller than previously calculated.

Figure 13.16 shows normalized plots of \( I_D \) versus \( V_{DS} \). Figure 13.16a is for the case of a constant mobility and Figure 13.16b is for the case of velocity saturation. Since the I-V characteristics change when velocity saturation occurs, the transconductance will also change—the transconductance will become smaller; hence, the effective gain of the transistor decreases when velocity saturation occurs.

### 13.3.3 Subthreshold and Gate Current Effects

The subthreshold current is the drain current in the JFET that exists when the gate voltage is below the pinchoff or threshold value. The subthreshold conduction is shown in Figure 13.14. When the JFET is biased in the saturation region, the drain current varies quadratically with gate-to-source voltage. When \( V_{GS} \) is below the threshold value, the drain current varies exponentially with gate-to-source voltage. Near threshold, the abrupt depletion approximation does not accurately model the channel region: A more detailed potential profile in the space charge region must be used. However, these calculations are beyond the scope of this chapter.

When the gate voltage is approximately 0.5 to 1.0 V below threshold in an n-channel MESFET, the drain current reaches a minimum value and then slowly increases as the gate voltage decreases. The drain current in this region is the gate leakage current. Figure 13.17 is a plot of the drain current versus \( V_{GS} \) for the three regions.

**Figure 13.16** | Normalized \( I_D \) versus \( V_{DS} \) plots for a constant mobility and field-dependent mobility.  
*(From Sze [19].)*

**Figure 13.17** | Measured drain current versus \( V_{GS} \) for a GaAs MESFET showing the normal drain current, subthreshold current, and gate leakage current.  
*(From Daring [21].)*

## 13.4 Equivalent Circuit and Frequency Limitations

In order to analyze a transistor circuit, one needs a mathematical model or equivalent circuit of the transistor. One of the most useful models is the small-signal equivalent circuit, which applies to transistors used in linear amplifier circuits. This equivalent circuit will introduce frequency effects in the transistor through the equivalent capacitor–resistor circuits. The various physical factors in the JFET affecting the frequency limitations are considered here and a transistor cutoff frequency, which is a figure of merit, is then defined.

### 13.4.1 Small-Signal Equivalent Circuit

The cross section of an n-channel pn JFET is shown in Figure 13.18, including source and drain series resistances. The substrate may be semi-insulating gallium arsenide or it may be a p+ type substrate.

Figure 13.19 shows a small-signal equivalent circuit for the JFET. The voltage \( V_{g'} \) is the internal gate-to-source voltage that controls the drain current. The \( r_g \) and \( C_{gs} \) parameters are the gate-to-source diffusion resistance and junction capacitance, respectively. The gate-to-source junction is reverse biased for depletion mode devices and has only a small forward-bias voltage for enhancement mode devices, so that normally \( r_g \) is large. The parameters \( r_d \) and \( C_{gd} \) are the gate-to-drain resistance and capacitance, respectively. The resistance \( r_{ds} \) is the finite drain resistance, which is a function of the channel length modulation effect. The \( C_d \) capacitance is mainly a drain-to-source parasitic capacitance and \( C_s \) is the drain-to-substrate capacitance.

The ideal small-signal equivalent circuit is shown in Figure 13.20a. All diffusion resistances are infinite, the series resistances are zero, and at low frequency the

**Figure 13.20** (a) Ideal low-frequency small-signal equivalent circuit. (b) Ideal equivalent circuit including \( r_s \).

Capacitances become open circuits. The small-signal drain current is now

\[
I_{ds} = g_m V_{gs}
\]

(13.57)

which is a function only of the transconductance and the input-signal voltage.

The effect of the source series resistance can be determined using Figure 13.20b. We have

\[
I_{ds} = g_m V_{gs}'
\]

(13.58)

The relation between \( V_{gs} \) and \( V_{gs}' \) can be found from

\[
V_{gs} = V_{gs}' + (g_m V_{gs}') r_s = (1 + g_m r_s) V_{gs}'
\]

(13.59)

Equation (13.58) can then be written as

\[
I_{ds} = \left( \frac{g_m}{1 + g_m r_s} \right) V_{gs} = g_m' V_{gs}
\]

(13.60)

The effect of the source resistance is to reduce the effective transconductance or transistor gain.

Recall that \( g_m \) is a function of the dc gate-to-source voltage, so \( g_m' \) will also be a function of \( V_{GS} \). Equation (13.41b) is the relation between \( g_m \) and \( V_{GS} \) when the...

**Figure 13.21** JFET transconductance versus \( V_{GS} \) (a) without and (b) with a source series resistance.

**Figure 13.22** A small-signal equivalent circuit with capacitance.

The transistor is biased in the saturation region. Figure 13.21 shows a comparison between the theoretical and experimental transconductance values using the parameters from Example 13.4 and letting \( r_s = 2000 \, \Omega \). (A value of \( r_s = 2000 \, \Omega \) may seem excessive, but keep in mind that the active thickness of the semiconductor may be on the order of 1 \(\mu m\) or less; thus, a large series resistance may result if special care is not taken.)

### 13.4.2 Frequency Limitation Factors and Cutoff Frequency

There are two frequency limitation factors in a JFET. The first is the channel transit time. If we assume a channel length of 1 \(\mu m\) and assume carriers are traveling at their saturation velocity, then the transit time is on the order of

\[
\tau_t = \frac{L}{v_s} = \frac{1 \times 10^{-4}}{1 \times 10^7} = 10 \, \text{ps}
\]

(13.61)

The channel transit time is normally not the limiting factor except in very high frequency devices.

The second frequency limitation factor is the capacitance charging time. Figure 13.22 is a simplified equivalent circuit that includes the primary capacitances and ignores the diffusion resistances. The output current will be the short-circuit current. As the frequency of the input-signal voltage \( V_g \) increases, the impedance of \( C_{gd} \) and \( C_{gp} \) decreases so the current through \( C_{gd} \) will increase. For a constant \( g_m V_{gs} \), the \( I_d \) current will then decrease. The output current then becomes a function of frequency.

If the capacitance charging time is the limiting factor, then the cutoff frequency \( f_T \) is defined as the frequency at which the magnitude of the input current \( I_i \) is equal to the magnitude of the ideal output current \( g_m V_{gs} \) of the intrinsic transistor. We have,

When the output is short-circuited,

\[
I_t = j\omega (C_P + C_{pd})V_{gs}
\]

(13.62)

If we let \( C_G = C_P + C_{pd} \), then at the cutoff frequency

\[
|I| = 2\pi f_T C_G V_{gs} = g_m V_{gs}
\]

(13.63)

or

\[
f_T = \frac{g_m}{2\pi C_G}
\]

(13.64)

From Equation (13.41b), the maximum possible transconductance is

\[
g_m (\text{max}) = g_{0m} = \frac{e\mu_n N_d w a}{L}
\]

(13.65)

and the minimum gate capacitance is

\[
C_G (\text{min}) = \frac{\varepsilon_s WL}{a}
\]

(13.66)

where \( a \) is the maximum space charge width. The maximum cutoff frequency can be written as

\[
f_T = \frac{e\mu_n N_d a^2}{2\pi \varepsilon_s L^2}
\]

(13.67)

For gallium arsenide JFETs or MESFETs with very small geometries, the cutoff frequency is even larger. The channel transit time may also become a factor in very high frequency devices, in which case the expression for cutoff frequency would need to be modified.


One application of GaAs FETs is in ultrafast digital integrated circuits. Conventional GaAs MESFET logic gates can achieve propagation delay times in the subnanosecond range. These delay times are at least comparable to, if not shorter than, fast ECL, but the power dissipation is three orders of magnitude smaller than in the ECL circuits. Enhancement mode GaAs JFETs have been used as drivers in logic circuits, and depletion mode devices may be used as loads. Propagation delay times of as low as 45 ps have been observed. Special JFET structures may be used to further increase the speed. These structures include the modulation-doped field-effect transistor, which is discussed in the following section.

## 13.5| High Electron Mobility Transistor

As frequency needs, power capacity, and low noise performance requirements increase, the gallium arsenide MESFET is pushed to its limit of design and performance. These requirements imply a very small FET with a short channel length, large saturation current, and large transconductance. These requirements are generally achieved by increasing the channel doping under the gate. In all of the devices we have considered, the channel region is in a doped layer of bulk semiconductor with the majority carriers and doping impurities in the same region. The majority carriers experience ionized impurity scattering, which reduces carrier mobility and degrades device performance.

The degradation in mobility and peak velocity in GaAs due to increased doping can be minimized by separating the majority carriers from the ionized impurities. This separation can be achieved in a heterostructure that has an abrupt discontinuity in conduction and valence bands. We considered the basic heterojunction properties in Chapter 9. Figure 13.23 shows the conduction-band energy relative to the Fermi energy of an N-AlGaAs-intrinsic GaAs heterojunction in thermal equilibrium. Thermal equilibrium is achieved when electrons from the wide-bandgap AlGaAs flow into the GaAs and are confined to the potential well. However, the electrons are free to move parallel to the heterojunction interface. In this structure, the majority carrier

**Figure 13.23** | Conduction-band edges for N-AlGaAs–intrinsic GaAs abrupt heterojunction.

The electrons in the potential well are now separated from the impurity dopant atoms in the AlGaAs; thus, impurity scattering tends to be minimized.

The FETs fabricated from these heterojunctions are known by several names. The term used here is the high electron mobility transistor (HEMT). Other names include modulation-doped field-effect transistor (MODFET), selectively doped heterojunction field-effect transistor (SDHT), and two-dimensional electron gas field-effect transistor (TEGFET).

### 13.5.1 Quantum Well Structures

Figure 13.23 shows the conduction-band energy of an N-AlGaAs–intrinsic GaAs heterojunction. A two-dimensional surface channel layer of electrons is formed in the thin potential well (~80Å) in the undoped GaAs. Electron sheet carrier densities on the order of \(10^{12} \, \text{cm}^{-2}\) have been obtained. An improvement in the low-field mobility of the carriers moving parallel to the heterojunction is observed since the impurity-scattering effects are reduced. At 300 K, mobilities have been reported in the range of 8500–9000 \(\text{cm}^2/\text{V·s}\), whereas GaAs MESFETs doped to \(N_d = 10^{17} \, \text{cm}^{-3}\) have low-field mobilities of less than 5000 \(\text{cm}^2/\text{V·s}\). The electron mobility in the heterojunction now tends to be dominated by lattice or phonon scattering, so that as the temperature is reduced, the mobility increases rapidly.

Impurity-scattering effects can be further reduced by increasing the separation of the electrons and ionized donor impurities. The electrons in the potential well of the abrupt heterojunction shown in Figure 13.23 are separated from the donor atoms, but are still close enough to be subjected to a coulomb attraction. A thin spacer layer of undoped AlGaAs can be placed between the doped AlGaAs and the undoped GaAs. Figure 13.24 shows the energy-band diagram for this structure.

Figure 13.24

Conduction-band edges for N-AlGaAs–undoped AlGaAs–undoped GaAs heterojunction.

(From Shur [13].)


- **(a)** 2-D electron gas
- **(b)** Energy-band diagram

- \(x = 0.35\)
- \((\text{Al}, \text{Ga})\text{As} = 1.5 \times 10^{18} \, \text{cm}^{-3}\)
- \(N_s = 1.14 \times 10^{12} \, \text{cm}^{-2}\)
- \(T = 300 \, K\)

- \(E_0\), \(E_i\), \(E_F\) are energy levels
- \(d_1\), \(d_2\), \(Z_1\), \(Z_2\) are distances

The diagram illustrates the separation of doped and undoped regions, showing the potential well and electron gas formation.
separation between the carriers and ionized donors increases further the electron mobility, since there is even less coulomb interaction. One disadvantage of this graded heterojunction is that the electron density in the potential well tends to be smaller than in the abrupt junction.

The molecular beam epitaxial process allows the growth of very thin layers of specific semiconductor materials with specific dopings. In particular, a multilayer modulation–doped heterostructure can be formed, as shown in Figure 13.25. Several surface channel layers of electrons are formed in parallel. This structure would be equivalent to increasing the channel electron density, which would increase the current capability of the FET.

### 13.5.2 Transistor Performance

A typical HEMT structure is shown in Figure 13.26. The N-AlGaAs is separated from the undoped GaAs by an undoped AlGaAs spacer. A Schottky contact to the N-AlGaAs forms the gate of the transistor. This structure is a “normal” MODFET. An “inverted” structure is shown in Figure 13.27. In this case the Schottky contact is made to the undoped GaAs layer. The inverted MODFET has been investigated less than the normal structure because the normal structure has yielded superior results.

The density of electrons in the two-dimensional electron gas layer in the potential well can be controlled by the gate voltage. The electric field of the Schottky gate depletes the two-dimensional electron gas layer in the potential well when a

**Figure 13.25** | Multilayer modulation–doped heterostructure.

| Layer                  | Thickness |
|------------------------|-----------|
| Doped AlGaAs           | 300 Å     |
| Undoped GaAs           | 300 Å     |
| Doped AlGaAs           | 250 Å     |
| Undoped GaAs           | 300 Å     |
| Doped AlGaAs           | 150 Å     |
| Undoped GaAs           | 300 Å     |
| Doped AlGaAs           | 50 Å      |
| Undoped AlGaAs spacer  | 0.2 μm    |
| Undoped buffer         | 1 μm      |

- Al = 0.2 to 0.25
- Semi-insulating substrate

**Figure 13.26** | A “normal” AlGaAs–GaAs HEMT.

- Source
- Drain
- n⁺ AlGaAs
- Undoped AlGaAs spacer
- Undoped GaAs buffer
- Semi-insulating GaAs substrate
- 2-D electron gas
- Ohmic contact
- n⁺GaAs Gate

**Figure 13.27** | An "inverted" GaAs–AlGaAs HEMT.  
*(From Shur [13].)*


**Figure 13.28** | Energy-band diagram of a normal HEMT  
(a) with zero gate bias and (b) with a negative gate bias.

Sufficiently large negative voltage is applied to the gate. Figure 13.28 shows the energy-band diagrams of the metal–AlGaAs–GaAs structure under zero bias and with a reverse bias applied to the gate. With zero bias, the conduction-band edge in the GaAs is below the Fermi level, implying a large density of the two-dimensional electron gas. With a negative voltage applied to the gate, the conduction-band edge in the GaAs is above the Fermi level, implying that the density of the two-dimensional electron gas is very small and the current in an FET would be essentially zero.

The Schottky barrier depletes the AlGaAs layer from the surface, and the heterojunction depletes the AlGaAs layer from the heterojunction interface. Ideally the device should be designed so that the two depletion regions just overlap to prevent electron conduction through the AlGaAs layer. For depletion mode devices, the depletion layer from the Schottky gate should extend only to the heterojunction depletion layer. For enhancement mode devices, the thickness of the doped AlGaAs layer is smaller and the Schottky gate built-in potential barrier will completely...


The density of the two-dimensional electron gas in a normal structure can be described using a charge control model. We may write:

\[
n_s = \frac{\varepsilon_l}{q(d + \Delta d)} (V_g - V_{\text{off}})
\]

(13.68)

where \(\varepsilon_l\) is the permittivity of the N-AlGaAs, \(d = d_t + d_i\) is the thickness of the doped-plus-undoped AlGaAs layer, and \(\Delta d\) is a correction factor given by:

\[
\Delta d = \frac{\varepsilon_a}{q} \approx 80 \, \text{Å}
\]

(13.69)

The threshold voltage \(V_{\text{off}}\) is given by:

\[
V_{\text{off}} = \phi_b - \frac{\Delta E_c}{q} - V_{pz}
\]

(13.70)

where \(\phi_b\) is the Schottky barrier height and \(V_{pz}\) is:

\[
V_{pz} = \frac{qN_ad_i^2}{2\varepsilon_l}
\]

(13.71)

A negative gate bias will reduce the two-dimensional electron gas concentration. If a positive gate voltage is applied, the density of the two-dimensional electron gas will increase. Increasing the gate voltage will increase the two-dimensional electron gas density until the conduction band of the AlGaAs crosses the Fermi level of the electron gas. Figure 13.29 shows this effect. At this point, the gate loses control over the electron gas since a parallel conduction path in the AlGaAs has been formed.

**Figure 13.29** | Energy-band diagram of an enhancement mode HEMT (a) with a slight forward gate voltage, and (b) with a larger forward gate voltage that creates a conduction channel in the AlGaAs. (From Fritzsche [5].)

The current–voltage characteristics of the MODFET can be found using the charge control model and the gradual channel approximation. The channel carrier concentration can be written as

\[
n_s(x) = \frac{q\epsilon_r \epsilon_0}{qd + \Delta d} [V_g - V_{off} - V(x)]
\]

where \(V(x)\) is the potential along the channel due to the drain-to-source voltage. The drain current is

\[
I_D = qn_s v(E)W
\]

where \(v(E)\) is the carrier drift velocity and \(W\) is the channel width. This analysis is very similar to that for the pn JFET in Section 13.2.2.

If we assume a constant mobility, then for low \(V_{DS}\) values, we have

\[
I_D = \frac{\epsilon_r \epsilon_0 \mu W}{2L(d + \Delta d)} [2(V_g - V_{off}) V_{DS} - V_{DS}^2]
\]

The form of this equation is the same as that for the pn JFET or MESFET operating in the nonsaturation region. If \(V_{DS}\) increases so that the carriers reach the saturation velocity, then

\[
I_{D(sat)} = \frac{\epsilon_r \epsilon_0 v_s W}{d + \Delta d} (V_g - V_{off} - V_0)_{v_{sat}}
\]

where \(v_{sat}\) is the saturation velocity and \(V_0 = E_s L\) with \(E_s\) being the electric field in the channel that produces the saturation velocity.

Various velocity versus electric field models can be used to derive different \( I\text{-}V \) expressions. However, Equations (13.74) and (13.75) yield satisfactory results for most situations. Figure 13.30 shows a comparison between experimental and calculated \( I\text{-}V \) characteristics. As observed in the figure, the current in these heterojunction devices can be quite large. The transconductance of the MODFET is defined as it was for the pn JFET and MESFET. Typical measured values at \( T = 300 \, \text{K} \) are in the range of 250 mS/mm. Higher values have been reported. These transconductance values are significantly larger than for either the pn JFET or the MESFET.

HEMTs may also be fabricated with multiple heterojunction layers. This device type is shown in Figure 13.31. A single heterojunction for an AlGaAs–GaAs interface has a maximum two-dimensional electron sheet density on the order of \( 1 \times 10^{12} \, \text{cm}^{-2} \). This concentration can be increased by fabricating two or more AlGaAs–GaAs interfaces in the same epitaxial layer. The device current capacity is increased, and power performance is improved. The multichannel HEMT behaves as multiple single-channel HEMTs connected in parallel and modulated by the same gate but with slightly different threshold voltages. The maximum transconductance will not scale directly with the number of channels because of the change in threshold voltage with each channel. In addition, the effective channel length increases as the distance between the gate and channel increases.

HEMTs can be used in high-speed logic circuits. They have been used in flip-flop circuits operating at clock frequencies of 5.5 GHz at \( T = 300 \, \text{K} \); the clock frequency can be increased at lower temperatures. Small-signal, high-frequency amplifiers have also been investigated. HEMTs showing low noise and reasonable gains have been operated at 35 GHz. The maximum frequency increases as the channel length decreases. Cutoff frequencies on the order of 100 GHz have been measured with channel lengths of 0.25 \(\mu\text{m}\).


**Figure 13.30** | Current–voltage characteristics of an enhancement mode HEMT, in which solid curves are numerical calculations and dots are measured points.  
*(From Shur [13].)*


**Figure 13.31** | A multilayer HEMT.

It seems clear that HEMTs are inherently superior to other FET technologies in terms of achieving higher speeds of operation, lower power dissipation, and lower noise. These advantages derive directly from the superior transport properties obtained by using undoped GaAs as the channel layer for the FET. One way to achieve an adequate carrier concentration in an undoped channel is to accumulate the carriers at a semiconductor heterojunction interface, as we have seen. The disadvantage of the HEMT is that the fabrication processes for the heterojunction are more complicated.

## 13.6 | SUMMARY

- The physics, characteristics, and operation of the junction field-effect transistor are considered in this chapter.
- The current in a JFET is controlled by an electric field applied perpendicular to the direction of current. The current is in the channel region between the source and drain contacts. In a pn JFET, the channel forms one side of a pn junction that is used to modulate the channel conductance.
- Two primary parameters of the JFET are the internal pinchoff voltage \( V_0 \) and the pinchoff voltage \( V_p \). The internal pinchoff voltage is defined as a positive quantity and is the total gate-to-channel potential that causes the junction space charge layer to completely fill the channel region. The pinchoff voltage is defined as the gate voltage that must be applied to achieve the pinchoff condition.
- The ideal current–voltage relationship is derived. The transconductance, or transistor gain, is the rate of change of drain current with respect to the corresponding change in gate-to-source voltage.
- Three nonideal effects are considered; channel-length modulation, velocity saturation, and subthreshold current. Each of these effects changes the ideal current–voltage relationship.
- A small-signal equivalent circuit of the JFET is developed. The equivalent circuit includes capacitances that introduce frequency effects in the transistor. Two physical factors affect the frequency limitation; channel transit time and capacitance charging time. The capacitance charging time constant is normally the limiting factor in short channel devices.
- The high-electron mobility transistor (HEMT) structure utilizes a heterojunction. A two-dimensional electron gas is confined to a potential well at the heterojunction interface. However, the electrons are free to move parallel to the interface. These electrons are separated from the ionized donors so that ionized impurity scattering effects are minimized, resulting in a high mobility.

## Glossary of Important Terms

**capacitance charging time**  
The time associated with charging or discharging the input gate capacitance with a change in the input gate signal.

**channel conductance**  
The ratio of a differential change in drain current to the corresponding differential change in drain-to-source voltage in the limit as the drain-to-source voltage approaches zero.

**channel conductance modulation**  
The process whereby the channel conductance changes with gate voltage; this is the basic field-effect transistor action.


### Key Terms

- **Channel length modulation**: The change in effective channel length with drain-to-source voltage with the JFET biased in the saturation region.

- **Conduction parameter**: The multiplying factor \( k_n \) in the expression for drain current versus gate-to-source voltage for the enhancement mode MESFET.

- **Cutoff frequency**: A figure of merit for the transistor defined to be the frequency at which the ratio of the small-signal input gate current to small-signal drain current is equal to unity.

- **Depletion mode JFET**: A JFET in which a gate-to-source voltage must be applied to create pinch-off and turn the device off.

- **Enhancement mode JFET**: A JFET in which pinch-off exists at zero gate voltage and a gate-to-source voltage must be applied to induce a channel, turning the device on.

- **Internal pinch-off voltage**: The total potential drop across the gate junction at pinch-off.

- **Output resistance**: The ratio of a differential change in drain-to-source voltage to the corresponding differential change in drain current at a constant gate-to-source voltage.

- **Pinch-off**: The condition whereby the gate junction space charge region extends completely through the channel so that the channel is completely depleted of free carriers.