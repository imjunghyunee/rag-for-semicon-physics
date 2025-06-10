# 15  
## Semiconductor Microwave and Power Devices

In previous chapters, we have discussed the basic physics, operation, and characteristics of diodes and transistors. We have analyzed the frequency response as well as the current–voltage characteristics of these semiconductor devices. However, we have not specifically considered the generation of microwave signals using semiconductor devices or the power capabilities of semiconductor transistors.

In this chapter, we first consider three semiconductor devices that are used to generate microwave signals. These devices include the tunnel diode, GUNN diode, and IMPATT diode. A basic principle of oscillators is that a region of negative differential resistance must exist. We consider the process by which a region of negative differential resistance is created in each device and discuss the basic operation of these devices.

Second, we discuss three specialized semiconductor power devices, including power bipolar transistors and power MOSFETs. We have considered the basic physics of these devices in previous chapters, and analyzed the current–voltage characteristics without specifically considering the current or voltage limitations or the power dissipation within the devices. In this chapter, we discuss the limitations in current and voltage, and the power capabilities of the devices. Finally, we discuss the operation and characteristics of a four-layered structure called a thyristor.

### 15.0 | PREVIEW

In this chapter, we will:

- Discuss the concept of negative differential resistance in a tunnel diode and derive an expression for the maximum resistance cutoff frequency.

- Discuss the concept of negative differential mobility in GaAs and discuss the process by which this characteristic can lead to microwave oscillations in a GUNN diode.

- Discuss the operation of an IMPATT diode oscillator and determine the process by which a dynamic negative resistance is created.

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

# Chapter 15: Semiconductor Microwave and Power Devices

!Figure 15.2  
**Figure 15.2** | Equivalent circuit of the tunnel diode.

Figure 15.1b shows an expanded plot of the I–V characteristics in the tunneling range. A point is shown on the curve where the minimum value of negative resistance occurs. (Note that \( R_{\text{min}} \) is a positive quantity.) The equivalent circuit of the tunnel diode for the case when the diode is biased at the \(-R_{\text{min}}\) point is shown in Figure 15.2. The parameter \( C_j \) is the junction capacitance, and the parameters \( L_p \) and \( R_p \) are the parasitic or interconnect line inductance and resistance, respectively.

The small signal input impedance can be written as

\[
Z = \left[ R_p - \frac{R_{\text{min}}}{1 + \omega^2 R_{\text{min}}^2 C_j^2} \right] + j\omega \left[ L_p - \frac{\omega R_{\text{min}}^2 C_j}{1 + \omega^2 R_{\text{min}}^2 C_j^2} \right]
\]

(15.1)

The resistive part of the impedance goes to zero at a frequency of

\[
f_r = \frac{1}{2\pi R_{\text{min}} C_j} \sqrt{\frac{R_{\text{min}}}{R_p} - 1}
\]

(15.2)

For frequencies \( f > f_r \), the resistive part of the impedance becomes positive so that the diode loses its negative differential resistance characteristic. The operating frequency must then occur at \( f_0 < f_r \). The frequency \( f_r \) is referred to as the maximum resistive cutoff frequency.

The tunneling process is a majority carrier effect so the diode does not exhibit time delays due to minority carrier diffusion, which means that the diode is capable of operating at microwave frequencies. However, due to the relatively small voltage range in which the diode exhibits the negative resistance characteristic, the tunnel diode is not used extensively.

## 15.2 Gunn Diode

Another negative differential resistance device is the GUNN diode, or Transferred-Electron Device (TED). The transferred-electron phenomenon is demonstrated in a few semiconductors in which conduction electrons in a high-mobility band are scattered to a low-mobility band by a high electric field. In Chapter 5, we discussed the drift velocity of electrons in GaAs versus electric field. Figure 15.3 again shows a plot of this characteristic. InP also shows this same characteristic.

Figure 15.4 shows an expanded plot of the energy-band structure in GaAs that is given in Figure 5.8. For small electric fields, essentially all of the electrons in the

# 15.2 GUNN Diode

!Graph of Electron Drift Velocity

**Figure 15.3.1** Electron drift velocity versus electric field for GaAs.

!Energy-band Structure of GaAs

**Figure 15.4.1** Energy-band structure of GaAs showing the lower valley and upper valley in the conduction band.

Conduction band exist in the lower valley of the E versus \( k \) diagram, where the density of states electron effective mass is small. A small effective mass leads to a large mobility value.

As the electric field increases above a threshold or critical value, \( E_{th} \), the electrons gain more than the 0.3 eV energy separating the two valleys so that electrons can be scattered into the upper valley, where the density of states electron effective mass is much larger. The larger effective mass yields a smaller mobility. The intervalley transfer mechanism with a change in mobility results in a decreasing average drift velocity of electrons with electric field, or a negative differential electron mobility. The maximum negative differential electron mobility in GaAs is approximately \(-2400 \, \text{cm}^2/\text{V}\cdot\text{s}\).

Consider a two-terminal n-type GaAs device with ohmic contacts at the ends that is biased in the negative mobility region (\( E_{\text{bias}} > E_{th} \)) as shown in Figure 15.5a. A small space charge region may develop in the material near the cathode as shown in Figure 15.5b. As a result, the electric field increases in this region as shown in Figure 15.5c. (Special device structures can be fabricated to ensure that the space charge fluctuations are generated near the cathode.)

# Chapter 15: Semiconductor Microwave and Power Devices

!Figure 15.5

**Figure 15.5** (a) A simplified two-terminal GaAs device. (b) Electron concentration versus distance showing a space charge formation. (c) Electric field versus distance.

In discussing excess carrier behavior in Chapter 6, we found the time behavior of a net charge density in a semiconductor to be given by

\[
\delta Q(t) = \delta Q(0)e^{-t/\tau_d}
\]

(15.3)

where \(\tau_d\) is the dielectric relaxation time constant and is on the order of a picosecond. Normally, a small space charge region would be quickly neutralized. The dielectric relaxation time constant is given by \(\tau_d = \varepsilon/\sigma\), where \(\sigma\) is the semiconductor conductivity. If the GaAs is biased in the negative mobility region, then the conductivity is negative and the exponent in Equation (15.3) becomes positive, so the space charge region, now called a domain, can actually build up as it drifts toward the anode. As the domain grows (Figure 15.6a), the electric field in this region increases which means that the electric field in the remaining material decreases. The E field in the material outside of the domain can drop below the critical value, as indicated in Figure 15.6b, while the E field within the domain remains above the critical value. For this reason, only one domain will normally be established in the material at any given time.

As the domain reaches the anode, a current pulse is induced in the external circuit. After the domain reaches the anode, another domain may form near the cathode and the process repeats itself. Thus, a series of current pulses may be generated as shown in Figure 15.7. The time between current pulses is the time for the domain to drift through the device. The oscillation frequency is given by

\[
f = 1/\tau = v_d/L
\]

(15.4)

where \(v_d\) is the average drift velocity and \(L\) is the length of the drift region.

!Figure 15.6  
**Figure 15.6** (a) Electron concentration versus distance showing a domain. (b) Electric field versus distance.

!Figure 15.7  
**Figure 15.7** Current pulses versus time in the GaAs device.

The oscillation mechanism just described is called the transit-time mode. More complex modes of operation are possible. Studies have shown that the efficiency of the transit-time device is largest when the product \( n_0 L \) is a few times \( 10^{12} \, \text{cm}^{-2} \). For this case, the domain fills about one-half of the drift region length and produces a current output that is nearly sinusoidal. The maximum dc-to-rf conversion efficiency is approximately 10 percent.

Oscillations in the frequency range of 1 to 100 GHz or higher can be obtained. If the device is operated in a pulsed mode, a peak output power in the range of hundreds of watts can be produced. Transferred-electron devices are now used as the microwave source in many radar systems.

## 15.3 IMPATT DIODE

The term IMPATT stands for **IM**pact ionization **A**valanche **T**ransit-**T**ime. The IMPATT diode consists of a high-field avalanche region and a drift region that produces a dynamic negative resistance at microwave frequencies. The negative resistance characteristic produced in this device is a result of a time delay so that the ac current and voltage components are out of phase, and is a different phenomenon compared to the tunnel diode, for example. The tunnel diode has a negative \( dI/dV \) region in the \( I-V \) characteristic.

One example of an IMPATT diode is a \( p^+ \)-\( n^- \)-\( i \)-\( n^+ \) structure as shown in Figure 15.8a. Typical doping concentrations (magnitudes) are shown in Figure 15.8b. The device is reverse biased so that the \( n \) and intrinsic regions are completely depleted. The electric field in the device is shown in Figure 15.8c. We may note that

\[
E \, dt = V_B
\]

where \( V_B \) is the applied reverse-biased voltage. The value of \( V_B \) is very close to the breakdown voltage. The avalanche region is localized near the pn junction. The electric field in the intrinsic region is nearly constant and the intrinsic layer provides the drift region.

# CHAPTER 15 Semiconductor Microwave and Power Devices

!Figure 15.8
**Figure 15.8** (a) An IMPATT diode structure. (b) Typical doping concentrations in the IMPATT diode. (c) Electric field versus distance through the IMPATT diode.

!Figure 15.9
**Figure 15.9** Circuit for an IMPATT diode oscillator.

Figure 15.9 shows the circuit for an IMPATT diode oscillator. An LC resonant circuit is required for the oscillator operation. During the positive ac voltage across the LC circuit as shown in the figure, the diode goes into breakdown and electron-hole pairs are generated at the p⁺n junction. The generated electrons flow back into the p⁺ region, while the holes start drifting through the depleted intrinsic region. In general, the holes will travel at their saturation velocity. During the negative ac voltage, the device operates below the breakdown voltage so electron-hole pairs are no longer produced.

There is an inherent \(\pi/2\) phase shift between the peak value of the avalanche voltage at the p⁺n junction and the injection of the holes into the intrinsic drift region due to the finite buildup time of the avalanche generated electron-hole pairs. A further delay of \(\pi/2\) is then required during the drift process to provide the total 180 degrees of phase shift between the current and voltage at the output terminal. The transit time of the holes is \(\tau = L/v_s\), where \(L\) is the length of the drift region and \(v_s\) is the saturation velocity of the holes. The LC circuit resonant frequency must be designed to be equal to the device resonant frequency, which is given by

\[
f = \frac{1}{2\tau} = \frac{v_s}{2L}
\]

(15.5)

When the holes reach the n⁺ cathode, the current is at a maximum value and the voltage is at its minimum value. The ac current and ac voltage are 180 degrees out of phase with respect to each other producing the dynamic negative resistance.

```markdown
# 15.4 Power Bipolar Transistors

Devices can be designed to operate in the 100 GHz or higher frequency range and produce power outputs of a few watts. The efficiency of these devices is in the range of 10 to 15 percent, and these devices provide the highest continuous output power of all the semiconductor microwave devices. As with most semiconductor device designs, other structures can be fabricated to provide specialized output characteristics.

## 15.4.1 Power Bipolar Transistors

In our previous discussions, we have ignored any physical transistor limitations in terms of maximum current, voltage, and power. We implicitly assumed that the transistors are capable of handling the current and voltage, and could handle the power dissipated within the device without suffering any damage.

However, with power transistors, we must be concerned with various transistor limitations. The limitations involve maximum rated current (on the order of amperes), maximum rated voltage (on the order of 100 V), and maximum rated power (on the order of watts or tens of watts).[^1]

### 15.4.1 Vertical Power Transistor Structure

Figure 15.10 shows the structure of a vertical npn power transistor. We have considered vertical npn bipolar transistors previously. However, with small switching devices, the collector terminal is still formed at the surface. In the vertical configuration for the power bipolar transistor, the collector terminal is at the “bottom” of the device. This configuration is preferred since it maximizes the cross-sectional area through which current is flowing in the device. In addition, the doping concentrations

!Figure 15.10

**Figure 15.10** | Cross section of typical vertical npn power BJT.

[^1]: We must note that, in general, the maximum rated current and maximum rated voltage cannot occur at the same time.
```

!Interdigitated Bipolar Transistor Structure

**Figure 15.11** | An interdigitated bipolar transistor structure showing the top view and cross-sectional view.

and dimensions are not the same as we have encountered in small switching transistors. The primary collector region has a low-doped impurity concentration so that a large base–collector voltage can be applied without initiating breakdown. Another n region, with a higher doping concentration, reduces collector resistance and makes contact with the external collector terminal. The base region is also much wider than normally encountered in small devices. A large base–collector voltage implies a relatively large space charge width being induced in both the collector and base regions. A relatively large base width is required to prevent punch-through breakdown.

Power transistors must also be large-area devices in order to handle large currents. We have previously considered the interdigitated structure that is repeated in Figure 15.11. Relatively small emitter widths are required to prevent the emitter current crowding effects that were discussed in Section 12.4.4.

### 15.4.2 Power Transistor Characteristics

The relatively wide base width implies a much smaller current gain \( \beta \) for power transistors compared to small switching transistors, and large area device implies a larger junction capacitance and hence lower cutoff frequency for a power transistor compared to a small switching transistor. Table 15.1 compares the parameters of a

# 15.4 Power Bipolar Transistors

## Table 15.1.1 Comparison of the characteristics and maximum ratings of small-signal and power BJTs

| Parameter                  | Small-signal BJT (2N2222A) | Power BJT (2N3055) | Power BJT (2N6078) |
|----------------------------|----------------------------|--------------------|--------------------|
| \( V_{CE} \) (max) (V)     | 40                         | 60                 | 250                |
| \( I_C \) (max) (A)        | 0.8                        | 15                 | 7                  |
| \( P_D \) (max) (W)        | 1.2                        | 115                | 45                 |
| (at \( T = 25^\circ C \))  |                            |                    |                    |
| \( \beta \)                | 35–100                     | 5–20               | 12–70              |
| \( f_T \) (MHz)            | 300                        | 0.8                | 1                  |

The table compares the characteristics of a general-purpose small-signal BJT to those of two power BJTs. The current gain is generally smaller in the power transistors, typically in the range of 20 to 100, and may be a strong function of collector current and temperature. Figure 15.12 shows typical current gain versus collector current characteristics for the 2N3055 power BJT at various temperatures.

The **maximum rated collector current** \( I_{C,max} \) may be related to the maximum current that the wires connecting the semiconductor to the external terminals can handle, the collector current at which the current gain falls below a minimum specified value, or the current that leads to the maximum power dissipation when the transistor is biased in saturation.

The **maximum rated voltage** in a BJT is generally associated with avalanche breakdown in the reverse-biased base–collector junction. In the common-emitter configuration, the breakdown voltage mechanism also involves the transistor gain, as well as the breakdown phenomenon in the pn junction. This is discussed in Section 12.4.6. Typical \( I_C \) versus \( V_{CE} \) characteristics are shown in Figure 15.13.

!Figure 15.12

**Figure 15.12** Typical dc beta characteristics (\( h_{FE} \) versus \( I_C \)) for 2N3055.

!Figure 15.13

**Figure 15.13** Typical collector current versus collector–emitter voltage characteristics of a bipolar transistor, showing breakdown effects.

# Semiconductor Microwave and Power Devices

## Safe Operating Area (SOA) of a Bipolar Transistor

### Figure 15.14

The safe operating area (SOA) of a bipolar transistor plotted on:

- (a) Linear scales
- (b) Logarithmic scales

#### (a) Linear Scales

- **Axes:**
  - \( I_C \) (A)
  - \( V_{CE} \) (V)

- **Key Points:**
  - \( I_{C,\text{max}} \)
  - \( P_T \)
  - Second breakdown
  - \( V_{CE,\text{max}} \)

#### (b) Logarithmic Scales

- **Axes:**
  - \( I_C \) (A)
  - \( V_{CE} \) (V)

- **Key Points:**
  - Maximum current limit
  - \( P_T \)
  - Second breakdown
  - \( V_{CE,\text{max}} \)

### Explanation

When the transistor is biased in the forward-active mode, the collector current begins to increase significantly before the actual breakdown voltage is reached. All the curves tend to merge to the same collector–emitter voltage once breakdown has occurred. This voltage, \( V_{CE,\text{max}} \), is the minimum voltage necessary to sustain the transistor in breakdown.

Another breakdown effect is called **second breakdown**, which occurs in a BJT operating at high voltage and high current. Slight nonuniformities in current density produce local regions of increased heating that increases the minority carrier concentrations in the semiconductor material, which in turn increases the current in these regions. This effect results in positive feedback, and the current continues to increase, producing a further increase in temperature, until the semiconductor material may actually melt, creating a short circuit between the collector and emitter.

The average power dissipated in a BJT must be kept below a specified maximum value, to ensure that the temperature of the device remains below a maximum value. If we assume the collector current and collector–emitter voltage are dc values, then at the maximum rated power \( P_T \) for the transistor, we can write:

\[
P_T = V_{CE} I_C
\]

Equation (15.6) neglects the \( V_{BE} I_B \) component of power dissipation in the transistor.

The maximum current, voltage, and power limitations can be illustrated on the \( I_C \) versus \( V_{CE} \) characteristics as shown in Figure 15.14. The average power limitation, \( P_T \), is a hyperbola described by Equation (15.6). The region where the transistor can be operated safely is known as the safe operating area (SOA) and is bounded by \( I_{C,\text{max}} \), \( V_{CE,\text{max}} \), \( P_T \), and the transistor’s second breakdown characteristic curve. Figure 15.14a shows the safe operating area using linear current and voltage scales. Figure 15.14b shows the same characteristics using log scales.

# 15.4 Power Bipolar Transistors

## Objective

Determine the required current, voltage, and power rating of a power BJT.

Consider the common-emitter circuit in Figure 15.15. The parameters are \( R_L = 10 \, \Omega \) and \( V_{CC} = 35 \, \text{V} \).

## Solution

For \( V_{CE} = 0 \), the maximum collector current is

\[
I_C (\text{max}) = \frac{V_{CC}}{R_L} = \frac{35}{10} = 3.5 \, \text{A}
\]

For \( I_C = 0 \), the maximum collector-emitter voltage is

\[
V_{CE} (\text{max}) = V_{CC} = 35 \, \text{V}
\]

The load line is given by

\[
V_{CE} = V_{CC} - I_C R_L
\]

and must remain within the SOA, as shown in Figure 15.16.

The transistor power dissipation is

\[
P_T = V_{CE} I_C = (V_{CC} - I_C R_L) I_C = V_{CC} I_C - I_C^2 R_L
\]

The current at which the maximum power occurs is found by setting the derivative of this equation equal to zero as follows:

\[
\frac{dP_T}{dI_C} = 0 = V_{CC} - 2I_C R_L
\]

which yields

\[
I_C = \frac{V_{CC}}{2R_L} = \frac{35}{2(10)} = 1.75 \, \text{A}
\]

!Figure 15.15

**Figure 15.15** | Bipolar common-emitter circuit.

!Figure 15.16

**Figure 15.16** | Load line and maximum power curve for Example 15.1.

### Graph Description

- **\( I_C \) (A)**: Y-axis ranging from 0 to 4.
- **\( V_{CE} \) (V)**: X-axis ranging from 0 to 40.
- **Load Line**: Starts at \( I_C (\text{max}) = 3.5 \, \text{A} \) and intersects \( V_{CE} = 35 \, \text{V} \).
- **Maximum Power Point**: Occurs at \( I_C = 1.75 \, \text{A} \).

# Chapter 15: Semiconductor Microwave and Power Devices

The collector–emitter voltage at this maximum power point is

\[
V_{CE} = V_{CC} - I_C R_C = 35 - (1.75)(10) = 17.5 \, \text{V}
\]

The maximum power dissipated in the transistor occurs at the center of the load line. The maximum transistor power dissipation is therefore

\[
P_T = V_{CE} I_C = (17.5)(1.75) = 30.6 \, \text{W}
\]

**Comment**

To find a transistor for a given application, safety factors are normally used. For this example, a transistor with a current rating greater than 3.5 A, a voltage rating greater than 35 V, and a power rating greater than 30.6 W would be required for the application just described.

**Exercise Problem**

**Ex 15.1** Assume the BJT in the common-emitter circuit shown in Figure 15.15 has limiting factors of \( I_{C, \text{max}} = 5 \, \text{A}, V_{CE, \text{max}} = 75 \, \text{V}, \) and \( P_T = 30 \, \text{W} \). Neglecting second breakdown effects, determine the minimum value of \( R_C \) such that the Q-point of the transistor always stays within the safe operating area for (a) \( V_{CC} = 60 \, \text{V}, \) (b) \( V_{CC} = 40 \, \text{V}, \) and (c) \( V_{CC} = 20 \, \text{V} \). In each case, determine the maximum collector current and maximum transistor power dissipation.

\[
\begin{align*}
\text{[M} & \, \text{C} = (\exists u)(d \, \forall s = (\exists u))^u \, \forall s = \gamma y' (s) \land \text{C} = (\exists u)(d \, \forall s = (\exists u))^u \\
& \, \forall s = \gamma y' (q) : \land \, 0 \text{E} = (\exists u)(d \, \forall x = (\exists u))^u \, 0 \text{E} = \gamma y' (q) \, \text{SU} \, \text{V}]
\end{align*}
\]

## 15.4.3 Darlington Pair Configuration

As mentioned, the base width of a power BJT is relatively wide so that the current gain is then relatively small. One method that is used to increase the effective current gain is to use a Darlington pair such as shown in Figure 15.17. Considering the currents, we see that

\[
i_C = i_{CA} + i_{CB} = \beta_A i_B + \beta_B i_{EA} = \beta_A i_B + \beta_B (\beta_A i_B) = \beta_A i_B (1 + \beta_B) = \beta i_B
\]

\[
(15.7)
\]

!Figure 15.17

**Figure 15.17** | An npn Darlington pair configuration.

# Power Bipolar Transistors

!Figure 15.18  
**Figure 15.18** | An integrated circuit implementation of the npn Darlington pair configuration.

The overall common-emitter current gain is then

\[
\frac{i_c}{i_b} = \beta_A \beta_B + \beta_A + \beta_B
\]

(15.8)

Thus, if the gain of each individual transistor is \(\beta_A = \beta_B = 15\), then the overall gain of the Darlington pair is \(\frac{i_c}{i_b} = 255\). This overall gain is then substantially larger than that of the individual device. A diode may be incorporated as shown in Figure 15.17 to aid in turning off the transistor \(Q_A\). A reverse current out of the base of \(Q_B\) through the diode will pull charge out of the base of this transistor and turn the device off faster than when no diode is used.

The Darlington pair shown in Figure 15.17 is typically used in the output stage of a power amplifier when an npn bipolar transistor is required. A pnp Darlington pair may also be used to increase the effective current gain of a power pnp device.

The integrated circuit configuration of the npn Darlington pair may be as shown in Figure 15.18. The silicon dioxide that is shown completely penetrates through the p-type base region so that the base regions of the two transistors are isolated.

----

## TEST YOUR UNDERSTANDING

**TYU 15.1**  
Consider the vertical power silicon BJT shown in Figure 15.10. Assume that a reverse-biased voltage of 200 V is applied to the base–collector junction. Calculate the space charge width that extends into the (a) collector region and (b) base region.  
(\( \epsilon_r = 11.7 \), \( q = 1.6 \times 10^{-19} \, \text{C} \), \( \epsilon_0 = 8.85 \times 10^{-12} \, \text{F/m} \))

**TYU 15.2**  
For the emitter–follower circuit in Figure 15.19, the parameters are \( V_{CC} = 10 \, \text{V} \) and \( R_E = 200 \, \Omega \). The transistor current gain is \(\beta = 150\), and the current and voltage limitations are \( I_{C,\text{max}} = 200 \, \text{mA} \) and \( V_{CE,\text{sat}} = 5 \, \text{V} \). Determine the minimum transistor power rating such that the transistor Q-point is always inside the safe operating area.  
(\( \lambda = 0 \), \( \epsilon = 11.7 \), \( q = 1.6 \times 10^{-19} \, \text{C} \))

!Figure 15.19  
**Figure 15.19** | Figure for Exercise TYU 15.2.

# 15.5 Power MOSFETs

The basic operation of the power MOSFET is the same as that of any MOSFET. However, the current handling capability of these devices is usually in the ampere range, and the drain-to-source blocking voltage may be in the range of 50 to 100 volts or even higher. One big advantage that a power MOSFET has over a bipolar power device is that the control signal is applied to the gate whose input impedance is extremely large. Even during switching between on and off states, the gate current is small, so that relatively large currents can be switched with very small control currents.

## 15.5.1 Power Transistor Structures

Large currents can be obtained in a MOSFET with a very large channel width. To achieve a large channel width device with good characteristics, power MOSFETs are fabricated with a repetitive pattern of small cells operating in parallel. To achieve a large blocking voltage, a vertical structure is used. There are two basic power MOSFET structures. The first is called a DMOS device and is shown in Figure 15.20. The DMOS device uses a double diffusion process: The p-base or the p-substrate region and the n⁺ source contact are diffused through a common window defined by the edge of the gate. The p-base region is diffused deeper than the n⁺ source, and the difference in the lateral diffusion distance between the p-base and the n⁺ source defines the surface channel length.

Electrons enter the source terminal and flow laterally through the inversion layer under the gate to the n-drift region. The electrons then flow vertically through the n-drift region to the drain terminal. The conventional current direction is from the drain to the source. The n-drift region must be moderately doped so that the drain breakdown voltage is sufficiently large. However, the thickness of the n-drift region should also be as thin as possible to minimize drain resistance.

The second power MOSFET structure, shown in Figure 15.21, is a VMOS structure. The vertical channel or VMOS power device is a nonplanar structure that requires a different type of fabrication process. In this case, a p-base or p⁺-substrate diffusion

!Figure 15.20  
**Figure 15.20** | Cross section of a double-diffused MOS (DMOS) transistor.

!Figure 15.21  
**Figure 15.21** | Cross section of a vertical channel MOS (VMOS) transistor.

```markdown
!Figure 15.22  
**Figure 15.22 | A HEXFET structure.**

is performed over the entire surface followed by the n\(^+\) source diffusion. A V-shaped groove is then formed, extending through the n-drift region. It has been found that certain chemical solutions etch the (111) planes in silicon at a much slower rate than the other planes. If (100) oriented silicon is etched through a window at the surface, these chemical etches will create a V-shaped groove. A gate oxide is then grown in the V-shaped groove and the metal gate material is deposited. An electron inversion layer is formed in the base or substrate so that current is again essentially a vertical current between the source and the drain. The relatively low-doped n-drift region supports the drain voltage since the depletion region extends mainly into this low-doped region.

We mentioned that many individual MOSFET cells are connected in parallel to fabricate a power MOSFET with the proper width-to-length ratio. Figure 15.22 shows a HEXFET structure. Each cell is a DMOS device with an n\(^+\) polysilicon gate. The HEXFET has a very high packing density—it may be on the order of 10\(^5\) cells per cm\(^2\). In the VMOS structure, the anisotropic etching of the grooves must be along the [110] direction on the (100) surface. This constraint limits the design options available for this type of device.

### 15.5.2 Power MOSFET Characteristics

Table 15.2 lists the basic parameters of two n-channel power MOSFETs. The drain currents are in the ampere range and the breakdown voltages are in the hundreds of volts range.

An important parameter of a power MOSFET is the **on resistance**, which can be written as

\[
R_{\text{on}} = R_S + R_{\text{CH}} + R_D \tag{15.9}
\]

where \( R_S \) is the resistance associated with the source contact, \( R_{\text{CH}} \) is the channel resistance, and \( R_D \) is the resistance associated with the drain contact. The \( R_S \) and \( R_D \)
```

# Chapter 15: Semiconductor Microwave and Power Devices

## Table 15.2 Characteristics of Two Power MOSFETs

| Parameter                  | 2N6757 | 2N6792 |
|----------------------------|--------|--------|
| \( V_{DS}(\text{max}) \) (V) | 150    | 400    |
| \( I_D(\text{max}) \) (at \( T = 25^\circ C \)) | 8      | 2      |
| \( P_D \) (W)              | 75     | 20     |

Resistance values are not necessarily negligible in power MOSFETs since small resistances and high currents can produce considerable power dissipation.

In the linear region of operation, we may write the channel resistance as:

\[
R_{CH} = \frac{L}{W \mu_n C_{ox} (V_{GS} - V_T)}
\]

We have noted in previous chapters that mobility decreases with increasing temperature. The threshold voltage varies only slightly with temperature so that, as current in a device increases and produces additional power dissipation, the temperature of the device increases, the carrier mobility decreases, and \( R_{CH} \) increases, which inherently limits the channel current. The resistances \( R_S \) and \( R_D \) are proportional to semiconductor resistivity and so are also inversely proportional to mobility and have the same temperature characteristics as \( R_{CH} \). Figure 15.23 shows a typical “on-resistance” characteristic as a function of drain current.

The increase in resistance with temperature provides stability for the power MOSFET. If the current in any particular cell begins to increase, the resulting temperature rise will increase the resistance, thus limiting the current. With this particular characteristic, the total current in a power MOSFET tends to be evenly distributed among the parallel cells, not concentrated in any single cell, a condition that can cause burnout.

!Figure 15.23

**Figure 15.23** Typical drain-to-source resistance versus drain current characteristics of a MOSFET.

- \( R_{DS(\text{on})} \) measured with current pulse of 2.0-μs duration, initial \( T_j = 25^\circ C \) (heating effect of 2.0-μs pulse is minimal)
- \( V_{GS} = 10 \, \text{V} \)
- \( V_{GS} = 20 \, \text{V} \)

The graph shows \( R_{DS(\text{on})} \) as a function of \( I_D \), drain current (A), with resistance (Ω) on the y-axis.

# 15.5 Power MOSFETs

!Graphs

**Figure 15.24** | Typical characteristics for high-power MOSFETs at various temperatures: (a) transconductance versus drain current; (b) drain current versus gate-to-source voltage.

## Description

Power MOSFETs differ from bipolar power transistors in both operating principles and performance. The superior performance characteristics of power MOSFETs are faster switching times, no second breakdown, and stable gain and response time over a wide temperature range. Figure 15.24a shows the transconductance of the 2N6757 versus temperature. The variation with temperature of the MOSFET transconductance is less than the variation in the BJT current gain that is shown in Figure 15.12. Figure 15.24b is a plot of drain current versus gate-to-source voltage at three different temperatures. We may note that at high current, the current decreases with temperature at a constant gate-to-source voltage, providing the stability that has been discussed.

Power MOSFETs must operate in a SOA. As with power BJTs, the SOA is defined by three factors: the maximum drain current, \(I_{D,\text{max}}\), rated breakdown voltage, \(BV_{\text{DSS}}\), and the maximum power dissipation given by \(P_T = V_{DS}I_D\). The SOA is shown in Figure 15.25a in which the current and voltage are plotted on linear scales.

!Graphs

**Figure 15.25** | The safe operating area (SOA) of a MOSFET plotted on (a) linear scales and (b) logarithmic scales.

### Equations

- Maximum power dissipation: \( P_T = V_{DS}I_D \)

### Graphs

- **Figure 15.24a**: Transconductance vs. Drain Current
  - \( V_{DS} = 15 \, \text{V} \)
  - \( T_J = -55^\circ \text{C}, 25^\circ \text{C}, 125^\circ \text{C} \)
  - 80-\(\mu\)s pulse test

- **Figure 15.24b**: Drain Current vs. Gate-to-Source Voltage
  - \( V_{DS} = 15 \, \text{V} \)
  - \( T_J = -55^\circ \text{C}, 25^\circ \text{C}, 125^\circ \text{C} \)
  - 80-\(\mu\)s pulse test

- **Figure 15.25a**: SOA on Linear Scales
  - \( I_D \) vs. \( V_{DS} \)
  - \( I_{D,\text{max}} \), \( BV_{\text{DSS}} \), \( P_T \)

- **Figure 15.25b**: SOA on Logarithmic Scales
  - \( \log I_D \) vs. \( \log V_{DS} \)
  - \( \log I_{D,\text{max}} \), \( \log BV_{\text{DSS}} \), \( P_T \)

```markdown
# CHAPTER 15 Semiconductor Microwave and Power Devices

**EXAMPLE 15.2**

**Objective:** Find the optimum drain resistor in a MOSFET inverter circuit.

A MOSFET inverter circuit is shown in Figure 15.26. Two different MOSFETs are being considered for use in the circuit. The parameters for devices A and B are given.

| Device A       | Device B       |
|----------------|----------------|
| \( BV_{\text{DSS}} = 35 \, \text{V} \) | \( BV_{\text{DSS}} = 35 \, \text{V} \) |
| \( P_T = 30 \, \text{W} \)     | \( P_T = 30 \, \text{W} \)     |
| \( I_{D, \text{max}} = 6 \, \text{A} \) | \( I_{D, \text{max}} = 4 \, \text{A} \) |

**Solution**

The SOA curves for the two devices are shown in Figure 15.27.

The load line for the inverter circuit using device A is shown as curve A. The load line intersects the voltage axis at \( V_{DD} = 24 \, \text{V} \). This curve is tangent to the maximum power curve and intersects the current axis at \( I_D = 5 \, \text{A} \). Note that, if we had wanted the load line to intersect the maximum rated current of \( I_{D, \text{max}} = 6 \, \text{A} \), the load line would have gone outside of the SOA.

For the load line A, the drain resistance is

\[
R_D = \frac{V_{DD}}{I_D} = \frac{24}{5} = 4.8 \, \Omega
\]

!Figure 15.26 | A MOSFET inverter circuit.

!Figure 15.27 | Safe operating area and load lines for devices in Example 15.2.

- **Figure 15.27** shows the safe operating area and load lines for devices in Example 15.2.
- The graph plots \( I_D \) (A) against \( V_{DS} \) (V).
- Key points include:
  - \( I_{D, \text{max}} = 6 \, \text{A} \)
  - \( I_{D, \text{max}} = 4 \, \text{A} \)
  - Maximum power dissipated
  - \( V_{DD} = 24 \, \text{V} \)
  - \( BV_{\text{DSS}} = 35 \, \text{V} \)
```

## 15.5 Power MOSFETs

The current at the maximum power point (using the results from Example 15.1) is

\[
I_D = \frac{V_{DD}}{2R_D} = \frac{24}{2(4.8)} = 2.5 \, \text{A}
\]

and the corresponding drain-to-source voltage is

\[
V_{DS} = V_{DD} - I_D R_D = 24 - (2.5)(4.8) = 12 \, \text{V}
\]

The maximum power that may be dissipated in the transistor is \( P = V_{DS} I_D = (12)(2.5) = 30 \, \text{W} = P_T \), which corresponds to the maximum rated power. This point is shown on the curve.

The load line for the inverter circuit using device B is shown as curve B. The load line intersects the voltage axis at \( V_{DD} = 24 \, \text{V} \) as before. This curve can now intersect the current axis at the maximum rated drain current of \( I_{D, \text{max}} = 4 \, \text{A} \). We see that the load line falls within the SOA of the transistor.

For load line B, the drain resistance is

\[
R_D = \frac{V_{DD}}{I_D} = \frac{24}{4} = 6 \, \Omega
\]

The current at the maximum power point is

\[
I_D = \frac{V_{DD}}{2R_D} = \frac{24}{2(6)} = 2 \, \text{A}
\]

and the corresponding drain-to-source voltage is

\[
V_{DS} = V_{DD} - I_D R_D = 24 - (2)(6) = 12 \, \text{V}
\]

The maximum power that may be dissipated in the transistor is \( P = V_{DS} I_D = (12)(2) = 24 \, \text{W} \), which is less than the maximum rated power. This point is also shown on the curve.

### Conclusion

We see that if device A is used, the drain resistor is determined by the maximum power. However, if device B is used, the drain resistor is determined by the maximum rated current of the device.

### EXERCISE PROBLEM

**Ex 15.2** Consider the common-source circuit shown in Figure 15.26. Determine the required current, voltage, and power ratings of the MOSFET for (a) \( R_D = 12 \, \Omega \), \( V_{DD} = 24 \, \text{V} \) and (b) \( R_D = 8 \, \Omega \), \( V_{DD} = 40 \, \text{V} \).

\[
\forall \, \xi = -\frac{wq}{\lambda} \, \alpha \, \theta = \frac{5qAQ}{\alpha} \, \theta : \lambda \, \Xi = \lambda \, \theta \, \forall \, \xi = -\frac{wq}{\lambda} \, \alpha \, \theta \, \Xi = \frac{5qAQ}{\alpha} \, \theta \, \text{SVU}
\]

## 15.5.3 Parasitic BJT

The MOSFET has a parasitic BJT as an inherent part of its structure. The parasitic BJT may be seen in both the DMOS and VMOS structures shown in Figures 15.20 and 15.21. The source terminal corresponds to the n-type emitter, the p-type base or substrate region corresponds to the p-type base, and the n-type drain corresponds to the n-type collector. This is also shown schematically in Figure 15.28. The channel length of the MOSFET corresponds to the base width of the parasitic BJT. Since this length is normally quite small, the current gain \(\beta\) of the BJT can be larger than unity.

# CHAPTER 15 Semiconductor Microwave and Power Devices

!Figure 15.28

**Figure 15.28** (a) Cross section of vertical MOSFET showing parasitic BJT and distributed resistance; (b) equivalent circuit of MOSFET and parasitic BJT with distributed parameters.

The BJT should be cutoff at all times, which means the source-to-body voltage (emitter-to-base voltage) should be as close to zero as possible. We see from the geometries of Figures 15.20 and 15.21 that the source ohmic contact also goes across the p-type body region so that this junction voltage is zero during steady-state operation of the transistor. However, the BJT may be turned on during high-speed switching of the MOSFET.

Figure 15.28b shows that the base and the collector of the parasitic BJT are connected by the gate-to-drain capacitance. A parasitic or distributed resistance also connects the base to the emitter of the BJT. When the MOSFET is being turned off, the drain-to-source voltage increases and induces a current in the gate-to-drain capacitance in the direction from the parasitic collector terminal to the parasitic base terminal. This induced current may be large enough to induce a voltage in the parasitic resistance that is sufficient to forward bias the base–emitter junction and therefore turn the BJT on. The turned on BJT may then induce a large drain current that can cause burnout of the MOSFET. This breakdown mechanism is known as **snapback breakdown** and has been discussed briefly in Section 11.4.1.

# 15.6 The Thyristor

The current–voltage characteristics are shown in Figure 11.22. Devices can be designed to minimize the parasitic or distributed base–emitter resistance to minimize this problem.

## 15.6.1 The Thyristor

One of the important applications of electronic devices is in switching between an off or blocking state to an on or low-impedance state. Thyristor is the name given to a general class of semiconductor pnpn switching devices that exhibit bistable regenerative switching characteristics. We have considered the transistor, which may be switched on with the application of a base drive or a gate voltage. The base drive or gate voltage must be applied as long as the transistor is to remain on. There are a number of applications in which it is useful to have a device remain in a blocking state until switched to the low-impedance state by a control signal, which then does not necessarily have to remain on. These devices are efficient in switching large currents at low frequencies, such as industrial control circuits operating at 60 Hz.

A Semiconductor Controlled Rectifier (SCR) is the common name given to a three-terminal thyristor. The SCR (sometimes referred to as a silicon controlled rectifier) is a four-layer pnpn structure with a gate control terminal. As with most semiconductor devices, there are several variations of the device structure. We consider the basic SCR operation and limitations, and then discuss some variations of the basic four-layer device.

### 15.6.1 The Basic Characteristics

The four-layer pnpn structure is shown in Figure 15.29a. The upper p region is called the anode and the lower n region is called the cathode. If a positive voltage is applied to the anode, the device is said to be forward biased. However, the junction \( J_2 \) is reverse biased so that only a very small current exists. If a negative voltage is applied to the anode, then junctions \( J_1 \) and \( J_3 \) are reverse biased—again only a very small current will exist. Figure 15.29b shows the I–V characteristics for these conditions. The voltage \( V_B \) is the breakdown voltage of the \( J_2 \) junction. For properly designed devices, the blocking voltage can be several thousand volts.

To consider the characteristics of the device as it goes into its conducting state, we can model the structure as coupled npn and pnp bipolar transistors. Figure 15.30a shows how we can split the four-layer structure and Figure 15.30b shows the two-transistor equivalent circuit with the associated currents. Since the base of the pnp device is the same as the collector of the npn transistor, the base current \( I_{B1} \) must in fact be the same as the collector current \( I_{C2} \). Similarly, since the collector of the pnp transistor is the same as the base of the npn device, the collector current \( I_{C1} \) must be the same as the base current \( I_{B2} \). In this bias configuration, the B–C of the pnp and the B–C of the npn devices are reverse biased, while the B–E junctions are both forward biased. The parameters \( \alpha_1 \) and \( \alpha_2 \) are the common base current gains of the pnp and npn transistors, respectively.

# Chapter 15: Semiconductor Microwave and Power Devices

!Figure 15.29

**Figure 15.29**  
(a) The basic four-layer pnpn structure.  
(b) The initial current–voltage characteristic of the pnpn device.

- **(a) Diagram:**
  - Anode (A)
  - Cathode (K)
  - Layers: p, n, p, n
  - Currents: \( I_A \), \( J_1 \), \( J_2 \), \( J_3 \)
  - Voltage: \( V_A \)

- **(b) Graph:**
  - Reverse blocking
  - Forward blocking
  - Voltage: \( V_A \)
  - Voltage point: \( V_p \)

!Figure 15.30

**Figure 15.30**  
(a) The splitting of the basic pnpn structure.  
(b) Two two-transistor equivalent circuit of the four-layer pnpn device.

- **(a) Diagram:**
  - Anode (A)
  - Cathode (K)
  - Layers: p, n, p, n
  - Currents: \( I_A \), \( I_K \)
  - Voltage: \( V_A \)

- **(b) Diagram:**
  - Equivalent circuit
  - Currents: \( I_{B1} = I_{C2} \), \( I_{C1} = I_{B2} \)

We can write:

\[
I_{C1} = \alpha_1 I_A + I_{C01} = I_{B2} \tag{15.11a}
\]

and

\[
I_{C2} = \alpha_2 I_K + I_{C02} = I_{B1} \tag{15.11b}
\]

# 15.6 The Thyristor

where \( I_{C01} \) and \( I_{C02} \) are the reverse B–C junction saturation currents in the two devices. In this particular configuration, \( I_A = I_K \) and \( I_{C1} + I_{C2} = I_A \). If we add Equations (15.11a) and (15.11b), we obtain

\[
I_{C1} + I_{C2} = I_A = (\alpha_1 + \alpha_2) I_A + I_{C01} + I_{C02}
\]

The anode current \( I_A \), from Equation (15.12), can be found as

\[
I_A = \frac{I_{C01} + I_{C02}}{1 - (\alpha_1 + \alpha_2)}
\]

As long as \( \alpha_1 + \alpha_2 \) is much smaller than unity, the anode current is small, as we have indicated in Figure 15.29b.

The common base current gains, \( \alpha_1 \) and \( \alpha_2 \), are very strong functions of collector current as we discussed in Chapter 12. For small values of \( V_A \), the collector current in each device is just the reverse saturation current, which is very small. The small collector current implies that both \( \alpha_1 \) and \( \alpha_2 \) are much smaller than unity. The four-layer structure maintains this blocking condition until the junction \( J_2 \) starts into breakdown or until a current is induced in the \( J_2 \) junction by some external means.

Consider, initially, the condition when the applied anode voltage is sufficiently large to cause the \( J_2 \) junction to start into avalanche breakdown. This effect is shown in Figure 15.31a. The electrons generated by impact ionization are swept into the \( n_1 \) region, making the \( n_1 \) region more negative, and the holes generated by impact ionization are swept into the \( p_2 \) region, making the \( p_2 \) region more positive. The more negative voltage of the \( n_1 \) region and the more positive voltage of the \( p_2 \) region means that the forward-bias junction voltages \( V_1 \) and \( V_3 \) both increase. The increase in the respective B–E junction voltages causes an increase in current, which results in an increase in the common-base current gains \( \alpha_1 \) and \( \alpha_2 \), causing a further increase in

!Figure 15.31

**Figure 15.31** (a) The pnpn device when the \( J_2 \) junction starts into avalanche breakdown. (b) The junction voltages in the pnpn structure when the device is in the high-current, low-impedance state.

|   |   |   |   |   |   |
|---|---|---|---|---|---|
| + \( V_1 \) | - |   | + \( V_3 \) | - |   |
| \( p_1 \) | \( n_1 \) | \( p_2 \) | \( n_2 \) |   |   |
| \( J_1 \) | \( J_2 \) | \( J_3 \) |   |   |   |
| \( V_A \) |   |   |   |   |   |
| \( I_A \) |   |   |   |   |   |

(a)

|   |   |   |   |   |   |
|---|---|---|---|---|---|
| + \( V_1 \) | - | - \( V_2 \) | + | + \( V_3 \) | - |
| \( p_1 \) | \( n_1 \) | \( p_2 \) | \( n_2 \) |   |   |
| \( J_1 \) | \( J_2 \) | \( J_3 \) |   |   |   |
| \( V_A \) |   |   |   |   |   |
| \( I_A \) |   |   |   |   |   |

(b)

# Semiconductor Microwave and Power Devices

!Figure 15.32  
**Figure 15.32** | The current–voltage characteristics of the pnpn device.

!Figure 15.33  
**Figure 15.33** (a) The three-terminal SCR. (b) The two-transistor equivalent circuit of the three-terminal SCR.

----

### Explanation

- **Forward Conducting**: The graph shows the transition from reverse blocking to forward conducting.
- **Reverse Blocking**: The device blocks current in the reverse direction.

### Equations

- **Equation (15.13)**: Describes the regenerative positive feedback situation, leading to a rapid increase in \( I_A \).

### Description

As the anode current \( I_A \) increases and \( \alpha_1 + \alpha_2 \) increases, the two equivalent bipolar transistors are driven into saturation, and the junction \( J_2 \) becomes forward biased. The total voltage across the device decreases and is approximately equal to one diode drop. The current in the device is limited by the external circuit. If the current is allowed to increase, ohmic losses may become important, slightly increasing the voltage drop across the device with current. The \( I_A \) versus \( V_A \) characteristic is shown in Figure 15.32.

## 15.6.2 Triggering the SCR

In the last section, we considered the case when the four-layer pnpn device is turned on by the avalanche breakdown process in the center junction. The turn-on condition can also be initiated by other means. Figure 15.33a shows three-terminal SCR in which the third terminal is the gate control. We can determine the effect of the gate current by reconsidering Equations (15.11a) and (15.11b).

Figure 15.33b again shows the two-transistor equivalent circuit including the gate current. We can write:

\[
I_{C1} = \alpha_1 I_A + I_{CO1} \tag{15.14a}
\]

and

\[
I_{C2} = \alpha_2 I_K + I_{CO2} \tag{15.14b}
\]

We now have \( I_K = I_A + I_G \), and we can still write \( I_{C1} + I_{C2} = I_A \). Adding Equations (15.14a) and (15.14b), we find that

\[
I_{C1} + I_{C2} = I_A = (\alpha_1 + \alpha_2) I_A + \alpha_2 I_G + I_{CO1} + I_{CO2} \tag{15.15}
\]

Solving for \( I_a \), we find

\[
I_a = \frac{\alpha_2 I_c + (I_{co1} + I_{co2})}{1 - (\alpha_1 + \alpha_2)}
\]

(15.16)

We can think of the gate current as the flow of holes into the \( p_2 \) region. The additional holes increase the potential of this region, which increases the forward-biased B–E voltage of the npn bipolar transistor, and the transistor action. The transistor action of the npn increases the collector current \( I_{c2} \), which starts the transistor action of the pnp bipolar transistor, and the entire pnnp device can be turned on into its low-impedance state. The gate current required to switch the SCR into its on condition is typically in the milliamp range. SCR can be turned on with a small gate current, which can control hundreds of amperes of anode current. The gate current can be turned off and the SCR will remain in its conducting state. The gate loses control of the device once the SCR is triggered into its conducting state. The current–voltage characteristics of the SCR as a function of gate current is shown in Figure 15.34.

A simple application of an SCR in a half-wave control circuit is shown in Figure 15.35a. The input signal is an ac voltage and a trigger pulse will control the turn-on of the SCR. We assume that the trigger pulse occurs at time \( t_1 \) during the ac voltage cycle. Prior to \( t_1 \), the SCR is off so that the current in the load is zero; thus, there is a zero output voltage. At \( t = t_1 \), the SCR is triggered on and the input voltage appears across the load (neglecting the voltage drop across the SCR). The SCR turns off when the anode-to-cathode voltage becomes zero even though the trigger pulse has been turned off prior to this time. The time at which the SCR is triggered during the voltage cycle can be varied, changing the amount of power delivered to the load. Full-wave control circuits can be designed to increase efficiency and degree of control.

The gate allows control of the turn-on of the SCR. However, the four-layer pnnp structure can also be triggered on by other means. In many integrated circuits, parasitic pnnp structures exist. One such example is the CMOS structure that we considered in Chapter 10. A transient ionizing radiation pulse can trigger the parasitic structure.

!Figure 15.34 | Current–voltage characteristics of an SCR.

**Figure 15.34 | Current–voltage characteristics of an SCR.**

| \( I_a \) | \( I_a = 0 \) | \( I_{g1} \) | \( I_{g2} \) | \( I_{g3} \) | \( I_{g4} = 0 \) |
|-----------|--------------|--------------|--------------|--------------|------------------|
|           | \( I_{g1} \) | \( I_{g2} \) | \( I_{g3} \) |              | \( V_{AK} \)     |
|           | \( 0 < I_{g1} < I_{g2} < I_{g3} \) | | | | |


# CHAPTER 15 Semiconductor Microwave and Power Devices

!SCR Circuit Diagram

**Figure 15.35** (a) Simple SCR circuit. (b) Input AC voltage signal and trigger pulse. (c) Output voltage versus time.

A four-layer device by generating electron–hole pairs, particularly in the \( J_2 \) junction, producing a photocurrent. The photocurrent is equivalent to a gate current in an SCR so the parasitic device can be switched into its conducting state. Again, once the device is switched on, it will remain in its conducting state even when the radiation ceases. An optical signal can also trigger the device in the same manner by generating electron–hole pairs.

Another triggering mechanism in the npnp device is \( dV/dt \) triggering. If the forward-bias anode voltage is applied rapidly, the voltage across the \( J_2 \) junction will also change quickly. This changing reverse-biased \( J_2 \) junction voltage means that the space charge region width is increasing; thus, electrons are being removed from the \( n_1 \) side of the junction and holes are being removed from the \( p_2 \) side of the junction.

```markdown
### 15.6 The Thyristor

If \( dV/dt \) is large, the rate of removal of these carriers is rapid, which leads to a large transient current that is equivalent to a gate current and can trigger the device into a low-impedance conducting state. In SCR devices, a \( dV/dt \) rating is usually specified. However, in parasitic pnpn structures, the \( dV/dt \) triggering mechanism is a potential problem.

#### 15.6.3 SCR Turn-Off

Switching the four-layer pnpn structure from its conducting state to its blocking state can be accomplished if the current \( I_A \) is reduced below the value creating the \( \alpha_1 + \alpha_2 = 1 \) condition. This critical \( I_A \) current is called the holding current. If a parasitic four-layer structure is triggered into the conducting state, the effective anode current in the device must be reduced below the corresponding holding current in order to turn off the device. This requirement essentially implies that all power supplies must be turned off in order to bring the parasitic device back into its blocking state.

The SCR can be triggered on by supplying holes to the \( p_2 \) region of the device. The SCR can perhaps be turned off by removing holes from this same region. If the reverse gate current is large enough to bring the npn bipolar transistor out of saturation, then the SCR can be switched from the conducting state into the blocking state. However, the lateral dimensions of the device may be large enough so that nonuniform biasing in the \( J_2 \) and \( J_3 \) junctions occurs during a negative gate current and the device will remain in the low-impedance conducting state. The four-layer pnpn device must be specifically designed for a turn-off capability.

#### 15.6.4 Device Structures

Many thyristor structures have been fabricated with specific characteristics for specific applications. We consider a few of these types of device to gain an appreciation for the variety of structures.

**Basic SCR**  
There are many variations of diffusion, implantation, and epitaxial growth that can be used in the fabrication of the SCR device. The basic structure is shown in Figure 15.36. The \( p_1 \) and \( p_2 \) regions are diffused into a fairly high resistivity \( n \) material. The \( n^+ \) cathode is formed and the \( p^+ \) gate contact is made. High thermal conductivity materials can be used for the anode and cathode ohmic contacts to aid in heat dissipation for high-power devices. The \( n_i \) region width may be on the order of 250 μm in order to support very large reverse-biased voltages across the \( J_2 \) junction. The \( p_1 \) and \( p_2 \) regions may be on the order of 75 μm wide, while the \( n^+ \) and \( p^+ \) regions are normally quite thin.

**Bilateral Thyristor**  
Since thyristors are often used in ac power applications, it may be useful to have a device that switches symmetrically in the positive and negative cycles of the ac voltage. There are a number of such devices, but the basic concept is to connect two conventional thyristors in antiparallel as shown in Figure 15.37a. The integration of this concept into a single device is shown in Figure 15.37b. Symmetrical \( n \) regions can be diffused into a pnp structure. Figure 15.37c shows the current–voltage characteristics in which the triggering into the conduction mode
```

# Semiconductor Microwave and Power Devices

## Figure Descriptions

### Figure 15.36
The basic SCR device structure.

### Figure 15.37
- **(a)** The antiparallel connection of two thyristors to form a bilateral device.
- **(b)** The bilateral thyristor as an integrated device.
- **(c)** The current–voltage characteristics of the bilateral thyristor. *(From Ghandhi [7].)*

## Text

Would be due to breakdown triggering. The two terminals alternately share the role of anode and cathode during successive half cycles of the AC voltage.

Triggering by a gate control is more complex for this device since a single gate region must serve for both of the antiparallel thyristors. One such device is known as a triac. Figure 15.38a shows the cross section of such a device. This device can be triggered into conduction by gate signals of either polarity and with anode-to-cathode voltages of either polarity.

# 15.6 The Thyristor

!Figure 15.38

**Figure 15.38** (a) The triac device. (b) The triac with a specific bias configuration. (From Ghandhi [7].)

!Figure 15.39

**Figure 15.39** The current–voltage characteristics of the triac.

One particular gate control situation is shown in Figure 15.38b. Terminal 1 is positive with respect to terminal 2, and a negative gate voltage is applied with respect to terminal 1, so the gate current is negative. This polarity arrangement induces the current \( I_1 \) and the junction \( J_4 \) becomes forward biased. Electrons are injected from \( n_3 \), diffuse across \( p_2 \), and are collected in the \( n_1 \) region. In this case \( n_3p_2n_1 \) behaves like a saturated transistor. The collected electrons in \( n_1 \) lower the potential of \( n_1 \) with respect to \( p_2 \). The current across the \( p_2n_1 \) junction increases, which can trigger the \( p_2n_1p_1n_3 \) thyristor into its conducting mode.

We can show that the other combinations of gate, anode, and cathode voltages will also trigger the triac into conduction. Figure 15.39 shows the terminal characteristics.

# CHAPTER 15 Semiconductor Microwave and Power Devices

!Figure 15.40  
**Figure 15.40** The V groove MOS gated thyristor.  
*(From Baliga [11])*

## MOS Gated Thyristor

The operation of a MOS gated thyristor is based upon controlling the gain of the npn bipolar transistor. Figure 15.40 shows a V-groove MOS gated thyristor. The MOS gate structure must extend into the n-drift region. If the gate voltage is zero, the depletion edge in the p-base remains essentially flat and parallel to the junction \( J_2 \); the gain of the npn transistor is low. This effect is shown in the figure by the dashed line. When a positive gate voltage is applied, the surface of the p base becomes depleted—the depletion region in the p base adjacent to the gate is shown by the dotted line. The undepleted base width \( W_b \) of the npn bipolar device narrows and the gain of the device increases.

At a gate voltage approximately equal to the threshold voltage, electrons from the n\(^+\) emitter are injected through the depletion region into the n-drift region. The potential of the n-drift region is lowered, which further forward biases the p\(^+\) anode to n-drift junction voltage, and the regenerative process is initiated. The gate voltage required to initiate turn-on is approximately the threshold voltage of the MOS device. One advantage of this device is that the input impedance to the control terminal is very high; relatively large currents can be switched with very small capacity coupled gate currents.

## MOS Turn-Off Thyristor

The MOS turn-off thyristor can both turn on and turn off the anode current by applying a signal to a MOS gate terminal. The basic device structure is shown in Figure 15.41. By applying a positive gate voltage, the n\(^+\)p bipolar transistor can be turned on as just discussed. Once the thyristor is turned on, the device can be turned off by applying a negative gate voltage: the negative gate voltage turns on the p-channel MOS transistor that effectively short circuits the B–E junction of the n\(^+\)pn bipolar transistor. Holes that now enter the p-base have an alternative path to the cathode. If the resistance of the p-channel MOSFET becomes low enough, all current will be diverted away from the n\(^+\)p emitter and the n\(^+\)pn device will effectively be turned off.

# 15.7 Summary

!Figure 15.41

**Figure 15.41** (a) The MOS turn-off thyristor. (b) Equivalent circuit for the MOS turn-off thyristor.  
*(From Baliga [11].)*

## 15.7 | SUMMARY

- The concept of a negative differential resistance in the \( I\text{-}V \) characteristic of the tunnel diode is used in the design of a microwave tunnel diode oscillator. The expression for the maximum resistance cutoff frequency is derived.

- The operation of a microwave GUNN diode oscillator is based on the concept of negative differential mobility.

- The IMPATT diode oscillator uses injection and drift time delays to create a region of differential negative resistance.

- The power BJT has a vertical configuration and an interdigitated base–emitter surface structure. The collector drift region (doping and width) determines the rated blocking voltage of the BJT, while the base width must be sufficiently wide to avoid punch-through breakdown at the rated blocking voltage.

- A power BJT is characterized by the maximum rated collector current, maximum rated voltage, and maximum rated power dissipation. These three parameters define the SOA of the transistor.

- A power MOSFET has a vertical configuration and an interdigitated gate–source surface structure. Two specific devices considered are the DMOS and VMOS structures. The drain-drift region (doping and width) determines the rated blocking voltage of the MOSFET, while the channel length of the base (body) must be sufficiently wide to avoid punch-through breakdown at the rated blocking voltage.

- A power MOSFET is characterized by the maximum rated drain current, maximum rated voltage, and maximum rated power dissipation. These three parameters define the SOA of the transistor.

# CHAPTER 15: Semiconductor Microwave and Power Devices

- The "on resistance" of a MOSFET has a positive temperature coefficient so that the power MOSFET is more stable versus temperature than a power BJT. This characteristic allows MOSFETs to be fabricated in parallel to increase the current capability of the device.
- The thyristor refers to a general class of pnpn switching devices that can be switched between a high-impedance, low-current state and a low-impedance, high-current state. These devices exhibit a bistable regenerative positive feedback switching characteristic.
- The basic pnpn device can be modeled as coupled npn and pnp bipolar transistors. In the "on" state, both bipolar transistors are driven into saturation, creating the high-current, low-voltage condition. In the "off" or blocking state, large voltages can be applied to the device and the current is essentially zero.
- The turn-on characteristics of the thyristor can be controlled through a gate control terminal. The three-terminal thyristors are referred to as semiconductor controlled rectifiers (SCRs).

## GLOSSARY OF IMPORTANT TERMS

**double-diffused MOSFET (DMOS)**  
A power MOSFET in which the source and channel regions are formed using a double diffusion process.

**HEXFET**  
The structure of a power MOSFET in which many individual MOSFETs are placed in parallel in a hexagonal configuration.

**maximum rated current**  
The maximum allowed current in a power transistor such that proper operation is maintained.

**maximum rated power**  
The maximum allowed power dissipation in a power transistor such that no permanent damage is done to the transistor.

**maximum rated voltage**  
The maximum allowed applied voltage to a power transistor such that breakdown is not initiated.

**negative differential mobility**  
A region in the drift velocity versus electric field characteristic of a semiconductor material in which the drift velocity decreases with an increase in the electric field.

**negative differential resistance**  
A region in the I–V characteristic of a device in which the current decreases while the voltage increases.

**on resistance**  
The effective resistance between source and drain of a power MOSFET.

**safe operating area**  
The allowed current–voltage regions of operation for a power transistor bounded by the maximum rated current, maximum rated voltage, and maximum power.

**second breakdown**  
A breakdown effect in a power BJT in which high temperature causes a thermal runaway process.

**SCR (semiconductor controlled rectifier)**  
The common name given to a three-terminal thyristor.

**thyristor**  
The name given to a general class of semiconductor pnpn switching devices exhibiting bistable regenerative switching characteristics.

**transferred-electron effect**  
The phenomenon in which conduction electrons are scattered from a lower energy, high-mobility band into a higher energy, low-mobility band.

**triac**  
The name of a bilateral three-terminal thyristor.

**V-groove MOSFET (VMOS)**  
A power MOSFET in which the channel region is formed along a V-shaped groove formed in the surface of the semiconductor.

# CHECKPOINT

After studying this chapter, the reader should have the ability to:

- Explain how a region of negative differential resistance is developed in the \( I\text{-}V \) characteristic of the tunnel diode.
- Discuss the concept of negative differential mobility in GaAs and discuss how this phenomenon leads to the generation of domains in a GUNN diode.
- Discuss the operation of an IMPATT diode oscillator.
- Sketch the cross section of a power BJT and discuss the voltage and current limitations of the device.
- Discuss the reason the current gain of a power BJT is generally smaller than that of a small switching BJT.
- Sketch the safe operating area of a power BJT.
- Describe the reason for and the operation of a Darlington configuration.
- Sketch the cross section of the DMOS and VMOS power MOSFET structures.
- Sketch the safe operating area of a power MOSFET.
- Describe why the “on resistance” of a power MOSFET has a positive temperature coefficient.
- Describe the switching characteristics of a pnpn device.
- Describe the switching characteristics of a semiconductor controlled rectifier.

# REVIEW QUESTIONS

1. Describe how a negative differential resistance region in the \( I\text{-}V \) characteristic of the tunnel diode is generated.
2. Describe how a negative differential mobility region in the drift velocity versus electric field characteristic in GaAs is developed.
3. Describe how a negative differential resistance characteristic is produced in the IMPATT diode.
4. Why is the doping concentration in the collector drift region low and why is the drift region width large in a power BJT?
5. Why does a power BJT have an interdigitated base–emitter structure?
6. Sketch the safe operating area of a power BJT.
7. Discuss how a DMOS structure of a power MOSFET is formed.
8. Discuss the voltage limitation of a power MOSFET.
9. Define the “on resistance” of a power MOSFET and show that the on resistance has a positive temperature coefficient.
10. Discuss how the gate terminal of a semiconductor controlled rectifier can control the switching characteristics.

# PROBLEMS

## Section 15.1 Tunnel Diode

**15.1** Sketch the energy band diagrams of a tunnel diode in which both the n and p regions are degenerately doped for the case of (a) zero bias, (b) \( 0 < V < V_p \), (c) \( V_p < V < V_n \), and (d) \( V > V_n \).

