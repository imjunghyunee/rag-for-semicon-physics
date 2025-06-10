# 13.4 Equivalent Circuit and Frequency Limitations

In order to analyze a transistor circuit, one needs a mathematical model or equivalent circuit of the transistor. One of the most useful models is the small-signal equivalent circuit, which applies to transistors used in linear amplifier circuits. This equivalent circuit will introduce frequency effects in the transistor through the equivalent capacitor–resistor circuits. The various physical factors in the JFET affecting the frequency limitations are considered here and a transistor cutoff frequency, which is a figure of merit, is then defined.

## 13.4.1 Small-Signal Equivalent Circuit

The cross section of an n-channel pn JFET is shown in Figure 13.18, including source and drain series resistances. The substrate may be semi-insulating gallium arsenide or it may be a p+ type substrate.

Figure 13.19 shows a small-signal equivalent circuit for the JFET. The voltage \( V_{g'} \) is the internal gate-to-source voltage that controls the drain current. The \( r_g \) and \( C_{gs} \) parameters are the gate-to-source diffusion resistance and junction capacitance, respectively. The gate-to-source junction is reverse biased for depletion mode devices and has only a small forward-bias voltage for enhancement mode devices, so that normally \( r_g \) is large. The parameters \( r_d \) and \( C_{gd} \) are the gate-to-drain resistance and capacitance, respectively. The resistance \( r_{ds} \) is the finite drain resistance, which is a function of the channel length modulation effect. The \( C_d \) capacitance is mainly a drain-to-source parasitic capacitance and \( C_s \) is the drain-to-substrate capacitance.

The ideal small-signal equivalent circuit is shown in Figure 13.20a. All diffusion resistances are infinite, the series resistances are zero, and at low frequency the

!Figure 13.18 | Cross section of JFET with source and drain series resistance.

!Figure 13.19 | Small-signal equivalent circuit of JFET.