# 12.6 Frequency Limitations

The hybrid-pi equivalent circuit, developed in the last section, introduces frequency effects through the capacitor–resistor circuits. We now discuss the various physical factors in the bipolar transistor affecting the frequency limitations of the device and then define the transistor cutoff frequency, which is a figure of merit for a transistor.

## 12.6.1 Time-Delay Factors

The bipolar transistor is a transit-time device. When the voltage across the B–E junction increases, for example, additional carriers from the emitter are injected into the base, diffuse across the base, and are collected in the collector region. As the frequency increases, this transit time can become comparable to the period of the input signal. At this point, the output response will no longer be in phase with the input and the magnitude of the current gain will decrease.

The total emitter-to-collector time constant or delay time is composed of four separate time constants. We can write

\[
\tau_{ec} = \tau_{\pi} + \tau_b + \tau_d + \tau_c
\]

(12.86)

where

- \(\tau_{ec}\) = emitter-to-collector time delay
- \(\tau_{\pi}\) = emitter–base junction capacitance charging time
- \(\tau_b\) = base transit time
- \(\tau_d\) = collector depletion region transit time
- \(\tau_c\) = collector capacitance charging time

The equivalent circuit of the forward-biased B–E junction is given in Figure 12.39a. The capacitance \(C_{\pi}\) is the junction capacitance. If we ignore the series resistance, then the emitter–base junction capacitance charging time is

\[
\tau_{\pi} = r'_e (C_{\pi} + C_p)
\]

(12.87)

where \(r'_e\) is the emitter junction or diffusion resistance. The capacitance \(C_p\) includes any parasitic capacitance between the base and emitter. The resistance \(r'_e\) is found as the inverse of the slope of the \(I_E\) versus \(V_{BE}\) curve. We obtain

\[
r'_e = \frac{kT}{e} \cdot \frac{1}{I_E}
\]

(12.88)

where \(I_E\) is the dc emitter current.

The second term, \(\tau_b\), is the base transit time, the time required for the minority carriers to diffuse across the neutral base region. The base transit time is related to the diffusion capacitance \(C_f\) of the B–E junction. For the npn transistor, the electron current density in the base can be written as

\[
J_n = -en(x)v(x)
\]

(12.89)

where \(v(x)\) is an average velocity. We can write

\[
v(x) = dx/dt \quad \text{or} \quad dt = dx/v(x)
\]

(12.90)