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