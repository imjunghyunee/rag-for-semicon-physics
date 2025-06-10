# 14.1 Optical Absorption

energy/cm\(^2\)-s and \(aL_\nu(x)\) is the rate at which energy is absorbed per unit volume. If we assume that one absorbed photon at an energy \(h\nu\) creates one electron–hole pair, then the generation rate of electron–hole pairs is

\[
g' = \frac{aL_\nu(x)}{h\nu}
\]

(14.6)

which is in units of #/cm\(^3\)-s. We may note that the ratio \(L_\nu(x)/h\nu\) is the photon flux. If, on the average, one absorbed photon produces less than one electron–hole pair, then Equation (14.6) must be multiplied by an efficiency factor.

----

**Objective:** Calculate the generation rate of electron–hole pairs given an incident intensity of photons.

**Example 14.2**

Consider gallium arsenide at \(T = 300 \, K\). Assume the photon intensity at a particular point is \(I_\nu(x) = 0.05 \, \text{W/cm}^2\) at a wavelength of \(\lambda = 0.75 \, \mu m\). This intensity is typical of sunlight, for example.

**Solution**

The absorption coefficient for gallium arsenide at this wavelength is \(\alpha \approx 0.9 \times 10^4 \, \text{cm}^{-1}\). The photon energy, using Equation (14.1), is

\[
E = h\nu = \frac{1.24}{0.75} = 1.65 \, \text{eV}
\]

Then, from Equation (14.6) and including the conversion factor between joules and eV, we have, for a unity efficiency factor,

\[
g' = \frac{aL_\nu(x)}{h\nu} = \frac{(0.9 \times 10^4)(0.05)}{(1.6 \times 10^{-19})(1.65)} = 1.70 \times 10^{21} \, \text{cm}^{-3}\text{-s}^{-1}
\]

If the incident photon intensity is a steady-state intensity, then, from Chapter 6, the steady-state excess carrier concentration is \(\delta n = g' \tau\), where \(\tau\) is the excess minority carrier lifetime. If \(\tau = 10^{-7} \, s\), for example, then

\[
\delta n = (1.70 \times 10^{21})(10^{-7}) = 1.70 \times 10^{14} \, \text{cm}^{-3}
\]

**Comment**

This example gives an indication of the magnitude of the electron–hole generation rate and the magnitude of the excess carrier concentration. Obviously, as the photon intensity decreases with distance in the semiconductor, the generation rate also decreases.

----

**Exercise Problem**

**Ex 14.2** A photon flux with an intensity of \(I_0 = 0.10 \, \text{W/cm}^2\) and at a wavelength of \(\lambda = 1 \, \mu m\) is incident on the surface of silicon. Neglecting any reflection from the surface, determine the generation rate of electron–hole pairs at a depth of (a) \(x = 5 \, \mu m\) and (b) \(x = 20 \, \mu m\) from the surface.

\[
[I_s = 0.10 \times E \, \text{(a)}; I_s = 0.10 \times 6L \, \text{(b)} \, \text{sun}]
\]