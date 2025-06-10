# Optical Devices

The intensity of the photon flux is denoted by \( I_{\nu}(x) \) and is expressed in terms of energy/cm\(^2\)-s. Figure 14.2 shows an incident photon intensity at a position \( x \) and the photon flux emerging at a distance \( x + dx \). The energy absorbed per unit time in the distance \( dx \) is given by

\[
\alpha I_{\nu}(x) dx
\]

(14.2)

where \( \alpha \) is the absorption coefficient. The absorption coefficient is the relative number of photons absorbed per unit distance, given in units of cm\(^{-1}\).

From Figure 14.2, we can write

\[
I_{\nu}(x + dx) - I_{\nu}(x) = \frac{dI_{\nu}(x)}{dx} \cdot dx = -\alpha I_{\nu}(x) dx
\]

(14.3)

or

\[
\frac{dI_{\nu}(x)}{dx} = -\alpha I_{\nu}(x)
\]

(14.4)

If the initial condition is given as \( I_{\nu}(0) = I_{\nu 0} \), then the solution to the differential equation, Equation (14.4), is

\[
I_{\nu}(x) = I_{\nu 0} e^{-\alpha x}
\]

(14.5)

The intensity of the photon flux decreases exponentially with distance through the semiconductor material. The photon intensity as a function of \( x \) for two general values of absorption coefficient is shown in Figure 14.3. If the absorption coefficient is large, the photons are absorbed over a relatively short distance.

The absorption coefficient in the semiconductor is a very strong function of photon energy and bandgap energy. Figure 14.4 shows the absorption coefficient \(\alpha\) plotted as a function of wavelength for several semiconductor materials. The absorption coefficient increases very rapidly for \( h\nu > E_g \), or for \( \lambda < 1.24/E_g \).

### Figures

- **Figure 14.2**: Optical absorption in a differential length.

- **Figure 14.3**: Photon intensity versus distance for two absorption coefficients.

| !Figure 14.2 | !Figure 14.3 |
|:--:|:--:|
| **Figure 14.2**: Optical absorption in a differential length. | **Figure 14.3**: Photon intensity versus distance for two absorption coefficients. |
