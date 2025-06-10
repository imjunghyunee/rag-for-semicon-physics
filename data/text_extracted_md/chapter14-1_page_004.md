# 14.1 Optical Absorption

!Absorption Coefficient Graph

**Figure 14.4** | Absorption coefficient as a function of wavelength for several semiconductors.  
*(From Shur [13].)*

The coefficients are very small for \( h\nu < E_g \), so the semiconductor appears transparent to photons in this energy range.

----

## Objective

Calculate the thickness of a semiconductor that will absorb 90 percent of the incident photon energy.

Consider silicon and assume that in the first case the incident wavelength is \( \lambda = 1.0 \, \mu m \) and in the second case, the incident wavelength is \( \lambda = 0.5 \, \mu m \).

### Solution

From Figure 14.4, the absorption coefficient is \( \alpha \approx 10^3 \, \text{cm}^{-1} \) for \( \lambda = 1.0 \, \mu m \). If 90 percent of the incident flux is to be absorbed in a distance \( d \), then the flux emerging at \( x = d \) will be 10 percent of the incident flux. We can write

\[
\frac{I_x(d)}{I_0} = 0.1 = e^{-\alpha d}
\]

Solving for the distance \( d \), we have

\[
d = \frac{1}{\alpha} \ln \left( \frac{1}{0.1} \right) = \frac{1}{10^3} \ln (10) = 0.0230 \, \text{cm}
\]