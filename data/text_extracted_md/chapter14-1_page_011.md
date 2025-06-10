# CHAPTER 14 Optical Devices

!Figure 14.9  
**Figure 14.9** | Solar spectral irradiance.  
*(From Sze [18].)*

!Figure 14.10  
**Figure 14.10** | Ideal solar cell efficiency at \( T = 300 \, K \) for \( C = 1 \) sun and for a \( C = 1000 \) sun concentrations as a function of bandgap energy.  
*(From Sze [18].)*

## 14.2.3 Nonuniform Absorption Effects

We have seen from the previous section that the photon absorption coefficient in a semiconductor is a very strong function of the incident photon energy or wavelength. Figure 14.4 shows the absorption coefficient as a function of wavelength for several semiconductor materials. As the absorption coefficient increases, more photon energy will be absorbed near the surface than deeper into the semiconductor. In this case, then, we will not have uniform excess carrier generation in a solar cell.

The number of photons absorbed per cm\(^2\) per second as a function of distance \( x \) from the surface can be written as

\[
\alpha \Phi_0 e^{-\alpha x}
\]

(14.15)

where \( \Phi_0 \) is the incident photon flux (cm\(^{-2}\) s\(^{-1}\)) on the surface of the semiconductor. We can also take into account the reflection of photons from the surface. Let \( R(\lambda) \) be the fraction of photons that are reflected. (For bare silicon, \( R \approx 35 \) percent.) If we assume that each photon absorbed creates one electron–hole pair, then the generation rate of electron–hole pairs as a function of distance \( x \) from the surface is

\[
G_L = \alpha(\lambda) \Phi_0(\lambda) [1 - R(\lambda)] e^{-\alpha x}
\]

(14.16)

where each parameter may be a function of the incident wavelength. Figure 14.11 shows the excess minority carrier concentrations in this pn solar cell for two values of wavelength and for the case when \( s = 0 \) at the surface.