# Chapter 14: Optical Devices

!Figure 14.32  
*Light propagating in z direction through a material with two energy levels.*

Figure 14.32 shows the two energy levels with a light wave at an intensity \( I_\nu \) propagating in the \( z \) direction. The change in intensity as a function of \( z \) can be written as

\[
\frac{dI_\nu}{dz} \propto \frac{\# \text{ photons emitted} - \# \text{ photons absorbed}}{\text{cm}^3}
\]

or

\[
\frac{dI_\nu}{dz} = N_2 W_i \cdot h\nu - N_1 W_i \cdot h\nu \tag{14.63}
\]

where \( W_i \) is the induced transition probability. Equation (14.63) assumes no loss mechanisms and neglects the spontaneous transitions.

Equation (14.63) can be written as

\[
\frac{dI_\nu}{dz} = \gamma(\nu)I_\nu \tag{14.64}
\]

where \( \gamma(\nu) \propto (N_2 - N_1) \) and is the amplification factor. From Equation (14.64), the intensity is

\[
I_\nu = I_\nu(0)e^{\gamma(\nu)z} \tag{14.65}
\]

Amplification occurs when \( \gamma(\nu) > 0 \) and absorption occurs when \( \gamma(\nu) < 0 \).

We can achieve population inversion and lasing in a forward-biased pn homojunction diode, if both sides of the junction are degenerately doped. Figure 14.33a shows the energy-band diagram of a degenerately doped pn junction in thermal equilibrium. The Fermi level is in the conduction band in the n-region and the Fermi level is in the valence band in the p region. Figure 14.33b shows the energy bands of the pn junction when a forward bias is applied. The gain factor in a pn homojunction diode is given by

\[
\gamma(\nu) \propto \left[ 1 - \exp\left(\frac{h\nu - (E_{Fn} - E_{Fp})}{kT}\right) \right] \tag{14.66}
\]

In order for \( \gamma(\nu) > 1 \), we must have \( h\nu < (E_{Fn} - E_{Fp}) \), which implies that the junction must be degenerately doped since we also have the requirement that \( h\nu \geq E_g \). In the vicinity of the junction, there is a region in which population inversion occurs. There are large numbers of electrons in the conduction band directly above a large number of empty states. If band-to-band recombination occurs, photons will be emitted with energies in the range \( E_g < h\nu < (E_{Fn} - E_{Fp}) \).