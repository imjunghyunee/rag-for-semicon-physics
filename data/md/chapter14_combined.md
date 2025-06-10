# Chapter 14

## Optical Devices

In previous chapters, we have considered the basic physics of transistors that are used to amplify or switch electrical signals. Semiconductor devices can be designed to convert optical energy into electrical energy, and to convert electrical signals into optical signals. These devices are used in broadband communications and data transmission over optical fibers. The general classification of these devices is called **optoelectronics**.

In this chapter, we discuss the basic principles of solar cells, several photodetectors, light emitting diodes, and laser diodes. Solar cells and photodetectors convert optical energy into electrical energy; light emitting diodes and laser diodes convert electrical signals into optical signals.

### 14.0 | PREVIEW

In this chapter, we will:

- Discuss and analyze photon absorption in a semiconductor and present absorption coefficient data for several semiconductor materials.
- Consider the basic principles of solar cells, analyze their I–V characteristics, and discuss the conversion efficiency.
- Present various types of solar cells, including homojunction, heterojunction, and amorphous silicon solar cells.
- Discuss the basic principles of photodetectors, including photoconductors, photodiodes, and phototransistors.
- Derive the output current characteristics of the various photodetectors.
- Present and analyze the basic operation of the Light Emitting Diode (LED).
- Discuss the basic principles and operation of the laser diode.

# 14.1 OPTICAL ABSORPTION

In Chapter 2, we discussed the wave–particle duality principle and indicated that light waves could be treated as particles, which are referred to as photons. The energy of a photon is \( E = h\nu \) where \( h \) is Planck’s constant and \( \nu \) is the frequency. We can also relate the wavelength and energy by

\[
\lambda = \frac{c}{\nu} = \frac{hc}{E} = \frac{1.24}{E} \, \text{μm}
\]

(14.1)

where \( E \) is the photon energy in eV and \( c \) is the speed of light.

There are several possible photon–semiconductor interaction mechanisms. For example, photons can interact with the semiconductor lattice whereby the photon energy is converted into heat. Photons can also interact with impurity atoms, either donors or acceptors, or they can interact with defects within the semiconductor. However, the basic photon interaction process of greatest interest is the interaction with valence electrons. When a photon collides with a valence electron, enough energy may be imparted to elevate the electron into the conduction band. Such a process generates electron–hole pairs and creates excess carrier concentrations. The behavior of excess carriers in a semiconductor was considered in Chapter 6.

## 14.1.1 Photon Absorption Coefficient

When a semiconductor is illuminated with light, the photons may be absorbed or they may propagate through the semiconductor, depending on the photon energy and on the bandgap energy \( E_g \). If the photon energy is less than \( E_g \), the photons are not readily absorbed. In this case, the light is transmitted through the material and the semiconductor appears to be transparent.

If \( E = h\nu > E_g \), the photon can interact with a valence electron and elevate the electron into the conduction band. The valence band contains many electrons and the conduction band contains many empty states, so the probability of this interaction is high when \( h\nu > E_g \). This interaction creates an electron in the conduction band and a hole in the valence band—an electron–hole pair. The basic absorption processes for different values of \( h\nu \) are shown in Figure 14.1. When \( h\nu > E_g \), an electron–hole pair is generated.

!Figure 14.1

**Figure 14.1.1** Optically generated electron–hole pair formation in a semiconductor.

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

# CHAPTER 14 Optical Devices

In the second case, the absorption coefficient is \( \alpha \approx 10^4 \, \text{cm}^{-1} \) for \( \lambda = 0.5 \, \mu\text{m} \). The distance \( d \), then, in which 90 percent of the incident flux is absorbed, is

\[
d = \frac{1}{10^4} \ln \left( \frac{1}{0.1} \right) = 2.30 \times 10^{-4} \, \text{cm} = 2.30 \, \mu\text{m}
\]

**Comment**  
As the incident photon energy increases, the absorption coefficient increases rapidly, so that the photon energy can be totally absorbed in a very narrow region at the surface of the semiconductor.

**EXERCISE PROBLEM**  
Ex 14.1 Consider a slab of silicon 5 \(\mu\text{m}\) thick. Determine the percentage of photon energy that will pass through the slab if the photon wavelength is (a) \(\lambda = 0.8 \, \mu\text{m}\) and (b) \(\lambda = 0.6 \, \mu\text{m}\).  
[%s '01 (q) '%s '09 (9) 'suv]

The relation between the bandgap energies of some of the common semiconductor materials and the light spectrum is shown in Figure 14.5. We may note that silicon and gallium arsenide will absorb all of the visible spectrum, whereas gallium phosphide, for example, will be transparent to the red spectrum.

## 14.1.2 Electron–Hole Pair Generation Rate

We have shown that photons with energy greater than \( E_g \) can be absorbed in a semiconductor, thereby creating electron–hole pairs. The intensity \( I(x) \) is in units of

!Figure 14.5

**Figure 14.5** Light spectrum versus wavelength and energy. Figure includes relative response of the human eye.  
*(From Sze [18].)*

|        | Infrared | Red | Orange | Yellow | Green | Violet | Blue | Ultraviolet |
|--------|----------|-----|--------|--------|-------|-------|------|-------------|
| Si     |          |     |        |        |       |       |      |             |
| GaAs   |          |     |        |        |       |       |      |             |
| CdSe   |          |     |        |        |       |       |      |             |
| GaP    |          |     |        |        |       |       |      |             |
| CdS    |          |     |        |        |       |       |      |             |
| SiC    |          |     |        |        |       |       |      |             |
| GaN    |          |     |        |        |       |       |      |             |
| GaAs\(_{1-x}\)P\(_x\) | | | | | | | | |

\[
\lambda_m = 0.555 \, \mu\text{m}
\]

\[
E_g \, (\text{eV})
\]

\[
\lambda \, (\mu\text{m})
\]

1.0 0.9 0.8 0.7 0.6 0.5 0.45 0.4 0.35  
1.2 1.4 1.6 1.8 2.0 2.2 2.4 2.6 2.8 3.0 3.2 3.4 3.6

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

# TEST YOUR UNDERSTANDING

**TYU 14.1**  
(a) A photon flux with an intensity of \( I_0 = 0.10 \, \text{W/cm}^2 \) is incident on the surface of silicon. The wavelength of the incident photon signal is \( \lambda = 1 \, \mu\text{m} \). Neglecting any reflection from the surface, determine the photon flux intensity at a depth of (i) \( x = 5 \, \mu\text{m} \) and (ii) \( x = 20 \, \mu\text{m} \) from the surface. (b) Repeat part (a) for a wavelength of \( \lambda = 0.60 \, \mu\text{m} \).

----

## 14.2 | SOLAR CELLS

A solar cell is a pn junction device with no voltage directly applied across the junction. The solar cell converts photon power into electrical power and delivers this power to a load. These devices have long been used for the power supply of satellites and space vehicles, and also as the power supply to some calculators. We will first consider the simple pn junction solar cell with uniform generation of excess carriers. We will also discuss briefly the heterojunction and amorphous silicon solar cells.

### 14.2.1 The pn Junction Solar Cell

Consider the pn junction shown in Figure 14.6 with a resistive load. Even with zero bias applied to the junction, an electric field exists in the space charge region as shown in the figure. Incident photon illumination can create electron–hole pairs in the space charge region that will be swept out producing the photocurrent \( I_L \) in the reverse-biased direction as shown.

The photocurrent \( I_L \) produces a voltage drop across the resistive load which forward biases the pn junction. The forward-bias voltage produces a forward-bias current \( I_F \) as indicated in the figure. The net pn junction current, in the reverse-biased direction, is

\[
I = I_L - I_F = I_L - I_S \left[ \exp \left( \frac{eV}{kT} \right) - 1 \right]
\]

(14.7)

!Figure 14.6

**Figure 14.6 | A pn junction solar cell with resistive load.**

# 14.2 Solar Cells

!I-V characteristics of a pn junction solar cell.

**Figure 14.7** I–V characteristics of a pn junction solar cell.

where the ideal diode equation has been used. As the diode becomes forward biased, the magnitude of the electric field in the space charge region decreases, but does not go to zero or change direction. The photocurrent is always in the reverse-biased direction and the net solar cell current is also always in the reverse-biased direction.

There are two limiting cases of interest. The short-circuit condition occurs when \( R = 0 \) so that \( V = 0 \). The current in this case is referred to as the **short-circuit current**, or

\[
I = I_{sc} = I_L
\]

(14.8)

The second limiting case is the open-circuit condition and occurs when \( R \rightarrow \infty \). The net current is zero and the voltage produced is the **open-circuit voltage**. The photocurrent is just balanced by the forward-biased junction current, so we have

\[
I = 0 = I_L - I_s \left[ \exp \left( \frac{eV}{kT} \right) - 1 \right]
\]

(14.9)

We can find the open circuit voltage \( V_{oc} \) as

\[
V_{oc} = V_t \ln \left( 1 + \frac{I_L}{I_s} \right)
\]

(14.10)

A plot of the diode current \( I \) as a function of the diode voltage \( V \) from Equation (14.7) is shown in Figure 14.7. We may note the short-circuit current and open-circuit voltage points on the figure.

----

**Objective:** Calculate the open-circuit voltage of a silicon pn junction solar cell.

**Example 14.3**

Consider a silicon pn junction at \( T = 300 \, \text{K} \) with the following parameters:

\[
\begin{align*}
N_d &= 5 \times 10^{18} \, \text{cm}^{-3} \\
N_a &= 10^{16} \, \text{cm}^{-3} \\
D_n &= 25 \, \text{cm}^2/\text{s} \\
D_p &= 10 \, \text{cm}^2/\text{s} \\
\tau_{n0} &= 5 \times 10^{-7} \, \text{s} \\
\tau_{p0} &= 10^{-7} \, \text{s}
\end{align*}
\]

Let the photocurrent density be \( J_L = I_L/A = 15 \, \text{mA/cm}^2 \).

# Chapter 14: Optical Devices

## Solution

We have that

\[
J_s = \frac{I_s}{A} = \left( \frac{eD_n n_{0e}}{L_n} + \frac{eD_p p_{0e}}{L_p} \right) = en_i^2 \left( \frac{D_n}{L_n N_D} + \frac{D_p}{L_p N_A} \right)
\]

We may calculate

\[
L_n = \sqrt{D_n \tau_{n0}} = \sqrt{25 \times (5 \times 10^{-7})} = 35.4 \, \mu m
\]

and

\[
L_p = \sqrt{D_p \tau_{p0}} = \sqrt{10 \times 10^{-7}} = 10.0 \, \mu m
\]

Then

\[
J_s = (1.6 \times 10^{-19})(1.5 \times 10^{19}) \times \left( \frac{25}{(35.4 \times 10^{-4} \times 5 \times 10^{15})} + \frac{10}{(10 \times 10^{-4} \times 10^{15})} \right)
\]

\[
= 3.6 \times 10^{-11} \, \text{A/cm}^2
\]

Then from Equation (14.10), we can find

\[
V_{oc} = V_t \ln \left( 1 + \frac{I_L}{I_s} \right) = V_t \ln \left( 1 + \frac{15 \times 10^{-3}}{3.6 \times 10^{-11}} \right) = (0.0259) \ln \left( 1 + \frac{15 \times 10^{-3}}{3.6 \times 10^{-11}} \right) = 0.514 \, \text{V}
\]

## Comment

We may determine the built-in potential barrier of this junction to be \( V_{bi} = 0.8556 \, \text{V} \). Taking the ratio of the open-circuit voltage to the built-in potential barrier, we find that \( V_{oc}/V_{bi} = 0.60 \). The open-circuit voltage will always be less than the built-in potential barrier.

## Exercise Problem

**Ex 14.3** Consider a GaAs pn junction solar cell with the following parameters:  
\( N_D = 10^{17} \, \text{cm}^{-3}, N_A = 2 \times 10^{16} \, \text{cm}^{-3}, D_n = 190 \, \text{cm}^2/\text{s}, D_p = 10 \, \text{cm}^2/\text{s}, \tau_{n0} = 10^{-7} \, \text{s}, \tau_{p0} = 10^{-8} \, \text{s} \). Assume a photocurrent density of \( J_L = 20 \, \text{mA/cm}^2 \) is generated in the solar cell. (a) Calculate the open-circuit voltage and (b) determine the ratio of open-circuit voltage to built-in potential barrier.  
\[ \text{[8GU 0 = "M/\lambda ^{(} \lambda 1.60 = ^{A} (b) 'SUVI} \]

The power delivered to the load is

\[
P = I \cdot V = I_L \cdot V - I_s \left[ \exp \left( \frac{eV}{kT} \right) - 1 \right] \cdot V \tag{14.11}
\]

We may find the current and voltage which will deliver the maximum power to the load by setting the derivative equal to zero, or \( dP/dV = 0 \). Using Equation (14.11), we find

\[
\frac{dP}{dV} = 0 = I_L - I_s \left[ \exp \left( \frac{eV_m}{kT} \right) - 1 \right] - I_s V_m \left( \frac{e}{kT} \right) \exp \left( \frac{eV_m}{kT} \right) \tag{14.12}
\]

where \( V_m \) is the voltage that produces the maximum power. We may rewrite Equation (14.12) in the form

\[
\left( 1 + \frac{V_m}{V_t} \right) \exp \left( \frac{eV_m}{kT} \right) = 1 + \frac{I_L}{I_s} \tag{14.13}
\]

The value of \( V_m \) may be determined by trial and error. Figure 14.8 shows the maximum power rectangle where \( I_m \) is the current when \( V = V_m \).

!Figure 14.8  
*Maximum power rectangle of the solar cell I-V characteristics.*

## 14.2.2 Conversion Efficiency and Solar Concentration

The conversion efficiency of a solar cell is defined as the ratio of output electrical power to incident optical power. For the maximum power output, we can write

\[
\eta = \frac{P_m}{P_{in}} \times 100\% = \frac{I_m V_m}{P_{in}} \times 100\%
\]

(14.14)

The maximum possible current and the maximum possible voltage in the solar cell are \(I_{sc}\) and \(V_{oc}\) respectively. The ratio \(I_m V_m / I_{sc} V_{oc}\) is called the fill factor and is a measure of the realizable power from a solar cell. Typically, the fill factor is between 0.7 and 0.8.

The conventional pn junction solar cell has a single semiconductor bandgap energy. When the cell is exposed to the solar spectrum, a photon with energy less than \(E_g\) will have no effect on the electrical output power of the solar cell. A photon with energy greater than \(E_g\) will contribute to the solar cell output power, but the fraction of photon energy that is greater than \(E_g\) will eventually only be dissipated as heat. Figure 14.9 shows the solar spectral irradiance (power per unit area per unit wavelength) where air mass zero represents the solar spectrum outside the earth’s atmosphere and air mass one is the solar spectrum at the earth’s surface at noon. The maximum efficiency of a silicon pn junction solar cell is approximately 28 percent. Nonideal factors, such as series resistance and reflection from the semiconductor surface, will lower the conversion efficiency typically to the range of 10 to 15 percent.

A large optical lens can be used to concentrate sunlight onto a solar cell so that the light intensity can be increased up to several hundred times. The short-circuit current increases linearly with light concentration while the open-circuit voltage increases only slightly with concentration. Figure 14.10 shows the ideal solar cell efficiency at 300 K for two values of solar concentration. We can see that the conversion efficiency increases only slightly with optical concentration. The primary advantage of using concentration techniques is to reduce the overall system cost since an optical lens is less expensive than an equivalent area of solar cells.

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

# 14.2 Solar Cells

!Graph of minority carrier concentration

**Figure 14.11** | Steady-state, photon-induced normalized minority carrier concentration in the pn junction solar cell for two values of incident photon wavelength (\(x_i = 2 \, \mu m\), \(W = 1 \, \mu m\), \(L_s = L_e = 40 \, \mu m\)).

----

!Energy-band diagram

**Figure 14.12** | The energy-band diagram of a pN heterojunction in thermal equilibrium.

## 14.2.4 The Heterojunction Solar Cell

As we have mentioned in previous chapters, a heterojunction is formed between two semiconductors with different bandgap energies. A typical pN heterojunction energy-band diagram in thermal equilibrium is shown in Figure 14.12. Assume that photons are incident on the wide-bandgap material. Photons with energy less than \(E_{gN}\) will pass through the wide-bandgap material, which acts as an optical window, and photons with energies greater than \(E_{gP}\) will be absorbed in the narrow bandgap material. On the average, excess carriers created in the depletion region and within a diffusion length of the junction will be collected and will contribute to the photocurrent. Photons with an energy greater than \(E_{gN}\) will be absorbed in the wide-bandgap material.

```markdown
### CHAPTER 14 Optical Devices

!Figure 14.13

**Figure 14.13** | The normalized spectral response of several AlGaAs/GaAs solar cells with different compositions. (From Sze [17].)

material, and excess carriers generated within one diffusion length of the junction will be collected. If \( E_{gV} \) is large enough, then the high-energy photons will be absorbed in the space charge region of the narrow-bandgap material. This heterojunction solar cell should have better characteristics than a homojunction cell, especially at the shorter wavelengths.

A variation of the heterojunction is shown in Figure 14.13. A pn homojunction is formed and then a wide-bandgap material is grown on top. Again, the wide-bandgap material acts as an optical window for photon energies \( h\nu < E_{gL} \). Photons with energies \( E_{gL} < h\nu < E_{gH} \) will create excess carriers in the homojunction and photons with energies \( h\nu > E_{gH} \) will create excess carriers in the window type material. If the absorption coefficient in the narrow bandgap material is high, then essentially all of the excess carriers will be generated within a diffusion length of the junction, so the collection efficiency will be very high. Figure 14.13 also shows the normalized spectral response for various mole fractions \( x \) in the Al\(_x\)Ga\(_{1-x}\)As.

### 14.2.5 Amorphous Silicon Solar Cells

Single-crystal silicon solar cells tend to be expensive and are limited to approximately 6 inches in diameter. A system powered by solar cells requires, in general,
```

# 14.2 Solar Cells

A very large area solar cell array to generate the required power. Amorphous silicon solar cells provide the possibility of fabricating large area and relatively inexpensive solar cell systems.

When silicon is deposited by CVD techniques at temperatures below 600°C, an amorphous film is formed regardless of the type of substrate. In amorphous silicon, there is only very short range order, and no crystalline regions are observed. Hydrogen may be incorporated in the silicon to reduce the number of dangling bonds, creating a material called hydrogenated amorphous silicon.

The density of states versus energy for amorphous silicon is shown in Figure 14.14. Amorphous silicon contains large numbers of electronic energy states within the normal bandgap of single-crystal silicon. However, because of the short-range order, the effective mobility is quite small, typically in the range between \(10^{-6}\) and \(10^{-3} \, \text{cm}^2/\text{V-s}\). The mobilities in the states above \(E_c\) and below \(E_v\) are between 1 and 10 \(\text{cm}^2/\text{V-s}\). Consequently, conduction through the energy states between \(E_c\) and \(E_v\) is negligible because of the low mobility. Because of the difference in mobility values, \(E_c\) and \(E_v\) are referred to as the mobility edges and the energy between \(E_c\) and \(E_v\) is referred to as the mobility gap. The mobility gap can be modified by adding specific types of impurities. Typically, the mobility gap is on the order of 1.7 eV.

Amorphous silicon has a very high optical absorption coefficient, so most sunlight is absorbed within approximately \(1 \, \mu m\) of the surface. Consequently, only a

!Figure 14.14

**Figure 14.14** | Density of states versus energy of amorphous silicon.  
(From Yang [22].)

- **E**: Energy
- **Conduction band**
- **Valence band**
- **Band-gap states**
- **Mobility edges**
- **Density of states \(N(E)\)**

# CHAPTER 14 Optical Devices

!Figure 14.15

**Figure 14.15** | The (a) cross section, (b) energy-band diagram at thermal equilibrium, and (c) energy-band diagram under photon illumination of an amorphous silicon PIN solar cell. (From Yang [22.1])

A very thin layer of amorphous silicon is required for a solar cell. A typical amorphous silicon solar cell is a PIN device shown in Figure 14.15. The amorphous silicon is deposited on an optically transparent indium tin oxide–coated glass substrate. If aluminum is used as the back contact, it will reflect any transmitted photons back through the PIN device. The n⁺ and p⁺ regions can be quite thin while the intrinsic region may be in the range of 0.5 to 1.0 μm thick. The energy-band diagram for the thermal equilibrium case is shown in the figure. Excess carriers generated in the intrinsic region are separated by the electric field and produce the photocurrent, as we have discussed. Conversion efficiencies are smaller than in single-crystal silicon, but the reduced cost makes this technology attractive. Amorphous silicon solar cells approximately 40 cm wide and many meters long have been fabricated.

----

## TEST YOUR UNDERSTANDING

**TYU 14.2**  
Consider a silicon pn junction solar cell with the parameters given in Example 14.3. Determine the required photocurrent density to produce an open-circuit voltage of \( V_{oc} = 0.60 \, \text{V} \).  
\((J_{sc} = 17 \, \text{mA/cm}^2)\)

**TYU 14.3**  
Consider the silicon pn junction solar cell described in Example 14.3. Let the solar intensity increase by a factor of 10. Calculate the open-circuit voltage.  
\((V_{oc} = 0.74 \, \text{V})\)

**TYU 14.4**  
The silicon pn junction solar cell described in TYU 14.2 has a cross-sectional area of 1 cm². Determine the maximum power that can be delivered to a load.  
\((P_{max} = 50 \, \text{mW})\)

# 14.3 PHOTODETECTORS

There are several semiconductor devices that can be used to detect the presence of photons. These devices are known as photodetectors; they convert optical signals into electrical signals. When excess electrons and holes are generated in a semiconductor, there is an increase in the conductivity of the material. This change in conductivity is the basis of the photoconductor, perhaps the simplest type of photodetector. If electrons and holes are generated within the space charge region of a pn junction, then they will be separated by the electric field and a current will be produced. The pn junction is the basis of several photodetector devices including the photodiode and the phototransistor.

## 14.3.1 Photoconductor

Figure 14.16 shows a bar of semiconductor material with ohmic contacts at each end and a voltage applied between the terminals. The initial thermal-equilibrium conductivity is

\[
\sigma_0 = e[\mu_n n_0 + \mu_p p_0]
\]

(14.17)

If excess carriers are generated in the semiconductor, the conductivity becomes

\[
\sigma = e[\mu_n(n_0 + \delta n) + \mu_p(p_0 + \delta p)]
\]

(14.18)

where \(\delta n\) and \(\delta p\) are the excess electron and hole concentrations, respectively. If we consider an n-type semiconductor, then, from charge neutrality, we can assume that

!Figure 14.16

**Figure 14.16 | A photoconductor.**

# CHAPTER 14 Optical Devices

\(\delta n = \delta p = \delta p\). We will use \(\delta p\) as the concentration of excess carriers. In steady state, the excess carrier concentration is given by \(\delta p = G \tau_r\), where \(G\) is the generation rate of excess carriers (cm\(^{-3}\)s\(^{-1}\)) and \(\tau_r\) is the excess minority carrier lifetime.

The conductivity from Equation (14.18) can be rewritten as

\[
\sigma = e(\mu_n n_0 + \mu_p p_0) + e(\delta p)(\mu_n + \mu_p)
\]

(14.19)

The change in conductivity due to the optical excitation, known as the **photoconductivity**, is then

\[
\Delta \sigma = e(\delta p)(\mu_n + \mu_p)
\]

(14.20)

An electric field is induced in the semiconductor by the applied voltage, which produces a current. The current density can be written as

\[
J = (J_0 + J_L) = (\sigma_0 + \Delta \sigma)E
\]

(14.21)

where \(J_0\) is the current density in the semiconductor prior to optical excitation and \(J_L\) is the photocurrent density. The photocurrent density is \(J_L = \Delta \sigma \cdot E\). If the excess electrons and holes are generated uniformly throughout the semiconductor, then the photocurrent is given by

\[
I_L = J_L \cdot A = \Delta \sigma \cdot A E = eG\tau_r(\mu_n + \mu_p)AE
\]

(14.22)

where \(A\) is the cross-sectional area of the device. The photocurrent is directly proportional to the excess carrier generation rate, which in turn is proportional to the incident photon flux.

If excess electrons and holes are not generated uniformly throughout the semiconductor material, then the total photocurrent is found by integrating the photoconductivity over the cross-sectional area.

Since \(\mu_n E\) is the electron drift velocity, the electron transit time, that is, the time required for an electron to flow through the photoconductor, is

\[
t_n = \frac{L}{\mu_n E}
\]

(14.23)

The photocurrent, from Equation (14.22), can be rewritten as

\[
I_L = eG\tau_r \left( \frac{T_p}{t_n} \right) \left( 1 + \frac{\mu_p}{\mu_n} \right) AL
\]

(14.24)

We may define a photoconductor gain, \(\Gamma_p\), as the ratio of the rate at which charge is collected by the contacts to the rate at which charge is generated within the photoconductor. We can write the gain as

\[
\Gamma_p = \frac{I_L}{eG\tau_r AL}
\]

(14.25)

which, using Equation (14.24), can be written

\[
\Gamma_p = \frac{T_p}{t_n} \left( 1 + \frac{\mu_p}{\mu_n} \right)
\]

(14.26)

## 14.3 Photodetectors

### Objective

Calculate the photoconductor gain of a silicon photoconductor.

Consider an n-type silicon photoconductor with a length \( L = 100 \, \mu m \), cross-sectional area \( A = 10^{-7} \, cm^2 \), and minority carrier lifetime \( \tau_r = 10^{-6} \, s \). Let the applied voltage be \( V = 10 \) volts.

### Solution

The electron transit time is determined as

\[
t_e = \frac{L}{\mu_e E} = \frac{L^2}{\mu_e V} = \frac{(100 \times 10^{-4})^2}{(1350)(10)} = 7.41 \times 10^{-9} \, s
\]

The photoconductor gain is then

\[
\Gamma_{ph} = \frac{\tau_r}{t_e} \left( 1 + \frac{\mu_h}{\mu_e} \right) = \frac{10^{-6}}{7.41 \times 10^{-9}} \left( 1 + \frac{480}{1350} \right) = 1.83 \times 10^2
\]

### Comment

The fact that a photoconductor—a bar of semiconductor material—has a gain may be surprising.

### EXERCISE PROBLEM

**Ex 14.4** Consider the photoconductor described in Example 14.4. Determine the photocurrent if \( G_t = 10^{21} \, cm^{-3} \cdot s^{-1} \) and \( E = 10 \, V/cm \). Also assume that \( \mu_e = 1000 \, cm^2/V \cdot s \) and \( \mu_p = 400 \, cm^2/V \cdot s \).

\((\text{Verify } \Gamma_{ph} = 7.1 \times 10^2)\)

----

Let's consider physically what happens to a photon-generated electron, for example. After the excess electron is generated, it drifts very quickly out of the photoconductor at the anode terminal. In order to maintain charge neutrality throughout the photoconductor, another electron immediately enters the photoconductor at the cathode and drifts toward the anode. This process will continue during a time period equal to the mean carrier lifetime. At the end of this period, on the average, the photoelectron will recombine with a hole.

The electron transit time, using the parameters from Example 14.4, is \( t_e = 7.41 \times 10^{-9} \, s \). In a simplistic sense, the photoelectron will circulate around the photoconductor circuit 135 times during the \( 10^{-6} \, s \) time duration, which is the mean carrier lifetime. If we take into account the photon-generated hole, the total number of charges collected at the photoconductor contacts for every electron generated is 183.

When the optical signal ends, the photocurrent will decay exponentially with a time constant equal to the minority carrier lifetime. From the photoconductor gain expression, we would like a large minority carrier lifetime, but the switching speed is enhanced by a small minority carrier lifetime. There is obviously a trade-off between gain and speed. In general, the performance of a photodiode, which we will discuss next, is superior to that of a photoconductor.

### 14.3.2 Photodiode

A photodiode is a pn junction diode operated with an applied reverse-biased voltage. We will initially consider a long diode in which excess carriers are generated.

# Optical Devices

!Figure 14.17

**Figure 14.17** (a) A reverse-biased pn junction. (b) Minority carrier concentration in the reverse-biased pn junction.

Uniformly throughout the semiconductor device. Figure 14.17a shows the reverse-biased diode and Figure 14.17b shows the minority carrier distribution in the reverse-biased junction prior to photon illumination.

Let \( G_L \) be the generation rate of excess carriers. The excess carriers generated within the space charge region are swept out of the depletion region very quickly by the electric field; the electrons are swept into the n region and the holes into the p region. The photon-generated current density from the space charge region is given by

\[
J_{L1} = e \int G_L \, dx
\]

(14.27)

where the integral is over the space charge region width. If \( G_L \) is constant throughout the space charge volume, then

\[
J_{L1} = e G_L W
\]

(14.28)

where \( W \) is the space charge width. We may note that \( J_{L1} \) is in the reverse-biased direction through the pn junction. This component of photocurrent responds very quickly to the photon illumination and is known as the prompt photocurrent.

We may note, by comparing Equations (14.28) and (14.25), that the photodiode gain is unity. The speed of the photodiode is limited by the carrier transport through the junction.

# 14.3 Photodetectors

The space charge region. If we assume that the saturation drift velocity is \(10^7 \, \text{cm/s}\) and the depletion width is \(2 \, \mu\text{m}\), the transit time is \(\tau_t = 20 \, \text{ps}\). The ideal modulating frequency has a period of \(2\tau_t\), so the frequency is \(f = 25 \, \text{GHz}\). This frequency response is substantially higher than that of photoconductors.

Excess carriers are also generated within the neutral n and p regions of the diode. The excess minority carrier electron distribution in the p region is found from the ambipolar transport equation, which is

\[
D_n \frac{\partial^2 (\delta n_p)}{\partial x^2} + G_t - \frac{\delta n_p}{\tau_{n0}} = \frac{\partial (\delta n_p)}{\partial t}
\]

(14.29)

We will assume that the E-field is zero in the neutral regions. In steady state, \(\partial (\delta n_p)/\partial t = 0\), so that Equation (14.29) can be written as

\[
\frac{d^2 (\delta n_p)}{dx^2} - \frac{\delta n_p}{L_n^2} = -\frac{G_t}{D_n}
\]

(14.30)

where \(L_n^2 = D_n \tau_{n0}\).

The solution to Equation (14.30) can be found as the sum of the homogeneous and particular solutions. The homogeneous solution is found from the equation

\[
\frac{d^2 (\delta n_{ph})}{dx^2} - \frac{\delta n_{ph}}{L_n^2} = 0
\]

(14.31)

where \(\delta n_{ph}\) is the homogeneous solution and is given by

\[
\delta n_{ph} = Ae^{-x/L_n} + Be^{x/L_n}, \quad (x \geq 0)
\]

(14.32)

One boundary condition is that \(\delta n_{ph}\) must remain finite, which implies that \(B = 0\) for the "long" diode.

The particular solution is found from

\[
-\frac{\delta n_{pp}}{L_n^2} = -\frac{G_t}{D_n}
\]

(14.33)

which yields

\[
\delta n_{pp} = \frac{G_t L_n^2}{D_n} = \frac{G_t (D_n \tau_{n0})}{D_n} = G_t \tau_{n0}
\]

(14.34)

The total steady-state solution for the excess minority carrier electron concentration in the p region is then

\[
\delta n_p = Ae^{-x/L_n} + G_t \tau_{n0}
\]

(14.35)

The total electron concentration is zero at \(x = 0\) for the reverse-biased junction. The excess electron concentration \(x = 0\) is then

\[
\delta n_p (x = 0) = -n_{p0}
\]

(14.36)

Using the boundary condition from Equation (14.36), the electron concentration given by Equation (14.35) becomes

\[
\delta n_p = G_t \tau_{n0} - (G_t \tau_{n0} + n_{p0}) e^{-x/L_n}
\]

(14.37)

# CHAPTER 14 Optical Devices

!Figure 14.18

**Figure 14.18** | Steady-state, photoinduced minority carrier concentrations and photocurrents in a “long” reverse-biased pn junction.

We can find the excess minority carrier hole concentration in the n region using the same type of analysis. Using the \( x' \) notation shown in Figure 14.17, we can write

\[
\delta p_n = G_T \tau_{p0} - (G_T \tau_{p0} + p_{n0}) e^{-x'/L_n}
\]

(14.38)

Equations (14.37) and (14.38) are plotted in Figure 14.18. We may note that the steady-state values far from the space charge region are the same as were given previously.

The gradient in the minority carrier concentrations will produce diffusion currents in the pn junction. The diffusion current density at \( x = 0 \) due to minority carrier electrons is

\[
J_{n1} = eD_n \left( \frac{d(\delta n_p)}{dx} \right)_{x=0} = eD_n \frac{d}{dx} [G_T \tau_{n0} - (G_T \tau_{n0} + n_{p0}) e^{-x/L_n}]_{x=0}
\]

\[
= \frac{eD_n}{L_n} (G_T \tau_{n0} + n_{p0})
\]

(14.39)

Equation (14.39) can be written as

\[
J_{n1} = eG_T L_n + \frac{eD_n n_{p0}}{L_n}
\]

(14.40)

The first term in Equation (14.40) is the steady-state photocurrent density while the second term is the ideal reverse saturation current density due to the minority carrier electrons.

The diffusion current density (in the x direction) at \( x' = 0 \) due to the minority carrier holes is

\[
J_{p1} = eG_T L_p + \frac{eD_p p_{n0}}{L_p}
\]

(14.41)

# 14.3 Photodetectors

Similarly, the first term is the steady-state photocurrent density and the second term is the ideal reverse saturation current density.

The total steady-state diode photocurrent density for the long diode is now

\[
J_L = eG_LW + eG_LL_n + eG_LL_p = e(W + L_n + L_p)G_L
\]

(14.42)

Again note that the photocurrent is in the reverse-biased direction through the diode. The photocurrent given by Equation (14.42) is the result of assuming uniform generation of excess carriers throughout the structure, a long diode, and steady state.

The time response of the diffusion components of the photocurrent is relatively slow, since these currents are the results of the diffusion of minority carriers toward the depletion region. The diffusion components of photocurrent are referred to as the delayed photocurrent.

----

**Objective:** Calculate the steady-state photocurrent density in a reverse-biased, long pn diode.

**Example 14.5**

Consider a silicon pn diode at \( T = 300 \, \text{K} \) with the following parameters:

\[
\begin{align*}
N_a &= 10^{16} \, \text{cm}^{-3} \\
N_d &= 10^{16} \, \text{cm}^{-3} \\
D_n &= 25 \, \text{cm}^2/\text{s} \\
D_p &= 10 \, \text{cm}^2/\text{s} \\
\tau_{n0} &= 5 \times 10^{-7} \, \text{s} \\
\tau_{p0} &= 10^{-7} \, \text{s}
\end{align*}
\]

Assume that a reverse-biased voltage of \( V_R = 5 \) volts is applied and let \( G_L = 10^{21} \, \text{cm}^{-3}\text{s}^{-1} \).

## Solution

We may calculate various parameters as follows:

\[
L_n = \sqrt{D_n \tau_{n0}} = \sqrt{25 \times 5 \times 10^{-7}} = 35.4 \, \mu\text{m}
\]

\[
L_p = \sqrt{D_p \tau_{p0}} = \sqrt{10 \times 10^{-7}} = 10.0 \, \mu\text{m}
\]

\[
V_{bi} = V_t \ln \left( \frac{N_a N_d}{n_i^2} \right) = (0.0259) \ln \left( \frac{(10^{16})(10^{16})}{(1.5 \times 10^{10})^2} \right) = 0.695 \, \text{V}
\]

\[
W = \left[ \frac{2 \varepsilon_s}{e} \left( \frac{N_a + N_d}{N_a N_d} \right) (V_R + V_{bi}) \right]^{1/2}
\]

\[
= \left[ \frac{2(11.7)(8.85 \times 10^{-14})}{1.6 \times 10^{-19}} \left( \frac{2 \times 10^{16}}{(10^{16})(10^{16})} \right) (0.695 + 5) \right]^{1/2} = 1.21 \, \mu\text{m}
\]

Finally, the steady-state photocurrent density is

\[
J_L = e(W + L_n + L_p)G_L
\]

\[
= (1.6 \times 10^{-19})(1.21 + 35.4 + 10.0) \times 10^{-4}(10^{21}) = 0.75 \, \text{A/cm}^2
\]

## Comment

Again, keep in mind that this photocurrent is in the reverse-biased direction through the diode and is many orders of magnitude larger than the reverse-biased saturation current density in the pn junction diode.

# CHAPTER 14 Optical Devices

## EXERCISE PROBLEM

**Ex 14.5** The doping concentrations of the photodiode described in Example 14.5 are changed to \( N_a = N_d = 10^{15} \, \text{cm}^{-3} \). (a) Determine the steady-state photocurrent density. (b) Calculate the ratio of prompt photocurrent to steady-state photocurrent.  
\[
\left[ \frac{L}{L_0} = \frac{\tau}{\tau'} (q) : \frac{J_{UV}}{J_{L0}} = \frac{\tau'}{\tau} (b) \, \text{SUV} \right]
\]

In this example calculation, \( L_n \gg W \) and \( L_p \gg W \). In many pn junction structures, the assumption of a long diode will not be valid, so the photocurrent expression will have to be modified. In addition, the photon energy absorption may not be uniform throughout the pn structure. The effect of nonuniform absorption will be considered in the next section.

### 14.3.3 PIN Photodiode

In many photodetector applications, the speed of response is important; therefore, the prompt photocurrent generated in the space charge region is the only photocurrent of interest. To increase the photodetector sensitivity, the depletion region width should be made as large as possible. This can be achieved in a PIN photodiode.

The PIN diode consists of a p region and an n region separated by an intrinsic region. A sketch of a PIN diode is shown in Figure 14.19a. The intrinsic region width \( W \) is much larger than the space charge width of a normal pn junction. If a reverse bias is applied to the PIN diode, the space charge region extends completely through the intrinsic region.

!Figure 14.19

**Figure 14.19** (a) A reverse-biased PIN photodiode. (b) Geometry showing nonuniform photon absorption.

## 14.3 Photodetectors

Assume that a photon flux \(\Phi_0\) is incident on the \(p^+\) region. If we assume that the \(p^+\) region width \(W_p\) is very thin, then the photon flux, as a function of distance, in the intrinsic region is \(\Phi(x) = \Phi_0 e^{-\alpha x}\), where \(\alpha\) is the photon absorption coefficient. This nonlinear photon absorption is shown in Figure 14.19b. The photocurrent density generated in the intrinsic region can be found as

\[
J_L = e \int_0^W G_L \, dx = e \int_0^W \Phi_0 e^{-\alpha x} \, dx = e \Phi_0 (1 - e^{-\alpha W})
\]

(14.43)

This equation assumes that there is no electron–hole recombination within the space charge region and also that each photon absorbed creates one electron–hole pair.

### Objective

**Calculate the photocurrent density in a PIN photodiode.**

**EXAMPLE 14.6**

Consider a silicon PIN diode with an intrinsic region width of \(W = 20 \, \mu m\). Assume that the photon flux is \(10^7 \, \text{cm}^{-2}\text{s}^{-1}\) and the absorption coefficient is \(\alpha = 10^3 \, \text{cm}^{-1}\).

#### Solution

The generation rate of electron–hole pairs at the front edge of the intrinsic region is

\[
G_{L1} = \alpha \Phi_0 = (10^3)(10^7) = 10^{10} \, \text{cm}^{-3}\text{s}^{-1}
\]

and the generation rate at the back edge of the intrinsic region is

\[
G_{L2} = \alpha \Phi_0 e^{-\alpha W} = (10^3)(10^7) \exp[-(10^3)(20 \times 10^{-4})]
\]

\[
= 0.135 \times 10^9 \, \text{cm}^{-3}\text{s}^{-1}
\]

The generation rate is obviously not uniform throughout the intrinsic region. The photocurrent density is then

\[
J_L = e \Phi_0 (1 - e^{-\alpha W})
\]

\[
= (1.6 \times 10^{-19})(10^7)[1 - \exp[-(10^3)(20 \times 10^{-4})]]
\]

\[
= 13.8 \, \text{mA/cm}^2
\]

#### Comment

The prompt photocurrent density of a PIN photodiode will be larger than that of a regular photodiode since the space charge region is larger in a PIN photodiode.

#### EXERCISE PROBLEM

**Ex 14.6** Repeat Example 14.6 for photon absorption coefficients of (a) \(\alpha = 10^2 \, \text{cm}^{-1}\) and (b) \(\alpha = 10^4 \, \text{cm}^{-1}\).

In most situations, we will not have a long diode; thus, the steady-state photocurrent described by Equation (14.42) will not apply for most photodiodes.

### 14.3.4 Avalanche Photodiode

The avalanche photodiode is similar to the pn or PIN photodiode except that the bias applied to the avalanche photodiode is sufficiently large to cause impact ionization.

# Chapter 14: Optical Devices

Electron–hole pairs are generated in the space charge region by photon absorption, as we have discussed previously. The photon-generated electrons and holes now generate additional electron–hole pairs through impact ionization. The avalanche photodiode now has a current gain introduced by the avalanche multiplication factor.

The electron–hole pairs generated by photon absorption and by impact ionization are swept out of the space charge region very quickly. If the saturation velocity is \(10^7 \, \text{cm/s}\) in a depletion region that is \(10 \, \mu\text{m}\) wide, then the transit time is

\[
\tau_t = \frac{10^7}{10 \times 10^4} = 100 \, \text{ps}
\]

The period of a modulation signal would be \(2\pi \tau_t\), so that the frequency would be

\[
f = \frac{1}{2\pi \tau_t} = \frac{1}{200 \times 10^{-12}} = 5 \, \text{GHz}
\]

If the avalanche photodiode current gain is 20, then the gain-bandwidth product is 100 GHz. The avalanche photodiode could respond to light waves modulated at microwave frequencies.

## 14.3.5 Phototransistor

A bipolar transistor can also be used as a photodetector. The phototransistor can have high gain through the transistor action. An npn bipolar phototransistor is shown in Figure 14.20a. This device has a large base–collector junction area and is usually operated with the base open circuited. Figure 14.20b shows the block diagram of the phototransistor. Electrons and holes generated in the reverse-biased B–C junction are swept out of the space charge region, producing a photocurrent \(I_L\). Holes are swept

!Figure 14.20

**Figure 14.20** (a) A bipolar phototransistor. (b) Block diagram of the open-base phototransistor.

# 14.4 Photoluminescence and Electroluminescence

into the p-type base, making the base positive with respect to the emitter. Since the B–E becomes forward-biased, electrons will be injected from the emitter back into the base, leading to the normal transistor action.

From Figure 14.20b, we see that

\[
I_E = \alpha I_C + I_L
\]

(14.44)

where \( I_L \) is the photon-generated current and \( \alpha \) is the common base current gain. Since the base is an open circuit, we have \( I_C = I_E \), so Equation (14.44) can be written as

\[
I_C = \alpha I_C + I_L
\]

(14.45)

Solving for \( I_C \), we find

\[
I_C = \frac{I_L}{1 - \alpha}
\]

(14.46)

Relating \( \alpha \) to \( \beta \), the dc common emitter current gain, Equation (14.46) becomes

\[
I_C = (1 + \beta) I_L
\]

(14.47)

Equation (14.47) shows that the basic B–C photocurrent is multiplied by the factor \((1 + \beta)\). The phototransistor, then, amplifies the basic photocurrent.

With the relatively large B–C junction area, the frequency response of the phototransistor is limited by the B–C junction capacitance. Since the base is essentially the input to the device, the large B–C capacitance is multiplied by the Miller effect, so the frequency response of the phototransistor is further reduced. The phototransistor, however, is a lower-noise device than the avalanche photodiode.

Phototransistors can also be fabricated in heterostructures. The injection efficiency is increased as a result of the bandgap differences, as we discussed in Chapter 12. With the bandgap difference, the lightly doped base restriction no longer applies. A fairly heavily doped, narrow-base device can be fabricated with a high blocking voltage and a high gain.

----

**TEST YOUR UNDERSTANDING**

**TYU 14.5** Consider a long silicon pn junction photodiode with the parameters given in Example 14.5. The cross-sectional area is \( A = 10^{-3} \, \text{cm}^2 \). Assume the photodiode is reverse biased by a 5-volt battery in series with a 5 k\(\Omega\) load resistor. An optical signal at a wavelength of \( \lambda = 1 \, \mu\text{m} \) is incident on the photodiode producing a uniform generation rate of excess carriers throughout the entire device. Determine the incident intensity such that the voltage across the load resistor is 0.5 V. \((\mu\text{W}/\lambda 9920 = \gamma \, \mu\text{V})\)

# 14.4 | PHOTOLUMINESCENCE AND ELECTROLUMINESCENCE

In the first section of this chapter, we have discussed the creation of excess electron–hole pairs by photon absorption. Eventually, excess electrons and holes recombine, and in direct bandgap materials the recombination process may result in the emission of a photon. The general property of light emission is referred to as luminescence.

# CHAPTER 14 Optical Devices

When excess electrons and holes are created by photon absorption, photon emission from the recombination process is called photoluminescence.

Electroluminescence is the process of generating photon emission when the excitation of excess carriers is a result of an electric current caused by an applied electric field. We are mainly concerned here with injection electroluminescence, the result of injecting carriers across a pn junction. The light emitting diode and the pn junction laser diode are examples of this phenomenon. In these devices, electric energy, in the form of a current, is converted directly into photon energy.

## 14.4.1 Basic Transitions

Once electron-hole pairs are formed, there are several possible processes by which the electrons and holes can recombine. Some recombination processes may result in photon emission from direct bandgap materials, whereas other recombination processes in the same material may not.

Figure 14.21a shows the basic interband transitions. Curve (i) corresponds to an intrinsic emission very close to the bandgap energy of the material. Curves (ii) and (iii) correspond to energetic electrons or holes. If either of these recombinations result in the emission of a photon, the energy of the emitted photon will be slightly larger than the bandgap energy. There will then be an emission spectrum and a bandwidth associated with the emission.

The possible recombination processes involving impurity or defect states are shown in Figure 14.21b. Curve (i) is the conduction band to acceptor transition, curve (ii) is the donor to valence-band transition, curve (iii) is the donor to acceptor transition, and curve (iv) is the recombination due to a deep trap. Curve (iv) is a nonradiative process corresponding to the Shockley–Read–Hall recombination process discussed in Chapter 6. The other recombination processes may or may not result in the emission of a photon.

Figure 14.21c shows the Auger recombination process, which can become important in direct bandgap materials with high doping concentrations. The Auger recombination process is a nonradiative process. The Auger recombination, in one case, shown in curve (i), is a recombination between an electron and hole, accompanied by the transfer of energy to another free hole. Similarly, in the second case, the recombination between an electron and hole can result in the transfer of energy to a free electron as shown in curve (ii). The third particle involved in this process will eventually lose its energy to the lattice in the form of heat. The process involving two holes and an electron would occur predominantly in heavily doped p-type materials, and the process involving two electrons and a hole would occur primarily in a heavily doped n-type material.

The recombination processes shown in Figure 14.21a indicate that the emission of a photon is not necessarily at a single, discrete energy, but can occur over a range of energies. The spontaneous emission rate generally has the form

\[
I(\nu) \propto \nu^2 (h\nu - E_g)^{1/2} \exp \left( -\frac{(h\nu - E_g)}{kT} \right)
\]

(14.48)

where \( E_g \) is the bandgap energy. Figure 14.22 shows the emission spectra from gallium arsenide. The peak photon energy decreases with temperature because the...

# 14.4 Photoluminescence and Electroluminescence

!Basic transitions in a semiconductor.

**Figure 14.21** | Basic transitions in a semiconductor.

**Figure 14.22** | GaAs diode emission spectra at \( T = 300 \, \text{K} \) and \( T = 77 \, \text{K} \).  
*(From Sze and Ng [17])*

The bandgap energy decreases with temperature. We will show that the bandwidth of the emission spectra can be greatly reduced in a laser diode by using an optical resonator.

## 14.4.2 Luminescent Efficiency

We have shown that not all recombination processes are radiative. An efficient luminescent material is one in which radiative transitions predominate. The quantum efficiency is defined as the ratio of the radiative recombination rate to the total recombination rate for all processes. We can write

\[
\eta_q = \frac{R_r}{R}
\]

(14.49)

where \( \eta_q \) is the quantum efficiency, \( R_r \) is the radiative recombination rate, and \( R \) is the total recombination rate of the excess carriers. Since the recombination rate is

!GaAs diode emission spectra graph.

# Chapter 14: Optical Devices

Inversely proportional to lifetime, we can write the quantum efficiency in terms of lifetimes as

\[
\eta_q = \frac{\tau_{nr}}{\tau_{nr} + \tau_r}
\]

(14.50)

where \(\tau_{nr}\) is the nonradiative lifetime and \(\tau_r\) is the radiative lifetime. For a high luminescent efficiency, the nonradiative lifetimes must be large; thus, the probability of a nonradiative recombination is small compared to the radiative recombination.

The interband recombination rate of electrons and holes will be directly proportional to the number of electrons available and directly proportional to the number of available empty states (holes). We can write

\[
R = Bnp
\]

(14.51)

where \(R\) is the band-to-band radiative recombination rate and \(B\) is the constant of proportionality. The values of \(B\) for direct-bandgap materials are on the order of \(10^6\) larger than for indirect bandgap materials. The probability of a direct band-to-band radiative recombination transition in an indirect bandgap material is very unlikely.

One problem encountered with the emission of photons from a direct bandgap material is the reabsorption of the emitted photons. In general, the emitted photons will have energies \(h\nu > E_p\), which means that the absorption coefficient is not zero for this energy. In order to generate a light output from a light emitting device, the process must take place near the surface. One possible solution to the reabsorption problem is to use heterojunction devices. These are discussed in later sections.

## 14.4.3 Materials

An important direct bandgap semiconductor material for optical devices is gallium arsenide. Another compound material that is of great interest is Al\(_x\)Ga\(_{1-x}\)As. This material is a compound semiconductor in which the ratio of aluminum atoms to gallium atoms can be varied to achieve specific characteristics. Figure 14.23 shows the bandgap energy as a function of the mole fraction between aluminum and gallium. We can note from the figure that for \(0 < x < 0.45\), the alloy material is a direct bandgap material. For \(x > 0.45\), the material becomes an indirect bandgap material, not suitable for optical devices. For \(0 < x < 0.35\), the bandgap energy can be expressed as

\[
E_g = 1.424 + 1.247x \, \text{eV}
\]

(14.52)

Another compound semiconductor used for optical devices is the GaAs\(_{1-x}\)P\(_x\) system. Figure 14.24a shows the bandgap energy as a function of the mole fraction \(x\). For \(0 \leq x \leq 0.45\), this material is also a direct bandgap material, and for \(x > 0.45\), the bandgap becomes indirect. Figure 14.24b is the \(E\) versus \(k\) diagram, showing how the bandgap changes from direct to indirect as the mole fraction changes.

# 14.4 Photoluminescence and Electroluminescence

!Bandgap Energy of Al\(_x\)Ga\(_{1-x}\)As

**Figure 14.23** | Bandgap energy of Al\(_x\)Ga\(_{1-x}\)As as a function of the mole fraction \(x\).  
*(From Sze [18].)*

- **\(E_g = 3.018\)**
- **\(E_g = 2.168\)**
- **\(E_g = 1.9\)**
- **\(E_g = 1.424\)**

**Mole fraction AlAs, \(x\):**  
- 0 (GaAs) to 1.0 (AlAs)

----

### Figure 14.24

**(a)** Bandgap energy of GaAs\(_{1-x}\)P\(_x\) as a function of mole fraction \(x\).

- **\(T = 300 \, K\)**
- **\(E_g = 2.261\)**
- **\(E_g = 1.977\)**
- **\(E_g = 1.424\)**

**Mole fraction GaP, \(x\):**  
- 0 (GaAs) to 1.0 (GaP)

**(b)** \(E\) versus \(\vec{p}\) diagram of GaAs\(_{1-x}\)P\(_x\) for various values of \(x\).

- **Conduction band**
- **Valence band**
- **\(h\nu\)**

**Crystal momentum \(\vec{p}\):**  
- \(\vec{p}_{\text{max}}\)

*(From Sze [18].)*

# CHAPTER 14 Optical Devices

## EXAMPLE 14.7

**Objective:** Determine the output wavelength of a GaAs\(_{1-x}\)P\(_x\) material for two different mole fractions.

Consider first GaAs and then GaAs\(_{1-x}\)P\(_x\).

**Solution**  
GaAs has a bandgap energy of \( E_g = 1.42 \, \text{eV} \). This material would produce a photon output at a wavelength of

\[
\lambda = \frac{1.24}{E} = \frac{1.24}{1.42} = 0.873 \, \mu\text{m}
\]

This wavelength is in the infrared range and not in the visible range. If we desire a visible output with a wavelength of \( \lambda = 0.653 \, \mu\text{m} \), for example, the bandgap energy would have to be

\[
E = \frac{1.24}{\lambda} = \frac{1.24}{0.653} = 1.90 \, \text{eV}
\]

This bandgap energy would correspond to a mole fraction of approximately \( x = 0.4 \).

**Comment**  
By changing the mole fraction in the GaAs\(_{1-x}\)P\(_x\) system, the output can change from the infrared to the red spectrum.

**EXERCISE PROBLEM**  
Ex 14.7 Determine the output wavelength of a GaAs\(_{1-x}\)P\(_x\) material for mole fractions of  
(a) \( x = 0.15 \) and (b) \( x = 0.30 \).

## 14.5 LIGHT EMITTING DIODES

Photodetectors and solar cells convert optical energy into electrical energy—the photons generate excess electrons and holes, which produce an electric current. We might also apply a voltage across a pn junction resulting in a diode current, which in turn can produce photons and a light output. This inverse mechanism is called injection electroluminescence. This device is known as a Light Emitting Diode (LED). The spectral output of an LED may have a relatively wide wavelength bandwidth of between 30 and 40 nm. However, this emission spectrum is narrow enough so that a particular color is observed, provided the output is in the visible range.

### 14.5.1 Generation of Light

As we have discussed previously, photons may be emitted if an electron and hole recombine by a direct band-to-band recombination process in a direct bandgap material. The emission wavelength, from Equation (14.1), is

\[
\lambda = \frac{hc}{E_g} = \frac{1.24}{E_g} \, \mu\text{m} \quad (14.53)
\]

where \( E_g \) is the bandgap energy measured in electron-volts.

# 14.5 Light Emitting Diodes

When a voltage is applied across a pn junction, electrons and holes are injected across the space charge region where they become excess minority carriers. These excess minority carriers diffuse into the neutral semiconductor regions where they recombine with majority carriers. If this recombination process is a direct band-to-band process, photons are emitted. The diode diffusion current is directly proportional to the recombination rate, so the output photon intensity will also be proportional to the ideal diode diffusion current. In gallium arsenide, electroluminescence originates primarily on the p side of the junction because the efficiency for electron injection is higher than that for hole injection.

## 14.5.2 Internal Quantum Efficiency

The **internal quantum efficiency** of an LED is the fraction of diode current that produces luminescence. The internal quantum efficiency is a function of the injection efficiency and a function of the percentage of radiative recombination events compared with the total number of recombination events.

The three current components in a forward-biased diode are the minority carrier electron diffusion current, the minority carrier hole diffusion current, and the space charge recombination current. These current densities can be written, respectively, as

\[
J_n = \frac{eD_n n_{p0}}{L_n} \left[ \exp\left(\frac{eV}{kT}\right) - 1 \right] \tag{14.54a}
\]

\[
J_p = \frac{eD_p p_{n0}}{L_p} \left[ \exp\left(\frac{eV}{kT}\right) - 1 \right] \tag{14.54b}
\]

and

\[
J_r = \frac{en_i W}{2\tau_0} \left[ \exp\left(\frac{eV}{2kT}\right) - 1 \right] \tag{14.54c}
\]

The recombination of electrons and holes within the space charge region is, in general, through traps near midgap and is a nonradiative process. Since luminescence is due primarily to the recombination of minority carrier electrons in GaAs, we can define an injection efficiency as the fraction of electron current to total current. Then

\[
\gamma = \frac{J_n}{J_n + J_p + J_r} \tag{14.55}
\]

where \(\gamma\) is the injection efficiency. We can make \(\gamma\) approach unity by using an n\(^+\)p diode so that \(J_p\) is a small fraction of the diode current and by forward biasing the diode sufficiently so that \(J_r\) is a small fraction of the total diode current.

Once the electrons are injected into the p region, not all electrons will recombine radiatively. We can define the radiative and nonradiative recombination rates as

\[
R_r = \frac{\delta n}{\tau_r} \tag{14.56a}
\]

and

\[
R_{nr} = \frac{\delta n}{\tau_{nr}}
\]

(14.56b)

where \(\tau_r\) and \(\tau_{nr}\) are the radiative and nonradiative recombination lifetimes, respectively, and \(\delta n\) is the excess carrier concentration. The total recombination rate is

\[
R = R_r + R_{nr} = \frac{\delta n}{\tau} = \frac{\delta n}{\tau_r} + \frac{\delta n}{\tau_{nr}}
\]

(14.57)

where \(\tau\) is the net excess carrier lifetime.

The radiative efficiency is defined as the fraction of recombinations that are radiative. We can write

\[
\eta = \frac{R_r}{R_r + R_{nr}} = \frac{1}{\frac{\tau}{\tau_r} + \frac{\tau}{\tau_{nr}}} = \frac{\tau_r}{\tau}
\]

(14.58)

where \(\eta\) is the radiative efficiency. The nonradiative recombination rate is proportional to \(N_t\), which is the density of nonradiative trapping sites within the forbidden bandgap. Obviously, the radiative efficiency increases as \(N_t\) is reduced.

The internal quantum efficiency is now written as

\[
\eta_i = \eta \eta
\]

(14.59)

The radiative recombination rate is proportional to the p-type doping. As the p-type doping increases, the radiative recombination rate increases. However, the injection efficiency decreases as the p-type doping increases; therefore, there is an optimum doping that maximizes the internal quantum efficiency.

## 14.5.3 External Quantum Efficiency

One very important parameter of the LED is the **external quantum efficiency**: the fraction of generated photons that are actually emitted from the semiconductor. The external quantum efficiency is normally a much smaller number than the internal quantum efficiency. Once a photon has been produced in the semiconductor, there are three loss mechanisms the photon may encounter: photon absorption within the semiconductor, Fresnel loss, and critical angle loss.

Figure 14.25 shows a pn junction LED. Photons can be emitted in any direction. Since the emitted photon energy must be \(h\nu \geq E_g\), these emitted photons can be reabsorbed within the semiconductor material. The majority of photons will actually be emitted away from the surface and reabsorbed in the semiconductor.

Photons must be emitted from the semiconductor into air; thus, the photons must be transmitted across a dielectric interface. Figure 14.26 shows the incident, reflected, and transmitted waves. The parameter \(n_2\) is the index of refraction for the semiconductor and \(n_1\) is the index of refraction for air. The reflection coefficient is

\[
\Gamma = \left( \frac{n_2 - n_1}{n_2 + n_1} \right)^2
\]

(14.60)

This effect is called Fresnel loss. The reflection coefficient \(\Gamma\) is the fraction of incident photons that are reflected back into the semiconductor.

!Figure 14.25  
**Figure 14.25** | Schematic of photon emission at the pn junction of an LED.

!Figure 14.26  
**Figure 14.26** | Schematic of incident, reflected, and transmitted photons at a dielectric interface.

# 14.5 Light Emitting Diodes

## Objective

Calculate the reflection coefficient at a semiconductor–air interface. Consider the interface between a GaAs semiconductor and air.

### Solution

The index of refraction for GaAs is \( n_2 = 3.8 \) at a wavelength of \( \lambda = 0.70 \, \mu m \) and the index of refraction for air is \( n_1 = 1.0 \). The reflection coefficient is

\[
\Gamma = \left( \frac{n_2 - n_1}{n_2 + n_1} \right)^2 = \left( \frac{3.8 - 1.0}{3.8 + 1.0} \right)^2 = 0.34
\]

### Comment

A reflection coefficient of \( \Gamma = 0.34 \) means that 34 percent of the photons incident from the gallium arsenide onto the semiconductor–air interface are reflected back into the semiconductor.

### EXERCISE PROBLEM

**Ex 14.8**  
At a wavelength of \( \lambda = 0.70 \, \mu m \), the index of refraction for GaAs is \( n_2 = 3.8 \) and that for GaP is \( n_2 = 3.2 \). Consider a GaAs\(_{1-x}\)P\(_x\) material with a mole fraction \( x = 0.40 \). Assuming the index of refraction is a linear function of the mole fraction, determine the reflection coefficient, \( \Gamma \), at the GaAs\(_{0.6}\)P\(_{0.4}\)–air interface.  
(\(1 \, \text{eV} = 1 \, \text{J}\))

Photons incident on the semiconductor–air interface at an angle are refracted as shown in Figure 14.27. If the photons are incident on the interface at an angle greater than the critical angle \( \theta_c \), the photons experience total internal reflection. The critical angle is determined from Snell’s law and is given by

\[
\theta_c = \sin^{-1} \left( \frac{n_1}{n_2} \right)
\]

(14.61)

# CHAPTER 14 Optical Devices

!Figure 14.27  
**Figure 14.27** | Schematic showing refraction and total internal reflection at the critical angle at a dielectric interface.

## EXAMPLE 14.9

**Objective:** Calculate the critical angle at a semiconductor–air interface. Consider the interface between GaAs and air.

### Solution

For GaAs, \( n_2 = 3.8 \) at a wavelength of \( \lambda = 0.70 \, \mu m \) and for air, \( n_1 = 1.0 \). The critical angle is

\[
\theta_c = \sin^{-1} \left( \frac{n_1}{n_2} \right) = \sin^{-1} \left( \frac{1.0}{3.8} \right) = 15.3^\circ
\]

### Comment

Any photon that is incident at an angle greater than 15.3° will be reflected back into the semiconductor.

### EXERCISE PROBLEM

**Ex 14.9** Repeat Example 14.9 for GaAs\(_{0.5}\)P\(_{0.4}\). See Exercise Problem Ex 14.8 for a discussion of the dielectric constant.  
(\(c. \, 91 = 2^\circ \) SUV)

Figure 14.28a shows the external quantum efficiency plotted as a function of the p-type doping concentration and Figure 14.28b is a plot of the external efficiency as a function of junction depth below the surface. Both figures show that the external quantum efficiency is in the range of 1 to 3 percent.

## 14.5.4 LED Devices

The wavelength of the output signal of an LED is determined by the bandgap energy of the semiconductor. Gallium arsenide, a direct bandgap material, has a bandgap energy of \( E_g = 1.42 \, \text{eV} \), which yields a wavelength of \( \lambda = 0.873 \, \mu m \). Comparing this wavelength to the visible spectrum, which is shown in Figure 14.5, the output...

# 14.5 Light Emitting Diodes

!Graphs

**Figure 14.28** (a) External quantum efficiency of a GaP LED versus acceptor doping. (b) External quantum efficiency of a GaAs LED versus junction depth.  
*(From Yang [22].)*

**Figure 14.29** Brightness of GaAsP diodes versus wavelength (or versus bandgap energy).  
*(From Yang [22].)*

The output of a GaAs LED is not in the visible range. For a visible output, the wavelength of the signal should be in the range of 0.4 to 0.72 μm. This range of wavelengths corresponds to bandgap energies between approximately 1.7 and 3.1 eV.

GaAs\(_{1-x}\)P\(_x\) is a direct bandgap material for \(0 \leq x \leq 0.45\), as shown in Figure 14.24. At \(x = 0.40\), the bandgap energy is approximately \(E_g = 1.9\) eV, which would produce an optical output in the red range. Figure 14.29 shows the brightness of GaAs\(_{1-x}\)P\(_x\) diodes for different values of \(x\). The peak also occurs in the red range. By using planar technology, GaAs\(_{0.6}\)P\(_{0.4}\) monolithic arrays have been fabricated for numeric and alphanumeric displays. When the mole fraction \(x\) is greater than 0.45, the material changes to an indirect bandgap semiconductor so that the quantum efficiency is greatly reduced.

GaAlAs\(_{1-x}\) can be used in a heterojunction structure to form an LED. A device structure is shown in Figure 14.30. Electrons are injected from the wide-bandgap \(N\)-GaAl\(_{0.3}\)As\(_{0.7}\) into the narrow-bandgap \(p\)-GaAl\(_{0.6}\)As\(_{0.4}\). The minority carrier electrons

# CHAPTER 14 Optical Devices

!Figure 14.30

**Figure 14.30** | The (a) cross section and (b) thermal equilibrium energy-band diagram of a GaAlAs heterojunction LED. (From Yang [22].)

In the p material can recombine radiatively. Since \( E_g < E_{gN} \), the photons are emitted through the wide-bandgap N material with essentially no absorption. The wide bandgap N material acts as an optical window and the external quantum efficiency increases.

## 14.6 LASER DIODES

The photon output of the LED is due to an electron giving up energy as it makes a transition from the conduction band to the valence band. The LED photon emission is spontaneous in that each band-to-band transition is an independent event. The spontaneous emission process yields a spectral output of the LED with a fairly wide bandwidth. If the structure and operating condition of the LED are modified, the device can operate in a new mode, producing a coherent spectral output with a bandwidth of wavelengths less than 0.1 nm. This new device is a laser diode, where laser stands for Light Amplification by Stimulated Emission of Radiation. Although there are many different types of lasers, we are here concerned only with the pn junction laser diode.

# 14.6.1 Stimulated Emission and Population Inversion

Figure 14.31a shows the case when an incident photon is absorbed and an electron is elevated from an energy state \( E_1 \) to an energy state \( E_2 \). This process is known as induced absorption. If the electron spontaneously makes the transition back to the lower energy level with a photon being emitted, we have a spontaneous emission process as indicated in Figure 14.31b. On the other hand, if there is an incident photon at a time when an electron is in the higher energy state as shown in Figure 14.31c, the incident photon can interact with the electron, causing the electron to make a transition downward. The downward transition produces a photon. Since this process was initiated by the incident photon, the process is called **stimulated or induced emission**. Note that this stimulated emission process has produced two photons; thus, we can have optical gain or amplification. The two emitted photons are in phase so that the spectral output will be coherent.

In thermal equilibrium, the electron distribution in a semiconductor is determined by the Fermi–Dirac statistics. If the Boltzmann approximation applies, then we can write

\[
\frac{N_2}{N_1} = \exp \left[ -\frac{(E_2 - E_1)}{kT} \right]
\]

(14.62)

where \( N_1 \) and \( N_2 \) are the electron concentrations in the energy levels \( E_1 \) and \( E_2 \), respectively, and where \( E_2 > E_1 \). In thermal equilibrium, \( N_2 < N_1 \). The probability of an induced absorption event is exactly the same as that of an induced emission event. The number of photons absorbed is proportional to \( N_1 \) and the number of additional photons emitted is proportional to \( N_2 \). In order to achieve optical amplification or for lasing action to occur, we must have \( N_2 > N_1 \); this is called population inversion. We cannot achieve lasing action at thermal equilibrium.

!Figure 14.31

**Figure 14.31** | Schematic diagram showing (a) induced absorption, (b) spontaneous emission, and (c) stimulated emission processes.

- **(a)** Induced absorption
- **(b)** Spontaneous emission
- **(c)** Stimulated or induced emission

- \( E_2 \)
- \( E_1 \)
- \( h\nu \)

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

# 14.6 Laser Diodes

!Figure 14.33

**Figure 14.33** (a) Degenerately doped pn junction at zero bias. (b) Degenerately doped pn junction under forward bias with photon emission.

!Figure 14.34

**Figure 14.34** A pn junction laser diode with cleaved (110) planes forming the Fabry-Perot cavity. (After Yang [22].)

## 14.6.2 Optical Cavity

Population inversion is one requirement for lasing action to occur. Coherent emission output is achieved by using an optical cavity. The cavity will cause a buildup of the optical intensity from positive feedback. A resonant cavity consisting of two parallel mirrors is known as a Fabry–Perot resonator. The resonant cavity can be fabricated, for example, by cleaving a gallium arsenide crystal along the (110) planes as shown in Figure 14.34. The optical wave propagates through the junction in the z direction, bouncing back and forth between the end mirrors. The mirrors are actually only partially reflecting so that a portion of the optical wave will be transmitted out of the junction.

For resonance, the length of the cavity \( L \) must be an integral number of half wavelengths, or

\[
N \left( \frac{\lambda}{2} \right) = L
\]

(14.67)

where \( N \) is an integer. Since \( \lambda \) is small and \( L \) is relatively large, there can be many resonant modes in the cavity. Figure 14.35a shows the resonant modes as a function of wavelength.

# CHAPTER 14 Optical Devices

!Figure 14.35

**Figure 14.35** | Schematic diagram showing (a) resonant modes of a cavity with length \( L \), (b) spontaneous emission curve, and (c) actual emission modes of a laser diode.  
(After Yang [22].)

When a forward-bias current is applied to the pn junction, spontaneous emission will initially occur. The spontaneous emission spectrum is relatively broadband and is superimposed on the possible lasing modes as shown in Figure 14.35b. In order for lasing to be initiated, the spontaneous emission gain must be larger than the optical losses. By positive feedback in the cavity, lasing can occur at several specific wavelengths as indicated in Figure 14.35c.

## 14.6.3 Threshold Current

The optical intensity in the device can be written from Equation (14.65) as

\[
I_{\nu} \propto e^{\gamma(\nu)x}
\]

where \(\gamma(\nu)\) is the amplification factor. We have two basic loss mechanisms. The first is the photon absorption in the semiconductor material. We can write

\[
I_{\nu} \propto e^{-\alpha(\nu)x} \tag{14.68}
\]

where \(\alpha(\nu)\) is the absorption coefficient. The second loss mechanism is due to the partial transmission of the optical signal through the ends, or through the partially reflecting mirrors.

# 14.6 Laser Diodes

At the onset of lasing, which is known as threshold, the optical loss of one round trip through the cavity is just offset by the optical gain. The threshold condition is then expressed as

\[
\Gamma \Gamma_1 \Gamma_2 \exp\left[ (2\gamma(\nu) - 2\alpha(\nu))L \right] = 1
\]

(14.69)

where \(\Gamma_1\) and \(\Gamma_2\) are the reflectivity coefficients of the two end mirrors. For the case when the optical mirrors are cleaved (110) surfaces of gallium arsenide, the reflectivity coefficients are given approximately by

\[
\Gamma_1 = \Gamma_2 = \left( \frac{\bar{n} - \bar{n}_i}{\bar{n} + \bar{n}_i} \right)^2
\]

(14.70)

where \(\bar{n}\) and \(\bar{n}_i\) are the index of refraction parameters for the semiconductor and air, respectively. The parameter \(\gamma(\nu)\) is the optical gain at threshold.

The optical gain at threshold, \(\gamma(\nu)\), may be determined from Equation (14.69) as

\[
\gamma(\nu) = \alpha + \frac{1}{2L} \ln \left( \frac{1}{\Gamma_1 \Gamma_2} \right)
\]

(14.71)

Since the optical gain is a function of the pn junction current, we can define a threshold current density as

\[
J_{th} = \frac{1}{\beta} \left[ \alpha + \frac{1}{2L} \ln \left( \frac{1}{\Gamma_1 \Gamma_2} \right) \right]
\]

(14.72)

where \(\beta\) can be determined theoretically or experimentally. Figure 14.36 shows the threshold current density as a function of the mirror losses. We may note the relatively high threshold current density for a pn junction laser diode.

!Threshold current density of a laser diode as a function of Fabry-Perot cavity end losses.

**Figure 14.36** | Threshold current density of a laser diode as a function of Fabry-Perot cavity end losses.  
*(After Yang [22].)*

### Table: Threshold Current Density

| \( \frac{1}{2L} \ln \frac{1}{\Gamma_1 \Gamma_2} \) (cm\(^{-1}\)) | \( J_{th} \) (A/cm\(^2\)) |
|:---------------------------------------------------------------:|:-------------------------:|
| 0                                                               | 0                         |
| 10                                                              | 500                       |
| 20                                                              | 1000                      |
| 30                                                              | 1500                      |
| 40                                                              | 2000                      |
| 50                                                              | 2500                      |
| 60                                                              | 3000                      |
| 70                                                              | 3500                      |

- \(\bar{\alpha} = 15 \, \text{cm}^{-1}\)
- \(\bar{\beta} = 2.1 \times 10^{-2} \, \text{cm/A}\)

# 14.6.4 Device Structures and Characteristics

We have seen that in a homojunction LED, the photons may be emitted in any direction, which lowers the external quantum efficiency. Significant improvement in device characteristics can be made if the emitted photons are confined to a region near the junction. This confinement can be achieved by using an optical dielectric waveguide. The basic device is a three-layered, double heterojunction structure known as a double heterojunction laser. A requirement for a dielectric waveguide is that the index of refraction of the center material be larger than that of the other two dielectrics. Figure 14.37 shows the index of refraction for the AlGaAs system. We may note that GaAs has the highest index of refraction.

An example of a double heterojunction laser is shown in Figure 14.38a. A thin p-GaAs layer is between P-AlGaAs and N-AlGaAs layers. A simplified energy-band diagram is shown in Figure 14.38b for the forward-biased diode. Electrons are injected from the N-AlGaAs into the p-GaAs. Population inversion is easily obtained since the conduction band potential barrier prevents the electrons from diffusing into the P-AlGaAs region. Radiative recombination is then confined to the p-GaAs region. Since the index of refraction of GaAs is larger than that of AlGaAs, the light wave is also confined to the GaAs region. An optical cavity can be formed by cleaving the semiconductor perpendicular to the N-AlGaAs–p-GaAs junction.

!Figure 14.37

**Figure 14.37** Index of refraction of Al\(_x\)Ga\(_{1-x}\)As as a function of mole fraction \(x\).  
*(From Sze [18].)*

### Graph Details

- **X-axis**: Mole fraction AlAs, \(x\)
  - Range: 0 (GaAs) to 1.0 (AlAs)
- **Y-axis**: Refractive index, \(n\)
  - Range: 2.9 to 3.6
- **Line**: \(h\nu = 1.38 \, \text{eV}\), \(T = 297 \, \text{K}\)

!Figure 14.38

**Figure 14.38** (a) Basic double heterojunction structure. (b) Energy-band diagram under forward bias. (c) Refractive index change through the structure. (d) Confinement of light in the dielectric waveguide.  
*(From Yang [22].)*

### Diagram Details

- **(a)** Structure: \(n\) Al\(_x\)Ga\(_{1-x}\)As, \(p\) GaAs, \(p\) Al\(_x\)Ga\(_{1-x}\)As
- **(b)** Energy-band diagram: \(J_n^-\), \(J_p^+\)
- **(c)** Refractive index: \(<1 \, \mu\text{m}\), \(\approx 5\%\)
- **(d)** Light confinement: \(\approx 0.1 \, \mu\text{m}\)

# 14.7 Summary

!Graph

**Figure 14.39** | Typical output power versus laser diode current at various temperatures.  
*(From Yang [22].)*

Typical optical output versus diode current characteristics are shown in Figure 14.39. The threshold current is defined to be the current at the breakpoint. At low currents, the output spectrum is very wide and is the result of the spontaneous transitions. When the diode current is slightly above the threshold value, the various resonant frequencies are observed. When the diode current becomes large, a single dominant mode with a narrow bandwidth is produced.

The performance of the laser diode can be further improved if a very narrow recombination region is used with a somewhat wider optical waveguide. Very complex structures using multilayers of compound semiconductor materials have been fabricated in a continuing effort to improve semiconductor laser performance.

## 14.7 | SUMMARY

- The absorption or emission of light (photons) in semiconductors leads to the study of a general class of devices called optoelectronics. A few of these devices have been discussed and analyzed in this chapter.

### Graph Details

- **AlGaAs DH laser**
  - \(\lambda (20^\circ C) = 827 \, \text{nm}\)
  - \(W = 12 \, \mu\text{m}\)
  - \(L = 130 \, \mu\text{m}\)

- **Graph Axes**
  - **Y-axis:** CW power emission (one facet) (mW)
  - **X-axis:** Diode current (mA)

- **Temperature Angles:**
  - \(0^\circ\)
  - \(10^\circ\)
  - \(20^\circ\)
  - \(30^\circ\)
  - \(40^\circ\)
  - \(50^\circ\)
  - \(60^\circ\)
  - \(70^\circ\)

- **Threshold current at 70°C**

# CHAPTER 14 Optical Devices

- The photon absorption process has been discussed and the absorption coefficient data for semiconductors has been presented.
- Solar cells convert optical power into electrical power. The simple pn junction solar cell was initially considered. The short-circuit current, open-circuit voltage, and maximum power were considered.
- Heterojunction and amorphous silicon solar cells were also considered. Heterojunction cells can be fabricated that tend to increase the conversion efficiency and produce relatively large open-circuit voltages. Amorphous silicon offers the possibility of low-cost, large-area solar cell arrays.
- Photodetectors are semiconductor devices that convert optical signals into electrical signals. The photoconductor is perhaps the simplest type of photodetector. The change in conductivity of the semiconductor due to the creation of excess electrons and holes by the incident photons is the basis of this device.
- Photodiodes are diodes that have reverse-biased voltages applied. Excess carriers that are created by incident photons in the space-charge region are swept out by the electric field creating a photocurrent. The photocurrent is directly proportional to the incident photon intensity. PIN and avalanche photodiodes are variations of the basic photodiode.
- The photocurrent generated in a phototransistor is multiplied by the transistor gain. However, the time response of the phototransistor may be slower than that of a photodiode because of the Miller effect and Miller capacitance.
- The inverse mechanism of photon absorption in a pn junction is injection electroluminescence. The recombination of excess electrons and holes in a direct bandgap semiconductor can result in the emission of photons.
- The light emitting diodes (LEDs) are the class of pn junction diodes whose photon output is a result of spontaneous recombinations of excess electrons and holes. A fairly wide bandwidth in the output signal, on the order of 30 nm, is a result of the spontaneous process.
- The output of a laser diode is the result of stimulated emission. An optical cavity, or Fabry–Perot resonator, is used in conjunction with a diode so that the photon output is in phase, or coherent. Multilayered heterojunction structures can be fabricated to improve the laser diode characteristics.

## GLOSSARY OF IMPORTANT TERMS

- **absorption coefficient**: The relative number of photons absorbed per unit distance in a semiconductor and denoted by the parameter \(\alpha\).

- **conversion efficiency**: The ratio of output electrical power to incident optical power in a solar cell.

- **delayed photocurrent**: The component of photocurrent in a semiconductor device due to diffusion currents.

- **external quantum efficiency**: The ratio of emitted photons to generated photons in a semiconductor device.

- **fill factor**: The ratio \(I_m V_m\) to \(I_{sc} V_{oc}\), which is a measure of the realizable power from a solar cell. The parameters \(I_m\) and \(V_m\) are the current and voltage at the maximum power point, respectively, and \(I_{sc}\) and \(V_{oc}\) are the short-circuit current and open-circuit voltage.

- **Fresnel loss**: The ratio of reflected to incident photons at an interface due to a change in the index of refraction.

# Review Questions

### Internal Quantum Efficiency
The fraction of diode current that produces luminescence.

### LASER Diode
An acronym for Light Amplification by Stimulated Emission of Radiation: the stimulated emission of photons produced in a forward-biased pn junction in conjunction with an optical cavity.

### LED
An acronym for Light Emitting Diode; the spontaneous photon emission due to electron–hole recombination in a forward-biased pn junction.

### Luminescence
The general property of light emission.

### Open-Circuit Voltage
The voltage generated across the open-circuited terminals of a solar cell.

### Photocurrent
The current generated in a semiconductor device due to the flow of excess carriers generated by the absorption of photons.

### Population Inversion
The condition whereby the concentration of electrons in one energy state is greater than that in a lower energy state; a nonequilibrium condition.

### Prompt Photocurrent
The component of photocurrent generated within the space charge region of a semiconductor device.

### Radiative Recombination
The recombination process of electrons and holes that produces a photon, such as the direct band-to-band transition in gallium arsenide.

### Short-Circuit Current
The current produced in a solar cell when the two terminals are shorted together.

### Stimulated Emission
The process whereby an electron is induced by an incident photon to make a transition to a lower energy state, emitting a second photon.

## CHECKPOINT

After studying this chapter, the reader should have the ability to:

- Describe the optical absorption process in semiconductors. When is optical absorption essentially zero?
- Describe the basic operation and characteristics of a solar cell, including the short-circuit current and open-circuit voltage.
- Discuss the factors that contribute to the solar cell conversion efficiency.
- Describe the advantages and disadvantages of an amorphous silicon solar cell.
- Describe the characteristics of a photoconductor, including the concept of the photoconductor gain.
- Discuss the operation and characteristics of a simple pn junction photodiode.
- Discuss the advantages of PIN and avalanche photodiodes compared to the simple pn junction photodiode.
- Discuss the operation and characteristics of a phototransistor.
- Describe the operation of an LED.
- Describe the operation of a laser diode.

## REVIEW QUESTIONS

1. Sketch the general shape of the optical absorption coefficient in a semiconductor as a function of wavelength. When does the absorption coefficient become zero?

2. Sketch the I–V characteristic of a pn junction solar cell. Define short-circuit current and open-circuit voltage.

# CHAPTER 14 Optical Devices

3. Discuss how a pn junction solar cell becomes forward biased.

4. Write an expression for the steady-state photocurrent in a simple photoconductor.

5. What is the source of prompt photocurrent in a photodiode? Does the prompt photocurrent depend on the reverse-biased voltage? Why or why not?

6. Sketch the cross section of a phototransistor and show the currents that are created by incident photons. Explain how current gain is achieved.

7. Explain the basic operation of an LED. State two factors that affect the efficiency of the device.

8. How can different colors be obtained in an LED?

9. Discuss the difference between an LED and a laser diode.

10. Discuss the concept of population inversion in a laser diode.

## PROBLEMS

### Section 14.1 Optical Absorption

**14.1** Determine the maximum wavelength \(\lambda\) of a light source that can generate electron-hole pairs in (a) Si, (b) Ge, (c) GaAs, and (d) InP.

**14.2** 
(a) Two sources generate light at wavelengths of \(\lambda = 480 \, \text{nm}\) and \(\lambda = 725 \, \text{nm}\), respectively. What are the corresponding photon energies?  
(b) Three sources generate light with photon energies of \(E = 0.87 \, \text{eV}, E = 1.32 \, \text{eV},\) and \(E = 1.90 \, \text{eV}\), respectively. What are the corresponding wavelengths?

**14.3** 
(a) A sample of GaAs is \(1.2 \, \mu \text{m}\) thick. The sample is illuminated with a light source that generates photons with energies of \(h\nu = 1.65 \, \text{eV}\). Determine the (i) absorption coefficient and (ii) fraction of energy that is absorbed in the material.  
(b) Repeat part (a) for a sample of GaAs that is \(0.80 \, \mu \text{m}\) thick and is illuminated with photons with energies of \(h\nu = 1.90 \, \text{eV}\).

**14.4** A light source with \(h\nu = 1.3 \, \text{eV}\) and at a power density of \(10^{-2} \, \text{W/cm}^2\) is incident on a thin slab of silicon. The excess minority carrier lifetime is \(10^{-6} \, \text{s}\). Determine the electron-hole generation rate and the steady-state excess carrier concentration. Neglect surface effects.

**14.5** An n-type GaAs sample has a minority carrier lifetime of \(\tau_p = 2 \times 10^{-7} \, \text{s}\). Incident photons with energies \(h\nu = 1.65 \, \text{eV}\) generate an excess carrier concentration of \(\delta p = 5 \times 10^{15} \, \text{cm}^{-3}\) at the surface of the semiconductor.  
(a) Determine the incident power required.  
(b) At what distance in the semiconductor does the generation rate drop to 10 percent of that at the surface?

**14.6** Consider a silicon semiconductor that is illuminated with photons with energies \(h\nu = 1.40 \, \text{eV}\).  
(a) Determine the thickness of the material such that 90 percent of the energy is absorbed.  
(b) Determine the thickness of the material such that 30 percent of the energy is transmitted through the material.

**14.7** If the thickness of a GaAs semiconductor is \(1 \, \mu \text{m}\) and 50 percent of the incident monochromatic photon energy is absorbed, determine the incident photon energy and wavelength.

**14.8** Consider monochromatic light at an intensity \(I_0\) incident on the surface at \(x = 0\) of an n-type semiconductor that extends to \(x = \infty\). Assume the electric field is zero in the semiconductor and assume a surface recombination velocity, \(s\). Taking into account...

*Note: Asterisks next to problems indicate problems that are more difficult.*

