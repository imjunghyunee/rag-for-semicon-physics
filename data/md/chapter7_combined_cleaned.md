# Chapter 7: The pn Junction

Up to this point in the text, we have been considering the properties of the semiconductor material. We calculated electron and hole concentrations in thermal equilibrium and determined the position of the Fermi level. We then considered the nonequilibrium condition in which excess electrons and holes are present in the semiconductor. We now wish to consider the situation in which a p-type and an n-type semiconductor are brought into contact with one another to form a pn junction.

Most semiconductor devices contain at least one junction between p-type and n-type semiconductor regions. Semiconductor device characteristics and operation are intimately connected to these pn junctions, so considerable attention is devoted initially to this basic device.

The electrostatics of the zero-biased and reverse-biased pn junction is considered in this chapter. The current–voltage characteristics of the pn junction diode are developed in the next chapter.

## 7.0 | PREVIEW

In this chapter, we will:

- Consider a uniformly doped pn junction, in which one region of the semiconductor is uniformly doped with acceptor atoms and the adjacent region is uniformly doped with donor atoms.
- Determine the energy-band diagram of a pn junction in thermal equilibrium.
- Discuss the creation of a space charge region between the p and n regions.
- Apply Poisson’s equation to determine the electric field in the space charge region and calculate the built-in potential barrier.
- Analyze the changes that occur in the pn junction when a reverse-biased voltage is applied. Derive expressions for space charge width and depletion capacitance.
- Analyze the voltage breakdown characteristics of a pn junction.
- Consider the properties of a nonuniformly doped pn junction. Specific doping profiles can lead to desirable properties of the pn junction.

## 7.1 BASIC STRUCTURE OF THE pn JUNCTION

Figure 7.1a schematically shows the pn junction. It is important to realize that the entire semiconductor is a single-crystal material in which one region is doped with acceptor impurity atoms to form the p region and the adjacent region is doped with donor atoms to form the n region. The interface separating the n and p regions is referred to as the **metallurgical junction**.

The impurity doping concentrations in the p and n regions are shown in Figure 7.1b. For simplicity, we will consider a **step junction** in which the doping concentration is uniform in each region and there is an abrupt change in doping at the junction. Initially, at the metallurgical junction, there is a very large density gradient in both electron and hole concentrations. Majority carrier electrons in the n region will begin diffusing into the p region, and majority carrier holes in the p region will begin diffusing into the n region. If we assume there are no external connections to the semiconductor, then this diffusion process cannot continue indefinitely. As electrons diffuse from the n region, positively charged donor atoms are left behind. Similarly, as holes diffuse from the p region, they uncover negatively charged acceptor atoms. The net positive and negative charges in the n and p regions induce an electric field in the region near the metallurgical junction, in the direction from the positive to the negative charge, or from the n to the p region.

The net positively and negatively charged regions are shown in Figure 7.2. These two regions are referred to as the **space charge region**. Essentially all electrons and holes are swept out of the space charge region by the electric field. Since the space charge region is depleted of any mobile charge, this region is also referred to as the **depletion region**; these two terms will be used interchangeably. Density gradients still exist in the majority carrier concentrations at each edge of the space charge region. We can think of a density gradient as producing a "diffusion force" that acts on the majority carriers. These diffusion forces, acting on the electrons and holes at the edges of the space charge region, are shown in the figure. The electric field in the space charge region produces another force on the electrons and holes, which is in the opposite direction to the diffusion force for each type of particle. In thermal equilibrium, the diffusion force and the E-field force exactly balance each other.


**Figure 7.1** (a) Simplified geometry of a pn junction; (b) doping profile of an ideal uniformly doped pn junction.


**Figure 7.2** The space charge region, the electric field, and the forces acting on the charged carriers.

- **Figure 7.1a**: Shows the metallurgical junction between p and n regions.
- **Figure 7.1b**: Illustrates the doping profile with \( N_a \) and \( N_d \) indicating acceptor and donor concentrations, respectively.

- **Figure 7.2**: Depicts the space charge region, electric field (E-field), and diffusion forces on holes and electrons. The space charge region is shown with \( N_a \) negative charge and \( N_d \) positive charge, indicating the direction of the electric field and diffusion forces.


## 7.2 Zero Applied Bias

We have considered the basic pn junction structure and discussed briefly how the space charge region is formed. In this section we will examine the properties of the step junction in thermal equilibrium, where no currents exist and no external excitation is applied. We will determine the space charge region width, electric field, and potential through the depletion region.

The analysis in this chapter is based on two assumptions that we have considered in previous chapters. The first assumption is that the Boltzmann approximation is valid, which means that each semiconductor region is nondegenerately doped. The second assumption is that complete ionization exists, which means that the temperature of the pn junction is not "too low."

### 7.2.1 Built-in Potential Barrier

If we assume that no voltage is applied across the pn junction, then the junction is in thermal equilibrium—the Fermi energy level is constant throughout the entire system. Figure 7.3 shows the energy-band diagram for the pn junction in thermal equilibrium. The conduction and valence band energies must bend as we go through the space charge region, since the relative position of the conduction and valence bands with respect to the Fermi energy changes between p and n regions.

**Figure 7.3** | Energy-band diagram of a pn junction in thermal equilibrium.

- \( E_c \) and \( E_v \) represent the conduction and valence band energies.
- \( E_{Fp} \) and \( E_{Fn} \) are the Fermi energy levels in the p and n regions.
- \( eV_0 \) is the built-in potential barrier.
- \( e\phi_{Fp} \) and \( e\phi_{Fn} \) are the energy differences between the Fermi level and the intrinsic level in the p and n regions, respectively.

Electrons in the conduction band of the n region see a potential barrier in trying to move into the conduction band of the p region. This potential barrier is referred to as the **built-in potential barrier** and is denoted by \( V_b \). The built-in potential barrier maintains equilibrium between majority carrier electrons in the n region and minority carrier electrons in the p region, and also between majority carrier holes in the p region and minority carrier holes in the n region. This potential difference across the junction cannot be measured with a voltmeter because new potential barriers will be formed between the probes and the semiconductor that will cancel \( V_b \). The potential \( V_b \) maintains equilibrium, so no current is produced by this voltage.

The intrinsic Fermi level is equidistant from the conduction band edge through the junction; thus, the built-in potential barrier can be determined as the difference between the intrinsic Fermi levels in the p and n regions. We can define the potentials \( \phi_{Fn} \) and \( \phi_{Fp} \) as shown in Figure 7.3, so we have

\[
V_{bi} = |\phi_{Fn}| + |\phi_{Fp}|
\]

(7.1)

In the n region, the electron concentration in the conduction band is given by

\[
n_0 = N_c \exp \left( \frac{-(E_c - E_F)}{kT} \right)
\]

(7.2)

which can also be written in the form

\[
n_0 = n_i \exp \left( \frac{E_F - E_{Fi}}{kT} \right)
\]

(7.3)

where \( n_i \) and \( E_{Fi} \) are the intrinsic carrier concentration and the intrinsic Fermi energy, respectively. We may define the potential \( \phi_{Fn} \) in the n region as

\[
e\phi_{Fn} = E_{Fi} - E_F
\]

(7.4)

Equation (7.3) may then be written as

\[
n_0 = n_i \exp \left( \frac{-e\phi_{Fn}}{kT} \right)
\]

(7.5)

Taking the natural log of both sides of Equation (7.5), setting \( n_0 = N_d \), and solving for the potential, we obtain

\[
\phi_{Fn} = -\frac{kT}{e} \ln \left( \frac{N_d}{n_i} \right)
\]

(7.6)

Similarly, in the p region, the hole concentration is given by

\[
p_0 = N_a = n_i \exp \left( \frac{E_{Fi} - E_F}{kT} \right)
\]

(7.7)

where \( N_a \) is the acceptor concentration. We can define the potential \( \phi_{Fp} \) in the p region as

\[
e\phi_{Fp} = E_{Fi} - E_F
\]

(7.8)

Combining Equations (7.7) and (7.8), we find that

\[
\phi_{fp} = \frac{kT}{e} \ln \left( \frac{N_a}{n_i} \right)
\]

(7.9)

Finally, the built-in potential barrier for the step junction is found by substituting Equations (7.6) and (7.9) into Equation (7.1), which yields

\[
V_0 = \frac{kT}{e} \ln \left( \frac{N_a N_d}{n_i^2} \right) = V_t \ln \left( \frac{N_a N_d}{n_i^2} \right)
\]

(7.10)

where \( V_t = kT/e \) and is defined as the thermal voltage.

At this time, we should note a subtle but important point concerning notation. Previously in the discussion of a semiconductor material, \( N_d \) and \( N_a \) denoted donor and acceptor impurity concentrations in the same region, thereby forming a compensated semiconductor. From this point on in the text, \( N_d \) and \( N_a \) will denote the net donor and acceptor concentrations in the individual n and p regions, respectively. If the p region, for example, is a compensated material, then \( N_a \) will represent the difference between the actual acceptor and donor impurity concentrations. The parameter \( N_d \) is defined in a similar manner for the n region.

## EXAMPLE 7.1

**Objective:** Calculate the built-in potential barrier in a pn junction.

Consider a silicon pn junction at \( T = 300 \, K \) with doping concentrations of \( N_d = 2 \times 10^{17} \, \text{cm}^{-3} \) and \( N_a = 10^{15} \, \text{cm}^{-3} \).

**Solution**

The built-in potential barrier is determined from Equation (7.10) as

\[
V_0 = V_t \ln \left( \frac{N_a N_d}{n_i^2} \right) = (0.0259) \ln \left( \frac{2 \times 10^{17} \times 10^{15}}{(1.5 \times 10^{10})^2} \right) = 0.713 \, V
\]

If we change the doping concentration in the p region of the pn junction such that the doping concentrations become \( N_a = 10^{16} \, \text{cm}^{-3} \) and \( N_d = 10^{15} \, \text{cm}^{-3} \), then the built-in potential barrier becomes \( V_0 = 0.635 \, V \).

**Comment**

The built-in potential barrier changes only slightly as the doping concentrations change by orders of magnitude because of the logarithmic dependence.

**EXERCISE PROBLEM**

Ex 7.1 (a) Calculate the built-in potential barrier in a silicon pn junction at \( T = 300 \, K \) for 
(i) \( N_a = 5 \times 10^{15} \, \text{cm}^{-3} \), \( N_d = 10^{17} \, \text{cm}^{-3} \) and (ii) \( N_a = 2 \times 10^{16} \, \text{cm}^{-3} \), \( N_d = 2 \times 10^{15} \, \text{cm}^{-3} \).

(b) Repeat part (a) for a GaAs pn junction.


**Figure 7.4** The space charge density in a uniformly doped pn junction assuming the abrupt junction approximation.

### 7.2.2 Electric Field

An electric field is created in the depletion region by the separation of positive and negative space charge densities. Figure 7.4 shows the volume charge density distribution in the pn junction assuming uniform doping and assuming an abrupt junction approximation. We will assume that the space charge region abruptly ends in the n region at \( x = +x_n \), and abruptly ends in the p region at \( x = -x_p \) (\( x_p \) is a positive quantity).

The electric field is determined from Poisson’s equation, which, for a one-dimensional analysis, is

\[
\frac{d^2 \phi(x)}{dx^2} = -\frac{\rho(x)}{\epsilon_s} = -\frac{dE(x)}{dx}
\]

(7.11)

where \( \phi(x) \) is the electric potential, \( E(x) \) is the electric field, \( \rho(x) \) is the volume charge density, and \( \epsilon_s \) is the permittivity of the semiconductor. From Figure 7.4, the charge densities are

\[
\rho(x) = -eN_a \quad -x_p < x < 0
\]

(7.12a)

and

\[
\rho(x) = eN_d \quad 0 < x < x_n
\]

(7.12b)

The electric field in the p region is found by integrating Equation (7.11). We have

\[
E = \int \frac{\rho(x)}{\epsilon_s} \, dx = -\int \frac{eN_a}{\epsilon_s} \, dx = -\frac{eN_a x}{\epsilon_s} + C_1
\]

(7.13)

where \( C_1 \) is a constant of integration. The electric field is assumed to be zero in the neutral p region for \( x < -x_p \), since the currents are zero in thermal equilibrium. Since there are no surface charge densities within the pn junction structure, the electric

Field is a continuous function. The constant of integration is determined by setting \( E = 0 \) at \( x = -x_p \). The electric field in the p region is then given by

\[
E = -\frac{eN_a}{\varepsilon_s}(x + x_p) \quad -x_p \leq x \leq 0
\]

(7.14)

In the n region, the electric field is determined from

\[
E = \int \frac{eN_d}{\varepsilon_s} \, dx = \frac{eN_d}{\varepsilon_s} x + C_2
\]

(7.15)

where \( C_2 \) is again a constant of integration and is determined by setting \( E = 0 \) at \( x = x_n \), since the E-field is assumed to be zero in the n region and is a continuous function. Then

\[
E = -\frac{eN_d}{\varepsilon_s}(x_n - x) \quad 0 \leq x \leq x_n
\]

(7.16)

The electric field is also continuous at the metallurgical junction, or at \( x = 0 \). Setting Equations (7.14) and (7.16) equal to each other at \( x = 0 \) gives

\[
N_a x_p = N_d x_n
\]

(7.17)

Equation (7.17) states that the number of negative charges per unit area in the p region is equal to the number of positive charges per unit area in the n region.

Figure 7.5 is a plot of the electric field in the depletion region. The electric field direction is from the n to the p region, or in the negative x direction for this geometry. For the uniformly doped pn junction, the E-field is a linear function of distance through the junction, and the maximum (magnitude) electric field occurs at the metallurgical junction. An electric field exists in the depletion region even when no voltage is applied between the p and n regions.

The potential in the junction is found by integrating the electric field. In the p region then, we have

\[
\phi(x) = -\int E(x) dx = \int \frac{eN_a}{\varepsilon_s}(x + x_p) dx
\]

(7.18)


**Figure 7.5** Electric field in the space charge region of a uniformly doped pn junction.

or

\[
\phi(x) = \frac{eN_a}{\varepsilon_s} \left( \frac{x^2}{2} + x_p \cdot x \right) + C'_1
\]

(7.19)

where \( C'_1 \) is again a constant of integration. The potential difference through the pn junction is the important parameter, rather than the absolute potential, so we may arbitrarily set the potential equal to zero at \( x = -x_p \). The constant of integration is then found as

\[
C'_1 = \frac{eN_a}{2\varepsilon_s} x_p^2
\]

(7.20)

so that the potential in the p region can now be written as

\[
\phi(x) = \frac{eN_a}{2\varepsilon_s} (x + x_p)^2 \quad (-x_p \leq x \leq 0)
\]

(7.21)

The potential in the n region is determined by integrating the electric field in the n region, or

\[
\phi(x) = \int \frac{eN_d}{\varepsilon_s} (x_n - x) dx
\]

(7.22)

Then

\[
\phi(x) = \frac{eN_d}{\varepsilon_s} \left( x \cdot x_n - \frac{x^2}{2} \right) + C'_2
\]

(7.23)

where \( C'_2 \) is another constant of integration. The potential is a continuous function, so setting Equation (7.21) equal to Equation (7.23) at the metallurgical junction, or at \( x = 0 \), gives

\[
C'_2 = \frac{eN_a}{2\varepsilon_s} x_p^2
\]

(7.24)

The potential in the n region can thus be written as

\[
\phi(x) = \frac{eN_d}{\varepsilon_s} \left( x \cdot x_n - \frac{x^2}{2} \right) + \frac{eN_a}{2\varepsilon_s} x_p^2 \quad (0 \leq x \leq x_n)
\]

(7.25)

Figure 7.6 is a plot of the potential through the junction and shows the quadratic dependence on distance. The magnitude of the potential at \( x = x_n \) is equal to the built-in potential barrier. Then from Equation (7.25), we have

\[
V_{bi} = |\phi(x = x_n)| = \frac{e}{2\varepsilon_s} (N_ax_p^2 + N_dx_n^2)
\]

(7.26)

The potential energy of an electron is given by \( E = -e\phi \), which means that the electron potential energy also varies as a quadratic function of distance through the space charge region. The quadratic dependence on distance was shown in the energy-band diagram of Figure 7.3, although we did not explicitly know the shape of the curve at that time.

**Figure 7.6** | Electric potential through the space charge region of a uniformly doped pn junction.

### 7.2.3 Space Charge Width

We can determine the distance that the space charge region extends into the p and n regions from the metallurgical junction. This distance is known as the space charge width. From Equation (7.17), we may write, for example,

\[
x_p = \frac{N_A x_n}{N_D}
\]

(7.27)

Then, substituting Equation (7.27) into Equation (7.26) and solving for \(x_n\), we obtain

\[
x_n = \left[ \frac{2 \varepsilon_s V_{bi}}{e} \left( \frac{N_A}{N_D} \right) \left( \frac{1}{N_A + N_D} \right) \right]^{1/2}
\]

(7.28)

Equation (7.28) gives the space charge width, or the width of the depletion region, \(x_n\), extending into the n-type region for the case of zero applied voltage.

Similarly, if we solve for \(x_p\) from Equation (7.17) and substitute into Equation (7.26), we find

\[
x_p = \left[ \frac{2 \varepsilon_s V_{bi}}{e} \left( \frac{N_D}{N_A} \right) \left( \frac{1}{N_A + N_D} \right) \right]^{1/2}
\]

(7.29)

where \(x_p\) is the width of the depletion region extending into the p region for the case of zero applied voltage.

The total depletion or space charge width \(W\) is the sum of the two components, or

\[
W = x_n + x_p
\]

(7.30)

Using Equations (7.28) and (7.29), we obtain

\[
W = \left[ \frac{2 \varepsilon_s V_{bi}}{e} \left( \frac{N_A + N_D}{N_A N_D} \right) \right]^{1/2}
\]

(7.31)

The built-in potential barrier can be determined from Equation (7.10), and then the total space charge region width is obtained using Equation (7.31).

## EXAMPLE 7.2

**Objective:** Calculate the space charge width and electric field in a pn junction for zero bias.

Consider a silicon pn junction at \( T = 300 \, \text{K} \) with doping concentrations of \( N_d = 10^{16} \, \text{cm}^{-3} \) and \( N_a = 10^{15} \, \text{cm}^{-3} \).

**Solution**

In Example 7.1, we determined the built-in potential barrier as \( V_b = 0.635 \, \text{V} \). From Equation (7.31), the space charge width is

\[
W = \left[ \frac{2 \varepsilon_s V_b}{e} \left( \frac{N_a + N_d}{N_a N_d} \right) \right]^{1/2}
\]

\[
= \left[ \frac{2(11.7)(8.85 \times 10^{-14})(0.635)}{1.6 \times 10^{-19}} \left( \frac{10^{16} + 10^{15}}{(10^{16})(10^{15})} \right) \right]^{1/2}
\]

\[
= 0.951 \times 10^{-4} \, \text{cm} = 0.951 \, \mu\text{m}
\]

Using Equations (7.28) and (7.29), we can find \( x_n = 0.8644 \, \mu\text{m} \), and \( x_p = 0.0864 \, \mu\text{m} \).

The peak electric field at the metallurgical junction, using Equation (7.16) for example, is

\[
E_{\text{max}} = -\frac{eN_dx_n}{\varepsilon_s} = -\frac{(1.6 \times 10^{-19})(10^{16})(0.8644 \times 10^{-4})}{(11.7)(8.85 \times 10^{-14})} = -1.34 \times 10^{4} \, \text{V/cm}
\]

**Comment**

The peak electric field in the space charge region of a pn junction is quite large. We must keep in mind, however, that there is no mobile charge in this region; hence there will be no drift current. We may also note, from this example, that the width of each space charge region is a reciprocal function of the doping concentration: The depletion region will extend further into the lower-doped region.

**Exercise Problem**

**Ex 7.2** A silicon pn junction at \( T = 300 \, \text{K} \) with zero applied bias has doping concentrations of \( N_d = 5 \times 10^{16} \, \text{cm}^{-3} \) and \( N_a = 5 \times 10^{15} \, \text{cm}^{-3} \). Determine \( x_n, x_p, W, \) and \( E_{\text{max}} \).

## TEST YOUR UNDERSTANDING

**TYU 7.1** Calculate \( V_b, x_n, x_p, W, \) and \( E_{\text{max}} \) for a silicon pn junction at zero bias and \( T = 300 \, \text{K} \) for doping concentrations of (a) \( N_a = 2 \times 10^{17} \, \text{cm}^{-3}, N_d = 10^{16} \, \text{cm}^{-3} \) and (b) \( N_a = 4 \times 10^{15} \, \text{cm}^{-3}, N_d = 3 \times 10^{16} \, \text{cm}^{-3} \).

**TYU 7.2** Repeat Exercise Problem Ex 7.2 for a GaAs pn junction.

## 7.3 Reverse Applied Bias

If we apply a potential between the p and n regions, we will no longer be in an equilibrium condition—the Fermi energy level will no longer be constant through the system. Figure 7.7 shows the energy-band diagram of the pn junction for the case when a positive voltage is applied to the n region with respect to the p region. As the positive potential is downward, the Fermi level on the n side is below the Fermi level on the p side. The difference between the two is equal to the applied voltage in units of energy.

The total potential barrier, indicated by \( V_{\text{total}} \), has increased. The applied potential is the reverse-biased condition. The total potential barrier is now given by

\[
V_{\text{total}} = |\phi_{p0}| + |\phi_{n0}| + V_R
\]

(7.32)

where \( V_R \) is the magnitude of the applied reverse-biased voltage. Equation (7.32) can be rewritten as

\[
V_{\text{total}} = V_{bi} + V_R
\]

(7.33)

where \( V_{bi} \) is the same built-in potential barrier we had defined in thermal equilibrium.

### 7.3.1 Space Charge Width and Electric Field

Figure 7.8 shows a pn junction with an applied reverse-biased voltage \( V_R \). Also indicated in the figure are the electric field in the space charge region and the electric field \( E_{\text{app}} \) induced by the applied voltage. The electric fields in the neutral p and n regions are essentially zero, or at least very small, which means that the magnitude of the electric field in the space charge region must increase above the thermal-equilibrium value due to the applied voltage. The electric field originates on positive charge and terminates on negative charge; this means that the number of positive and negative


**Figure 7.7** | Energy-band diagram of a pn junction under reverse bias.


**Figure 7.8.1** A pn junction, with an applied reverse-biased voltage, showing the directions of the electric field induced by \( V_R \) and the space charge electric field.

charges must increase if the electric field increases. For given impurity doping concentrations, the number of positive and negative charges in the depletion region can be increased only if the space charge width \( W \) increases. The space charge width \( W \) increases, therefore, with an increasing reverse-biased voltage \( V_R \). We are assuming that the electric field in the bulk n and p regions is zero. This assumption will become clearer in the next chapter when we discuss the current–voltage characteristics.

In all of the previous equations, the built-in potential barrier can be replaced by the total potential barrier. The total space charge width can be written from Equation (7.31) as

\[
W = \left[ \frac{2 \varepsilon_s (V_{bi} + V_R)}{e} \left( \frac{N_a + N_d}{N_a N_d} \right) \right]^{1/2}
\]

(7.34)

showing that the total space charge width increases as we apply a reverse-biased voltage. By substituting the total potential barrier \( V_{\text{total}} \) into Equations (7.28) and (7.29), the space charge widths in the n and p regions, respectively, can be found as a function of applied reverse-biased voltage.


## EXAMPLE 7.3

**Objective:** Calculate the width of the space charge region in a pn junction when a reverse-biased voltage is applied.

Again consider a silicon pn junction at \( T = 300 \, \text{K} \) with doping concentrations of \( N_a = 10^{16} \, \text{cm}^{-3} \) and \( N_d = 10^{15} \, \text{cm}^{-3} \). Assume that \( n_i = 1.5 \times 10^{10} \, \text{cm}^{-3} \) and \( V_R = 5 \, \text{V} \).

**Solution**

The built-in potential barrier was calculated in Example 7.1 for this case and is \( V_{bi} = 0.635 \, \text{V} \). The space charge width is determined from Equation (7.34). We have

\[
W = \left[ \frac{2(11.7)(8.85 \times 10^{-14})(0.635 + 5)}{1.6 \times 10^{-19}} \right]^{1/2} \left[ \frac{10^{16} + 10^{15}}{(10^{16})(10^{15})} \right]^{1/2}
\]

so that

\[
W = 2.83 \times 10^{-4} \, \text{cm} = 2.83 \, \mu\text{m}
\]

**Comment**  
The space charge width has increased from 0.951 μm at zero bias to 2.83 μm at a reverse bias of 5 V.

**Exercise Problem**

**Ex 7.3**  
(a) A silicon pn junction at \( T = 300 \, K \) has doping concentrations of \( N_d = 5 \times 10^{15} \, \text{cm}^{-3} \) and \( N_a = 5 \times 10^{16} \, \text{cm}^{-3} \). A reverse-biased voltage of \( V_R = 4 \, V \) is applied.  
Determine \( V_{bi}, x_n, x_p, \) and \( W \). (b) Repeat part (a) for \( V_R = 8 \, V \).

The magnitude of the electric field in the depletion region increases with an applied reverse-biased voltage. The electric field is still given by Equations (7.14) and (7.16) and is still a linear function of distance through the space charge region. Since \( x_n \) and \( x_p \) increase with reverse-biased voltage, the magnitude of the electric field also increases. The maximum electric field still occurs at the metallurgical junction.

The maximum electric field at the metallurgical junction, from Equations (7.14) and (7.16), is

\[
E_{\text{max}} = -\frac{eN_dx_n}{\varepsilon_s} = -\frac{eN_ax_p}{\varepsilon_s}
\]

(7.35)

If we use either Equation (7.28) or (7.29) in conjunction with the total potential barrier, \( V_{bi} + V_R \), then

\[
E_{\text{max}} = -\left\{ \frac{2e(V_{bi} + V_R)}{\varepsilon_s} \left( \frac{N_aN_d}{N_a + N_d} \right) \right\}^{1/2}
\]

(7.36)

We can show that the maximum electric field in the pn junction can also be written as

\[
E_{\text{max}} = -\frac{2(V_{bi} + V_R)}{W}
\]

(7.37)

where \( W \) is the total space charge width.

## DESIGN EXAMPLE 7.4
**Objective:** Design a pn junction to meet maximum electric field and voltage specifications.

Consider a silicon pn junction at \( T = 300 \, K \) with a p-type doping concentration of \( N_a = 2 \times 10^{17} \, \text{cm}^{-3} \). Determine the n-type doping concentration such that the maximum electric field is \( |E_{\text{max}}| = 2.5 \times 10^6 \, \text{V/cm} \) at a reverse-biased voltage of \( V_R = 25 \, V \).

**Solution**  
The maximum electric field is given by Equation (7.36). Neglecting \( V_{bi} \) compared to \( V_R \), we can write

\[
|E_{\text{max}}| = \left\{ \frac{2eV_R}{\varepsilon_s} \left( \frac{N_aN_d}{N_a + N_d} \right) \right\}^{1/2}
\]

or

\[
2.5 \times 10^8 = \left\{ \frac{2(1.6 \times 10^{-19})(25)}{(11.7)(8.85 \times 10^{-14})} \left[ \frac{(2 \times 10^{17})N_d}{2 \times 10^7 + N_d} \right] \right\}^{1/2}
\]

which yields

\[
N_d = 8.43 \times 10^{15} \, \text{cm}^{-3}
\]

**Conclusion**

A smaller value of \( N_d \) results in a smaller value of \( E_{\text{max}} \) for a given reverse-biased voltage. The value of \( N_d \) determined in this example, then, is the maximum value that will meet the specifications.

**Exercise Problem**

**Ex. 7.4** The maximum electric field in a reverse-biased GaAs pn junction at \( T = 300 \, \text{K} \) is to be limited to \( |E_{\text{max}}| = 7.2 \times 10^4 \, \text{V/cm} \). The doping concentrations are \( N_a = 5 \times 10^{15} \, \text{cm}^{-3} \) and \( N_d = 3 \times 10^{16} \, \text{cm}^{-3} \). Determine the maximum reverse-biased voltage that can be applied. (\( \lambda \, 1\% \, \varepsilon = \lambda \, 1\% \, \text{SUV} \))

### 7.3.2 Junction Capacitance

Since we have a separation of positive and negative charges in the depletion region, a capacitance is associated with the pn junction. Figure 7.9 shows the charge densities in the depletion region for applied reverse-biased voltages of \( V_R \) and \( V_R + dV_R \). An increase


**Figure 7.9** | Differential change in the space charge width with a differential change in reverse-biased voltage for a uniformly doped pn junction.

- **p** region: \(-dQ'\)
- **n** region: \(+dQ'\)
- Charge densities: \(-eN_a\) and \(+eN_d\)
- Widths: \(dx_p\) and \(dx_n\)
- Applied voltages: \(V_R\) and \(V_R + dV_R\)

In the reverse-biased voltage \(dV_R\) will uncover additional positive charges in the n region and additional negative charges in the p region. The junction capacitance is defined as

\[
C' = \frac{dQ'}{dV_R}
\]

(7.38)

where

\[
dQ' = eN_d \, dx_n = eN_a \, dx_p
\]

(7.39)

The differential charge \(dQ'\) is in units of C/cm\(^2\) so that the capacitance \(C'\) is in units of farads per square centimeter F/cm\(^2\), or capacitance per unit area.

For the total potential barrier, Equation (7.28) may be written as

\[
x_n = \left\{ \frac{2 \varepsilon_s (V_{bi} + V_R)}{e} \left[ \frac{N_a}{N_d} \right] \left[ \frac{1}{N_a + N_d} \right] \right\}^{1/2}
\]

(7.40)

The junction capacitance can be written as

\[
C' = \frac{dQ'}{dV_R} = eN_d \frac{dx_n}{dV_R}
\]

(7.41)

so that

\[
C' = \left\{ \frac{e \varepsilon_s N_d N_a}{2(V_{bi} + V_R)(N_a + N_d)} \right\}^{1/2}
\]

(7.42)

Exactly the same capacitance expression is obtained by considering the space charge region extending into the p region \(x_p\). The junction capacitance is also referred to as the **depletion layer capacitance**.


## Example 7.5

**Objective**

Calculate the junction capacitance of a pn junction.

Consider the same pn junction as that in Example 7.3. Again assume that \(V_R = 5 \, \text{V}\).

**Solution**

The junction capacitance is found from Equation (7.42) as

\[
C' = \left\{ \frac{(1.6 \times 10^{-19})(11.7)(8.85 \times 10^{-14})(10^{16})(10^{15})}{2(0.635 + 5)(10^{16} + 10^{15})} \right\}^{1/2}
\]

or

\[
C' = 3.66 \times 10^{-7} \, \text{F/cm}^2
\]

If the cross-sectional area of the pn junction is, for example, \(A = 10^{-4} \, \text{cm}^2\), then the total junction capacitance is

\[
C = C' \cdot A = 0.366 \times 10^{-12} \, \text{F} = 0.366 \, \text{pF}
\]

**Comment**

The value of junction capacitance is usually in the pF, or smaller, range.

**Exercise Problem**

**Ex 7.5** Consider a GaAs pn junction at \( T = 300 \, \text{K} \) doped to \( N_d = 5 \times 10^{15} \, \text{cm}^{-3} \) and \( N_a = 2 \times 10^{16} \, \text{cm}^{-3} \).  
(a) Calculate \( V_{bi} \).  
(b) Determine the junction capacitance \( C' \) for \( V_R = 4 \, \text{V} \).  
(c) Repeat part (b) for \( V_R = 8 \, \text{V} \).

\[
\left[ \frac{\varepsilon_s}{W} \right] \cdot 6.01 \times 9.9 = \lambda \left( \frac{\varepsilon_s}{W} \right) \cdot 6.01 \times 8.7 = \lambda \left( \frac{\varepsilon_s}{W} \right) \cdot 9.1 = \lambda \left( \frac{\varepsilon_s}{W} \right) \cdot \text{suV}
\]

If we compare Equation (7.34) for the total depletion width \( W \) of the space charge region under reverse bias and Equation (7.42) for the junction capacitance \( C' \), we find that we can write

\[
C' = \frac{\varepsilon_s}{W}
\]

(7.43)

Equation (7.43) is the same as the capacitance per unit area of a parallel plate capacitor. Considering Figure 7.9, we may have come to this same conclusion earlier. Keep in mind that the space charge width is a function of the reverse-biased voltage so that the junction capacitance is also a function of the reverse-biased voltage applied to the pn junction.

### 7.3.3 One-Sided Junctions

Consider a special pn junction called the one-sided junction. If, for example, \( N_a \gg N_d \), this junction is referred to as a p\(^+\)n junction. The total space charge width, from Equation (7.34), reduces to

\[
W \approx \left[ \frac{2 \varepsilon_s (V_R + V_{bi})}{e N_d} \right]^{1/2}
\]

(7.44)

Considering the expressions for \( x_n \) and \( x_p \), we have for the p\(^+\)n junction

\[
x_p \ll x_n
\]

(7.45)

and

\[
W \approx x_n
\]

(7.46)

Almost the entire space charge layer extends into the low-doped region of the junction. This effect can be seen in Figure 7.10.

The junction capacitance of the p\(^+\)n junction reduces to

\[
C' \approx \left[ \frac{e \varepsilon_s N_d}{2(V_R + V_{bi})} \right]^{1/2}
\]

(7.47)

The depletion layer capacitance of a one-sided junction is a function of the doping concentration in the low-doped region. Equation (7.47) may be manipulated to give

\[
\left( \frac{1}{C'} \right)^2 = \frac{2(V_R + V_{bi})}{e \varepsilon_s N_d}
\]

(7.48)

which shows that the inverse capacitance squared is a linear function of applied reverse-biased voltage.

**Figure 7.10** | Space charge density of a one-sided p<sup>+</sup>n junction.

**Figure 7.11** | \( (1/C^2) \) versus \( V_R \) of a uniformly doped pn junction.

Figure 7.11 shows a plot of Equation (7.48). The built-in potential of the junction can be determined by extrapolating the curve to the point where \( (1/C^2) = 0 \). The slope of the curve is inversely proportional to the doping concentration of the low-doped region in the junction; thus, this doping concentration can be experimentally determined. The assumptions used in the derivation of this capacitance include uniform doping in both semiconductor regions, the abrupt junction approximation, and a planar junction.

## EXAMPLE 7.6

**Objective:** Determine the impurity doping concentrations in a p<sup>+</sup>n junction given the parameters from Figure 7.11.

Assume that the intercept and the slope of the curve in Figure 7.11 are \( V_{bi} = 0.725 \, \text{V} \) and \( 6.15 \times 10^{15} \, (\text{F/cm}^2)^{-2} \, (\text{V}^{-1}) \), respectively, for a silicon p<sup>+</sup>n junction at \( T = 300 \, \text{K} \).

**Solution**

The slope of the curve in Figure 7.11 is given by \( 2/\varepsilon_s N_d \), so we may write

\[
N_d = \frac{2}{\varepsilon_s \cdot \text{slope}} = \frac{2}{(1.6 \times 10^{-19})(11.7)(8.85 \times 10^{-14})(6.15 \times 10^{15})}
\]

or

\[
N_d = 1.96 \times 10^{15} \, \text{cm}^{-3}
\]

From the expression for \( V_{bi} \), which is

\[
V_{bi} = V_t \ln \left( \frac{N_a N_d}{n_i^2} \right)
\]

we can solve for \( N_a \) as

\[
N_a = \frac{n_i^2}{N_d} \exp \left( \frac{V_{bi}}{V_t} \right) = \frac{(1.5 \times 10^{10})^2}{1.963 \times 10^{15}} \exp \left( \frac{0.725}{0.0259} \right)
\]

which yields 

\[ N_a = 1.64 \times 10^{17} \, \text{cm}^{-3} \]

**Comment**

The results of this example show that \( N_a \gg N_d \); therefore the assumption of a one-sided junction was valid.

**EXERCISE PROBLEM**

**Ex 7.6** The experimentally measured junction capacitance of a one-sided silicon n⁺p-junction biased at \( V_R = 3 \, \text{V} \) and at \( T = 300 \, \text{K} \) is \( C = 0.105 \, \text{pF} \). The built-in potential barrier is found to be \( V_b = 0.765 \, \text{V} \). The cross-ctional area is \( A = 10^{-5} \, \text{cm}^2 \). Find the doping concentrations.

A one-sided pn junction is useful for experimentally determining the doping concentrations and built-in potential.


## TEST YOUR UNDERSTANDING

**TYU 7.3** 

(a) A silicon pn junction at \( T = 300 \, \text{K} \) is reverse biased at \( V_R = 8 \, \text{V} \). The doping concentrations are \( N_d = 5 \times 10^{16} \, \text{cm}^{-3} \) and \( N_a = 5 \times 10^{15} \, \text{cm}^{-3} \). Determine \( x_n, x_p, W, \) and \( E_{\text{max}} \).

(b) Repeat part (a) for a reverse-biased voltage of \( V_R = 12 \, \text{V} \).

**TYU 7.4** 

A silicon pn junction at \( T = 300 \, \text{K} \) has doping concentrations of \( N_d = 3 \times 10^{16} \, \text{cm}^{-3} \) and \( N_a = 8 \times 10^{15} \, \text{cm}^{-3} \), and has a cross-sectional area of \( A = 5 \times 10^{-5} \, \text{cm}^2 \). Determine the junction capacitance at (a) \( V_R = 2 \, \text{V} \) and (b) \( V_R = 5 \, \text{V} \).


## 7.4 | JUNCTION BREAKDOWN

In the last section, we determined the effects of applying a reverse-biased voltage across the pn junction. However, the reverse-biased voltage may not increase without limit; at some particular voltage, the reverse-biased current will increase rapidly. The applied voltage at this point is called the **breakdown voltage**.

Two physical mechanisms give rise to the reverse-biased breakdown in a pn junction: the **Zener effect** and the **avalanche effect**. Zener breakdown occurs in highly doped pn junctions through a tunneling mechanism. In a highly doped junction, the conduction and valence bands on opposite sides of the junction are sufficiently close during reverse bias that electrons may tunnel directly from the valence band on the p side into the conduction band on the n side. This tunneling process is schematically shown in Figure 7.12a.

The avalanche breakdown process occurs when electrons and/or holes, moving across the space charge region, acquire sufficient energy from the electric field to create electron–hole pairs by colliding with atomic electrons within the depletion region. The avalanche process is schematically shown in Figure 7.12b. The newly

**Figure 7.12** (a) Zener breakdown mechanism in a reverse-biased pn junction; (b) avalanche breakdown process in a reverse-biased pn junction.


**Figure 7.13** Electron and hole current components through the space charge region during avalanche multiplication.

Created electrons and holes move in opposite directions due to the electric field and thereby create a reverse-biased current. In addition, the newly generated electrons and/or holes may acquire sufficient energy to ionize other atoms, leading to the avalanche process. For most pn junctions, the predominant breakdown mechanism will be the avalanche effect.

If we assume that a reverse-biased electron current \( I_{n0} \) enters the depletion region at \( x = 0 \) as shown in Figure 7.13, the electron current \( I_n \) will increase with distance through the depletion region due to the avalanche process. At \( x = W \), the electron current may be written as

\[
I_n(W) = M_n I_{n0}
\]

(7.49)

where \( M_n \) is a multiplication factor. The hole current is increasing through the depletion region from the n to p region and reaches a maximum value at \( x = 0 \). The total current is constant through the pn junction in steady state.

We can write an expression for the incremental electron current at some point \( x \) as

\[
dI_n(x) = I_n(x) \alpha_n dx + I_p(x) \alpha_p dx \tag{7.50}
\]

where \( \alpha_n \) and \( \alpha_p \) are the electron and hole ionization rates, respectively. The ionization rates are the number of electron-hole pairs generated per unit length by an electron (\( \alpha_n \)) or by a hole (\( \alpha_p \)). Equation (7.50) may be written as

\[
\frac{dI_n(x)}{dx} = I_n(x) \alpha_n + I_p(x) \alpha_p \tag{7.51}
\]

The total current \( I \) is given by

\[
I = I_n(x) + I_p(x) \tag{7.52}
\]

which is a constant. Solving for \( I_p(x) \) from Equation (7.52) and substituting into Equation (7.51), we obtain

\[
\frac{dI_n(x)}{dx} + (\alpha_p - \alpha_n) I_n(x) = \alpha_p I \tag{7.53}
\]

If we make the assumption that the electron and hole ionization rates are equal so that

\[
\alpha_n = \alpha_p = \alpha \tag{7.54}
\]

then Equation (7.53) may be simplified and integrated through the space charge region. We will obtain

\[
I_n(W) - I_n(0) = I \int_0^W \alpha \, dx \tag{7.55}
\]

Using Equation (7.49), Equation (7.55) may be written as

\[
\frac{M_n I_{n0} - I_n(0)}{I} = \int_0^W \alpha \, dx \tag{7.56}
\]

Since \( M_n I_{n0} \approx I \) and since \( I_n(0) = I_{n0} \), Equation (7.56) becomes

\[
1 - \frac{1}{M_n} = \int_0^W \alpha \, dx \tag{7.57}
\]

The avalanche breakdown voltage is defined to be the voltage at which \( M_n \) approaches infinity. The avalanche breakdown condition is then given by

\[
\int_0^W \alpha \, dx = 1 \tag{7.58}
\]

The ionization rates are strong functions of electric field and, since the electric field is not constant through the space charge region, Equation (7.58) is not easy to evaluate.

If we consider, for example, a one-sided p\(^+\)n junction, the maximum electric field is given by

\[
E_{\text{max}} = \frac{eN_d x_n}{\varepsilon_s} \tag{7.59}
\]

**Figure 7.14** Critical electric field at breakdown in a one-sided junction as a function of impurity doping concentrations. (From Sze and Ng [14].)

The depletion width \( x_n \) is given approximately as

\[
x_n \approx \left[ \frac{2 \varepsilon_s V_r}{e \cdot N_d} \right]^{1/2}
\]

(7.60)

where \( V_r \) is the magnitude of the applied reverse-biased voltage. We have neglected the built-in potential \( V_{bi} \).

If we now define \( V_B \) to be the breakdown voltage, \( V_B \), the maximum electric field, \( E_{\text{max}} \), will be defined as a critical electric field, \( E_{\text{crit}} \), at breakdown. Combining Equations (7.59) and (7.60), we may write

\[
V_B = \frac{\varepsilon_s E_{\text{crit}}}{2eN_B}
\]

(7.61)

where \( N_B \) is the semiconductor doping in the low-doped region of the one-sided junction. The critical electric field, plotted in Figure 7.14, is a slight function of doping.

We have been considering a uniformly doped planar junction. The breakdown voltage will decrease for a linearly graded junction. (See Section 7.5.) Figure 7.15 shows a plot of the breakdown voltage for a one-sided abrupt junction and a linearly graded junction. If we take into account the curvature of a diffused junction as well, the breakdown voltage will be further degraded.

## DESIGN EXAMPLE 7.7

**Objective:** Design an ideal one-sided n\(^+\)p junction diode to meet a breakdown voltage specification.

Consider a silicon pn junction diode at \( T = 300 \, K \). Assume that \( N_d = 3 \times 10^{18} \, \text{cm}^{-3} \). Design the diode such that the breakdown voltage is \( V_B = 100 \, V \).

**Solution**

From Figure 7.15, we find that the doping concentration in the low-doped side of a one-sided abrupt junction should be approximately \( 4 \times 10^{15} \, \text{cm}^{-3} \) for a breakdown voltage of 100 V.

**Figure 7.15** | Breakdown voltage versus impurity concentration in uniformly doped and linearly graded junctions. (From Sze [14].)

For a doping concentration of \(4 \times 10^{15} \, \text{cm}^{-3}\), the critical electric field, from Figure 7.14, is approximately \(3.7 \times 10^5 \, \text{V/cm}\). Then, using Equation (7.61), we find the breakdown voltage as

\[
V_b = \frac{\varepsilon_c E_{\text{crit}}^2}{2eN_d} = \frac{(11.7)(8.85 \times 10^{-14})(3.7 \times 10^5)^2}{2(1.6 \times 10^{-19})(4 \times 10^{15})} = 110 \, \text{V}
\]

which correlates very well with the results from Figure 7.15.

**Conclusion**

As Figure 7.15 shows, the breakdown voltage increases as the doping concentration decreases in the low-doped region.

**Exercise Problem**

**Ex. 7.7** A one-sided, planar, uniformly doped silicon pn junction diode is required to have a reverse-biased breakdown voltage of \(V_b = 60 \, \text{V}\). What is the maximum doping concentration in the low-doped region such that this specification is met?

## 7.5 Nonuniformly Doped Junctions

In the pn junctions considered so far, we have assumed that each semiconductor region has been uniformly doped. In actual pn junction structures, this is rarely true. In some electronic applications, specific nonuniform doping profiles are used to obtain special pn junction capacitance characteristics.

### 7.5.1 Linearly Graded Junctions

If we start with a uniformly doped n-type semiconductor, for example, and diffuse acceptor atoms through the surface, the impurity concentrations will tend to be like those shown in Figure 7.16. The point at \( x = x' \) on the figure corresponds to the metallurgical junction. The depletion region extends into the p and n regions from the metallurgical junction as we have discussed previously. The net p-type doping concentration near the metallurgical junction may be approximated as a linear function of distance from the metallurgical junction. Likewise, as a first approximation, the net n-type doping concentration is also a linear function of distance extending into the n region from the metallurgical junction. This effective doping profile is referred to as a linearly graded junction.

Figure 7.17 shows the space charge density in the depletion region of the linearly graded junction. For convenience, the metallurgical junction is placed at \( x = 0 \). The space charge density can be written as

\[
\rho(x) = e\alpha x
\]

(7.62)

where \( \alpha \) is the gradient of the net impurity concentration.

The electric field and potential in the space charge region can be determined from Poisson’s equation. We can write

\[
\frac{dE}{dx} = \frac{\rho(x)}{\varepsilon_s} = \frac{e\alpha x}{\varepsilon_s}
\]

(7.63)

so that the electric field can be found by integration as

\[
E = \int \frac{e\alpha x}{\varepsilon_s} \, dx = \frac{e\alpha}{2\varepsilon_s} (x^2 - x_0^2)
\]

(7.64)

The electric field in the linearly graded junction is a quadratic function of distance rather than the linear function found in the uniformly doped junction. The maximum electric field again occurs at the metallurgical junction. We may note that the electric field is zero at both \( x = +x_0 \) and at \( x = -x_0 \). The electric field in a nonuniformly doped junction doped semiconductor is not exactly zero, but the magnitude of this field is small, so setting \( E = 0 \) in the bulk regions is still a good approximation.

**Figure 7.16** Impurity concentrations of a pn junction with a nonuniformly doped p region.

**Figure 7.17** Space charge density in a linearly graded pn junction.

The potential is again found by integrating the electric field as

\[
\phi(x) = - \int E \, dx
\]

(7.65)

If we arbitrarily set \(\phi = 0\) at \(x = -x_0\), then the potential through the junction is

\[
\phi(x) = -\frac{ea}{2\epsilon_s} \left( \frac{x^3}{3} - x_0^2 x \right) + \frac{ea}{3\epsilon_s} x_0^3
\]

(7.66)

The magnitude of the potential at \(x = +x_0\) will equal the built-in potential barrier for this function. We then have that

\[
\phi(x_0) = \frac{2}{3} \cdot \frac{ea x_0^3}{\epsilon_s} = V_{bi}
\]

(7.67)

Another expression for the built-in potential barrier for a linearly graded junction can be approximated from the expression used for a uniformly doped junction. We can write

\[
V_{bi} = V_t \ln \left[ \frac{N_d(x_0) N_a(-x_0)}{n_i^2} \right]
\]

(7.68)

where \(N_d(x_0)\) and \(N_a(-x_0)\) are the doping concentrations at the edges of the space charge region. We can relate these doping concentrations to the gradient, so that

\[
N_d(x_0) = ax_0
\]

(7.69a)

and

\[
N_a(-x_0) = ax_0
\]

(7.69b)

Then the built-in potential barrier for the linearly graded junction becomes

\[
V_{bi} = V_t \ln \left( \frac{(ax_0)^2}{n_i^2} \right)
\]

(7.70)

There may be situations in which the doping gradient is not the same on either side of the junction, but we will not consider that condition here.

If a reverse-biased voltage is applied to the junction, the potential barrier increases. The built-in potential barrier \(V_{bi}\) in the above equations is then replaced by the total potential barrier \(V_{bi} + V_R\). Solving for \(x_0\) from Equation (7.67) and using the total potential barrier, we obtain

\[
x_0 = \left[ \frac{3}{2} \cdot \frac{\epsilon_s}{ea} (V_{bi} + V_R) \right]^{1/3}
\]

(7.71)

The junction capacitance per unit area can be determined by the same method that we used for the uniformly doped junction. Figure 7.18 shows the differential charge

**Figure 7.18** Differential change in space charge width with a differential change in reverse-biased voltage for a linearly graded pn junction.

\( dQ' \), which is uncovered as a differential voltage \( dV_R \) is applied. The junction capacitance is then

\[
C' = \frac{dQ'}{dV_R} = (ea\alpha) \frac{dx_0}{dV_R}
\]

(7.72)

Using Equation (7.71), we obtain\(^1\)

\[
C' = \left\{ \frac{ea\epsilon^2}{12(V_{bi} + V_R)} \right\}^{1/3}
\]

(7.73)

We may note that \( C' \) is proportional to \( (V_{bi} + V_R)^{-1/3} \) for the linearly graded junction as compared to \( C' \propto (V_{bi} + V_R)^{-1/2} \) for the uniformly doped junction. In the linearly graded junction, the capacitance is less dependent on reverse-biased voltage than in the uniformly doped junction.

### 7.5.2 Hyperabrupt Junctions

The uniformly doped junction and linearly graded junction are not the only possible doping profiles. Figure 7.19 shows a generalized one-sided p\(^+\)n junction where the generalized n-type doping concentration for \( x > 0 \) is given by

\[
N = Bx^m
\]

(7.74)

The case of \( m = 0 \) corresponds to the uniformly doped junction, and \( m = +1 \) corresponds to the linearly graded junction just discussed. The cases of \( m = +2 \) and \( m = +3 \) would approximate a fairly low-doped epitaxial n-type layer grown


\(^1\)In a more exact analysis, \( V_R \) in Equation (7.73) is replaced by a gradient voltage. However, this analysis is beyond the scope of this text.

!Generalized doping profiles of a one-sided p\(^+\)n junction.

*Figure 7.19* Generalized doping profiles of a one-sided p\(^+\)n junction.  
(From Sze [14].)


On a much more heavily doped n\(^+\) substrate layer. When the value of \(m\) is negative, we have what is referred to as a **hyperabrupt junction**. In this case, the n-type doping is larger near the metallurgical junction than in the bulk semiconductor. Equation (7.74) is used to approximate the n-type doping over a small region near \(x = x_0\) and does not hold at \(x = 0\) when \(m\) is negative.

The junction capacitance can be derived using the same analysis method as before and is given by

\[
C' = \left( \frac{eB\epsilon (m+1)}{(m + 2)\sqrt{V_{bi} + V}} \right)^{1/(m+2)}
\]

(7.75)

When \(m\) is negative, the capacitance becomes a very strong function of reverse-biased voltage, a desired characteristic in **varactor diodes**. The term varactor comes from the words **variable reactor** and means a device whose reactance can be varied in a controlled manner with bias voltage.

If a varactor diode and an inductance are in parallel, the resonant frequency of the LC circuit is

\[
f_r = \frac{1}{2\pi\sqrt{LC}}
\]

(7.76)

The capacitance of the diode, from Equation (7.75), can be written in the form

\[
C = C_0(V_{bi} + V_R)^{-1/(m+2)}
\]

(7.77)

In a circuit application, we would, in general, like to have the resonant frequency be a linear function of reverse-biased voltage \( V_R \), so we need

\[
C \propto V^{-2}
\]

(7.78)

From Equation (7.77), the parameter \( m \) required is found from

\[
\frac{1}{m + 2} = 2
\]

(7.79a)

or

\[
m = -\frac{3}{2}
\]

(7.79b)

A specific doping profile will yield the desired capacitance characteristic.

## 7.6 Summary

- A uniformly doped pn junction is initially considered, in which one region of a semiconductor is uniformly doped with acceptor impurities and the adjacent region is uniformly doped with donor impurities.
- A space charge region, or depletion region, is formed on either side of the metallurgical junction separating the n and p regions. This region is essentially depleted of any mobile electrons or holes. A net positive charge density, due to the positively charged donor impurity ions, exists in the n region and a net negative charge density, due to the negatively charged acceptor impurity ions, exists in the p region.
- An electric field exists in the depletion region due to the net space charge density. The direction of the electric field is from the n region to the p region.
- A potential difference exists across the space charge region. Under zero applied bias, this potential difference, known as the built-in potential barrier, maintains thermal equilibrium and holds back majority carrier electrons in the n region and majority carrier holes in the p region.
- An applied reverse-biased voltage (n region positive with respect to the p region) increases the potential barrier, the space charge width, and the magnitude of the electric field.
- As the reverse-biased voltage changes, the amount of charge in the depletion region changes. This change in charge with voltage defines the junction capacitance.
- Avalanche breakdown occurs when a sufficiently large reverse-biased voltage is applied to the pn junction. A large reverse-biased current may then be induced in the pn junction. The breakdown voltage, as a function of the doping concentrations in the pn junction, is derived. In a one-sided pn junction, the breakdown voltage is a function of the doping concentration in the low-doped region.
- The linearly graded junction represents a nonuniformly doped pn junction. Expressions for the electric field, built-in potential barrier, and junction capacitance are derived. The functional relationships differ from those of the uniformly doped junction.
- Specific doping profiles can be used to obtain specific capacitance characteristics. A hyperabrupt junction is one in which the doping decreases away from the metallurgical junction. This type of junction is advantageous in varactor diodes that are used in resonant circuits.

## Glossary of Important Terms

**abrupt junction approximation**  
The assumption that there is an abrupt discontinuity in space charge density between the space charge region and the neutral semiconductor region.

**avalanche breakdown**  
The process whereby a large reverse-biased pn junction current is created due to the generation of electron–hole pairs by the collision of electrons and/or holes with atomic electrons within the space charge region.

**built-in potential barrier**  
The electrostatic potential difference between the p and n regions of a pn junction in thermal equilibrium.

**critical electric field**  
The peak electric field in the space charge region at breakdown.

**depletion layer capacitance**  
Another term for junction capacitance.

**depletion region**  
Another term for space charge region.

**hyperabrupt junction**  
A pn junction in which the doping concentration on one side decreases away from the metallurgical junction to achieve a specific capacitance–voltage characteristic.

**junction capacitance**  
The capacitance of the pn junction under reverse bias.

**linearly graded junction**  
A pn junction in which the doping concentrations on either side of the metallurgical junction are approximated by a linear distribution.

**metallurgical junction**  
The interface between the p- and n-doped regions of a pn junction.

**one-sided junction**  
A pn junction in which one side of the junction is much more heavily doped than the adjacent side.

**reverse bias**  
The condition in which a positive voltage is applied to the n region with respect to the p region of a pn junction so that the potential barrier between the two regions increases above the thermal-equilibrium built-in potential barrier.

**space charge region**  
The region on either side of the metallurgical junction in which there is a net charge density due to ionized donors in the n region and ionized acceptors in the p region.

**space charge width**  
The width of the space charge region, a function of doping concentrations and applied voltage.

**varactor diode**  
A diode whose reactance can be varied in a controlled manner with bias voltage.

## Checkpoint

After studying this chapter, the reader should have the ability to:

- Describe why and how the space charge region is formed.
- Draw the energy-band diagram of a zero-biased and reverse-biased pn junction.
- Define and derive the expression of the built-in potential barrier voltage.
- Derive the expression for the electric field in space charge region of the pn junction.
- Describe what happens to the parameters of the space charge region when a reverse-biased voltage is applied.
- Define and explain the junction capacitance.
- Describe the characteristics and properties of a one-sided pn junction.
- Describe the avalanche breakdown mechanism in a reverse-biased pn junction.
- Describe how a linearly graded junction is formed.
- Define a hyperabrupt junction.

## REVIEW QUESTIONS

1. Define the built-in potential voltage and describe how it maintains thermal equilibrium.

2. Why is an electric field formed in the space charge region? Why is the electric field a linear function of distance in a uniformly doped pn junction?

3. Where does the maximum electric field occur in the space charge region?

4. Why is the space charge width larger in the lower doped side of a pn junction?

5. What is the functional dependence of the space charge width on reverse-biased voltage?

6. Why does the space charge width increase with reverse-biased voltage?

7. Why does a capacitance exist in a reverse-biased pn junction? Why does the capacitance decrease with increasing reverse-biased voltage?

8. What is a one-sided pn junction? What parameters can be determined in a one-sided pn junction?

9. Why does the breakdown voltage of a pn junction decrease as the doping concentration increases?

10. What is a linearly graded junction?

11. What is a hyperabrupt junction and what is one advantage or characteristic of such a junction?