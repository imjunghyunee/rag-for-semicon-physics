# Chapter 4: The Semiconductor in Equilibrium

So far, we have been considering a general crystal and applying to it the concepts of quantum mechanics in order to determine a few of the characteristics of electrons in a single-crystal lattice. In this chapter, we apply these concepts specifically to a semiconductor material. **In particular, we use the density of quantum states in the conduction band and the density of quantum states in the valence band along with the Fermi–Dirac probability function to determine the concentration of electrons and holes in the conduction and valence bands, respectively.** We also apply the concept of the Fermi energy to the semiconductor material.

This chapter deals with the semiconductor in equilibrium. **Equilibrium, or thermal equilibrium, implies that no external forces such as voltages, electric fields, magnetic fields, or temperature gradients are acting on the semiconductor. All properties of the semiconductor will be independent of time** in this case.

## 4.0 | Preview

In this chapter, we will:

- Derive the thermal-equilibrium concentrations of electrons and holes in a semiconductor as a function of the Fermi energy level.
- Discuss the process by which the properties of a semiconductor material can be favorably altered by adding specific impurity atoms to the semiconductor.
- Determine the thermal-equilibrium concentrations of electrons and holes in a semiconductor as a function of the concentration of dopant atoms added to the semiconductor.
- Determine the position of the Fermi energy level as a function of the concentrations of dopant atoms added to the semiconductor.

## 4.1 Charge Carriers in Semiconductors

Current is the rate at which charge flows. In a semiconductor, two types of charge carriers, the electron and the hole, can contribute to a current. Since the current in a semiconductor is determined largely by the number of electrons in the conduction band and the number of holes in the valence band, an important characteristic of the semiconductor is the density of these charge carriers. The density of electrons and holes is related to the density of states function and the Fermi distribution function, both of which we have considered. A qualitative discussion of these relationships will be followed by a more rigorous mathematical derivation of the thermal-equilibrium concentration of electrons and holes.

### 4.1.1 Equilibrium Distribution of Electrons and Holes

The distribution (with respect to energy) of electrons in the conduction band is given by the density of allowed quantum states times the probability that a state is occupied by an electron. This statement is written in equation form as

\[
n(E) = g_c(E)f_r(E)
\]

(4.1)

where \( f_r(E) \) is the Fermi–Dirac probability function and \( g_c(E) \) is the density of quantum states in the conduction band. The total electron concentration per unit volume in the conduction band is then found by integrating Equation (4.1) over the entire conduction-band energy.

Similarly, the distribution (with respect to energy) of holes in the valence band is the density of allowed quantum states in the valence band multiplied by the probability that a state is not occupied by an electron. We may express this as

\[
p(E) = g_v(E)[1 - f_r(E)]
\]

(4.2)

The total hole concentration per unit volume is found by integrating this function over the entire valence-band energy.

To find the thermal-equilibrium electron and hole concentrations, we need to determine the position of the Fermi energy \( E_f \) with respect to the bottom of the conduction-band energy \( E_c \) and the top of the valence-band energy \( E_v \). To address this question, we will initially consider an intrinsic semiconductor. An ideal intrinsic semiconductor is a pure semiconductor with no impurity atoms and no lattice defects in the crystal (e.g., pure silicon). We have argued in the previous chapter that, for an intrinsic semiconductor at \( T = 0 \, K \), all energy states in the valence band are filled with electrons and all energy states in the conduction band are empty of electrons. The Fermi energy must, therefore, be somewhere between \( E_v \) and \( E_c \). (The Fermi energy does not need to correspond to an allowed energy.)

As the temperature begins to increase above 0 K, the valence electrons will gain thermal energy. A few electrons in the valence band may gain sufficient energy to jump to the conduction band. As an electron jumps from the valence band to the conduction band, an empty state, or hole, is created in the valence band. In an intrinsic


**Figure 4.1** (a) Density of states functions, Fermi–Dirac probability function, and areas representing electron and hole concentrations for the case when \( E_F \) is near the midgap energy; (b) expanded view near the conduction-band energy; and (c) expanded view near the valence-band energy.

- **(a)** 
  - \( g_c(E)f_r(E) = n(E) \)
  - Area = \( n_0 \) = electron concentration
  - \( f_r(E) = 0 \)
  - \( f_r(E) = 1 \)
  - \( g_v(E)[1 - f_r(E)] = p(E) \)
  - Area = \( p_0 \) = hole concentration

- **(b)**
  - \( f_r(E) \)
  - \( g_c(E) \)

- **(c)**
  - \( g_v(E) \)
  - \( [1 - f_r(E)] \)

In a semiconductor, then, **electrons and holes are created in pairs by the thermal energy so that the number of electrons in the conduction band is equal to the number of holes in the valence band.**

Figure 4.1a shows a plot of the density of states function in the conduction-band \( g_c(E) \), the density of states function in the valence-band \( g_v(E) \), and the Fermi-Dirac probability function for \( T > 0 \) K when \( E_F \) is approximately halfway between \( E_c \) and \( E_v \). **If we assume, for the moment, that the electron and hole effective masses are equal, then \( g_c(E) \) and \( g_v(E) \) are symmetrical functions about the midgap energy (the energy midway between \( E_c \) and \( E_v \)).** We noted previously that the function \( f_r(E) \) for \( E > E_F \) is symmetrical to the function \( 1 - f_r(E) \) for \( E < E_F \) about the energy \( E = E_F \). This also means that the function \( f_r(E) \) for \( E = E_F + dE \) is equal to the function \( 1 - f_r(E) \) for \( E = E_F - dE \).

Figure 4.1b is an expanded view of the plot in Figure 4.1a showing \( g_c(E) \) and \( f(E) \) above the conduction-band energy \( E_c \). The product of \( g_c(E) \) and \( f(E) \) is the distribution of electrons \( n(E) \) in the conduction band given by Equation (4.1). This product is plotted in Figure 4.1a. Figure 4.1c, which is an expanded view of the plot in Figure 4.1a shows \([1 - f_f(E)]\) and \( g_v(E) \) below the valence-band energy \( E_v \). The product of \( g_v(E) \) and \([1 - f_f(E)]\) is the distribution of holes \( p(E) \) in the valence band given by Equation (4.2). This product is also plotted in Figure 4.1a. The areas under these curves are then the total density of electrons in the conduction band and the total density of holes in the valence band. From this we see that if \( g_c(E) \) and \( g_v(E) \) are symmetrical, the Fermi energy must be at the midgap energy in order to obtain equal electron and hole concentrations. If the effective masses of the electron and hole are not exactly equal, then the effective density of states functions \( g_c(E) \) and \( g_v(E) \) will not be exactly symmetrical about the midgap energy. The Fermi level for the intrinsic semiconductor will then shift slightly from the midgap energy in order to obtain equal electron and hole concentrations.

### 4.1.2 The \( n_0 \) and \( p_0 \) Equations

We have argued that the Fermi energy for an intrinsic semiconductor is near midgap. In deriving the equations for the thermal-equilibrium concentration of electrons \( n_0 \) and the thermal-equilibrium concentration of holes \( p_0 \), we will not be quite so restrictive. We will see later that, in particular situations, the Fermi energy can deviate from this midgap energy. We will assume initially, however, that the Fermi level remains within the bandgap energy.

**Thermal-Equilibrium Electron Concentration** The equation for the thermal-equilibrium concentration of electrons may be found by integrating Equation (4.1) over the conduction band energy, or

\[
n_0 = \int_{E_c}^{\infty} g_c(E)f_f(E) \, dE
\]

(4.3)

The lower limit of integration is \( E_c \) and the upper limit of integration should be the top of the allowed conduction band energy. However, since the Fermi probability function rapidly approaches zero with increasing energy as indicated in Figure 4.1a, we can take the upper limit of integration to be infinity.

We are assuming that the Fermi energy is within the forbidden-energy bandgap. For electrons in the conduction band, we have \( E > E_c \). If \((E - E_f) \gg kT\), then \((E - E_c) \gg kT\), so that the Fermi probability function reduces to the Boltzmann approximation, which is

\[
f_f(E) = \frac{1}{1 + \exp\left(\frac{E - E_f}{kT}\right)} \approx \exp\left[-\frac{(E - E_f)}{kT}\right]
\]

(4.4)

*The Maxwell–Boltzmann and Fermi–Dirac distribution functions are within 5 percent of each other when \( E - E_f > 3kT \) (see Figure 3.35). The \(\gg\) notation is then somewhat misleading to indicate when the Boltzmann approximation is valid, although it is commonly used.*

Applying the Boltzmann Approximation to Equation (4.3), the thermal-equilibrium density of electrons in the conduction band is found from

\[
n_0 = \int_{E_c}^{\infty} \frac{4\pi(2m_n^*)^{3/2}}{h^3} \sqrt{E - E_c} \exp \left[ -\frac{(E - E_f)}{kT} \right] dE
\]

The integral of Equation (4.5) may be solved more easily by making a change of variable. If we let:

\[
\eta = \frac{E - E_c}{kT}
\]

then Equation (4.5) becomes:

\[
n_0 = \frac{4\pi(2m_n^*kT)^{3/2}}{h^3} \exp \left[ -\frac{(E_c - E_f)}{kT} \right] \int_0^{\infty} \eta^{1/2} \exp (-\eta) d\eta
\]

The integral is the gamma function, with a value of:

\[
\int_0^{\infty} \eta^{1/2} \exp (-\eta) d\eta = \frac{1}{2} \sqrt{\pi}
\]

Then Equation (4.7) becomes:

\[
n_0 = 2 \left( \frac{2\pi m_n^* kT}{h^2} \right)^{3/2} \exp \left[ -\frac{(E_c - E_f)}{kT} \right]
\]

We may define a parameter \( N_c \) as:

\[
N_c \equiv 2 \left( \frac{2\pi m_n^* kT}{h^2} \right)^{3/2}
\]

The parameter \( m_n^* \) is the density of states effective mass of the electron. The thermal-equilibrium electron concentration in the conduction band can be written as:

\[
n_0 = N_c \exp \left[ -\frac{(E_c - E_f)}{kT} \right]
\]

The parameter \( N_c \) is called the effective density of states function in the conduction band. If we were to assume that \( m_n^* = m_0 \), then the value of the effective density of states function at \( T = 300 \, \text{K} \) is \( N_c = 2.5 \times 10^{19} \, \text{cm}^{-3} \), which is of the order of magnitude of \( N_c \) for most semiconductors. If the effective mass of the electron is larger or smaller than \( m_0 \), then the value of the effective density of states function changes accordingly, but is still of the same order of magnitude.

## Example 4.1

**Objective:** Calculate the probability that a quantum state in the conduction band at \( E = E_c + kT/2 \) is occupied by an electron, and calculate the thermal-equilibrium electron concentration in silicon at \( T = 300 \, \text{K} \).

Assume the Fermi energy is 0.25 eV below the conduction band. The value of \( N_c \) for silicon at \( T = 300 \, \text{K} \) is \( N_c = 2.8 \times 10^{19} \, \text{cm}^{-3} \) (see Appendix B).

**Solution**

The probability that a quantum state at \( E = E_c + kT/2 \) is occupied by an electron is given by:

\[
f(kT) = \frac{1}{1 + \exp \left[ \frac{E - E_f}{kT} \right]} \exp \left[ -\frac{(E - E_f)}{kT} \right] = \exp \left[ -\frac{(E_c + (kT/2) - E_f)}{kT} \right]
\]

or

\[
f(E) = \exp\left[\frac{-0.25 + (0.0259/2)}{0.0259}\right] = 3.90 \times 10^{-5}
\]

The electron concentration is given by

\[
n_0 = N_c \exp\left[\frac{-(E_c - E_F)}{kT}\right] = (2.8 \times 10^{19}) \exp\left[\frac{-0.25}{0.0259}\right]
\]

or

\[
n_0 = 1.80 \times 10^{15} \text{ cm}^{-3}
\]


**Comment**

The probability of a state being occupied can be quite small, but the fact that there are a large number of states means that the electron concentration is a reasonable value.


**EXERCISE PROBLEM**

**Ex 4.1** Determine the probability that a quantum state at energy \( E = E_c + kT \) is occupied by an electron, and calculate the electron concentration in GaAs at \( T = 300 \, K \) if the Fermi energy level is 0.25 eV below \( E_c \).


**Thermal-Equilibrium Hole Concentration**

The thermal-equilibrium concentration of holes in the valence band is found by integrating Equation (4.2) over the valence-band energy, or

\[
p_0 = \int g_v(E)[1 - f(E)] \, dE \tag{4.12}
\]

We may note that

\[
1 - f(E) = \frac{1}{1 + \exp\left[\frac{E - E_F}{kT}\right]} \tag{4.13a}
\]

For energy states in the valence band, \( E \ll E_F \). If \( (E_F - E) \gg kT \) (the Fermi function is still assumed to be within the bandgap), then we have a slightly different form of the Boltzmann approximation. Equation (4.13a) may be written as

\[
1 - f(E) = \frac{1}{1 + \exp\left[\frac{E_F - E}{kT}\right]} \approx \exp\left[\frac{-(E_F - E)}{kT}\right] \tag{4.13b}
\]

Applying the Boltzmann approximation of Equation (4.13b) to Equation (4.12), we find the thermal-equilibrium concentration of holes in the valence band is

\[
p_0 = \int \frac{4\pi (2m_p)^{3/2}}{h^3} \sqrt{E_v - E} \exp\left[\frac{-(E_F - E)}{kT}\right] \, dE \tag{4.14}
\]

where the lower limit of integration is taken as minus infinity instead of the bottom of the valence band. The exponential term decays fast enough so that this approximation is valid.


Equation (4.14) may be solved more easily by again making a change of variable. If we let

\[
\eta' = \frac{E_v - E}{kT}
\]

then Equation (4.14) becomes

\[
p_0 = \frac{4\pi (2m_p^* kT)^{3/2}}{h^3} \exp \left( \frac{-(E_F - E_v)}{kT} \right) \int_0^\infty (\eta')^{1/2} \exp(-\eta') \, d\eta'
\]

where the negative sign comes from the differential \(dE = -kT d\eta'\). Note that the lower limit of \(\eta'\) becomes \(+\infty\) when \(E = -\infty\). If we change the order of integration, we introduce another minus sign. From Equation (4.8), Equation (4.16) becomes

\[
p_0 = 2 \left( \frac{2\pi m_p^* kT}{h^2} \right)^{3/2} \exp \left( \frac{-(E_F - E_v)}{kT} \right)
\]

We may define a parameter \(N_v\) as

\[
N_v = 2 \left( \frac{2\pi m_p^* kT}{h^2} \right)^{3/2}
\]

which is called the **effective density of states function in the valence band**. The parameter \(m_p^*\) is the density of states effective mass of the hole. The thermal-equilibrium concentration of holes in the valence band may now be written as

\[
p_0 = N_v \exp \left( \frac{-(E_F - E_v)}{kT} \right)
\]

The magnitude of \(N_v\) is also on the order of \(10^{19} \, \text{cm}^{-3}\) at \(T = 300 \, \text{K}\) for most semiconductors.


## Example 4.2

**Objective:** Calculate the thermal-equilibrium hole concentration in silicon at \(T = 400 \, \text{K}\).

Assume that the Fermi energy is 0.27 eV above the valence-band energy. The value of \(N_v\) for silicon at \(T = 300 \, \text{K}\) is \(N_v = 1.04 \times 10^{19} \, \text{cm}^{-3}\). (See Appendix B)

**Solution**

The parameter values at \(T = 400 \, \text{K}\) are found as:

\[
N_v = (1.04 \times 10^{19}) \left( \frac{400}{300} \right)^{3/2} = 1.60 \times 10^{19} \, \text{cm}^{-3}
\]

and

\[
kT = (0.0259) \left( \frac{400}{300} \right) = 0.03453 \, \text{eV}
\]

The hole concentration is then

\[
p_0 = N_v \exp \left[ \frac{-(E_F - E_v)}{kT} \right] = (1.60 \times 10^{19}) \exp \left( \frac{-0.27}{0.03453} \right)
\]

or

\[
p_0 = 6.43 \times 10^5 \, \text{cm}^{-3}
\]

**Comment**

The parameter values at any temperature can easily be found by using the 300 K values and the temperature dependence.

**Exercise Problem**

**Ex 4.2**  
(a) Repeat Example 4.2 at \( T = 250 \, \text{K} \).  
(b) What is the ratio of \( p_0 \) at \( T = 250 \, \text{K} \) to that at \( T = 400 \, \text{K} \)?

The effective density of states functions, \( N_c \) and \( N_v \), are constant for a given semiconductor material at a fixed temperature. Table 4.1 gives the values of the density of states function and of the density of states effective masses for silicon, gallium arsenide, and germanium. Note that the value of \( N_v \) for gallium arsenide is smaller than the typical \( 10^{19} \, \text{cm}^{-3} \) value. This difference is due to the small electron effective mass in gallium arsenide.

The thermal-equilibrium concentrations of electrons in the conduction band and of holes in the valence band are directly related to the effective density of states constants and to the Fermi energy level.

## TEST YOUR UNDERSTANDING

**TYU 4.1**  
Calculate the thermal equilibrium electron and hole concentration in silicon at \( T = 300 \, \text{K} \) for the case when the Fermi energy level is 0.22 eV below the conduction-band energy \( E_c \). The value of \( E_g \) is given in Appendix B.4.

**TYU 4.2**  
Determine the thermal equilibrium electron and hole concentration in GaAs at \( T = 300 \, \text{K} \) for the case when the Fermi energy level is 0.30 eV above the valence-band energy \( E_v \). The value of \( E_g \) is given in Appendix B.4.

### 4.1.3 The Intrinsic Carrier Concentration

For an intrinsic semiconductor, the concentration of electrons in the conduction band is equal to the concentration of holes in the valence band. We may denote \( n_i \) and \( p_i \) as the electron and hole concentrations, respectively, in the intrinsic semiconductor. These parameters are usually referred to as the intrinsic electron concentration and intrinsic hole concentration. However, \( n_i = p_i \), so normally we simply use the parameter \( n_i \) as the intrinsic carrier concentration, which refers to either the intrinsic electron or hole concentration.

**Table 4.1**

Effective density of states function and density of states effective mass values

| Material          | \( N_c \, (\text{cm}^{-3}) \) | \( N_v \, (\text{cm}^{-3}) \) | \( m_e^*/m_0 \) | \( m_h^*/m_0 \) |
|-------------------|-------------------------------|-------------------------------|-----------------|-----------------|
| Silicon           | \( 2.8 \times 10^{19} \)      | \( 1.04 \times 10^{19} \)     | 1.08            | 0.56            |
| Gallium arsenide  | \( 4.7 \times 10^{17} \)      | \( 7.0 \times 10^{18} \)      | 0.067           | 0.48            |
| Germanium         | \( 1.04 \times 10^{19} \)     | \( 6.0 \times 10^{18} \)      | 0.55            | 0.37            |



The Fermi energy level for the intrinsic semiconductor is called the intrinsic Fermi energy, or \( E_F = E_{Fi} \). If we apply Equations (4.11) and (4.19) to the intrinsic semiconductor, then we can write

\[
n_0 = n_i = N_c \exp \left( \frac{-(E_c - E_{Fi})}{kT} \right)
\]
(4.20)

and

\[
p_0 = p_i = n_i = N_v \exp \left( \frac{-(E_{Fi} - E_v)}{kT} \right)
\]
(4.21)

If we take the product of Equations (4.20) and (4.21), we obtain

\[
n_i^2 = N_c N_v \exp \left( \frac{-(E_c - E_{Fi})}{kT} \right) \cdot \exp \left( \frac{-(E_{Fi} - E_v)}{kT} \right)
\]
(4.22)

or

\[
n_i^2 = N_c N_v \exp \left( \frac{-(E_c - E_v)}{kT} \right) = N_c N_v \exp \left( \frac{-E_g}{kT} \right)
\]
(4.23)

where \( E_g \) is the bandgap energy. For a given semiconductor material at a constant temperature, the value of \( n_i \) is a constant, and independent of the Fermi energy.

The intrinsic carrier concentration for silicon at \( T = 300 \, \text{K} \) may be calculated by using the effective density of states function values from Table 4.1. The value of \( n_i \) calculated from Equation (4.23) for \( E_g = 1.12 \, \text{eV} \) is \( n_i = 6.95 \times 10^9 \, \text{cm}^{-3} \). The commonly accepted value of \( n_i \) for silicon at \( T = 300 \, \text{K} \) is approximately \( 1.5 \times 10^{10} \, \text{cm}^{-3} \). This discrepancy may arise from several sources. First, the values of the effective masses are determined at a low temperature where the cyclotron resonance experiments are performed. Since the effective mass is an experimentally determined parameter, and since the effective mass is a measure of how well a particle moves in a crystal, this parameter may be a slight function of temperature. Next, the density of states function for a semiconductor was obtained by generalizing the model of an electron in a three-dimensional infinite potential well. This theoretical function may also not agree exactly with experiment. However, the difference between the theoretical value and the experimental value of \( n_i \) is approximately a factor of 2, which, in many cases, is not significant. Table 4.2 lists the commonly accepted values of \( n_i \) for silicon, gallium arsenide, and germanium at \( T = 300 \, \text{K} \).

The intrinsic carrier concentration is a very strong function of temperature.

**Table 4.2** | Commonly accepted values of \( n_i \) at \( T = 300 \, \text{K} \)

| Material        | \( n_i \) (\(\text{cm}^{-3}\)) |
|-----------------|-------------------------------|
| Silicon         | \( n_i = 1.5 \times 10^{10} \) |
| Gallium arsenide| \( n_i = 1.8 \times 10^6 \)    |
| Germanium       | \( n_i = 2.4 \times 10^{13} \) |

*Various references may list slightly different values of the intrinsic silicon concentration at room temperature. In general, they are all between \( 1 \times 10^{10} \) and \( 1.5 \times 10^{10} \, \text{cm}^{-3} \). This difference is, in most cases, not significant.*


## EXAMPLE 4.3
**Objective**: 
Calculate the intrinsic carrier concentration in silicon at \( T = 250 \, \text{K} \) and at \( T = 400 \, \text{K} \)

The values of \( N_c \) and \( N_v \) for silicon at \( T = 300 \, \text{K} \) are \( 2.8 \times 10^{19} \, \text{cm}^{-3} \) and \( 1.04 \times 10^{19} \, \text{cm}^{-3} \), respectively. Both \( N_c \) and \( N_v \) vary as \( T^{3/2} \). Assume the bandgap energy of silicon is 1.12 eV and does not vary over this temperature range.

**Solution**

Using Equation (4.23), we find, at \( T = 250 \, \text{K} \)

\[
n_i^2 = (2.8 \times 10^{19} \times 1.04 \times 10^{19}) \left(\frac{250}{300}\right)^3 \exp\left(\frac{-1.12}{(0.0259)(250/300)}\right)
\]

\[
= 4.90 \times 10^{15}
\]

or

\[
n_i = 7.0 \times 10^7 \, \text{cm}^{-3}
\]

At \( T = 400 \, \text{K} \), we find

\[
n_i^2 = (2.8 \times 10^{19} \times 1.04 \times 10^{19}) \left(\frac{400}{300}\right)^3 \exp\left(\frac{-1.12}{(0.0259)(400/300)}\right)
\]

\[
= 5.67 \times 10^{24}
\]

or

\[
n_i = 2.38 \times 10^{12} \, \text{cm}^{-3}
\]

**Comment**

We may note from this example that the intrinsic carrier concentration increased by over 4 orders of magnitude as the temperature increased by 150°C.

**Exercise Problem**

**Ex 4.3** (a) Calculate the intrinsic carrier concentration in GaAs at \( T = 400 \, \text{K} \) and at \( T = 250 \, \text{K} \). Assume that \( E_g = 1.42 \, \text{eV} \) is constant over this temperature range.  
(b) What is the ratio of \( n_i \) at \( T = 400 \, \text{K} \) to that at \( T = 250 \, \text{K} \)?

\[
[01 \times 10^{19} \, \text{cm}^{-3} \times 1.01 \times 10^{19} \, \text{cm}^{-3} \times (0.0259)(1/u) \times 1.01 \times 6.2 \times (00)^{1/2} \, \text{cm}^{-3}]
\]

Figure 4.2 is a plot of \( n_i \) from Equation (4.23) for silicon, gallium arsenide, and germanium as a function of temperature. As seen in the figure, the value of \( n_i \) for these semiconductors may easily vary over several orders of magnitude as the temperature changes over a reasonable range.


## TEST YOUR UNDERSTANDING

**TYU 4.3** Calculate the intrinsic concentration in silicon at (a) \( T = 200 \, \text{K} \) and (b) \( T = 450 \, \text{K} \). Determine the ratio of \( n_i \) at \( T = 450 \, \text{K} \) to that at \( T = 200 \, \text{K} \).

\[
[01 \times 9.2 \times (1) \times 1.01 \times 1 \times 1.01 \times 9.9 \times 1 = 1 \, \text{cm}^{-3}]
\]

**TYU 4.4** Repeat TYU 4.3 for GaAs.

\[
[01 \times 1.82 \times (1) \times 1.01 \times 9.8 \times 1 = 1 \, \text{cm}^{-3}]
\]

**TYU 4.5** Repeat TYU 4.3 for Ge.

\[
[01 \times 8.1 \times (1) \times 1.01 \times 6.2 \times 1 = 1 \, \text{cm}^{-3}]
\]

**Figure 4.2** The intrinsic carrier concentration of Ge, Si, and GaAs as a function of temperature. (From Sze [14].)

### 4.1.4 The Intrinsic Fermi-Level Position

We have qualitatively argued that the Fermi energy level is located near the center of the forbidden bandgap for the intrinsic semiconductor. We can specifically calculate the intrinsic Fermi-level position. Since the electron and hole concentrations are equal, setting Equations (4.20) and (4.21) equal to each other, we have

\[
N_c \exp\left[-\frac{(E_c - E_F)}{kT}\right] = N_v \exp\left[-\frac{(E_F - E_v)}{kT}\right]
\]

(4.24)

If we take the natural log of both sides of this equation and solve for \( E_{Fi} \), we obtain

\[
E_{Fi} = \frac{1}{2} (E_c + E_v) + \frac{1}{2} kT \ln \left( \frac{N_v}{N_c} \right)
\]

(4.25)

From the definitions for \( N_c \) and \( N_v \) given by Equations (4.10) and (4.18), respectively, Equation (4.25) may be written as

\[
E_{Fi} = \frac{1}{2} (E_c + E_v) + \frac{3}{4} kT \ln \left( \frac{m_h^*}{m_e^*} \right)
\]

(4.26a)

The first term, \(\frac{1}{2} (E_c + E_v)\), is the energy exactly midway between \( E_c \) and \( E_v \), or the midgap energy. We can define

\[
\frac{1}{2} (E_c + E_v) = E_{\text{midgap}}
\]

so that

\[
E_{Fi} - E_{\text{midgap}} = \frac{3}{4} kT \ln \left( \frac{m_h^*}{m_e^*} \right)
\]

(4.26b)

If the electron and hole effective masses are equal so that \( m_h^* = m_e^* \), then the intrinsic Fermi level is exactly in the center of the bandgap. If \( m_h^* > m_e^* \), the intrinsic Fermi level is slightly above the center, and if \( m_h^* < m_e^* \), it is slightly below the center of the bandgap. The density of states function is directly related to the carrier effective mass; thus, a larger effective mass means a larger density of states function. The intrinsic Fermi level must shift away from the band with the larger density of states in order to maintain equal numbers of electrons and holes.

## EXAMPLE 4.4

**Objective:** Calculate the position of the intrinsic Fermi level with respect to the center of the bandgap in silicon at \( T = 300 \, \text{K} \).

The density of states effective carrier masses in silicon are \( m_e^* = 1.08m_0 \) and \( m_h^* = 0.56m_0 \).

**Solution**

The intrinsic Fermi level with respect to the center of the bandgap is

\[
E_{Fi} - E_{\text{midgap}} = \frac{3}{4} kT \ln \left( \frac{m_h^*}{m_e^*} \right) = \frac{3}{4} (0.0259) \ln \left( \frac{0.56}{1.08} \right)
\]

or

\[
E_{Fi} - E_{\text{midgap}} = -0.0128 \, \text{eV} = -12.8 \, \text{meV}
\]

**Comment**

The intrinsic Fermi level in silicon is 12.8 meV below the midgap energy. If we compare 12.8 meV to 560 meV, which is one-half of the bandgap energy of silicon, we can, in many applications, simply approximate the intrinsic Fermi level to be in the center of the bandgap.

**EXERCISE PROBLEM**

Ex 4.4 Determine the position of the intrinsic Fermi level at \( T = 300 \, \text{K} \) with respect to the center of the bandgap for (a) GaAs and (b) Ge. \[ \text{[Assume } 0.7 \, \text{eV} - (q) : \text{Assume } 0.8 \, \text{eV} + (q) \text{]} \]

## TEST YOUR UNDERSTANDING

**TYU 4.6** Determine the position of the intrinsic Fermi level with respect to the center of the bandgap in silicon at (a) \( T = 200 \, \text{K} \) and (b) \( T = 400 \, \text{K} \). Assume the effective masses are constant over this temperature range. \[ \text{[AAU10 L1—(q); AAU10 SQS8—(q); SUV1]} \]

## 4.2 DOPANT ATOMS AND ENERGY LEVELS

The intrinsic semiconductor may be an interesting material, but the real power of semiconductors is realized by adding small, controlled amounts of specific dopant, or impurity, atoms. This doping process, described briefly in Chapter 1, can greatly alter the electrical characteristics of the semiconductor. The doped semiconductor, called an extrinsic material, is the primary reason we can fabricate the various semiconductor devices that we will consider in later chapters.

### 4.2.1 Qualitative Description

In Chapter 3, we discussed the covalent bonding of silicon and considered the simple two-dimensional representation of the single-crystal silicon lattice as shown in Figure 4.3. Now consider adding a group V element, such as phosphorus, as a substitutional impurity. The group V element has five valence electrons. Four of these will contribute to the covalent bonding with the silicon atoms, leaving the fifth more loosely bound to the phosphorus atom. This effect is schematically shown in Figure 4.4. We refer to the fifth valence electron as a donor electron.

The phosphorus atom without the donor electron is positively charged. At very low temperatures, the donor electron is bound to the phosphorus atom. However, by intuition, it should seem clear that the energy required to elevate the donor electron into the conduction band is considerably less than that for the electrons involved in the covalent bonding. Figure 4.5 shows the energy-band diagram that we would expect. The energy level, \( E_D \), is the energy state of the donor electron.

**Figure 4.5** | The energy-band diagram showing (a) the discrete donor energy state and (b) the effect of a donor state being ionized.

If a small amount of energy, such as thermal energy, is added to the donor electron, it can be elevated into the conduction band, leaving behind a positively charged phosphorus ion. The electron in the conduction band can now move through the crystal generating a current, while the positively charged ion is fixed in the crystal. This type of impurity atom donates an electron to the conduction band and so is called a **donor impurity atom**. The donor impurity atoms add electrons to the conduction band without creating holes in the valence band. The resulting material is referred to as an **n-type semiconductor** (n for the negatively charged electron).

Now consider adding a group III element, such as boron, as a **substitutional impurity** to silicon. The group III element has three valence electrons, which are all taken up in the covalent bonding. As shown in Figure 4.6a, one covalent bonding position appears to be empty. If an electron were to occupy this “empty” position, its energy would have to be greater than that of the valence electrons, since the net charge state of the boron atom would now be negative. However, the electron occupying this “empty” position does not have sufficient energy to be in the conduction band, so its energy is far smaller than the conduction-band energy. Figure 4.6b shows how valence electrons may gain a small amount of thermal energy and move about in the lattice.

!Silicon lattice representation

**Figure 4.6** | Two-dimensional representation of a silicon lattice (a) doped with a boron atom and (b) showing the ionization of the boron atom resulting in a hole.

**Figure 4.7**

The energy-band diagram showing (a) the discrete acceptor energy state and (b) the effect of an acceptor state being ionized.


The crystal. The "empty" position associated with the boron atom becomes occupied, and other valence electron positions become vacated. These other vacated electron positions can be thought of as holes in the semiconductor material.

Figure 4.7 shows the expected energy state of the "empty" position and also the formation of a hole in the valence band. The hole can move through the crystal generating a current, while the negatively charged boron atom is fixed in the crystal. The group III atom accepts an electron from the valence band and so is referred to as an **acceptor impurity atom**. The acceptor atom can generate holes in the valence band without generating electrons in the conduction band. This type of semiconductor material is referred to as a **p type material** (p for the positively charged hole).

The pure single-crystal semiconductor material is called an intrinsic material. Adding controlled amounts of dopant atoms, either donors or acceptors, creates a material called an **extrinsic semiconductor**. An extrinsic semiconductor will have either a preponderance of electrons (n type) or a preponderance of holes (p type).

### 4.2.2 Ionization Energy

We can calculate the approximate distance of the donor electron from the donor impurity ion, and also the approximate energy required to elevate the donor electron into the conduction band. This energy is referred to as the **ionization energy**. We will use the Bohr model of the atom for these calculations. The justification for using this model is that the most probable distance of an electron from the nucleus in a hydrogen atom, determined from quantum mechanics, is the same as the Bohr radius. The energy levels in the hydrogen atom determined from quantum mechanics are also the same as obtained from the Bohr theory.

In the case of the donor impurity atom, we may visualize the donor electron orbiting the donor ion, which is embedded in the semiconductor material. We will need to use the permittivity of the semiconductor material in the calculations rather than the permittivity of free space as is used in the case of the hydrogen atom. We will also use the effective mass of the electron in the calculations.

The analysis begins by setting the coulomb force of attraction between the electron and ion equal to the centripetal force of the orbiting electron. This condition will give a steady orbit. We have

\[
\frac{e^2}{4\pi \varepsilon r} = \frac{mv^2}{r}
\]

(4.27)

where \( v \) is the magnitude of the velocity and \( r \) is the radius of the orbit. If we assume the angular momentum is also quantized, then we can write

\[
m^* r_n v = n\hbar
\]

(4.28)

where \( n \) is a positive integer. Solving for \( v \) from Equation (4.28), substituting into Equation (4.27), and solving for the radius, we obtain

\[
r_n = \frac{n^2 \hbar^2 4\pi \varepsilon}{m^* e^2}
\]

(4.29)

The assumption of the angular momentum being quantized leads to the radius also being quantized.

The Bohr radius is defined as

\[
a_0 = \frac{4\pi \varepsilon \hbar^2}{m_0 e^2} = 0.53 \, \text{Å}
\]

(4.30)

We can normalize the radius of the donor orbital to that of the Bohr radius, which gives

\[
\frac{r_n}{a_0} = n^2 \varepsilon_r \left( \frac{m_0}{m^*} \right)
\]

(4.31)

where \( \varepsilon_r \) is the relative dielectric constant of the semiconductor material, \( m_0 \) is the rest mass of an electron, and \( m^* \) is the conductivity effective mass of the electron in the semiconductor. 

If we consider the lowest energy state in which \( n = 1 \) and if we consider silicon in which \( \varepsilon_r = 11.7 \) and the conductivity effective mass is \( m^*/m_0 = 0.26 \), then we have that

\[
\frac{r_1}{a_0} = 45
\]

(4.32)

or \( r_1 = 23.9 \, \text{Å} \). This radius corresponds to approximately four lattice constants of silicon. Recall that one unit cell in silicon effectively contains eight atoms, so the radius of the orbiting donor electron encompasses many silicon atoms. The donor electron is not tightly bound to the donor atom.

The total energy of the orbiting electron is given by

\[
E = T + V
\]

(4.33)

where \( T \) is the kinetic energy and \( V \) is the potential energy of the electron. The kinetic energy is

\[
T = \frac{1}{2} m^* v^2
\]

(4.34)


*The conductivity effective mass is used when electrons and holes are in motion. See Appendix F for a discussion of effective mass concepts.*

Using the velocity \( v \) from Equation (4.28) and the radius \( r_n \) from Equation (4.29), the kinetic energy becomes

\[
T = \frac{m^* e^4}{2(n\hbar)^2(4\pi \varepsilon)^2}
\]

(4.35)

The potential energy is

\[
V = -\frac{e^2}{4\pi \varepsilon} = -\frac{m^* e^4}{(n\hbar)^2(4\pi \varepsilon)^2}
\]

(4.36)

The total energy is the sum of the kinetic and potential energies, so that

\[
E = T + V = -\frac{m^* e^4}{2(n\hbar)^2(4\pi \varepsilon)^2}
\]

(4.37)

For the hydrogen atom, \( m^* = m_0 \) and \( \varepsilon = \varepsilon_0 \). The ionization energy of the hydrogen atom in the lowest energy state is then \( E = -13.6 \, \text{eV} \). If we consider silicon, the ionization energy is \( E = -25.8 \, \text{meV} \), much less than the bandgap energy of silicon. This energy is the approximate ionization energy of the donor atom, or the energy required to elevate the donor electron into the conduction band.

For ordinary donor impurities such as phosphorus or arsenic in silicon or germanium, this hydrogenic model works quite well and gives some indication of the magnitudes of the ionization energies involved. Table 4.3 lists the actual experimentally measured ionization energies for a few impurities in silicon and germanium. Germanium and silicon have different relative dielectric constants and effective masses; thus, we expect the ionization energies to differ.

### 4.2.3 Group III–V Semiconductors

In the previous sections, we have discussed the donor and acceptor impurities in a group IV semiconductor, such as silicon. The situation in the group III–V compound semiconductors, such as gallium arsenide, is more complicated. Group II elements, such as beryllium, zinc, and cadmium, can enter the lattice as substitutional impurities, replacing the group III gallium element to become acceptor impurities. Similarly, group VI elements, such as selenium and tellurium, can enter the lattice substitutionally, replacing the group V arsenic element to become donor impurities. The corresponding ionization energies for these impurities are smaller than those for the impurities in silicon. The ionization energies for the donors in gallium arsenide.

**Table 4.3** Impurity ionization energies in silicon and germanium

| Impurity  | Ionization energy (eV) | |
|-----------|------------------------|---|
|           | Si     | Ge            |
| **Donors**|        |               |
| Phosphorus| 0.045  | 0.012         |
| Arsenic   | 0.05   | 0.0127        |
| **Acceptors**|     |               |
| Boron     | 0.045  | 0.0104        |
| Aluminum  | 0.06   | 0.0102        |


## 4.3 The Extrinsic Semiconductor

**Table 4.4** Impurity Ionization Energies in Gallium Arsenide

| Impurity   | Ionization Energy (eV) |
|------------|------------------------|
| **Donors** |                        |
| Selenium   | 0.0059                 |
| Tellurium  | 0.0058                 |
| Silicon    | 0.0058                 |
| Germanium  | 0.0061                 |
| **Acceptors** |                     |
| Beryllium  | 0.028                  |
| Zinc       | 0.0307                 |
| Cadmium    | 0.0347                 |
| Silicon    | 0.0345                 |
| Germanium  | 0.0404                 |

The ionization energies for donors are smaller than those for the acceptors, because of the smaller effective mass of the electron compared to that of the hole.

Group IV elements, such as silicon and germanium, can also be impurity atoms in gallium arsenide. If a silicon atom replaces a gallium atom, the silicon impurity will act as a donor, but if the silicon atom replaces an arsenic atom, then the silicon impurity will act as an acceptor. The same is true for germanium as an impurity atom. Such impurities are called amphoteric. Experimentally in gallium arsenide, it is found that germanium is predominantly an acceptor and silicon is predominantly a donor. Table 4.4 lists the ionization energies for the various impurity atoms in gallium arsenide.


## TEST YOUR UNDERSTANDING

**TYU 4.7** (a) Calculate the ionization energy and the radius (normalized to the Bohr radius) of a donor electron in its lowest energy state in GaAs. (b) Repeat part (a) for Ge.

\[
\text{[} \epsilon \text{]} \frac{9q}{\mu} \Lambda \epsilon^9 - (q) \frac{S61 = 9q}{\mu} \Lambda \mu 0 \epsilon - (b) \frac{9q}{\mu}
\]

## 4.3 The Extrinsic Semiconductor

We defined an intrinsic semiconductor as a material with no impurity atoms present in the crystal. An extrinsic semiconductor is defined as a semiconductor in which controlled amounts of specific dopant or impurity atoms have been added so that the thermal-equilibrium electron and hole concentrations are different from the intrinsic carrier concentration. One type of carrier will predominate in an extrinsic semiconductor.

### 4.3.1 Equilibrium Distribution of Electrons and Holes

Adding donor or acceptor impurity atoms to a semiconductor will change the distribution of electrons and holes in the material. Since the Fermi energy is related to the distribution function, the Fermi energy will change as dopant atoms are added.

**Figure 4.8** | Density of states functions, Fermi–Dirac probability function, and areas representing electron and hole concentrations for the case when \( E_F \) is above the intrinsic Fermi energy.


If the Fermi energy changes from near the midgap value, the density of electrons in the conduction band and the density of holes in the valence band will change. These effects are shown in Figures 4.8 and 4.9. Figure 4.8 shows the case for \( E_F > E_{Fi} \) and Figure 4.9 shows the case for \( E_F < E_{Fi} \). When \( E_F > E_{Fi} \), the electron concentration is larger than the hole concentration, and when \( E_F < E_{Fi} \), the hole concentration is larger than the electron concentration. When the density of electrons is greater than the density of holes, the semiconductor is n type; donor impurity atoms have been added. When the density of holes is greater than the density of electrons, the semiconductor is p type; acceptor impurity atoms have been added. The Fermi energy level in a semiconductor changes as the electron and hole concentrations change and, again, the Fermi energy changes as donor or acceptor impurities are added. The change in the Fermi level as a function of impurity concentrations is considered in Section 4.6.

**Figure 4.9** Density of states functions, Fermi–Dirac probability function, and areas representing electron and hole concentrations for the case when \( E_F \) is below the intrinsic Fermi energy.

The expressions previously derived for the **thermal-equilibrium concentration of electrons and holes**, given by Equations (4.11) and (4.19), are general equations for \( n_0 \) and \( p_0 \) in terms of the Fermi energy. These equations are again given as

\[
n_0 = N_c \exp\left(\frac{-(E_c - E_F)}{kT}\right)
\]

and

\[
p_0 = N_v \exp\left(\frac{-(E_F - E_v)}{kT}\right)
\]

As we just discussed, the Fermi energy may vary through the bandgap energy, which will then change the values of \( n_0 \) and \( p_0 \).

## Example 4.5

**Objective:** Calculate the thermal equilibrium concentrations of electrons and holes for a given Fermi energy.

Consider silicon at \( T = 300 \, \text{K} \) so that \( N_c = 2.8 \times 10^{19} \, \text{cm}^{-3} \) and \( N_v = 1.04 \times 10^{19} \, \text{cm}^{-3} \). Assume that the Fermi energy is 0.25 eV below the conduction band. If we assume that the bandgap energy of silicon is 1.12 eV, then the Fermi energy will be 0.87 eV above the valence band.

**Solution**

Using Equation (4.11), we have

\[
n_0 = (2.8 \times 10^{19}) \exp\left(\frac{-0.25}{0.0259}\right) = 1.8 \times 10^{15} \, \text{cm}^{-3}
\]

From Equation (4.19), we can write

\[
p_0 = (1.04 \times 10^{19}) \exp\left(\frac{-0.87}{0.0259}\right) = 2.7 \times 10^4 \, \text{cm}^{-3}
\]

**Comment**

The change in the Fermi level is actually a function of the donor or acceptor impurity concentrations that are added to the semiconductor. However, this example shows that electron and hole concentrations change by orders of magnitude from the intrinsic carrier concentration as the Fermi energy changes by a few tenths of an electron-volt.

**Exercise Problem**

**Ex 4.5** Determine the thermal-equilibrium concentrations of electrons and holes in silicon at \( T = 300 \, \text{K} \) if the Fermi energy level is 0.215 eV above the valence-band energy \( E_v \).


In the previous example, since \( n_0 > p_0 \), the semiconductor is n-type. In an n-type semiconductor, electrons are referred to as the majority carrier and holes as the minority carrier. By comparing the relative values of \( n_0 \) and \( p_0 \) in the example, it is easy to see how this designation came about. Similarly, in a p-type semiconductor where \( p_0 > n_0 \), holes are the majority carrier and electrons are the minority carrier.

We may derive another form of the equations for the thermal-equilibrium concentrations of electrons and holes. If we add and subtract an intrinsic Fermi energy in the exponent of Equation (4.11), we can write

\[
n_0 = N_c \exp\left(\frac{-(E_c - E_{Fi}) + (E_F - E_{Fi})}{kT}\right) \tag{4.38a}
\]

or

\[
n_0 = N_c \exp\left(\frac{-(E_c - E_{Fi})}{kT}\right) \exp\left(\frac{E_F - E_{Fi}}{kT}\right) \tag{4.38b}
\]

The intrinsic carrier concentration is given by Equation (4.20) as

\[
n_i = N_c \exp\left(\frac{-(E_c - E_{Fi})}{kT}\right)
\]

so that the thermal-equilibrium electron concentration can be written as

\[
n_0 = n_i \exp \left( \frac{E_F - E_{Fi}}{kT} \right)
\]

(4.39)

Similarly, if we add and subtract an intrinsic Fermi energy in the exponent of Equation (4.19), we will obtain

\[
p_0 = n_i \exp \left( \frac{-(E_F - E_{Fi})}{kT} \right)
\]

(4.40)

As we will see, the Fermi level changes when donors and acceptors are added, but Equations (4.39) and (4.40) show that, as the Fermi level changes from the intrinsic Fermi level, \( n_0 \) and \( p_0 \) change from the \( n_i \) value. If \( E_F > E_{Fi} \), then we will have \( n_0 > n_i \) and \( p_0 < n_i \). One characteristic of an n-type semiconductor is that \( E_F > E_{Fi} \) so that \( n_0 > p_0 \). Similarly, in a p-type semiconductor, \( E_F < E_{Fi} \) so that \( p_0 > n_i \) and \( p_0 < n_i \); thus, \( p_0 > n_0 \).

We can see the functional dependence of \( n_0 \) and \( p_0 \) with \( E_F \) in Figures 4.8 and 4.9. As \( E_F \) moves above or below \( E_{Fi} \), the overlapping probability function with the density of states functions in the conduction band and valence band changes. As \( E_F \) moves above \( E_{Fi} \), the probability function in the conduction band increases, while the probability, \( 1 - f_F(E) \), of an empty state (hole) in the valence band decreases. As \( E_F \) moves below \( E_{Fi} \), the opposite occurs.

### 4.3.2 The \( n_0 p_0 \) Product

We may take the product of the general expressions for \( n_0 \) and \( p_0 \) as given in Equations (4.11) and (4.19), respectively. The result is

\[
n_0 p_0 = N_c N_v \exp \left( \frac{-(E_c - E_F)}{kT} \right) \exp \left( \frac{-(E_F - E_v)}{kT} \right)
\]

(4.41)

which may be written as

\[
n_0 p_0 = N_c N_v \exp \left( \frac{-E_g}{kT} \right)
\]

(4.42)

As Equation (4.42) was derived for a general value of Fermi energy, the values of \( n_0 \) and \( p_0 \) are not necessarily equal. However, Equation (4.42) is exactly the same as Equation (4.23), which we derived for the case of an intrinsic semiconductor. We then have that, for the semiconductor in thermal equilibrium,

\[
n_0 p_0 = n_i^2
\]

(4.43)

Equation (4.43) states that the product of \( n_0 \) and \( p_0 \) is always a constant for a given semiconductor material at a given temperature. Although this equation seems


very simple. It is one of the fundamental principles of semiconductors in thermal equilibrium. The significance of this relation will become more apparent in the chapters that follow. It is important to keep in mind that Equation (4.43) was derived using the Boltzmann approximation. If the Boltzmann approximation is not valid, then likewise, Equation (4.43) is not valid.

An extrinsic semiconductor in thermal equilibrium does not, strictly speaking, contain an intrinsic carrier concentration, although some thermally generated carriers are present. The intrinsic electron and hole carrier concentrations are modified by the donor or acceptor impurities. However, we may think of the intrinsic concentration \( n_i \) in Equation (4.43) simply as a parameter of the semiconductor material.

### 4.3.3 The Fermi–Dirac Integral

In the derivation of the Equations (4.11) and (4.19) for the thermal equilibrium electron and hole concentrations, we assumed that the Boltzmann approximation was valid. If the Boltzmann approximation does not hold, the thermal equilibrium electron concentration is written from Equation (4.43) as

\[
n_0 = \frac{4 \pi}{h^3} (2m_n^*)^{3/2} \int_{E_c}^{\infty} \frac{(E - E_c)^{1/2} \, dE}{1 + \exp \left( \frac{E - E_f}{kT} \right)}
\]

(4.44)

If we again make a change of variable and let

\[
\eta = \frac{E - E_c}{kT}
\]

(4.45a)

and also define

\[
\eta_f = \frac{E_f - E_c}{kT}
\]

(4.45b)

then we can rewrite Equation (4.44) as

\[
n_0 = 4 \pi \left( \frac{2m_n^* kT}{h^2} \right)^{3/2} \int_0^{\infty} \frac{\eta^{1/2} \, d\eta}{1 + \exp (\eta - \eta_f)}
\]

(4.46)

The integral is defined as

\[
F_{1/2}(\eta_f) = \int_0^{\infty} \frac{\eta^{1/2} \, d\eta}{1 + \exp (\eta - \eta_f)}
\]

(4.47)

This function, called the Fermi–Dirac integral, is a tabulated function of the variable \( \eta_f \). Figure 4.10 is a plot of the Fermi–Dirac integral. Note that if \( \eta_f > 0 \), then \( E_f > E_c \); thus, the Fermi energy is actually in the conduction band.


## Example 4.6

**Objective:** Calculate the electron concentration using the Fermi–Dirac integral.

Let \( \eta_f = 2 \) so that the Fermi energy is above the conduction band by approximately 52 meV at \( T = 300 \, \text{K} \).

**Solution**

Equation (4.46) can be written as

\[
n_0 = \frac{2}{\sqrt{\pi}} N_c F_{1/2} (\eta_f)
\]

For silicon at \( T = 300 \, \text{K} \), \( N_c = 2.8 \times 10^{19} \, \text{cm}^{-3} \) and, from Figure 4.10, the Fermi–Dirac integral has a value of \( F_{1/2} (2) = 2.7 \). Then

\[
n_0 = 2 \left( \frac{2.8 \times 10^{19} \times 2.7}{\sqrt{\pi}} \right) = 8.53 \times 10^{19} \, \text{cm}^{-3}
\]

**Comment**

Note that if we had used Equation (4.11), the thermal equilibrium value of \( n_0 \) would be \( n_0 = 2.08 \times 10^{20} \, \text{cm}^{-3} \), which is incorrect since the Boltzmann approximation is not valid for this case.

**EXERCISE PROBLEM**

Ex. 4.6 If \( n_0 = 1.5 \times 10^{20} \, \text{cm}^{-3} \) in silicon at \( T = 300 \, \text{K} \), determine the position of the Fermi level relative to the conduction-band energy \( E_c \).

We may use the same general method to calculate the thermal equilibrium concentration of holes. We obtain

\[
p_0 = 4 \left( \frac{2m_p^* kT}{h^2} \right)^{3/2} \int_0^\infty \frac{(\eta')^{1/2} d\eta'}{1 + \exp(\eta' - \eta_p)} \tag{4.48}
\]


**Figure 4.10** The Fermi–Dirac integral \( F_{1/2} \) as a function of the Fermi energy.  
*(From Sze [14].)*

\[
F_{1/2}(\eta_F) = \int_0^\infty \frac{\eta^{1/2} d\eta}{1 + \exp(\eta - \eta_F)}
\]

where

\[
\eta' = \frac{E_c - E}{kT}
\]

(4.49a)

and

\[
\eta'_F = \frac{E_c - E_F}{kT}
\]

(4.49b)

The integral in Equation (4.48) is the same Fermi–Dirac integral defined by Equation (4.47), although the variables have slightly different definitions. We may note that if \(\eta'_F > 0\), then the Fermi level is in the valence band.


## TEST YOUR UNDERSTANDING

**TYU 4.8**  
(a) Calculate the thermal-equilibrium electron concentration in silicon at \(T = 300 \, \text{K}\) for the case when \(E_F = E_c\).  
(b) Calculate the thermal-equilibrium hole concentration in silicon at \(T = 300 \, \text{K}\) for the case when \(E_F = E_v\).


### 4.3.4 Degenerate and Nondegenerate Semiconductors

In our discussion of adding dopant atoms to a semiconductor, we have implicitly assumed that the concentration of dopant atoms added is small when compared to the density of host or semiconductor atoms. The small number of impurity atoms are spread far enough apart so that there is no interaction between donor electrons, for example, in an n-type material. We have assumed that the impurities introduce discrete, noninteracting donor energy states in the n-type semiconductor and discrete, noninteracting acceptor states in the p-type semiconductor. These types of semiconductors are referred to as nondegenerate semiconductors.

If the impurity concentration increases, the distance between the impurity atoms decreases and a point will be reached when donor electrons, for example, will begin to interact with each other. When this occurs, the single discrete donor energy will split into a band of energies. As the donor concentration further increases, the band of donor states widens and may overlap the bottom of the conduction band. This overlap occurs when the donor concentration becomes comparable with the effective density of states. When the concentration of electrons in the conduction band exceeds the density of states \(N_c\), the Fermi energy lies within the conduction band. This type of semiconductor is called a degenerate n-type semiconductor.

In a similar way, as the acceptor doping concentration increases in a p-type semiconductor, the discrete acceptor energy states will split into a band of energies and may overlap the top of the valence band. The Fermi energy will lie in the valence band when the concentration of holes exceeds the density of states \(N_v\). This type of semiconductor is called a degenerate p-type semiconductor.

Schematic models of the energy-band diagrams for a degenerate n-type and degenerate p-type semiconductor are shown in Figure 4.11. The energy states below \(E_F\)


**Figure 4.11** | Simplified energy-band diagrams for degenerately doped (a) n-type and (b) p-type semiconductors.

are mostly filled with electrons and the energy states above \( E_F \) are mostly empty. In the degenerate n-type semiconductor, the states between \( E_F \) and \( E_c \) are mostly filled with electrons; thus, the electron concentration in the conduction band is very large. Similarly, in the degenerate p-type semiconductor, the energy states between \( E_v \) and \( E_F \) are mostly empty; thus, the hole concentration in the valence band is very large.

## 4.4 Statistics of Donors and Acceptors

In the previous chapter, we discussed the Fermi–Dirac distribution function, which gives the probability that a particular energy state will be occupied by an electron. We need to reconsider this function and apply the probability statistics to the donor and acceptor energy states.

### 4.4.1 Probability Function

One postulate used in the derivation of the Fermi–Dirac probability function was the Pauli exclusion principle, which states that only one particle is permitted in each quantum state. The Pauli exclusion principle also applies to the donor and acceptor states.

Suppose we have \( N_d \) electrons and \( g_i \) quantum states, where the subscript \( i \) indicates the \( i \)th energy level. There are \( g_i \) ways of choosing where to put the first particle. Each donor level has two possible spin orientations for the donor electron; thus, each donor level has two quantum states. The insertion of an electron into one quantum state, however, precludes putting an electron into the second quantum state. By adding one electron, the vacancy requirement of the atom is satisfied, and the addition of a second electron in the donor level is not possible. The distribution function of donor electrons in the donor energy states is then slightly different than the Fermi–Dirac function.

The probability function of electrons occupying the donor state is

\[
n_d = \frac{N_d}{1 + \frac{1}{2} \exp \left( \frac{E_d - E_F}{kT} \right)}
\]

(4.50)

where \( n_d \) is the density of electrons occupying the donor level and \( E_d \) is the energy of the donor level. The factor \(\frac{1}{2}\) in this equation is a direct result of the spin factor just mentioned. The \(\frac{1}{g}\) factor is sometimes written as \(1/g\), where \(g\) is called a degeneracy factor.

Equation (4.50) can also be written in the form

\[
n_d = N_d - N_d^+
\]

(4.51)

where \( N_d^+ \) is the concentration of ionized donors. In many applications, we will be interested more in the concentration of ionized donors than in the concentration of electrons remaining in the donor states.

If we do the same type of analysis for acceptor atoms, we obtain the expression

\[
p_a = \frac{N_a}{1 + \frac{1}{g} \exp \left( \frac{E_F - E_a}{kT} \right)} = N_a - N_a^-
\]

(4.52)

where \( N_a \) is the concentration of acceptor atoms, \( E_a \) is the acceptor energy level, \( p_a \) is the concentration of holes in the acceptor states, and \( N_a^- \) is the concentration of ionized acceptors. A hole in an acceptor state corresponds to an acceptor atom that is neutrally charged and still has an “empty” bonding position as we have discussed in Section 4.2.1. The parameter \( g \) is, again, a degeneracy factor. The ground state degeneracy factor \( g \) is normally taken as 4 for the acceptor level in silicon and gallium arsenide because of the detailed band structure.

### 4.4.2 Complete Ionization and Freeze-Out

The probability function for electrons in the donor energy state was just given by Equation (4.50). If we assume that \((E_d - E_F) \gg kT\) then

\[
n_d \approx \frac{N_d}{2} \exp \left( \frac{E_d - E_F}{kT} \right) = 2N_d \exp \left( \frac{-(E_d - E_F)}{kT} \right)
\]

(4.53)

If \((E_d - E_F) \gg kT\), then the Boltzmann approximation is also valid for the electrons in the conduction band so that, from Equation (4.11),

\[
n_0 = N_c \exp \left( \frac{-(E_c - E_F)}{kT} \right)
\]

We can determine the relative number of electrons in the donor state compared with the total number of electrons; therefore, we can consider the ratio of electrons in the donor state to the total number of electrons in the conduction band plus donor state. Using the expressions of Equations (4.53) and (4.11), we write

\[
\frac{n_d}{n_d + n_0} = \frac{2N_d \exp \left( \frac{-(E_d - E_F)}{kT} \right)}{2N_d \exp \left( \frac{-(E_d - E_F)}{kT} \right) + N_c \exp \left( \frac{-(E_c - E_F)}{kT} \right)}
\]

(4.54)


The Fermi energy cancels out of this expression. Dividing by the numerator term, we obtain

\[
\frac{n_d}{n_u + n_0} = \frac{1}{1 + \frac{N_c}{2N_D} \exp \left( \frac{-(E_c - E_d)}{kT} \right)}
\]

(4.55)

The factor \((E_c - E_d)\) is just the ionization energy of the donor electrons.

## EXAMPLE 4.7

**Objective**: 
Determine the fraction of total electrons still in the donor states at \( T = 300 \, \text{K} \)

Consider phosphorus doping in silicon, for \( T = 300 \, \text{K} \), at a concentration of \( N_d = 10^{16} \, \text{cm}^{-3} \).

**Solution**

Using Equation (4.55), we find

\[
\frac{n_d}{n_0 + n_d} = \frac{1}{1 + \frac{2.8 \times 10^{19}}{2(10^{16})} \exp \left( \frac{-0.045}{0.0259} \right)} = 0.0041 = 0.41\%
\]

**Comment**

This example shows that there are very few electrons in the donor state compared with the conduction band. Essentially all of the electrons from the donor states are in the conduction band and, since only about 0.4 percent of the donor states contain electrons, the donor states are said to be completely ionized.

**Exercise Problem**

**Ex 4.7** Repeat Example 4.7 for (a) \( T = 250 \, \text{K} \) and (b) \( T = 200 \, \text{K} \). (c) What can be said about the fraction as the temperature decreases?


At room temperature, then, the donor states are essentially completely ionized and, for a typical doping of \( 10^{16} \, \text{cm}^{-3} \), almost all donor impurity atoms have donated an electron to the conduction band.

At room temperature, there is also essentially complete ionization of the acceptor atoms. This means that each acceptor atom has accepted an electron from the valence band so that \( p_a \) is zero. At typical acceptor doping concentrations, a hole is created in the valence band for each acceptor atom. This ionization effect and the creation of electrons and holes in the conduction band and valence band, respectively, are shown in Figure 4.12.

The opposite of complete ionization occurs at \( T = 0 \, \text{K} \). At absolute zero degrees, all electrons are in their lowest possible energy state; that is, for an n-type semiconductor, each donor state must contain an electron, therefore \( n_d = N_d \) or \( N_d - n_d = 0 \). We must have, then, from Equation (4.50) that \(\exp \left[ \frac{(E_d - E_f)}{kT} \right] = 0\). Since \( T = 0 \, \text{K} \), this will occur for \(\exp(-\infty) = 0\), which means that \( E_f > E_d \). The Fermi energy level must be above the donor energy level at absolute zero. In the case of a p-type semiconductor at absolute zero temperature, the impurity atoms will not contain any electrons, so that the Fermi energy level must be below the acceptor energy state. The distribution of electrons among the various energy states, and hence the Fermi energy, is a function of temperature.

**Figure 4.12** Energy-band diagrams showing complete ionization of (a) donor states and (b) acceptor states.

**Figure 4.13** Energy-band diagram at \( T = 0 \, \text{K} \) for (a) n-type and (b) p-type semiconductors.

A detailed analysis, not given in this text, shows that at \( T = 0 \, \text{K} \), the Fermi energy is halfway between \( E_c \) and \( E_d \) for the n-type material and halfway between \( E_a \) and \( E_v \) for the p-type material. Figure 4.13 shows these effects. No electrons from the donor state are thermally elevated into the conduction band; this effect is called freeze-out. Similarly, when no electrons from the valence band are elevated into the acceptor states, the effect is also called freeze-out.

Between \( T = 0 \, \text{K} \), freeze-out, and \( T = 300 \, \text{K} \), complete ionization, we have partial ionization of donor or acceptor atoms.

## EXAMPLE 4.8

**Objective:** Determine the temperature at which 90 percent of acceptor atoms are ionized.

Consider p-type silicon doped with boron at a concentration of \( N_a = 10^{16} \, \text{cm}^{-3} \).

**Solution**

Find the ratio of holes in the acceptor state to the total number of holes in the valence band plus acceptor state. Taking into account the Boltzmann approximation and assuming the degeneracy factor is \( g = 4 \), we write

\[
\frac{p_s}{p_0 + p_s} = \frac{1}{1 + \frac{N_v}{4N_a} \cdot \exp\left(\frac{- (E_a - E_v)}{kT}\right)}
\]

For 90 percent ionization,

\[
\frac{p_x}{p_0 + p_x} = 0.10 = \frac{1}{1 + \left(1.04 \times 10^{19}\right) \left(\frac{T}{300}\right)^{3/2} \cdot \frac{1}{4 \times 10^{14}} \cdot \exp\left(\frac{-0.045}{0.0259 \left(\frac{T}{300}\right)}\right)}
\]

Using trial and error, we find that \( T = 193 \, \text{K} \).

**Comment**

This example shows that at approximately 100°C below room temperature, we still have 90 percent of the acceptor atoms ionized; in other words, 90 percent of the acceptor atoms have "donated" a hole to the valence band.

**EXERCISE PROBLEM**

Ex 4.8 Determine the fraction of total holes still in the acceptor states in silicon for \( N_a = 10^{16} \, \text{cm}^{-3} \) at (a) \( T = 250 \, \text{K} \) and (b) \( T = 200 \, \text{K} \).

\[ [x-01 \times 9E L 8 (q) \div -01 \times 16 6 (p) \, \text{suV}] \]


## TEST YOUR UNDERSTANDING

**TYU 4.9** Determine the fraction of total holes still in the acceptor states in silicon at \( T = 300 \, \text{K} \) for a boron impurity concentration of \( N_a = 10^{17} \, \text{cm}^{-3} \). \( (6L1'0 \, \text{suV}) \)

**TYU 4.10** Consider silicon with a phosphorus impurity concentration of \( N_d = 10^{15} \, \text{cm}^{-3} \).

Determine the percent of ionized phosphorus atoms at (a) \( T = 100 \, \text{K} \),

(b) \( T = 200 \, \text{K} \), (c) \( T = 300 \, \text{K} \), and (d) \( T = 400 \, \text{K} \).

\[ [\%866 66 (p) :\%966 66 (q) :\%286 66 (q) :\%296 66 (p) \, \text{suV}] \]


## 4.5 | CHARGE NEUTRALITY

In thermal equilibrium, the semiconductor crystal is electrically neutral. The electrons are distributed among the various energy states, creating negative and positive charges, but the net charge density is zero. This charge-neutrality condition is used to determine the thermal-equilibrium electron and hole concentrations as a function of the impurity doping concentration. We will define a compensated semiconductor and then determine the electron and hole concentrations as a function of the donor and acceptor concentrations.

### 4.5.1 Compensated Semiconductors

A **compensated semiconductor** is one that contains both donor and acceptor impurity atoms in the same region. A compensated semiconductor can be formed, for example, by diffusing acceptor impurities into an n-type material or by diffusing donor impurities into a p-type material. An n-type compensated semiconductor occurs when \( N_d > N_a \), and a p-type compensated semiconductor occurs when \( N_a > N_d \). If \( N_d = N_a \), we have a completely compensated semiconductor that has, as we will show, the characteristics of an intrinsic material. Compensated semiconductors are created quite naturally during device fabrication as we will see later.

### 4.5.2 Equilibrium Electron and Hole Concentrations

!Energy-band diagram of a compensated semiconductor

**Figure 4.14** Energy-band diagram of a compensated semiconductor showing ionized and un-ionized donors and acceptors.

Figure 4.14 shows the energy-band diagram of a semiconductor when both donor and acceptor impurity atoms are added to the same region to form a compensated semiconductor. The figure shows how the electrons and holes can be distributed among the various states.

The charge neutrality condition is expressed by equating the density of negative charges to the density of positive charges. We then have

\[
n_0 + N_D^+ = p_0 + N_A^-
\]

(4.56)

or

\[
n_0 + (N_D - n_d) = p_0 + (N_A - n_a)
\]

(4.57)

where \( n_0 \) and \( p_0 \) are the thermal-equilibrium concentrations of electrons and holes in the conduction band and valence band, respectively. The parameter \( n_d \) is the concentration of electrons in the donor energy states, so \( N_D^+ = N_D - n_d \) is the concentration of positively charged donor states. Similarly, \( p_a \) is the concentration of holes in the acceptor states, so \( N_A^- = N_A - p_a \) is the concentration of negatively charged acceptor states. We have expressions for \( n_0, p_0, n_d, \) and \( p_a \) in terms of the Fermi energy and temperature.


**Thermal-Equilibrium Electron Concentration**

If we assume complete ionization, \( n_d \) and \( p_a \) are both zero, and Equation (4.57) becomes:

\[
n_0 + N_a = p_0 + \frac{n_i^2}{n_0} = N_d
\]

(4.58)

If we express \( p_0 \) as \( n_i^2/n_0 \), then Equation (4.58) can be written as:

\[
n_0 + N_a = \frac{n_i^2}{n_0} + N_d
\]

(4.59a)

which in turn can be written as:

\[
n_0^2 = (N_d - N_a)n_0 - n_i^2 = 0
\]

(4.59b)

The electron concentration \( n_0 \) can be determined using the quadratic formula, or:

\[
n_0 = \frac{(N_d - N_a)}{2} + \sqrt{\left(\frac{N_d - N_a}{2}\right)^2 + n_i^2}
\]

(4.60)

The **positive sign** in the quadratic formula must be used, since, in the limit of an intrinsic semiconductor when \( N_d = N_a = 0 \), the electron concentration must be a positive quantity, or \( n_0 = n_i \).

Equation (4.60) is used to calculate the electron concentration in an n-type semiconductor, or when \( N_d > N_a \). Although Equation (4.60) was derived for a compensated semiconductor, the equation is also valid for \( N_a = 0 \).

## EXAMPLE 4.9
**Objective**: 
Determine the thermal-equilibrium electron and hole concentrations in silicon at \( T = 300 \, K \) for given doping concentrations.

(a) Let \( N_d = 10^{16} \, \text{cm}^{-3} \) and \( N_a = 0 \).

(b) Let \( N_d = 5 \times 10^{15} \, \text{cm}^{-3} \) and \( N_a = 2 \times 10^{15} \, \text{cm}^{-3} \).

Recall that \( n_i = 1.5 \times 10^{10} \, \text{cm}^{-3} \) in silicon at \( T = 300 \, K \).

**Solution**

(a) From Equation (4.60), the majority carrier electron concentration is:

\[
n_0 = \frac{10^{16}}{2} + \sqrt{\left(\frac{10^{16}}{2}\right)^2 + (1.5 \times 10^{10})^2} \approx 10^{16} \, \text{cm}^{-3}
\]

The minority carrier hole concentration is found to be:

\[
p_0 = \frac{n_i^2}{n_0} = \frac{(1.5 \times 10^{10})^2}{10^{16}} = 2.25 \times 10^4 \, \text{cm}^{-3}
\]

(b) Again, from Equation (4.60), the majority carrier electron concentration is:

\[
n_0 = \frac{5 \times 10^{15} - 2 \times 10^{15}}{2} + \sqrt{\left(\frac{5 \times 10^{15} - 2 \times 10^{15}}{2}\right)^2 + (1.5 \times 10^{10})^2} \approx 3 \times 10^{15} \, \text{cm}^{-3}
\]

The minority carrier hole concentration is:

\[
p_0 = \frac{n_i^2}{n_0} = \frac{(1.5 \times 10^{10})^2}{3 \times 10^{15}} = 7.5 \times 10^4 \, \text{cm}^{-3}
\]

**Comment**

In both parts of this example, \((N_d - N_a) \gg n_i\), so the **thermal-equilibrium majority carrier electron concentration** is essentially equal to the difference between the donor and acceptor concentrations. Also, in both cases, the majority carrier electron concentration is orders of magnitude larger than the minority carrier hole concentration.

**Exercise Problem**

**Ex 4.9** Find the thermal-equilibrium electron and hole concentrations in silicon with doping concentrations of \(N_d = 7 \times 10^{15} \, \text{cm}^{-3}\) and \(N_a = 3 \times 10^{15} \, \text{cm}^{-3}\) for (a) \(T = 250 \, \text{K}\) and (b) \(T = 400 \, \text{K}\).

\[
\begin{align*}
&\text{[} q \cdot \text{(} w_s \cdot s)01 \times y = w \cdot (q) \cdot i_w \cdot \Sigma \text{]} = q \cdot t_w \cdot (w_s)01 \times y = w \cdot (y) \cdot stv \text{]}
\end{align*}
\]

We have argued in our discussion and we may note from the results of Example 4.9 that the concentration of electrons in the conduction band increases above the intrinsic carrier concentration as we add donor impurity atoms. At the same time, the minority carrier hole concentration decreases below the intrinsic carrier concentration as we add donor atoms. We must keep in mind that as we add donor impurity atoms and the corresponding donor electrons, there is a redistribution of electrons among available energy states. Figure 4.15 shows a schematic of this physical redistribution. A few of the donor electrons will fall into the empty states in the valence band and, in doing so, will annihilate some of the intrinsic holes. The minority carrier hole concentration will therefore decrease as we have seen in Example 4.9. At the same time, because of this redistribution, the net electron concentration in the conduction band is not simply equal to the donor concentration plus the intrinsic electron concentration.


**Figure 4.15** | Energy-band diagram showing the redistribution of electrons when donors are added.


## EXAMPLE 4.10

**Objective**: 
Calculate the thermal-equilibrium electron and hole concentrations in germanium for a given doping concentration

Consider a germanium sample at \( T = 300 \, \text{K} \) in which \( N_D = 2 \times 10^{14} \, \text{cm}^{-3} \) and \( N_A = 0 \). Assume that \( n_i = 2.4 \times 10^{13} \, \text{cm}^{-3} \).

**Solution**

Again, from Equation (4.60), the majority carrier electron concentration is

\[
n_0 = \frac{2 \times 10^{14}}{2} + \sqrt{\left(\frac{2 \times 10^{14}}{2}\right)^2 + (2.4 \times 10^{13})^2} \approx 2.028 \times 10^{14} \, \text{cm}^{-3}
\]

The minority carrier hole concentration is

\[
p_0 = \frac{n_i^2}{n_0} = \frac{(2.4 \times 10^{13})^2}{2.028 \times 10^{14}} = 2.84 \times 10^{12} \, \text{cm}^{-3}
\]

**Comment**

If the donor impurity concentration is not too different in magnitude from the intrinsic carrier concentration, then the thermal-equilibrium majority carrier electron concentration is influenced by the intrinsic concentration.

**Exercise Problem**

**Ex 4.10** Repeat Example 4.10 for (a) \( T = 250 \, \text{K} \) and (b) \( T = 350 \, \text{K} \). (c) What can be said about a very low-doped material as the temperature increases?


We have seen that the intrinsic carrier concentration \( n_i \) is a very strong function of temperature. As the temperature increases, additional electron-hole pairs are thermally generated so that the \( n_i^2 \) term in Equation (4.60) may begin to dominate. The semiconductor will eventually lose its extrinsic characteristics. Figure 4.16 shows the electron concentration versus temperature in silicon doped with \( 5 \times 10^{14} \) donors per cm\(^3\). As the temperature increases, we can see where the intrinsic concentration begins to dominate. Also shown is the partial ionization, or the onset of freeze-out, at the low temperature.

**Thermal-Equilibrium Hole Concentration**

If we reconsider Equation (4.58) and express \( n_i^2/p_0 \), then we have

\[
\frac{n_i^2}{p_0} + N_D = p_0 + N_A \tag{4.61a}
\]

which we can write as

\[
p_0^2 - (N_A - N_D)p_0 - n_i^2 = 0 \tag{4.61b}
\]

**Figure 4.16** Electron concentration versus temperature showing the three regions: partial ionization, extrinsic, and intrinsic.

Using the quadratic formula, the hole concentration is given by

\[
p_0 = \frac{N_A}{2} + \sqrt{\left(\frac{N_A - N_D}{2}\right)^2 + n_i^2}
\]

(4.62)

where the positive sign, again, must be used. Equation (4.62) is used to calculate the thermal-equilibrium majority carrier hole concentration in a p-type semiconductor, or when \(N_A > N_D\). This equation also applies for \(N_D = 0\).

## EXAMPLE 4.11

**Objective:** Calculate the thermal-equilibrium electron and hole concentrations in a compensated p-type semiconductor.

Consider a silicon semiconductor at \(T = 300 \, K\) in which \(N_D = 10^{16} \, \text{cm}^{-3}\) and \(N_A = 3 \times 10^{15} \, \text{cm}^{-3}\). Assume \(n_i = 1.5 \times 10^{10} \, \text{cm}^{-3}\).

**Solution**

Since \(N_D > N_A\), the compensated semiconductor is p-type and the thermal-equilibrium majority carrier hole concentration is given by Equation (4.62) as

\[
p_0 = \frac{10^{16} - 3 \times 10^{15}}{2} + \sqrt{\left(\frac{10^{16} - 3 \times 10^{15}}{2}\right)^2 + (1.5 \times 10^{10})^2}
\]

so that

\[
p_0 \approx 7 \times 10^{15} \, \text{cm}^{-3}
\]

The minority carrier electron concentration is

\[
n_0 = \frac{n_i^2}{p_0} = \frac{(1.5 \times 10^{10})^2}{7 \times 10^{15}} = 3.21 \times 10^4 \, \text{cm}^{-3}
\]

**Comment**

If we assume complete ionization and if \((N_D - N_A) \gg n_i\), then the majority carrier hole concentration is, to a very good approximation, just the difference between the acceptor and donor concentrations.

**Exercise Problem**

**Ex 4.11**  
Consider silicon at \( T = 300 \, \text{K} \). Calculate the thermal-equilibrium electron and hole concentrations for impurity concentrations of (a) \( N_A = 4 \times 10^{16} \, \text{cm}^{-3} \), \( N_D = 8 \times 10^{15} \, \text{cm}^{-3} \) and (b) \( N_D = N_A = 3 \times 10^{15} \, \text{cm}^{-3} \).  
\[
\begin{align*}
\text{(a)} & \quad \underline{n_0 = 0.01 \times 1 = 0.01} \quad \underline{p_0 = 0.01 \times 20 = 0.2} \quad \underline{n_0 = 0.01 \times 7 = 0.07} \\
\text{(b)} & \quad \underline{n_0 = 0.01 \times 1 = 0.01} \quad \underline{p_0 = 0.01 \times 7 = 0.07}
\end{align*}
\]

We may note that, for a compensated p-type semiconductor, the minority carrier electron concentration is determined from
\[
n_0 = \frac{n_i^2}{p_0} = \frac{n_i^2}{(N_A - N_D)}
\]

Equations (4.60) and (4.62) are used to calculate the majority carrier electron concentration in an n-type semiconductor and majority carrier hole concentration in a p-type semiconductor, respectively. The minority carrier hole concentration in an n-type semiconductor could, theoretically, be calculated from Equation (4.62). However, we would be subtracting two numbers on the order of \( 10^{16} \, \text{cm}^{-3} \), for example, to obtain a number on the order of \( 10^4 \, \text{cm}^{-3} \), which from a practical point of view is not possible. The minority carrier concentrations are calculated from \( n_0p_0 = n_i^2 \) once the majority carrier concentration has been determined.


## TEST YOUR UNDERSTANDING

**TYU 4.11**  
Consider a compensated GaAs semiconductor at \( T = 300 \, \text{K} \) doped at \( N_A = 5 \times 10^{15} \, \text{cm}^{-3} \) and \( N_D = 2 \times 10^{16} \, \text{cm}^{-3} \). Calculate the thermal equilibrium electron and hole concentrations.  
\[
\underline{n_0 = 0.01 \times 9 = 0.09} \quad \underline{p_0 = 0.01 \times 1 = 0.01}
\]

**TYU 4.12**  
Silicon is doped at \( N_A = 10^{15} \, \text{cm}^{-3} \) and \( N_D = 0 \). (a) Plot the concentration of electrons versus temperature over the range 300 K to 600 K. (b) Calculate the temperature at which the electron concentration is equal to \( 1.1 \times 10^{15} \, \text{cm}^{-3} \).  
\[
\underline{n_0 = 0.01 \times 1 = 0.01}
\]

**TYU 4.13**  
A silicon device with n-type material is to be operated at \( T = 550 \, \text{K} \). At this temperature, the intrinsic carrier concentration must contribute no more than 5 percent of the total electron concentration. Determine the minimum donor concentration required to meet this specification.  
\[
\underline{n_0 = 0.01 \times 0.1 = 0.01}
\]


## 4.6 | POSITION OF FERMI ENERGY LEVEL

We have discussed qualitatively in Section 4.3.1 how the electron and hole concentrations change as the Fermi energy level moves through the bandgap energy. Then, in Section 4.5, we calculated the electron and hole concentrations as a function of donor and acceptor impurity concentrations. We can now determine the position of the Fermi energy level as a function of the doping concentrations and as a function of temperature. The relevance of the Fermi energy level will be further discussed after the mathematical derivations.

### 4.6.1 Mathematical Derivation

The position of the Fermi energy level within the bandgap can be determined by using the equations already developed for the thermal-equilibrium electron and hole concentrations. **If we assume the Boltzmann approximation to be valid**, then from Equation (4.11) we have \( n_0 = N_c \exp \left[ \frac{-(E_c - E_F)}{kT} \right] \). We can solve for \( E_c - E_F \) from this equation and obtain

\[
E_c - E_F = kT \ln \left( \frac{N_c}{n_0} \right)
\]

(4.63)

where \( n_0 \) is given by Equation (4.60). If we consider an n-type semiconductor in which \( N_D \gg p_0 \), then \( n_0 \approx N_D \), so that

\[
E_c - E_F = kT \ln \left( \frac{N_c}{N_D} \right)
\]

(4.64)

The distance between the bottom of the conduction band and the Fermi energy is a logarithmic function of the donor concentration. As the donor concentration increases, the Fermi level moves closer to the conduction band. Conversely, if the Fermi level moves closer to the conduction band, then the electron concentration in the conduction band is increasing. We may note that if we have a compensated semiconductor, then the \( N_D \) term in Equation (4.64) is simply replaced by \( N_D - N_A \), or the net effective donor concentration.


## DESIGN EXAMPLE 4.12

**Objective:** Determine the required donor impurity concentration to obtain a specified Fermi energy.

Silicon at \( T = 300 \, \text{K} \) contains an acceptor impurity concentration of \( N_A = 10^{16} \, \text{cm}^{-3} \). Determine the concentration of donor impurity atoms that must be added so that the silicon is n-type and the Fermi energy is 0.20 eV below the conduction-band edge.

**Solution**

From Equation (4.64), we have

\[
E_c - E_F = kT \ln \left( \frac{N_c}{N_D - N_A} \right)
\]

which can be rewritten as

\[
N_D - N_A = N_c \exp \left[ \frac{-(E_c - E_F)}{kT} \right]
\]

Then

\[
N_D - N_A = 2.8 \times 10^{19} \exp \left[ \frac{-0.20}{0.0259} \right] = 1.24 \times 10^{16} \, \text{cm}^{-3}
\]

or

\[
N_D = 1.24 \times 10^{16} + N_A = 2.24 \times 10^{16} \, \text{cm}^{-3}
\]

**Comment**

A compensated semiconductor can be fabricated to provide a specific Fermi energy level.

**Exercise Problem**

**Ex 4.12** Consider silicon at \( T = 300 \, \text{K} \) with doping concentrations of \( N_D = 8 \times 10^{15} \, \text{cm}^{-3} \) and \( N_A = 5 \times 10^{15} \, \text{cm}^{-3} \). Determine the position of the Fermi energy level with respect to \( E_c \). (\( kT = 0.0259 \, \text{eV} \))


We may develop a slightly different expression for the position of the Fermi level. We had from Equation (4.39) that \( n_0 = n_i \exp \left[ \frac{(E_F - E_i)}{kT} \right] \). We can solve for \( E_F - E_i \) as

\[
E_F - E_i = kT \ln \left( \frac{n_0}{n_i} \right)
\]

(4.65)

Equation (4.65) can be used specifically for an n-type semiconductor, where \( n_0 \) is given by Equation (4.60), to find the difference between the Fermi level and the intrinsic Fermi level as a function of the donor concentration. We may note that, if the net effective donor concentration is zero, that is, \( N_D - N_A = 0 \), then \( n_0 = n_i \) and \( E_F = E_i \). A completely compensated semiconductor has the characteristics of an intrinsic material in terms of carrier concentration and Fermi-level position.

We can derive the same types of equations for a p-type semiconductor. From Equation (4.19), we have \( p_0 = N_A \exp \left[ \frac{-(E_F - E_v)}{kT} \right] \), so that

\[
E_F - E_v = kT \ln \left( \frac{N_v}{p_0} \right)
\]

(4.66)

If we assume that \( N_A \gg n_i \), then Equation (4.66) can be written as

\[
E_F - E_v = kT \ln \left( \frac{N_v}{N_A} \right)
\]

(4.67)

The distance between the Fermi level and the top of the valence-band energy for a p-type semiconductor is a logarithmic function of the acceptor concentration: as the acceptor concentration increases, the Fermi level moves closer to the valence band. Equation (4.67) still assumes that the Boltzmann approximation is valid. Again, if we have a compensated p-type semiconductor, then the \( N_A \) term in Equation (4.67) is replaced by \( N_A - N_D \) or the net effective acceptor concentration.

We can also derive an expression for the relationship between the Fermi level and the intrinsic Fermi level in terms of the hole concentration. We have from Equation (4.40) that \( p_0 = n_i \exp \left[ \frac{-(E_F - E_i)}{kT} \right] \), which yields

\[
E_F - E_i = kT \ln \left( \frac{p_0}{n_i} \right)
\]

(4.68)

Equation (4.68) can be used to find the difference between the intrinsic Fermi level and the Fermi energy in terms of the acceptor concentration. The hole concentration \( p_0 \) in Equation (4.68) is given by Equation (4.62).

**Figure 4.17** | Position of Fermi level for an (a) n-type \((N_D > N_A)\) and (b) p-type \((N_A > N_D)\) semiconductor.

We may again note from Equation (4.65) that, for an n-type semiconductor, \(n_0 > p_0\) and \(E_F > E_Fi\). The Fermi level for an n-type semiconductor is above \(E_Fi\). For a p-type semiconductor, \(p_0 > n_0\), and from Equation (4.68) we see that \(E_Fi > E_F\). The Fermi level for a p-type semiconductor is below \(E_Fi\). These results are shown in Figure 4.17.

### 4.6.2 Variation of \(E_F\) with Doping Concentration and Temperature

We may plot the position of the Fermi energy level as a function of the doping concentration. **Figure 4.18** shows the Fermi energy level as a function of donor concentration (n-type) and as a function of acceptor concentration (p-type) for silicon at \(T = 300 \, K\). As the doping levels increase, the Fermi energy level moves closer to the conduction band for the n-type material and closer to the valence band for the p-type material. Keep in mind that the equations for the Fermi energy level that we have derived assume that the Boltzmann approximation is valid.

**Figure 4.18** | Position of Fermi level as a function of donor concentration (n-type) and acceptor concentration (p-type).

| \(N_D \, (\text{cm}^{-3})\) | \(N_A \, (\text{cm}^{-3})\) |
|-----------------------------|-----------------------------|
| \(10^{12}\)                 | \(10^{12}\)                 |
| \(10^{13}\)                 | \(10^{13}\)                 |
| \(10^{14}\)                 | \(10^{14}\)                 |
| \(10^{15}\)                 | \(10^{15}\)                 |
| \(10^{16}\)                 | \(10^{16}\)                 |
| \(10^{17}\)                 | \(10^{17}\)                 |
| \(10^{18}\)                 | \(10^{18}\)                 |

- **n-type**: Fermi level moves closer to \(E_c\).
- **p-type**: Fermi level moves closer to \(E_v\).


## EXAMPLE 4.13
**Objective**: Determine the Fermi energy level and the maximum doping concentration at which the Boltzmann approximation is still valid.

Consider p-type silicon, at \( T = 300 \, \text{K} \), doped with boron. We may assume that the limit of the Boltzmann approximation occurs when \( E_F - E_i = 3kT \). (See Section 4.1.2.)

**Solution**

From Table 4.3, we find the ionization energy is \( E_c - E_i = 0.045 \, \text{eV} \) for boron in silicon. If we assume that \( E_F \approx E_{\text{acceptor}} \), then from Equation (4.68), the position of the Fermi level at the maximum doping is given by

\[
E_F - E_i = \frac{E_c - E_i}{2} - (E_c - E_i) = kT \ln \left( \frac{N_a}{n_i} \right)
\]

or

\[
0.56 - 0.045 - 3(0.0259) = 0.437 = (0.0259) \ln \left( \frac{N_a}{n_i} \right)
\]

We can then solve for the doping as

\[
N_a = n_i \exp \left( \frac{0.437}{0.0259} \right) = 3.2 \times 10^{17} \, \text{cm}^{-3}
\]

**Comment**

If the acceptor (or donor) concentration in silicon is greater than approximately \( 3 \times 10^{17} \, \text{cm}^{-3} \), then the Boltzmann approximation of the distribution function becomes less valid and the equations for the Fermi-level position are no longer quite as accurate.

**Exercise Problem**

**Ex 4.13** Consider n-type silicon at \( T = 300 \, \text{K} \) doped with arsenic. Determine the maximum doping at which the Boltzmann approximation is still valid. Assume the limit is such that \( E_F - E_i = 3kT. \left( c_{ud} = 0.01 \times 0.70 \, \text{e} = 0.1 \, \text{suV} \right) \)

The intrinsic carrier concentration, \( n_i \), in Equations (4.65) and (4.68), is a strong function of temperature, so that \( E_F \) is a function of temperature also. Figure 4.19 shows the variation of the Fermi energy level in silicon with temperature for several donor and acceptor concentrations. As the temperature increases, \( n_i \) increases, and \( E_F \) moves closer to the intrinsic Fermi level. At high temperature, the semiconductor material begins to lose its extrinsic characteristics and begins to behave more like an intrinsic semiconductor. At the very low temperature, freeze-out occurs; the Boltzmann approximation is no longer valid and the equations we derived for the Fermi-level position no longer apply. At the low temperature where freeze-out occurs, the Fermi level goes above \( E_F \) for the n-type material and below \( E_F \) for the p-type material. At absolute zero degrees, all energy states below \( E_F \) are full and all energy states above \( E_F \) are empty.

### 4.6.3 Relevance of the Fermi Energy

We have been calculating the position of the Fermi energy level as a function of doping concentrations and temperature. This analysis may seem somewhat arbitrary and fictitious. However, these relations do become significant later in our discussion of pn junctions and the other semiconductor devices we consider. An important...

**Figure 4.19** Position of Fermi level as a function of temperature for various doping concentrations.  
*(From Sze [14].)*

The point is that, in thermal equilibrium, the Fermi energy level is a constant throughout a system. We will not prove this statement, but we can intuitively see its validity by considering the following example.

Suppose we have a particular material, A, whose electrons are distributed in the energy states of an allowed band as shown in Figure 4.20a. Most of the energy states below \( E_F \) contain electrons and most of the energy states above \( E_F \) are empty of electrons. Consider another material, B, whose electrons are distributed in the energy states of an allowed band as shown in Figure 4.20b. The energy states below \( E_F \) are mostly full and the energy states above \( E_F \) are mostly empty. If these two materials are brought into intimate contact, the electrons in the entire system will tend to seek the lowest possible energy. Electrons from material A will flow into the lower energy states of material B, as indicated in Figure 4.20c, until thermal equilibrium is reached. Thermal equilibrium occurs when the distribution of electrons, as a function of energy, is the same in the two materials. This equilibrium state occurs when the Fermi energy is the same in the two materials as shown in Figure 4.20d. The Fermi energy, important in the physics of the semiconductor, also provides a good pictorial representation of the characteristics of the semiconductor materials and devices.

## TEST YOUR UNDERSTANDING

**TYU 4.14** Determine the position of the Fermi level with respect to the valence-band energy in p-type GaAs at \( T = 300 \, K \). The doping concentrations are \( N_d = 5 \times 10^{16} \, \text{cm}^{-3} \) and \( N_a = 4 \times 10^{15} \, \text{cm}^{-3} \). (\( \Lambda P \langle 0 \rangle = -q \langle 3q \rangle \))

**TYU 4.15** Calculate the position of the Fermi energy level in n-type silicon at \( T = 300 \, K \) with respect to the intrinsic Fermi energy level. The doping concentrations are \( N_d = 2 \times 10^{17} \, \text{cm}^{-3} \) and \( N_a = 3 \times 10^{16} \, \text{cm}^{-3} \). (\( \Lambda P \langle 0 \rangle = -q \langle 3q \rangle \))


**Figure 4.20** The Fermi energy of (a) material A in thermal equilibrium, (b) material B in thermal equilibrium, (c) materials A and B at the instant they are placed in contact, and (d) materials A and B in contact at thermal equilibrium.

## 4.7 | SUMMARY

- The concentration of electrons in the conduction band is the integral over the conduction-band energy of the product of the density of states function in the conduction band and the Fermi–Dirac probability function.

- The concentration of holes in the valence band is the integral over the valence-band energy of the product of the density of states function in the valence band and the probability of a state being empty, which is \([1 - f_F(E)]\).

- Using the Maxwell–Boltzmann approximation, the thermal-equilibrium concentration of electrons in the conduction band is given by

  \[
  n_0 = N_c \exp\left[-\frac{(E_c - E_F)}{kT}\right]
  \]

  where \(N_c\) is the effective density of states in the conduction band.

- Using the Maxwell–Boltzmann approximation, the thermal-equilibrium concentration of holes in the valence band is given by

  \[
  p_0 = N_v \exp\left[-\frac{(E_F - E_v)}{kT}\right]
  \]

  where \(N_v\) is the effective density of states in the valence band.

- The intrinsic carrier concentration is found from

  \[
  n_i^2 = N_c N_v \exp\left[-\frac{E_g}{kT}\right]
  \]
- The concept of doping the semiconductor with donor (group V elements) impurities and acceptor (group III elements) impurities to form n-type and p-type extrinsic semiconductors was discussed.
- The fundamental relationship of \( n_0 p_0 = n_i^2 \) was derived.
- Using the concepts of complete ionization and charge neutrality, equations for the electron and hole concentrations as a function of impurity doping concentrations were derived.
- The position of the Fermi energy level as a function of impurity doping concentrations was derived.
- The relevance of the Fermi energy was discussed. The Fermi energy is a constant throughout a semiconductor that is in thermal equilibrium.

## Glossary of Important Terms

**acceptor atoms**  
Impurity atoms added to a semiconductor to create a p-type material.

**charge carrier**  
The electron and/or hole that moves inside the semiconductor and gives rise to electrical currents.

**compensated semiconductor**  
A semiconductor that contains both donors and acceptors in the same semiconductor region.

**complete ionization**  
The condition when all donor atoms are positively charged by giving up their donor electrons and all acceptor atoms are negatively charged by accepting electrons.

**degenerate semiconductor**  
A semiconductor whose electron concentration or hole concentration is greater than the effective density of states, so that the Fermi level is in the conduction band (n type) or in the valence band (p type).

**donor atoms**  
Impurity atoms added to a semiconductor to create an n-type material.

**effective density of states**  
The parameter \( N_c \), which results from integrating the density of quantum states \( g_c(E) \) times the Fermi function \( f(E) \) over the conduction-band energy, and the parameter \( N_v \), which results from integrating the density of quantum states \( g_v(E) \) times \([1 - f_v(E)]\) over the valence-band energy.

**extrinsic semiconductor**  
A semiconductor in which controlled amounts of donors and/or acceptors have been added so that the electron and hole concentrations change from the intrinsic carrier concentration and a preponderance of either electrons (n type) or holes (p type) is created.

**freeze-out**  
The condition that occurs in a semiconductor when the temperature is lowered and the donors and acceptors become neutrally charged. The electron and hole concentrations become very small.

## CHECKPOINT

After studying this chapter, the reader should have the ability to:

- Derive the equations for the thermal equilibrium concentrations of electrons and holes in terms of the Fermi energy.
- Derive the equation for the intrinsic carrier concentration.
- Discuss what is meant by the effective density of states for electrons and holes.
- Describe the effect of adding donor and acceptor impurity atoms to a semiconductor.

## REVIEW QUESTIONS

1. How does the electron concentration in the conduction band change with energy \( E \) for \( E > E_c \)?

2. In deriving the equation for \( n_0 \) in terms of the Fermi function, the upper limit of the integral should be the energy at the top of the conduction band. Justify using infinity instead.

3. Assuming the Boltzmann approximation applies, write the equations for \( n_0 \) and \( p_0 \) in terms of the Fermi energy.

4. What is the source of electrons and holes in an intrinsic semiconductor?

5. Under what condition would the intrinsic Fermi level be at the midgap energy?

6. What is a donor impurity? What is an acceptor impurity?

7. What is meant by complete ionization? What is meant by freeze-out?

8. What is the product of \( n_0 \) and \( p_0 \) equal to?

9. Write the equation for charge neutrality for the condition of complete ionization.

10. Sketch a graph of \( n_0 \) versus temperature for an n-type material.

11. Sketch graphs of the Fermi energy versus donor impurity concentration and versus temperature.

12. What is the relevance of the Fermi energy?