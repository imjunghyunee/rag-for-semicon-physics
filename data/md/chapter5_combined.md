# Chapter 5: Carrier Transport Phenomena

In the previous chapter, we considered the semiconductor in equilibrium and determined electron and hole concentrations in the conduction and valence bands, respectively. A knowledge of the densities of these charged particles is important toward an understanding of the electrical properties of a semiconductor material. The net flow of the electrons and holes in a semiconductor will generate currents. The process by which these charged particles move is called **transport**. In this chapter we consider two basic transport mechanisms in a semiconductor crystal: **drift**—the movement of charge due to electric fields, and **diffusion**—the flow of charge due to density gradients.

The carrier transport phenomena are the foundation for finally determining the current–voltage characteristics of semiconductor devices. We will implicitly assume in this chapter that, although there will be a net flow of electrons and holes due to the transport processes, thermal equilibrium will not be substantially disturbed. Nonequilibrium processes are considered in the next chapter.

## 5.0 | PREVIEW

In this chapter, we will:

- Describe the mechanism of carrier drift and induced drift current due to an applied electric field.
- Define and describe the characteristics of carrier mobility.
- Describe the mechanism of carrier diffusion and induced diffusion current due to a gradient in the carrier concentration.
- Define the carrier diffusion coefficient.
- Describe the effects of a nonuniform impurity doping concentration in a semiconductor material.
- Discuss and analyze the Hall effect in a semiconductor material.

# 5.1 CARRIER DRIFT

An electric field applied to a semiconductor will produce a force on electrons and holes so that they will experience a net acceleration and net movement, provided there are available energy states in the conduction and valence bands. This net movement of charge due to an electric field is called **drift**. The net drift of charge gives rise to a **drift current**.

## 5.1.1 Drift Current Density

If we have a **positive volume charge density** \( \rho \) moving at an **average drift velocity** \( v_d \), the **drift current density** is given by

\[
J_{dr} = \rho v_d
\]

(5.1a)

In terms of units, we have

\[
J_{dr} = \left( \frac{\text{Coul}}{\text{cm}^3} \right) \left( \frac{\text{cm}}{\text{s}} \right) = \frac{\text{Coul}}{\text{cm}^2 \cdot \text{s}} = \frac{\text{A}}{\text{cm}^2}
\]

(5.1b)

If the **volume charge density is due to positively charged holes**, then

\[
J_{pdr} = (eP)v_{dp}
\]

(5.2)

where \( J_{pdr} \) is the drift current density due to holes and \( v_{dp} \) is the average drift velocity of the holes.

The equation of motion of a positively charged hole in the presence of an electric field is

\[
F = m_p^* a = eE
\]

(5.3)

where \( e \) is the magnitude of the electronic charge, \( a \) is the acceleration, \( E \) is the electric field, and \( m_p^* \) is the conductivity effective mass of the hole. If the electric field is constant, then we expect the velocity to increase linearly with time. However, charged particles in a semiconductor are involved in collisions with ionized impurity atoms and with thermally vibrating lattice atoms. These collisions, or scattering events, alter the velocity characteristics of the particle.

As the hole accelerates in a crystal due to the electric field, the velocity increases. When the charged particle collides with an atom in the crystal, for example, the particle loses most, or all, of its energy. The particle will again begin to accelerate and gain energy until it is again involved in a scattering process. This continues over and over again. Throughout this process, the particle will gain an average drift velocity which, for low electric fields, is directly proportional to the electric field. We may then write

\[
v_{dp} = \mu_p E
\]

(5.4)

where \( \mu_p \) is the proportionality factor and is called the **hole mobility**. The mobility is an important parameter of the semiconductor since it describes how well a particle can move through the material.

*The conductivity effective mass is used when carriers are in motion. See Appendix F for further discussion of effective mass concepts.*

# Chapter 5: Carrier Transport Phenomena

## Table 5.1: Typical mobility values at \( T = 300 \, K \) and low doping concentrations

| Material          | \(\mu_n\) (cm\(^2\)/V·s) | \(\mu_p\) (cm\(^2\)/V·s) |
|-------------------|--------------------------|--------------------------|
| Silicon           | 1350                     | 480                      |
| Gallium arsenide  | 8500                     | 400                      |
| Germanium         | 3900                     | 1900                     |

The unit of mobility is usually expressed in terms of cm\(^2\)/V·s.

By combining Equations (5.2) and (5.4), we may write the drift current density due to holes as

\[
J_{\text{drift}} = (ep)v_{\text{dp}} = e\mu_p pE
\]

(5.5)

The drift current due to holes is in the same direction as the applied electric field. The same discussion of drift applies to electrons. We may write

\[
J_{\text{drift}} = pv_{\text{dn}} = (-en)v_{\text{dn}}
\]

(5.6)

where \( J_{\text{drift}} \) is the drift current density due to electrons and \( v_{\text{dn}} \) is the average drift velocity of electrons. The net charge density of electrons is negative.

The average drift velocity of an electron is also proportional to the electric field for small fields. However, since the electron is negatively charged, the net motion of the electron is opposite to the electric field direction. We can then write

\[
v_{\text{dn}} = -\mu_n E
\]

(5.7)

where \(\mu_n\) is the electron mobility and is a positive quantity. Equation (5.6) may now be written as

\[
J_{\text{drift}} = (-en)(-\mu_n E) = e\mu_n nE
\]

(5.8)

The conventional drift current due to electrons is also in the same direction as the applied electric field even though the electron movement is in the opposite direction.

Electron and hole mobilities are functions of temperature and doping concentrations, as we will see in the next section. Table 5.1 shows some typical mobility values at \( T = 300 \, K \) for low doping concentrations.

Since both electrons and holes contribute to the drift current, the total drift current density is the sum of the individual electron and hole drift current densities, so we may write

\[
J_{\text{drift}} = e(\mu_n n + \mu_p p)E
\]

(5.9)

----

### Example 5.1

**Objective:** Calculate the drift current density in a semiconductor for a given electric field.

Consider a gallium arsenide sample at \( T = 300 \, K \) with doping concentrations of \( N_d = 0 \) and \( N_a = 10^{16} \, \text{cm}^{-3} \). Assume complete ionization and assume electron and hole mobilities given in Table 5.1. Calculate the drift current density if the applied electric field is \( E = 10 \, \text{V/cm} \).

# Solution

Since \( N_D > N_A \), the semiconductor is n-type and the majority carrier electron concentration, from Chapter 4 is given by

\[
n = \frac{N_D - N_A}{2} + \sqrt{\left(\frac{N_D - N_A}{2}\right)^2 + n_i^2} \approx 10^{16} \, \text{cm}^{-3}
\]

The minority carrier hole concentration is

\[
p = \frac{n_i^2}{n} = \frac{(1.8 \times 10^{20})}{10^{16}} = 3.24 \times 10^{-4} \, \text{cm}^{-3}
\]

For this extrinsic n-type semiconductor, the drift current density is

\[
J_{\text{drift}} = e(\mu_n n + \mu_p p)E \approx e\mu_n N_D E
\]

Then

\[
J_{\text{drift}} = (1.6 \times 10^{-19})(8500)(10^{16})(10) = 136 \, \text{A/cm}^2
\]

# Comment

Significant drift current densities can be obtained in a semiconductor applying relatively small electric fields. We may note from this example that the drift current will usually be due primarily to the majority carrier in an extrinsic semiconductor.

# EXERCISE PROBLEM

**Ex 5.1**  
A drift current density of \( J_{\text{drift}} = 75 \, \text{A/cm}^2 \) is required in a device using p-type silicon when an electric field of \( E = 120 \, \text{V/cm} \) is applied. Determine the required impurity doping concentration to achieve this specification. Assume that electron and hole mobilities given in Table 5.1 apply. (\( \mu_n = 0.1 \times 10^3 = \text{N s/V} \))

## 5.1.2 Mobility Effects

In the previous section, we defined mobility, which relates the average drift velocity of a carrier to the electric field. Electron and hole mobilities are important semiconductor parameters in the characterization of carrier drift, as seen in Equation (5.9).

Equation (5.3) related the acceleration of a hole to a force such as an electric field. We may write this equation as

\[
F = m_n \frac{dv}{dt} = eE
\]

(5.10)

where \( v \) is the velocity of the particle due to the electric field and does not include the random thermal velocity. If we assume that the conductivity effective mass and electric field are constants, then we may integrate Equation (5.10) and obtain

\[
v = \frac{eEt}{m_n^*}
\]

(5.11)

where we have assumed the initial drift velocity to be zero.

Figure 5.1a shows a schematic model of the random thermal velocity and motion of a hole in a semiconductor with zero electric field. There is a mean time

# Chapter 5: Carrier Transport Phenomena

!Figure 5.1

**Figure 5.1** Typical random behavior of a hole in a semiconductor (a) without an electric field and (b) with an electric field.

Between collisions which may be denoted by \( \tau_{mp} \). If a small electric field (E-field) is applied as indicated in Figure 5.1b, there will be a net drift of the hole in the direction of the E-field, and the net drift velocity will be a small perturbation on the random thermal velocity, so the time between collisions will not be altered appreciably. **If we use the mean time between collisions \( \tau_{mp} \) in place of the time \( t \) in Equation (5.11), then the mean peak velocity just prior to a collision or scattering event is**

\[
v_{\text{peak}} = \left( \frac{e \tau_{mp}}{m_{hp}} \right) E
\]

(5.12a)

The average drift velocity is one half the peak value so that we can write

\[
\langle v_d \rangle = \frac{1}{2} \left( \frac{e \tau_{mp}}{m_{hp}} \right) E
\]

(5.12b)

However, the collision process is not as simple as this model, but is statistical in nature. In a more accurate model including the effect of a statistical distribution, the factor \( \frac{1}{2} \) in Equation (5.12b) does not appear. The hole mobility is then given by

\[
\mu_p = \frac{v_d}{E} = \frac{e \tau_{mp}}{m_{hp}}
\]

(5.13)

The same analysis applies to electrons; thus, we can write the electron mobility as

\[
\mu_n = \frac{e \tau_{mn}}{m_{hn}}
\]

(5.14)

where \( \tau_{mn} \) is the mean time between collisions for an electron.

There are two collision or scattering mechanisms that dominate in a semiconductor and affect the carrier mobility: phonon or lattice scattering, and ionized impurity scattering.

The atoms in a semiconductor crystal have a certain amount of thermal energy at temperatures above absolute zero that causes the atoms to randomly vibrate about their lattice position within the crystal. The lattice vibrations cause a disruption in the perfect periodic potential function. A perfect periodic potential in a solid allows...

# 5.1 Carrier Drift

electrons to move unimpeded, or with no scattering, through the crystal. But the thermal vibrations cause a disruption of the potential function, resulting in an interaction between the electrons or holes and the vibrating lattice atoms. This lattice scattering is also referred to as **phonon scattering**.

Since lattice scattering is related to the thermal motion of atoms, the rate at which the scattering occurs is a function of temperature. If we denote \( \mu_L \) as the mobility that would be observed if only lattice scattering existed, then the scattering theory states that to first order

\[
\mu_L \propto T^{-3/2}
\]

(5.15)

Mobility that is due to lattice scattering increases as the temperature decreases. Intuitively, we expect the lattice vibrations to decrease as the temperature decreases, which implies that the probability of a scattering event also decreases, thus increasing mobility.

Figure 5.2 shows the temperature dependence of electron and hole mobilities in silicon. In lightly doped semiconductors, lattice scattering dominates and the carrier mobility decreases with temperature as we have discussed. The temperature dependence of mobility is proportional to \( T^{-n} \). The inserts in the figure show that the parameter \( n \) is not equal to \( \frac{3}{2} \) as the first-order scattering theory predicted. However, mobility does increase as the temperature decreases.

The second interaction mechanism affecting carrier mobility is called **ionized impurity scattering**. We have seen that impurity atoms are added to the semiconductor to control or alter its characteristics. These impurities are ionized at room temperature so that a coulomb interaction exists between the electrons or holes and the ionized impurities. This coulomb interaction produces scattering or collisions and also alters the velocity characteristics of the charge carrier. If we denote \( \mu_i \) as the mobility that would be observed if only ionized impurity scattering existed, then to first order we have

\[
\mu_i \propto \frac{T^{3/2}}{N_i}
\]

(5.16)

where \( N_i = N_i^+ + N_i^- \) is the total ionized impurity concentration in the semiconductor. If temperature increases, the random thermal velocity of a carrier increases, reducing the time the carrier spends in the vicinity of the ionized impurity center. The less time spent in the vicinity of a coulomb force, the smaller the scattering effect and the larger the expected value of \( \mu_i \). If the number of ionized impurity centers increases, then the probability of a carrier encountering an ionized impurity center increases, implying a smaller value of \( \mu_i \).

Figure 5.3 is a plot of electron and hole mobilities in germanium, silicon, and gallium arsenide at \( T = 300 \, \text{K} \) as a function of impurity concentration. More accurately, these curves are of mobility versus ionized impurity concentration \( N_i \). As the impurity concentration increases, the number of impurity scattering centers increases, thus reducing mobility.

If \( \tau_L \) is the mean time between collisions due to lattice scattering, then \( dt/\tau_L \) is the probability of a lattice scattering event occurring in a differential time \( dt \). Likewise, if \( \tau_i \) is the mean time between collisions due to ionized impurity scattering, then \( dt/\tau_i \) is the probability of an ionized impurity scattering event occurring in the differential time \( dt \).

```markdown
!Figure 5-12

**Figure 5-12** (a) Electron and (b) hole mobilities in silicon versus temperature for various doping concentrations. Insets show temperature dependence of almost intrinsic silicon. (From Pierret [8].)

### (a) Electron Mobility

| Temperature (°C) | N = 10^13 | N = 10^15 | N = 10^17 | N = 10^19 |
|------------------|-----------|-----------|-----------|-----------|
| -50              |           |           |           |           |
| 0                |           |           |           |           |
| 50               |           |           |           |           |
| 100              |           |           |           |           |
| 150              |           |           |           |           |
| 200              |           |           |           |           |

**Inset:**

| Temperature (°C) | N = 10^13 |
|------------------|-----------|
| 100              |           |
| 150              |           |
| 200              |           |
| 250              |           |

### (b) Hole Mobility

| Temperature (°C) | N = 10^13 | N = 10^15 | N = 10^17 | N = 10^19 |
|------------------|-----------|-----------|-----------|-----------|
| -50              |           |           |           |           |
| 0                |           |           |           |           |
| 50               |           |           |           |           |
| 100              |           |           |           |           |
| 150              |           |           |           |           |
| 200              |           |           |           |           |

**Inset:**

| Temperature (°C) | N = 10^13 |
|------------------|-----------|
| 100              |           |
| 150              |           |
| 200              |           |
| 250              |           |

**Note:** The values for electron and hole mobilities are represented in \([ \text{cm}^2/\text{V}\cdot\text{s} ]\).
```

# 5.1 Carrier Drift

!Graphs of Electron and Hole Mobilities

**Figure 5.3** Electron and hole mobilities versus impurity concentrations for germanium, silicon, and gallium arsenide at \( T = 300 \, \text{K} \).  
*(From Sze [14].)*

If these two scattering processes are independent, then the total probability of a scattering event occurring in the differential time \( dt \) is the sum of the individual events, or

\[
\frac{dt}{\tau} = \frac{dt}{\tau_i} + \frac{dt}{\tau_L}
\]

(5.17)

where \( \tau \) is the mean time between any scattering event.

Comparing Equation (5.17) with the definitions of mobility given by Equation (5.13) or (5.14), we can write

\[
\frac{1}{\mu} = \frac{1}{\mu_i} + \frac{1}{\mu_L}
\]

(5.18)

where \( \mu_i \) is the mobility due to the ionized impurity scattering process and \( \mu_L \) is the mobility due to the lattice scattering process. The parameter \( \mu \) is the net mobility. With two or more independent scattering mechanisms, the inverse mobilities add, which means that the net mobility decreases.

## Example 5.2

**Objective:** Determine the electron mobility in silicon at various doping concentrations and various temperatures.

Using Figure 5.2, find the electron mobility in silicon for:

(a) \( T = 25^\circ C \) for (i) \( N_d = 10^{16} \, \text{cm}^{-3} \) and (ii) \( N_d = 10^{17} \, \text{cm}^{-3} \).

(b) \( N_d = 10^{16} \, \text{cm}^{-3} \) for (i) \( T = 0^\circ C \) and (ii) \( T = 100^\circ C \).

### Solution:

From Figure 5.2, we find the following:

(a) \( T = 25^\circ C \):

- (i) \( N_d = 10^{16} \, \text{cm}^{-3} \) \(\Rightarrow \mu_n = 1200 \, \text{cm}^2/\text{V-s}\).
- (ii) \( N_d = 10^{17} \, \text{cm}^{-3} \) \(\Rightarrow \mu_n = 800 \, \text{cm}^2/\text{V-s}\).

(b) \( N_d = 10^{16} \, \text{cm}^{-3} \):

- (i) \( T = 0^\circ C \) \(\Rightarrow \mu_n = 1400 \, \text{cm}^2/\text{V-s}\).
- (ii) \( T = 100^\circ C \) \(\Rightarrow \mu_n = 780 \, \text{cm}^2/\text{V-s}\).

### Comment

The results of this example show that the mobility values are strong functions of the doping concentration and temperature. These variations must be taken into account in the design of semiconductor devices.

### Exercise Problem

Ex 5.2 Using Figure 5.2, find the hole mobility in silicon for:

(a) \( T = 25^\circ C \) for (i) \( N_d = 10^{16} \, \text{cm}^{-3} \) and (ii) \( N_d = 10^{18} \, \text{cm}^{-3} \), and

(b) \( N_d = 10^{16} \, \text{cm}^{-3} \) for (i) \( T = 0^\circ C \) and (ii) \( T = 100^\circ C \).

## 5.1.3 Conductivity

The drift current density, given by Equation (5.9), may be written as

\[
J_{\text{dr}} = e(\mu_n n + \mu_p p)E = \sigma E
\]

(5.19)

where \(\sigma\) is the conductivity of the semiconductor material. The conductivity is given in units of \((\Omega\cdot\text{cm})^{-1}\) and is a function of the electron and hole concentrations and mobilities. We have just seen that the mobilities are functions of impurity concentrations; conductivity, then is a somewhat complicated function of impurity concentration.

The reciprocal of conductivity is **resistivity**, which is denoted by \(\rho\) and is given in units of ohm-cm. We can write the formula for resistivity as

\[
\rho = \frac{1}{\sigma} = \frac{1}{e(\mu_n n + \mu_p p)}
\]

(5.20)

Figure 5.4 is a plot of resistivity as a function of impurity concentration in silicon, germanium, gallium arsenide, and gallium phosphide at \( T = 300 \, K \). Obviously, the curves are not linear functions of \( N_d \) or \( N_a \) because of mobility effects.

*The symbol \(\rho\) is also used for volume charge density. The context in which \(\rho\) is used should make it clear whether it stands for charge density or resistivity.*

```markdown
!Resistivity vs Impurity Concentration Graphs

**Figure 5.4** Resistivity versus impurity concentration at \( T = 300 \, \text{K} \) in (a) silicon and (b) germanium, gallium arsenide, and gallium phosphide.  
(From Sze [14].)

### Graph (a): Silicon at 300 K

- **Y-axis:** Resistivity \((\Omega \cdot \text{cm})\)
- **X-axis:** Impurity concentration \((\text{cm}^{-3})\)
- **Curves:**
  - n-type (phosphorus)
  - p-type (boron)

### Graph (b): Germanium, Gallium Arsenide, and Gallium Phosphide at 300 K

- **Y-axis:** Resistivity \((\Omega \cdot \text{cm})\)
- **X-axis:** Impurity concentration \((\text{cm}^{-3})\)
- **Curves:**
  - n-type and p-type for:
    - p-Ge, n-Ge
    - p-GaP, n-GaP
    - n-GaAs, p-GaAs

- **Legend:**
  - Solid line: n-type
  - Dashed line: p-type
```

# Carrier Transport Phenomena

!Figure 5.5  
**Figure 5.5** | Bar of semiconductor material as a resistor.

If we have a bar of semiconductor material as shown in Figure 5.5 with a voltage applied that produces a current \( I \), then we can write

\[
J = \frac{I}{A}
\]
(5.21a)

and

\[
E = \frac{V}{L}
\]
(5.21b)

We can now rewrite Equation (5.19) as

\[
\frac{I}{A} = \sigma \left( \frac{V}{L} \right)
\]
(5.22a)

or

\[
V = \left( \frac{L}{\sigma A} \right) I = \left( \frac{\rho L}{A} \right) I = IR
\]
(5.22b)

Equation (5.22b) is Ohm’s law for a semiconductor. The resistance is a function of resistivity, or conductivity, as well as the geometry of the semiconductor.

If we consider, for example, a p-type semiconductor with an acceptor doping \( N_a(N_d = 0) \) in which \( N_a \gg n_i \) and if we assume that the electron and hole mobilities are of the same order of magnitude, then the conductivity becomes

\[
\sigma = e(\mu_n n + \mu_p p) \approx e \mu_p p
\]
(5.23)

If we also assume complete ionization, then Equation (5.23) becomes

\[
\sigma \approx e \mu_p N_a \approx \frac{1}{\rho}
\]
(5.24)

The conductivity and resistivity of an extrinsic semiconductor are a function primarily of the majority carrier parameters.

# 5.1 Carrier Drift

!Graph

**Figure 5.6** | Electron concentration and conductivity versus inverse temperature for silicon.  
(After Sze [14].)

We may plot the carrier concentration and conductivity of a semiconductor as a function of temperature for a particular doping concentration. Figure 5.6 shows the electron concentration and conductivity of silicon as a function of inverse temperature for the case when \( N_d = 10^{15} \, \text{cm}^{-3} \). In the midtemperature range, or extrinsic range, as shown, we have complete ionization—the electron concentration remains essentially constant. However, the mobility is a function of temperature so the conductivity varies with temperature in this range. At higher temperatures, the intrinsic carrier concentration increases and begins to dominate the electron concentration as well as the conductivity. In the lower temperature range, freeze-out begins to occur; the electron concentration and conductivity decrease with decreasing temperature.

----

**Objective:** Determine the doping concentration and majority carrier mobility given the type and conductivity of a compensated semiconductor

**Example 5.3**

Consider compensated n-type silicon at \( T = 300 \, \text{K} \), with a conductivity of \( \sigma = 16 \, (\Omega \, \text{cm})^{-1} \) and an acceptor doping concentration of \( 10^{17} \, \text{cm}^{-3} \). Determine the donor concentration and the electron mobility.

**Solution**

For n-type silicon at \( T = 300 \, \text{K} \), we can assume complete ionization; therefore the conductivity, assuming \( N_d - N_a \gg n_i \), is given by

\[
\sigma = e \mu_n n = e \mu_n (N_d - N_a)
\]

We have that

\[
16 = (1.6 \times 10^{-19}) \mu_n (N_d - 10^{17})
\]

# CHAPTER 5 Carrier Transport Phenomena

Since mobility is a function of the ionized impurity concentration, we can use Figure 5.3 along with trial and error to determine \( \mu_n \) and \( N_d \). For example, if we choose \( N_d = 2 \times 10^{17} \), then \( N_i = N_d + N_a = 3 \times 10^{17} \) so that \( \mu_n \approx 510 \, \text{cm}^2/\text{V}\cdot\text{s} \) which gives \( \sigma = 8.16 \, (\Omega\cdot\text{cm})^{-1} \). If we choose \( N_d = 5 \times 10^{17} \), then \( N_i = 6 \times 10^{17} \) so that \( \mu_n \approx 325 \, \text{cm}^2/\text{V}\cdot\text{s} \), which gives \( \sigma = 20.8 \, (\Omega\cdot\text{cm})^{-1} \). The doping is bounded between these two values. Further trial and error yields

\[ N_d \approx 3.5 \times 10^{17} \, \text{cm}^{-3} \quad \text{and} \quad \mu_n \approx 400 \, \text{cm}^2/\text{V}\cdot\text{s} \]

which gives

\[ \sigma \approx 16 \, (\Omega\cdot\text{cm})^{-1} \]

**Comment**

We can see from this example that, in a high-conductivity semiconductor material, mobility is a strong function of carrier concentration.

## EXERCISE PROBLEM

**Ex 5.3** A compensated p-type silicon material at \( T = 300 \, \text{K} \) has impurity doping concentrations of \( N_i = 2.8 \times 10^{17} \, \text{cm}^{-3} \) and \( N_a = 8 \times 10^{16} \, \text{cm}^{-3} \). Determine the (a) hole mobility, (b) conductivity, and (c) resistivity.

----

## DESIGN EXAMPLE 5.4

**Objective:** Design a semiconductor resistor with a specified resistance to handle a given current density.

A silicon semiconductor at \( T = 300 \, \text{K} \) is initially doped with donors at a concentration of \( N_d = 5 \times 10^{15} \, \text{cm}^{-3} \). Acceptors are to be added to form a compensated p-type material. The resistor is to have a resistance of 10 k\(\Omega\) and handle a current density of 50 A/cm\(^2\) when 5 V is applied.

### Solution

For 5 V applied to a 10-k\(\Omega\) resistor, the total current is

\[ I = \frac{V}{R} = \frac{5}{10} = 0.5 \, \text{mA} \]

If the current density is limited to 50 A/cm\(^2\), then the cross-sectional area is

\[ A = \frac{I}{J} = \frac{0.5 \times 10^{-3}}{50} = 10^{-5} \, \text{cm}^2 \]

If we, somewhat arbitrarily at this point, limit the electric field to \( E = 100 \, \text{V/cm} \), then the length of the resistor is

\[ L = \frac{V}{E} = \frac{5}{100} = 5 \times 10^{-2} \, \text{cm} \]

From Equation (5.22b), the conductivity of the semiconductor is

\[ \sigma = \frac{L}{RA} = \frac{5 \times 10^{-2}}{(10)(10^{-5})} = 0.50 \, (\Omega\cdot\text{cm})^{-1} \]

# 5.1 Carrier Drift

The conductivity of a compensated p-type semiconductor is

\[
\sigma \approx e\mu_p p = e\mu_p (N_a - N_d)
\]

where the mobility is a function of the total ionized impurity concentration \(N_a + N_d\).

Using trial and error, if \(N_a = 1.25 \times 10^{16} \, \text{cm}^{-3}\), then \(N_a + N_d = 1.75 \times 10^{16} \, \text{cm}^{-3}\), and the hole mobility, from Figure 5.3, is approximately \(\mu_p = 410 \, \text{cm}^2/\text{V}\cdot\text{s}\). The conductivity is then

\[
\sigma = e\mu_p(N_a - N_d) = (1.6 \times 10^{-19})(410)(1.25 \times 10^{16} - 5 \times 10^{15}) = 0.492
\]

which is very close to the value we need.

## Comment

Since the mobility is related to the total ionized impurity concentration, the determination of the impurity concentration to achieve a particular conductivity is not straightforward.

## EXERCISE PROBLEM

**Ex 5.4** A bar of p-type silicon, such as shown in Figure 5.5, has a cross-sectional area \(A = 10^{-6} \, \text{cm}^2\) and a length \(L = 1.2 \times 10^{-1} \, \text{cm}\). For an applied voltage of 5 V, a current of 2 mA is required. What is the required (a) resistance, (b) resistivity, and (c) impurity doping concentration? (d) What is the resulting hole mobility?

----

For an intrinsic material, the conductivity can be written as

\[
\sigma_i = e(\mu_n + \mu_p) n_i \tag{5.25}
\]

The concentrations of electrons and holes are equal in an intrinsic semiconductor, so the intrinsic conductivity includes both the electron and hole mobility. Since, in general, the electron and hole mobilities are not equal, the intrinsic conductivity is not the minimum value possible at a given temperature.

## 5.1.4 Velocity Saturation

So far in our discussion of drift velocity, we have assumed that mobility is not a function of electric field, meaning that the drift velocity will increase linearly with applied electric field. The total velocity of a particle is the sum of the random thermal velocity and drift velocity. At \(T = 300 \, \text{K}\), the average random thermal energy is given by

\[
\frac{1}{2} m_n v_t^2 = \frac{3}{2} kT = \frac{3}{2} (0.0259) = 0.03885 \, \text{eV} \tag{5.26}
\]

This energy translates into a mean thermal velocity of approximately \(10^7 \, \text{cm/s}\) for an electron in silicon. If we assume an electron mobility of \(\mu_n = 1350 \, \text{cm}^2/\text{V}\cdot\text{s}\) in low-doped silicon, a drift velocity of \(10^5 \, \text{cm/s}\), or 1 percent of the thermal velocity, is achieved if the applied electric field is approximately \(75 \, \text{V/cm}\). This applied electric field does not appreciably alter the energy of the electron.

Figure 5.7 is a plot of average drift velocity as a function of applied electric field for electrons and holes in silicon, gallium arsenide, and germanium. At low electric fields, where there is a linear variation of velocity with electric field, the slope of

# Carrier Transport Phenomena

!Carrier drift velocity versus electric field

**Figure 5.7** Carrier drift velocity versus electric field for high-purity silicon, germanium, and gallium arsenide. (From Sze [14].)

The **drift velocity versus electric field curve** is the mobility. The behavior of the drift velocity of carriers at high electric fields deviates substantially from the linear relationship observed at low fields. The drift velocity of electrons in silicon, for example, saturates at approximately \(10^7 \, \text{cm/s}\) at an electric field of approximately 30 kV/cm. If the drift velocity of a charge carrier saturates, then the drift current density also saturates and becomes independent of the applied electric field.

The experimental carrier drift velocity versus electric field in silicon can be approximated for electrons by [2]

\[
v_n = \frac{v_s}{1 + \left(\frac{E_{eo}}{E}\right)^{1/2}}
\]

(5.27a)

and for holes by

\[
v_p = \frac{v_s}{1 + \left(\frac{E_{op}}{E}\right)^{1/2}}
\]

(5.27b)

The variables are \(v_s = 10^7 \, \text{cm/s}\) at \(T = 300 \, \text{K}\), \(E_{eo} = 7 \times 10^3 \, \text{V/cm}\), and \(E_{op} = 2 \times 10^4 \, \text{V/cm}\).

We may note that for small electric fields, the drift velocities reduce to

\[
v_n = \left(\frac{E}{E_{eo}}\right) \cdot v_s
\]

(5.28a)

and

\[
v_p = \left(\frac{E}{E_{op}}\right) \cdot v_s
\]

(5.28b)

# 5.1 Carrier Drift

At low electric fields, the drift velocities are linear functions of the electric field as we have discussed. However, for large electric fields, the drift velocities approach the saturation value.

The drift velocity versus electric field characteristic of gallium arsenide is more complicated than for silicon or germanium. At low fields, the slope of the drift velocity versus E-field is constant and is the low-field electron mobility, which is approximately 8500 cm\(^2\)/V·s for gallium arsenide. The low-field electron mobility in gallium arsenide is much larger than in silicon. As the field increases, the electron drift velocity in gallium arsenide reaches a peak and then decreases. A differential mobility is the slope of the \(v_d\) versus E curve at a particular point on the curve and the negative slope of the drift velocity versus electric field represents a negative differential mobility. The negative differential mobility produces a negative differential resistance; this characteristic is used in the design of oscillators.

The negative differential mobility can be understood by considering the E versus k diagram for gallium arsenide, which is shown again in Figure 5.8. The density of states effective mass of the electron in the lower valley is \(m_n^* = 0.067 m_0\). The small effective mass leads to a large mobility. As the E-field increases, the energy of the electron increases and the electron can be scattered into the upper valley, where the density of states effective mass is 0.55\(m_0\). The larger effective mass in the upper valley yields a smaller mobility. This intervalley transfer mechanism results in a decreasing average drift velocity of electrons with electric field, or the negative differential mobility characteristic.

!Energy-band structure for gallium arsenide

**Figure 5.8** Energy-band structure for gallium arsenide showing the upper valley and lower valley in the conduction band.  
*(From Sze [15].)*

# TEST YOUR UNDERSTANDING

**TYU 5.1**  
Consider a sample of silicon at \( T = 300 \, \text{K} \) doped at an impurity concentration of \( N_d = 10^{15} \, \text{cm}^{-3} \) and \( N_a = 10^{14} \, \text{cm}^{-3} \). Assume electron and hole mobilities given in Table 5.1. Calculate the drift current density if the applied electric field is \( E = 35 \, \text{V/cm} \).

**TYU 5.2**  
Silicon at \( T = 300 \, \text{K} \) is doped with impurity concentrations of \( N_d = 5 \times 10^{16} \, \text{cm}^{-3} \) and \( N_a = 2 \times 10^{16} \, \text{cm}^{-3} \).  
(a) What are the electron and hole mobilities?  
(b) Determine the conductivity and resistivity of the material.

**TYU 5.3**  
For a particular silicon semiconductor device at \( T = 300 \, \text{K} \), the required material is n type with a resistivity of \( 0.10 \, \Omega \cdot \text{cm} \).  
(a) Determine the required impurity doping concentration and  
(b) the resulting electron mobility.

----

## 5.2 | CARRIER DIFFUSION

There is a second mechanism, in addition to drift, that can induce a current in a semiconductor. We may consider a classic physics example in which a container, as shown in Figure 5.9, is divided into two parts by a membrane. The left side contains gas molecules at a particular temperature and the right side is initially empty. The gas molecules are in continual random thermal motion so that, when the membrane is broken, the gas molecules flow into the right side of the container. **Diffusion is the process whereby particles flow from a region of high concentration toward a region of low concentration.** If the gas molecules were electrically charged, the net flow of charge would result in a diffusion current.

### 5.2.1 Diffusion Current Density

To begin to understand the diffusion process in a semiconductor, we will consider a simplified analysis. Assume that an electron concentration varies in one dimension as shown in Figure 5.10. The temperature is assumed to be uniform so that the average thermal velocity of electrons is independent of \( x \). To calculate the current, we will determine the net flow of electrons per unit time per unit area crossing the plane at \( x = 0 \). If the distance \( l \) shown in Figure 5.10 is less than the mean-free path of an electron, that is, the average

!Figure 5.9

**Figure 5.9** | Container divided by a membrane with gas molecules on one side.

```markdown
!Figure 5.10

**Figure 5.10** | Electron concentration versus distance.

If the distance an electron travels between collisions (\(l < v_n \tau_n\)), then on the average, electrons moving to the right at \(x = -l\) and electrons moving to the left at \(x = +l\) will cross the \(x = 0\) plane. One half of the electrons at \(x = -l\) will be traveling to the right at any instant of time and one half of the electrons at \(x = +l\) will be traveling to the left at any given time. The net rate of electron flow, \(F_n\), in the \(+x\) direction at \(x = 0\) is given by

\[
F_n = \frac{1}{2} n(-l)v_{th,n} - \frac{1}{2} n(+l)v_{th,n} = \frac{1}{2} v_{th}[n(-l) - n(+l)]
\]

(5.29)

If we expand the electron concentration in a Taylor series about \(x = 0\) keeping only the first two terms, then we can write Equation (5.29) as

\[
F_n = \frac{1}{2} v_{th} \left[ n(0) - \frac{dn}{dx} - n(0) + \frac{dn}{dx} \right]
\]

(5.30)

which becomes

\[
F_n = -v_{th} l \frac{dn}{dx}
\]

(5.31)

Each electron has a charge \((-e)\), so the current is

\[
J = -eF_n = e v_{th} l \frac{dn}{dx}
\]

(5.32)

The current described by Equation (5.32) is the electron diffusion current and is proportional to the spatial derivative, or density gradient, of the electron concentration.

The diffusion of electrons from a region of high concentration to a region of low concentration produces a flux of electrons flowing in the negative \(x\) direction for this example. Since electrons have a negative charge, the conventional current direction is in the positive \(x\) direction. Figure 5.11a shows these one-dimensional flux and
```

# Chapter 5: Carrier Transport Phenomena

!Figure 5.11

**Figure 5.11** (a) Diffusion of electrons due to a density gradient. (b) Diffusion of holes due to a density gradient.

We may write the electron diffusion current density for this one-dimensional case, in the form

\[
J_{\text{n-diff}} = eD_n \frac{dn}{dx}
\]

(5.33)

where \( D_n \) is called the **electron diffusion coefficient**, has units of cm\(^2\)/s, and is a positive quantity. If the electron density gradient becomes negative, the electron diffusion current density will be in the negative \( x \) direction.

Figure 5.11b shows an example of a hole concentration as a function of distance in a semiconductor. The diffusion of holes, from a region of high concentration to a region of low concentration, produces a flux of holes in the negative \( x \) direction. Since holes are positively charged particles, the conventional diffusion current density is also in the negative \( x \) direction. The hole diffusion current density is proportional to the hole density gradient and to the electronic charge, so we may write

\[
J_{\text{p-diff}} = -eD_p \frac{dp}{dx}
\]

(5.34)

for the one-dimensional case. The parameter \( D_p \) is called the **hole diffusion coefficient**, has units of cm\(^2\)/s, and is a positive quantity. If the hole density gradient becomes negative, the hole diffusion current density will be in the positive \( x \) direction.

## Objective: Calculate the diffusion current density given a density gradient.

Assume that, in an n-type gallium arsenide semiconductor at \( T = 300 \, \text{K} \), the electron concentration varies linearly from \( 1 \times 10^{18} \) to \( 7 \times 10^{17} \, \text{cm}^{-3} \) over a distance of 0.10 cm. Calculate the diffusion current density if the electron diffusion coefficient is \( D_n = 225 \, \text{cm}^2/\text{s} \).

### Solution

The diffusion current density is given by

\[
J_{\text{n,diff}} = eD_n \frac{dn}{dx} = eD_n \frac{\Delta n}{\Delta x}
\]

\[
= (1.6 \times 10^{-19})(225) \left( \frac{1 \times 10^{18} - 7 \times 10^{17}}{0.10} \right) = 108 \, \text{A/cm}^2
\]

### Comment

A significant diffusion current density can be generated in a semiconductor material with only a modest density gradient.

### EXERCISE PROBLEM

**Ex 5.5** The hole density in silicon is given by \( p(x) = 10^{16} e^{-x/L_p} \, (x \geq 0) \) where \( L_p = 2 \times 10^{-4} \, \text{cm} \). Assume the hole diffusion coefficient is \( D_p = 8 \, \text{cm}^2/\text{s} \). Determine the hole diffusion current density at (a) \( x = 0 \), (b) \( x = 2 \times 10^{-4} \, \text{cm} \), and (c) \( x = 10^{-3} \, \text{cm} \).

----

### 5.2.2 Total Current Density

We now have four possible independent current mechanisms in a semiconductor. These components are electron drift and diffusion currents and hole drift and diffusion currents. The **total current density** is the sum of these four components, or, for the one-dimensional case,

\[
J = en_n \mu_n E_x + ep_n \mu_p E_x + eD_n \frac{dn}{dx} - eD_p \frac{dp}{dx} \tag{5.35}
\]

This equation may be generalized to three dimensions as

\[
J = en_n \mu_n \mathbf{E} + ep_n \mu_p \mathbf{E} + eD_n \nabla n - eD_p \nabla p \tag{5.36}
\]

The electron mobility gives an indication of how well an electron moves in a semiconductor as a result of the force of an electric field. The electron diffusion coefficient gives an indication of how well an electron moves in a semiconductor as a result of a density gradient. The electron mobility and diffusion coefficient are not independent parameters. Similarly, the hole mobility and diffusion coefficient are not independent parameters. The relationship between mobility and the diffusion coefficient is developed in the next section.

The expression for the total current in a semiconductor contains four terms. Fortunately in most situations, we will only need to consider one term at any one time at a particular point in a semiconductor.

# CHAPTER 5 Carrier Transport Phenomena

## TEST YOUR UNDERSTANDING

**TYU 5.4** The electron concentration in silicon is given by \( n(x) = 10^{15} e^{-x/L_n} \, \text{cm}^{-3} \, (x \geq 0) \) where \( L_n = 10^{-4} \, \text{cm} \). The electron diffusion coefficient is \( D_n = 25 \, \text{cm}^2/\text{s} \). Determine the electron diffusion current density at (a) \( x = 0 \), (b) \( x = 10^{-4} \, \text{cm} \), and (c) \( x \rightarrow \infty \). (a) \(\uparrow\); \(\uparrow\) \( \downarrow \) \( \uparrow \); \(\uparrow\)\( \uparrow \) \( \downarrow \) \( \uparrow \) \( \downarrow \) \( \uparrow \)

**TYU 5.5** The hole concentration in silicon varies linearly from \( x = 0 \) to \( x = 0.01 \, \text{cm} \). The hole diffusion coefficient is \( D_p = 10 \, \text{cm}^2/\text{s} \), the hole diffusion current density is \( 20 \, \text{A/cm}^2 \), and the hole concentration at \( x = 0 \) is \( p = 4 \times 10^{17} \, \text{cm}^{-3} \). What is the value of the hole concentration at \( x = 0.01 \, \text{cm} \)? (\(\uparrow\), \(\uparrow\) \( \downarrow \) \( \uparrow \) \( \downarrow \) \( \uparrow \))

## 5.3 GRADED IMPURITY DISTRIBUTION

In most cases so far, we have assumed that the semiconductor is uniformly doped. **In many semiconductor devices, however, there may be regions that are nonuniformly doped.** We will investigate how a nonuniformly doped semiconductor reaches thermal equilibrium and, from this analysis, we will derive the **Einstein relation**, which relates mobility and the diffusion coefficient.

### 5.3.1 Induced Electric Field

Consider a semiconductor that is nonuniformly doped with donor impurity atoms. If the semiconductor is in thermal equilibrium, the Fermi energy level is constant through the crystal so the energy-band diagram may qualitatively look like that shown in Figure 5.12. The doping concentration decreases as \( x \) increases in this case. There will be a diffusion of majority carrier electrons from the region of high concentration to the region of low concentration, which is in the \( +x \) direction. The flow of negative electrons leaves behind positively charged donor ions. The separation of

!Energy-band diagram

**Figure 5.12** | Energy-band diagram for a semiconductor in thermal equilibrium with a nonuniform donor impurity concentration.

## 5.3 Graded Impurity Distribution

Positive and negative charge induces an electric field that is in a direction to oppose the diffusion process. When equilibrium is reached, the mobile carrier concentration is not exactly equal to the fixed impurity concentration and the induced electric field prevents any further separation of charge. In most cases of interest, the space charge induced by this diffusion process is a small fraction of the impurity concentration, thus the mobile carrier concentration is not too different from the impurity dopant density.

The electric potential \( \phi \) is related to electron potential energy by the charge \((-e)\), so we can write

\[
\phi = + \frac{1}{e} (E_F - E_{Fn})
\]

(5.37)

The electric field for the one-dimensional situation is defined as

\[
E_x = -\frac{d\phi}{dx} = \frac{1}{e} \frac{dE_{Fn}}{dx}
\]

(5.38)

If the intrinsic Fermi-level changes as a function of distance through a semiconductor in thermal equilibrium, an electric field exists in the semiconductor.

If we assume a quasi-neutrality condition in which the electron concentration is almost equal to the donor impurity concentration, then we can still write

\[
n_0 = n_i \exp \left[ \frac{E_F - E_{Fi}}{kT} \right] \approx N_d(x)
\]

(5.39)

Solving for \(E_F - E_{Fi}\), we obtain

\[
E_F - E_{Fi} = kT \ln \left( \frac{N_d(x)}{n_i} \right)
\]

(5.40)

The Fermi level is constant for thermal equilibrium so when we take the derivative with respect to \(x\) we obtain

\[
-\frac{dE_{Fi}}{dx} = -\frac{kT}{N_d(x)} \frac{dN_d(x)}{dx}
\]

(5.41)

The electric field can then be written, combining Equations (5.41) and (5.38), as

\[
E_x = -\left( \frac{kT}{e} \right) \frac{1}{N_d(x)} \frac{dN_d(x)}{dx}
\]

(5.42)

Since we have an electric field, there will be a potential difference through the semiconductor due to the nonuniform doping.

----

**Objective:** Determine the induced electric field in a semiconductor in thermal equilibrium, given a linear variation in doping concentration.

**Example 5.6**

Assume that the donor concentration in an n-type semiconductor at \(T = 300 \, \text{K}\) is given by

\[
N_d(x) = 10^{16} - 10^{19} \times x \quad (\text{cm}^{-3})
\]

where \(x\) is given in cm and ranges between \(0 \leq x \leq 1 \, \mu\text{m}\).

# CHAPTER 5 Carrier Transport Phenomena

## Solution

Taking the derivative of the donor concentration, we have

\[
\frac{dN_D(x)}{dx} = -10^9 \quad (\text{cm}^{-4})
\]

The electric field is given by Equation (5.42), so we have

\[
E_x = -\frac{(0.0259)(-10^9)}{(10^{16} - 10^9)x}
\]

At \( x = 0 \), for example, we find

\[
E_x = 25.9 \, \text{V/cm}
\]

## Comment

We may recall from our previous discussion of drift current that fairly small electric fields can produce significant drift current densities, so that an induced electric field from nonuniform doping can significantly influence semiconductor device characteristics.

## EXERCISE PROBLEM

**Ex 5.6** Assume the donor concentration in an n-type semiconductor at \( T = 300 \, \text{K} \) is given by \( N_D(x) = 10^{16} e^{-x/L} \) where \( L = 2 \times 10^{-2} \, \text{cm} \). Determine the induced electric field in the semiconductor at (a) \( x = 0 \) and (b) \( x = 10^{-4} \, \text{cm} \).

----

## 5.3.2 The Einstein Relation

If we consider the nonuniformly doped semiconductor represented by the energy-band diagram shown in Figure 5.12 and assume there are no electrical connections so that the semiconductor is in thermal equilibrium, then the individual electron and hole currents must be zero. We can write

\[
J_n = 0 = en\mu_n E_x + eD_n \frac{dn}{dx} \tag{5.43}
\]

If we assume quasi-neutrality so that \( n \approx N_d(x) \), then we can rewrite Equation (5.43) as

\[
J_n \equiv 0 = e\mu_n N_d(x)E_x + eD_n \frac{dN_d(x)}{dx} \tag{5.44}
\]

Substituting the expression for the electric field from Equation (5.42) into Equation (5.44), we obtain

\[
0 = -e\mu_n N_d(x) \left( \frac{kT}{e} \right) \frac{1}{N_d(x)} \frac{dN_d(x)}{dx} + eD_n \frac{dN_d(x)}{dx} \tag{5.45}
\]

Equation (5.45) is valid for the condition

\[
\frac{D_n}{\mu_n} = \frac{kT}{e} \tag{5.46a}
\]

The hole current must also be zero in the semiconductor. From this condition, we can show that

\[
\frac{D_p}{\mu_p} = \frac{kT}{e} \tag{5.46b}
\]

Combining Equations (5.46a) and (5.46b) gives

\[
\frac{D_n}{\mu_n} = \frac{D_p}{\mu_p} = \frac{kT}{e}
\]

(5.47)

The diffusion coefficient and mobility are not independent parameters. This relation between the mobility and diffusion coefficient, given by Equation (5.47), is known as the **Einstein relation**.

----

### Objective: Determine the diffusion coefficient given the carrier mobility.

Assume that the mobility of a particular carrier is 1000 cm\(^2\)/V·s at \( T = 300 \) K.

**Solution**

Using the Einstein relation, we have that

\[
D = \left( \frac{kT}{e} \right) \mu = (0.0259)(1000) = 25.9 \text{ cm}^2/\text{s}
\]

**Comment**

Although this example is fairly simple and straightforward, it is important to keep in mind the relative orders of magnitude of the mobility and diffusion coefficient. The diffusion coefficient is approximately 40 times smaller than the mobility at room temperature.

**EXERCISE PROBLEM**

Ex. 5.7 Assume the electron diffusion coefficient of a semiconductor at \( T = 300 \) K is \( D_n = 215 \) cm\(^2\)/s. Determine the electron mobility. (S·N·ω3 10E8 = *ηf s·νv*)

----

Table 5.2 shows the diffusion coefficient values at \( T = 300 \) K corresponding to the mobilities listed in Table 5.1 for silicon, gallium arsenide, and germanium.

The relation between the mobility and diffusion coefficient given by Equation (5.47) contains temperature. It is important to keep in mind that the major temperature effects are a result of lattice scattering and ionized impurity scattering processes, as discussed in Section 5.1.2. As the mobilities are strong functions of temperature because of the scattering processes, the diffusion coefficients are also strong functions of temperature. The specific temperature dependence given in Equation (5.47) is a small fraction of the real temperature characteristic.

----

### Table 5.2 | Typical mobility and diffusion coefficient values at \( T = 300 \) K (\(\mu = \text{cm}^2/\text{V·s}\) and \( D = \text{cm}^2/\text{s}\))

| Material          | \(\mu_e\) | \(D_n\) | \(\mu_p\) | \(D_p\) |
|-------------------|-----------|--------|-----------|--------|
| Silicon           | 1350      | 35     | 480       | 12.4   |
| Gallium arsenide  | 8500      | 220    | 400       | 10.4   |
| Germanium         | 3900      | 101    | 1900      | 49.2   |


# 5.4 The Hall Effect

The Hall effect is a consequence of the forces that are exerted on moving charges by electric and magnetic fields. The Hall effect is used to distinguish whether a semiconductor is type n or type p and to measure the majority carrier concentration and majority carrier mobility. The Hall effect device, as discussed in this section, is used to experimentally measure semiconductor parameters. However, it is also used extensively in engineering applications as a magnetic probe and in other circuit applications.

The force on a particle having a charge \( q \) and moving in a magnetic field is given by

\[
F = q \mathbf{v} \times \mathbf{B}
\]

(5.48)

where the cross product is taken between velocity and magnetic field so that the force vector is perpendicular to both the velocity and magnetic field.

Figure 5.13 illustrates the Hall effect. A semiconductor with a current \( I_x \) is placed in a magnetic field perpendicular to the current. In this case, the magnetic field is in the \( z \) direction. Electrons and holes flowing in the semiconductor will experience a force as indicated in the figure. The force on both electrons and holes is in the (−\( y \)) direction. In a p-type semiconductor (\( p_0 > n_0 \)), there will be a buildup of positive charge on the \( y = 0 \) surface of the semiconductor and, in an n-type semiconductor (\( n_0 > p_0 \)), there will be a buildup of negative charge on the \( y = 0 \) surface. This net charge induces an electric field in the \( y \) direction as shown in the figure. In steady

!Figure 5.13

**Figure 5.13** | Geometry for measuring the Hall effect.

----

*Indicates sections that will aid in the total summation of understanding of semiconductor devices, but may be skipped the first time through the text without loss of continuity.

*We will assume an extrinsic semiconductor material in which the majority carrier concentration is much larger than the minority carrier concentration.

# 5.4 The Hall Effect

The magnetic field force will be exactly balanced by the induced electric field force. This balance may be written as

\[
F = q[E + v \times B] = 0
\]

which becomes

\[
qE_y = qv_xB_z
\]

The induced electric field in the y direction is called the **Hall field**. The Hall field produces a voltage across the semiconductor which is called the **Hall voltage**. We can write

\[
V_{H} = +E_yW
\]

where \(E_y\) is assumed positive in the +y direction and \(V_H\) is positive with the polarity shown.

In a p-type semiconductor, in which holes are the majority carrier, the Hall voltage will be positive as defined in Figure 5.13. In an n-type semiconductor, in which electrons are the majority carrier, the Hall voltage will have the opposite polarity. The polarity of the Hall voltage is used to determine whether an extrinsic semiconductor is n type or p type.

Substituting Equation (5.50) into Equation (5.49) gives

\[
V_H = v_xWB_z
\]

For a p-type semiconductor, the drift velocity of holes can be written as

\[
v_x = \frac{J_x}{-ep} = \frac{I}{(ep)Wd}
\]

where \(e\) is the magnitude of the electronic charge. Combining Equations (5.52) and (5.59), we have

\[
V_H = \frac{I B_z}{epd}
\]

or, solving for the hole concentration, we obtain

\[
p = \frac{I B_z}{edV_H}
\]

The majority carrier hole concentration is determined from the current, magnetic field, and Hall voltage.

For an n-type semiconductor, the Hall voltage is given by

\[
V_H = -\frac{I B_z}{ned}
\]

so that the electron concentration is

\[
n = -\frac{I B_z}{edV_H}
\]

Note that the Hall voltage is negative for the n-type semiconductor; therefore, the electron concentration determined from Equation (5.56) is actually a positive quantity.

```markdown
# CHAPTER 5 Carrier Transport Phenomena

Once the majority carrier concentration has been determined, we can calculate the low-field majority carrier mobility. For a p-type semiconductor, we can write

\[
J_p = e p \mu_p E_x
\]

(5.57)

The current density and electric field can be converted to current and voltage so that Equation (5.57) becomes

\[
\frac{I_x}{Wd} = \frac{ep\mu_p V_t}{L}
\]

(5.58)

The hole mobility is then given by

\[
\mu_p = \frac{I_x L}{epV_t Wd}
\]

(5.59)

Similarly for an n-type semiconductor, the low-field electron mobility is determined from

\[
\mu_n = \frac{I_x L}{enV_t Wd}
\]

(5.60)

## EXAMPLE 5.8

**Objective:** Determine the majority carrier concentration and mobility, given Hall effect parameters.

Consider the geometry shown in Figure 5.13. Let \( L = 10^{-1} \) cm, \( W = 10^{-2} \) cm, and \( d = 10^{-3} \) cm. Also assume that \( I_x = 1.0 \) mA, \( V_t = 12.5 \) V, \( B_z = 500 \) gauss \( = 5 \times 10^{-2} \) tesla, and \( V_H = -6.25 \) mV.

### Solution

A negative Hall voltage for this geometry implies that we have an n-type semiconductor. Using Equation (5.56), we can calculate the electron concentration as

\[
n = \frac{- (10^{-3}) \times 5 \times 10^{-2}}{(1.6 \times 10^{-19})(10^{-3})(-6.25 \times 10^{-3})} = 5 \times 10^{21} \, \text{m}^{-3} = 5 \times 10^{15} \, \text{cm}^{-3}
\]

The electron mobility is then determined from Equation (5.60) as

\[
\mu_n = \frac{(10^{-3})(10^{-3})}{(1.6 \times 10^{-19})(5 \times 10^{21})(12.5 \times 10^{-3})} = 0.10 \, \text{m}^2/\text{V}\cdot\text{s}
\]

or

\[
\mu_n = 1000 \, \text{cm}^2/\text{V}\cdot\text{s}
\]

### Comment

It is important to note that the MKS units must be used consistently in the Hall effect equations to yield correct results.

### EXERCISE PROBLEM

**Ex 5.8** A p-type silicon sample with the geometry shown in Figure 5.13 has parameters \( L = 0.2 \) cm, \( W = 10^{-2} \) cm, and \( d = 8 \times 10^{-4} \) cm. The semiconductor parameters are \( p = 10^{16} \, \text{cm}^{-3} \) and \( \mu_p = 320 \, \text{cm}^2/\text{V}\cdot\text{s} \). For \( V_t = 10 \) V and \( B_z = 500 \) gauss \( = 5 \times 10^{-2} \) tesla, determine \( I_x \) and \( V_H \). (AU\# 080 = "A \# \nu 8f0 0 = 7 \text{suV}")
```

# 5.5.1 Summary

- The two basic transport mechanisms are drift, due to an applied electric field, and diffusion, due to a density gradient.
- Carriers reach an average drift velocity in the presence of an applied electric field, due to scattering events. Two scattering processes within a semiconductor are lattice scattering and impurity scattering.
- The average drift velocity is a linear function of the applied electric field for small values of electric field, but the drift velocity reaches a saturation limit that is on the order of \(10^7 \, \text{cm/s}\) at high electric fields.
- Carrier mobility is the ratio of the average drift velocity and applied electric field. The electron and hole mobilities are functions of temperature and of the ionized impurity concentration.
- The drift current density is the product of conductivity and electric field (a form of Ohm’s law). Conductivity is a function of the carrier concentrations and mobilities. Resistivity is the inverse of conductivity.
- The diffusion current density is proportional to the carrier diffusion coefficient and the carrier density gradient.
- The diffusion coefficient and mobility are related through the Einstein relation.
- The Hall effect is a consequence of a charged carrier moving in the presence of perpendicular electric and magnetic fields. The charged carrier is deflected, inducing a Hall voltage. The polarity of the Hall voltage is a function of the semiconductor conductivity type. The majority carrier concentration and mobility can be determined from the Hall voltage.

# Glossary of Important Terms

- **conductivity**: A material parameter related to carrier drift; quantitatively, the ratio of drift current density to electric field.
- **diffusion**: The process whereby particles flow from a region of high concentration to a region of low concentration.
- **diffusion coefficient**: The parameter relating particle flux to the particle density gradient.
- **diffusion current**: The current that results from the diffusion of charged particles.
- **drift**: The process whereby charged particles move while under the influence of an electric field.
- **drift current**: The current that results from the drift of charged particles.
- **drift velocity**: The average velocity of charged particles in the presence of an electric field.
- **Einstein relation**: The relation between the mobility and the diffusion coefficient.
- **Hall voltage**: The voltage induced across a semiconductor in a Hall effect measurement.
- **ionized impurity scattering**: The interaction between a charged carrier and an ionized impurity center.
- **lattice scattering**: The interaction between a charged carrier and a thermally vibrating lattice atom.
- **mobility**: The parameter relating carrier drift velocity and electric field.
- **resistivity**: The reciprocal of conductivity; a material parameter that is a measure of the resistance to current.
- **velocity saturation**: The saturation of carrier drift velocity with increasing electric field.

# CHECKPOINT

After studying this chapter, the reader should have the ability to:

- Discuss carrier drift current density.
- Explain why carriers reach an average drift velocity in the presence of an applied electric field.
- Discuss the mechanisms of lattice scattering and impurity scattering.
- Define mobility and discuss the temperature and ionized impurity concentration dependence on mobility.
- Define conductivity and resistivity.
- Discuss velocity saturation.
- Discuss carrier diffusion current density.
- State the Einstein relation.
- Describe the Hall effect.

# REVIEW QUESTIONS

1. Write the equation for the total drift current density. Is the linear relationship between drift current density and electric field always valid? Why or why not.
2. Define electron and hole mobility. What is the unit of mobility?
3. Explain the temperature dependence of mobility. Why is the carrier mobility a function of the ionized impurity concentrations?
4. Define conductivity. Define resistivity. What are the units of conductivity and resistivity?
5. Sketch the drift velocity of electrons in silicon versus electric field. Repeat for GaAs.
6. Write the equations for the diffusion current densities of electrons and holes.
7. What is the Einstein relation?
8. What is the direction of the induce electric field in a semiconductor with a graded donor impurity concentration? Repeat for a graded acceptor impurity concentration.
9. Describe the Hall effect.
10. Explain why the polarity of the Hall voltage changes depending on the conductivity type (n type or p type) of the semiconductor.

# PROBLEMS

*(Note: Use the semiconductor parameters given in Appendix B if the parameters are not specifically given in a problem.)*

## Section 5.1 Carrier Drift

**5.1**  
The concentration of donor impurity atoms in silicon is \( N_D = 10^{15} \, \text{cm}^{-3} \). Assume an electron mobility of \( \mu_n = 1300 \, \text{cm}^2/\text{V}\cdot\text{s} \) and a hole mobility of \( \mu_p = 450 \, \text{cm}^2/\text{V}\cdot\text{s} \).

(a) Calculate the resistivity of the material.  
(b) What is the conductivity of the material?

**5.2**  
A p-type silicon material is to have a conductivity of \( \sigma = 1.80 \, (\Omega\cdot\text{cm})^{-1} \). If the mobility values are \( \mu_n = 1250 \, \text{cm}^2/\text{V}\cdot\text{s} \) and \( \mu_p = 380 \, \text{cm}^2/\text{V}\cdot\text{s} \), what must be the acceptor impurity concentration in the material?

