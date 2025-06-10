# Chapter 6: Nonequilibrium Excess Carriers in Semiconductors

Our discussion of the physics of semiconductors in Chapter 4 was based on thermal equilibrium. When a voltage is applied or a current exists in a semiconductor device, the semiconductor is operating under nonequilibrium conditions. In our discussion of current transport in Chapter 5, we did not address nonequilibrium conditions but implicitly assumed that equilibrium was not significantly disturbed. Excess electrons in the conduction band and excess holes in the valence band may exist in addition to the thermal-equilibrium concentrations if an external excitation is applied to the semiconductor. In this chapter, we discuss the behavior of nonequilibrium electron and hole concentrations as functions of time and space coordinates.

Excess electrons and excess holes do not move independently of each other. These excess carriers diffuse, drift, and recombine with the same effective diffusion coefficient, drift mobility, and lifetime. This phenomenon is called ambipolar transport. We develop the ambipolar transport equation that describes the behavior of excess electrons and holes. Excess carriers dominate the electrical properties of a semiconductor material, and the behavior of excess carriers is fundamental to the operation of semiconductor devices.

## 6.0 | PREVIEW

In this chapter, we will:

- Describe the process of generation and recombination of excess carriers in a semiconductor.
- Define the recombination rate and generation rate of excess carriers, and define the excess carrier lifetime.
- Discuss why excess electrons and excess holes do not move independently of each other. The movement of excess carriers is called **ambipolar transport**, and the ambipolar transport equation is derived.

# 6.1 Carrier Generation and Recombination

- Apply the ambipolar transport equation to various situations to determine the time behavior and spatial behavior of excess carriers.
- Define the quasi-Fermi energy level.
- Analyze the effect of defects in a semiconductor on the excess carrier lifetime.
- Analyze the effect of defects at a semiconductor surface on the excess carrier concentration.

## 6.1.1 Carrier Generation and Recombination

In this chapter, we discuss carrier generation and recombination, which we can define as follows: **generation** is the process whereby electrons and holes are created, and **recombination** is the process whereby electrons and holes are annihilated.

Any deviation from thermal equilibrium will tend to change the electron and hole concentrations in a semiconductor. A sudden increase in temperature, for example, will increase the rate at which electrons and holes are thermally generated so that their concentrations will change with time until new equilibrium values are reached. An external excitation, such as light (a flux of photons), can also generate electrons and holes, creating a nonequilibrium condition. To understand the generation and recombination processes, we first consider direct band-to-band generation and recombination, and then, later, the effect of allowed electronic energy states within the bandgap, referred to as traps or recombination centers.

### 6.1.1 The Semiconductor in Equilibrium

We have determined the thermal-equilibrium concentration of electrons and holes in the conduction and valence bands, respectively. In thermal equilibrium, these concentrations are independent of time. However, electrons are continually being thermally excited from the valence band into the conduction band by the random nature of the thermal process. At the same time, electrons moving randomly through the crystal in the conduction band may come in close proximity to holes and “fall” into the empty states in the valence band. This recombination process annihilates both the electron and hole. Since the net carrier concentrations are independent of time in thermal equilibrium, the rate at which electrons and holes are generated and the rate at which they recombine must be equal. The generation and recombination processes are schematically shown in Figure 6.1.

!Figure 6.1

**Figure 6.1** | Electron–hole generation and recombination.

NO_CONTENT_HERE

# 6.1 Carrier Generation and Recombination

When excess electrons and holes are created, the concentration of electrons in the conduction band and of holes in the valence band increase above their thermal-equilibrium value. We may write

\[
n = n_0 + \delta n \tag{6.5a}
\]

and

\[
p = p_0 + \delta p \tag{6.5b}
\]

where \( n_0 \) and \( p_0 \) are the thermal-equilibrium concentrations, and \( \delta n \) and \( \delta p \) are the excess electron and hole concentrations. Figure 6.2 shows the excess electron–hole generation process and the resulting carrier concentrations. The external force has perturbed the equilibrium condition so that the semiconductor is no longer in thermal equilibrium. We may note from Equations (6.5a) and (6.5b) that, in a nonequilibrium condition, \( np \neq n_0p_0 = n_i^2 \).

A steady-state generation of excess electrons and holes will not cause a continual buildup of the carrier concentrations. As in the case of thermal equilibrium, an electron in the conduction band may “fall down” into the valence band, leading to the process of excess electron–hole recombination. Figure 6.3 shows this process. The

!Figure 6.2

**Figure 6.2** | Creation of excess electron and hole densities by photons.

!Figure 6.3

**Figure 6.3** | Recombination of excess carriers reestablishing thermal equilibrium.

## CHAPTER 6 Non-equilibrium Excess Carriers in Semiconductors

Recombination rate for excess electrons is denoted by \( R'_n \) and for excess holes by \( R'_p \). Both parameters have units of \(\#/\text{cm}^3\cdot\text{s}\). The excess electrons and holes recombine in pairs, so the recombination rates must be equal. We can then write

\[
R'_n = R'_p
\tag{6.6}
\]

In the direct band-to-band recombination that we are considering, the recombination occurs spontaneously; thus, the probability of an electron and hole recombining is constant with time. The rate at which electrons recombine must be proportional to the electron concentration and must also be proportional to the hole concentration. If there are no electrons or holes, there can be no recombination.

The net rate of change in the electron concentration can be written as

\[
\frac{dn(t)}{dt} = \alpha_r \left[ n_i^2 - n(t)p(t) \right]
\tag{6.7}
\]

where

\[
n(t) = n_0 + \delta n(t)
\tag{6.8a}
\]

and

\[
p(t) = p_0 + \delta p(t)
\tag{6.8b}
\]

The first term, \(\alpha_r n_i^2\), in Equation (6.7) is the thermal-equilibrium generation rate. Since excess electrons and holes are created and recombine in pairs, we have that \(\delta n(t) = \delta p(t)\). (Excess electron and hole concentrations are equal so we can simply use the phrase excess carriers to mean either.) The thermal-equilibrium parameters, \(n_0\) and \(p_0\), are independent of time; therefore, Equation (6.7) becomes

\[
\frac{d[\delta n(t)]}{dt} = \alpha_r \left[ n_i^2 - (n_0 + \delta n(t))(p_0 + \delta p(t)) \right]
\]

\[
= -\alpha_r \delta n(t)(n_0 + p_0) + \delta n(t)
\tag{6.9}
\]

Equation (6.9) can easily be solved if we impose the condition of low-level injection. Low-level injection puts limits on the magnitude of the excess carrier concentration compared with the thermal-equilibrium carrier concentrations. In an extrinsic n-type material, we generally have \(n_0 \gg p_0\) and, in an extrinsic p-type material, we generally have \(p_0 \gg n_0\). Low-level injection means that the excess carrier concentration is much less than the thermal-equilibrium majority carrier concentration. Conversely, high-level injection occurs when the excess carrier concentration becomes comparable to or greater than the thermal-equilibrium majority carrier concentrations.

If we consider a p-type material (\(p_0 \gg n_0\)) under low-level injection (\(\delta n(t) \ll p_0\)), then Equation (6.9) becomes

\[
\frac{d[\delta n(t)]}{dt} = -\alpha_r p_0 \delta n(t)
\tag{6.10}
\]

The solution to the equation is an exponential decay from the initial excess concentration, or

\[
\delta n(t) = \delta n(0)e^{-\alpha_r p_0 t} = \delta n(0)e^{-t/\tau}
\tag{6.11}
\]

where \(\tau_0 = ( \alpha_r p_0 )^{-1}\) and is a constant for the low-level injection. Equation (6.11) describes the decay of excess minority carrier electrons so that \(\tau_0\) is often referred to as the **excess minority carrier lifetime**. 

The recombination rate—which is defined as a positive quantity—of excess minority carrier electrons can be written, using Equation (6.10), as

\[
R'_n = -\frac{d(\delta n(t))}{dt} = \alpha_r p_0 \delta n(t) = \frac{\delta n(t)}{\tau_{n0}}
\]

(6.12)

For the direct band-to-band recombination, the excess majority carrier holes recombine at the same rate, so that for the p-type material

\[
R'_n = R'_p = \frac{\delta n(t)}{\tau_{n0}}
\]

(6.13)

In the case of an n-type material \((n_0 \gg p_0)\) under low-level injection \((\delta n(t) \ll n_0)\), the decay of minority carrier holes occurs with a time constant \(\tau_{p0} = (\alpha_r n_0)^{-1}\), where \(\tau_{p0}\) is also referred to as the excess minority carrier lifetime. The recombination rate of the majority carrier electrons will be the same as that of the minority carrier holes, so we have

\[
R'_n = R'_p = \frac{\delta n(t)}{\tau_{p0}}
\]

(6.14)

The generation rates of excess carriers are not functions of electron or hole concentrations. In general, the generation and recombination rates may be functions of the space coordinates and time.

----

### Objective: Determine the behavior of excess carriers as a function of time.

**EXAMPLE 6.1**

Assume that excess carriers have been generated uniformly in a semiconductor to a concentration of \(\delta n(0) = 10^{15} \, \text{cm}^{-3}\). The forcing function generating the excess carriers turns off at time \(t = 0\). Assuming the excess carrier lifetime is \(\tau_0 = 10^{-6} \, \text{s}\), determine \(\delta n(t)\) for \(t > 0\).

#### Solution

From Equation (6.11), we have

\[
\delta n(t) = \delta n(0) e^{-t/\tau_0} = 10^{15} e^{-t/10^{-6}} \, \text{cm}^{-3}
\]

For example, at:

- \(t = 0\), \(\delta n = 10^{15} \, \text{cm}^{-3}\)
- \(t = 1 \, \mu\text{s}\), \(\delta n = 10^{15} e^{-1} = 3.68 \times 10^{14} \, \text{cm}^{-3}\)
- \(t = 4 \, \mu\text{s}\), \(\delta n = 10^{15} e^{-4} = 1.83 \times 10^{13} \, \text{cm}^{-3}\)
- \(t = 10 \, \mu\text{s}\), \(\delta n = 10^{15} e^{-10} = 4.54 \times 10^{9} \, \text{cm}^{-3}\)

#### Comment

These results simply demonstrate the exponential decay of excess carriers with time after an excitation source is removed.

----

*In Chapter 5 we defined \(\tau\) as a mean time between collisions. We define \(\tau\) here as the mean time before a recombination event occurs. The two parameters are not related.*

```markdown
# CHAPTER 6 Non-equilibrium Excess Carriers in Semiconductors

## EXERCISE PROBLEM

**Ex 6.1** Using the parameters in Example 6.1, calculate the recombination rate of the excess carriers for (a) \( t = 0 \), (b) \( t = 1 \mu s \), (c) \( t = 4 \mu s \), and (d) \( t = 10 \mu s \).

\[ 
\text{[1.5 \times 10^{16} \, \text{cm}^{-3}]} \times e^{-t/\tau} \, (p) : i_{s} = u_{0} \, 0 \, 1 \, < \, 8 \, 1 \, (p) : i_{s} = u_{0} \, 0 \, 1 \, \times \, 8 \, 9 \, (q) : i_{s} = u_{0} \, r(0 \, 1) \, \text{suV}
\]

## 6.2 CHARACTERISTICS OF EXCESS CARRIERS

The generation and recombination rates of excess carriers are important parameters, but how the excess carriers behave with time and in space in the presence of electric fields and density gradients is of equal importance. As mentioned at the beginning of this chapter, the excess electrons and holes do not move independently of each other, but they diffuse and drift with the same effective diffusion coefficient and with the same effective mobility. This phenomenon is called ambipolar transport. The question that must be answered is what is the effective diffusion coefficient and what is the effective mobility that characterizes the behavior of these excess carriers? To answer these questions, we must develop the continuity equations for the carriers and then develop the ambipolar transport equations.

The final results show that, for an extrinsic semiconductor under low injection (this concept will be defined in the analysis), the effective diffusion coefficient and mobility parameters are those of the minority carrier. This result is thoroughly developed in the following derivations. As will be seen in the following chapters, the behavior of the excess carriers has a profound impact on the characteristics of semiconductor devices.

### 6.2.1 Continuity Equations

The continuity equations for electrons and holes are developed in this section. Figure 6.4 shows a differential volume element in which a one-dimensional hole-particle flux is entering the differential element at \( x \) and is leaving the element at \( x + dx \). The parameter \( F_{px}^{+} \) is the hole-particle flux, or flow, and has units of number of holes/cm\(^2\)-s. For the \( x \) component of the particle current density shown, we may write

\[
F_{px}^{+}(x + dx) = F_{px}^{+}(x) + \frac{\partial F_{px}^{+}}{\partial x} \cdot dx
\]

\[
(6.15)
\]

!Figure 6.4

**Figure 6.4** | Differential volume showing \( x \) component of the hole-particle flux.
```

## 6.2 Characteristics of Excess Carriers

This equation is a Taylor expansion of \( F^+_p(x + dx) \), where the differential length \( dx \) is small, so that only the first two terms in the expansion are significant. The net increase in the number of holes per unit time within the differential volume element due to the \( x \)-component of hole flux is given by

\[
\frac{\partial p}{\partial t} dx \, dy \, dz = [F^+_p(x) - F^+_p(x + dx)] dy \, dz = -\frac{\partial F^+_p}{\partial x} dx \, dy \, dz
\]

(6.16)

If \( F^+_p(x) > F^+_p(x + dx) \), for example, there will be a net increase in the number of holes in the differential volume element with time. If we generalize to a three-dimensional hole flux, then the right side of Equation (6.16) may be written as \(-\nabla \cdot F^+_p \, dx \, dy \, dz\), where \(\nabla \cdot F^+_p\) is the divergence of the flux vector. We will limit ourselves to a one-dimensional analysis.

The generation rate and recombination rate of holes will also affect the hole concentration in the differential volume. The net increase in the number of holes per unit time in the differential volume element is then given by

\[
\frac{\partial p}{\partial t} dx \, dy \, dz = -\frac{\partial F^+_p}{\partial x} dx \, dy \, dz + g_p \, dx \, dy \, dz - \frac{p}{\tau_{p0}} dx \, dy \, dz
\]

(6.17)

where \( p \) is the density of holes. The first term on the right side of Equation (6.17) is the increase in the number of holes per unit time due to the hole flux, the second term is the increase in the number of holes per unit time due to the generation of holes, and the last term is the decrease in the number of holes per unit time due to the recombination of holes. The recombination rate for holes is given by \( p/\tau_{p0} \), where \(\tau_{p0}\) includes the thermal-equilibrium carrier lifetime and the excess carrier lifetime.

If we divide both sides of Equation (6.17) by the differential volume \( dx \, dy \, dz \), the net increase in the hole concentration per unit time is

\[
\frac{\partial p}{\partial t} = -\frac{\partial F^+_p}{\partial x} + g_p - \frac{p}{\tau_{p0}}
\]

(6.18)

Equation (6.18) is known as the continuity equation for holes.

Similarly, the one-dimensional continuity equation for electrons is given by

\[
\frac{\partial n}{\partial t} = -\frac{\partial F^-_n}{\partial x} + g_n - \frac{n}{\tau_{n0}}
\]

(6.19)

where \( F^-_n \) is the electron-particle flow, or flux, also given in units of number of electrons/cm\(^2\)-s.

### 6.2.2 Time-Dependent Diffusion Equations

In Chapter 5, we derived the hole and electron current densities, which are given, in one dimension, by

\[
J_p = e\mu_p pE - eD_p \frac{\partial p}{\partial x}
\]

(6.20)

and

\[
J_n = e\mu_n nE + eD_n \frac{\partial n}{\partial x}
\]

(6.21)

# CHAPTER 6 Non-equilibrium Excess Carriers in Semiconductors

If we divide the hole current density by \( (+e) \) and the electron current density by \( (-e) \), we obtain each particle flux. These equations become

\[
\frac{J_p}{(+e)} = F^+ = \mu_p pE - D_p \frac{\partial p}{\partial x}
\]
(6.22)

and

\[
\frac{J_n}{(-e)} = F^- = -\mu_n nE - D_n \frac{\partial n}{\partial x}
\]
(6.23)

Taking the divergence of Equations (6.22) and (6.23), and substituting back into the continuity equations of (6.18) and (6.19), we obtain

\[
\frac{\partial p}{\partial t} = -\mu_p \frac{\partial (pE)}{\partial x} + D_p \frac{\partial^2 p}{\partial x^2} + g_p - \frac{p}{\tau_p}
\]
(6.24)

and

\[
\frac{\partial n}{\partial t} = \mu_n \frac{\partial (nE)}{\partial x} + D_n \frac{\partial^2 n}{\partial x^2} + g_n - \frac{n}{\tau_n}
\]
(6.25)

Keeping in mind that we are limiting ourselves to a one-dimensional analysis, we can expand the derivative of the product as

\[
\frac{\partial (pE)}{\partial x} = E \frac{\partial p}{\partial x} + p \frac{\partial E}{\partial x}
\]
(6.26)

In a more generalized three-dimensional analysis, Equation (6.26) would have to be replaced by a vector identity. Equations (6.24) and (6.25) can be written in the form

\[
D_p \frac{\partial^2 p}{\partial x^2} - \mu_p \left( E \frac{\partial p}{\partial x} + p \frac{\partial E}{\partial x} \right) + g_p - \frac{p}{\tau_p} = \frac{\partial p}{\partial t}
\]
(6.27)

and

\[
D_n \frac{\partial^2 n}{\partial x^2} + \mu_n \left( E \frac{\partial n}{\partial x} + n \frac{\partial E}{\partial x} \right) + g_n - \frac{n}{\tau_n} = \frac{\partial n}{\partial t}
\]
(6.28)

Equations (6.27) and (6.28) are the time-dependent diffusion equations for holes and electrons, respectively. Since both the hole concentration \( p \) and the electron concentration \( n \) contain the excess concentrations, Equations (6.27) and (6.28) describe the space and time behavior of the excess carriers.

The hole and electron concentrations are functions of both the thermal equilibrium and the excess values, which are given in Equations (6.5a) and (6.5b). The thermal-equilibrium concentrations, \( n_0 \) and \( p_0 \), are not functions of time. For the special case of a homogeneous semiconductor, \( n_0 \) and \( p_0 \) are also independent of the space coordinates. Equations (6.27) and (6.28) may then be written in the form

\[
D_p \frac{\partial^2 (\delta p)}{\partial x^2} - \mu_p \left( E \frac{\partial (\delta p)}{\partial x} + p \frac{\partial E}{\partial x} \right) + g_p - \frac{p}{\tau_p} = \frac{\partial (\delta p)}{\partial t}
\]
(6.29)

and

\[
D_n \frac{\partial^2 (\delta n)}{\partial x^2} + \mu_n \left( E \frac{\partial (\delta n)}{\partial x} + n \frac{\partial E}{\partial x} \right) + g_n - \frac{n}{\tau_n} = \frac{\partial (\delta n)}{\partial t}
\]
(6.30)

# 6.3 Ambipolar Transport

Note that Equations (6.29) and (6.30) contain terms involving the total concentrations, \( p \) and \( n \), and terms involving only the excess concentrations, \( \delta p \) and \( \delta n \).

## 6.3.1 Ambipolar Transport

Originally, we assumed that the electric field in the current Equations (6.20) and (6.21) was an applied electric field. This electric field term appears in the time-dependent diffusion equations given by Equations (6.29) and (6.30). If a pulse of excess electrons and a pulse of excess holes are created at a particular point in a semiconductor with an applied electric field, the excess holes and electrons will tend to drift in opposite directions. However, because the electrons and holes are charged particles, any separation will induce an internal electric field between the two sets of particles. This internal electric field will create a force attracting the electrons and holes back toward each other. This effect is shown in Figure 6.5. The electric field term in Equations (6.29) and (6.30) is then composed of the externally applied field plus the induced internal field. This E-field may be written as

\[
E = E_{\text{app}} + E_{\text{int}}
\]

(6.31)

where \( E_{\text{app}} \) is the applied electric field and \( E_{\text{int}} \) is the induced internal electric field.

Since the internal E-field creates a force attracting the electrons and holes, this E-field will hold the pulses of excess electrons and excess holes together. The negatively charged electrons and positively charged holes then will drift or diffuse together with a single effective mobility or diffusion coefficient. This phenomenon is called **ambipolar diffusion** or **ambipolar transport**.

### 6.3.1 Derivation of the Ambipolar Transport Equation

The time-dependent diffusion Equations (6.29) and (6.30) describe the behavior of the excess carriers. However, a third equation is required to relate the excess electron and hole concentrations to the internal electric field. This relation is Poisson’s equation, which may be written as

\[
\nabla \cdot E_{\text{int}} = \frac{e(\delta p - \delta n)}{\varepsilon_s} = \frac{\partial E_{\text{int}}}{\partial x}
\]

(6.32)

where \( \varepsilon_s \) is the permittivity of the semiconductor material.

!Figure 6.5

**Figure 6.5** | The creation of an internal electric field as excess electrons and holes tend to separate.

To make the solution of Equations (6.29), (6.30), and (6.32) more tractable, we need to make some approximations. We can show that only a relatively small internal electric field is sufficient to keep the excess electrons and holes drifting and diffusing together. Hence, we can assume that

\[
|\mathbf{E}_{\text{int}}| \ll |\mathbf{E}_{\text{appl}}|
\]

(6.33)

However, the \(\nabla \cdot \mathbf{E}_{\text{int}}\) term may not be negligible. We will impose the condition of charge neutrality: We will assume that the excess electron concentration is just balanced by an equal excess hole concentration at any point in space and time. If this assumption were exactly true, there would be no induced internal electric field to keep the two sets of particles together. However, only a very small difference in the excess electron concentration and excess hole concentration will set up an internal E-field sufficient to keep the particles diffusing and drifting together. We can show that a 1 percent difference in \(\delta p\) and \(\delta n\), for example, will result in non-negligible values of the \(\nabla \cdot \mathbf{E} = \nabla \cdot \mathbf{E}_{\text{int}}\) term in Equations (6.29) and (6.30).

We can combine Equations (6.29) and (6.30) to eliminate the \(\nabla \cdot \mathbf{E}\) term. Considering Equations (6.1) and (6.4), we can define

\[
g_n = g_p = g
\]

(6.34)

and considering Equations (6.2) and (6.6), we can define

\[
R_n = \frac{n}{\tau_{n}} = R_p = \frac{p}{\tau_{p}} \equiv R
\]

(6.35)

The lifetimes in Equation (6.35) include the thermal-equilibrium carrier lifetimes and the excess carrier lifetimes. If we impose the charge neutrality condition, then \(\delta n \approx \delta p\). We will denote both the excess electron and excess hole concentrations in Equations (6.29) and (6.30) by \(\delta n\). We may then rewrite Equations (6.29) and (6.30) as

\[
D_n \frac{\partial^2 (\delta n)}{\partial x^2} - \mu_p \left[ E \frac{\partial (\delta n)}{\partial x} + p \frac{\partial E}{\partial x} \right] + g - R = \frac{\partial (\delta n)}{\partial t}
\]

(6.36)

and

\[
D_p \frac{\partial^2 (\delta n)}{\partial x^2} + \mu_e \left[ E \frac{\partial (\delta n)}{\partial x} + n \frac{\partial E}{\partial x} \right] + g - R = \frac{\partial (\delta n)}{\partial t}
\]

(6.37)

If we multiply Equation (6.36) by \(\mu_n\), multiply Equation (6.37) by \(\mu_p\), and add the two equations, the \(\nabla \cdot \mathbf{E} = \partial E / \partial x\) term will be eliminated. The result of this addition gives

\[
(\mu_n n D_n + \mu_p D_p) \frac{\partial^2 (\delta n)}{\partial x^2} + (\mu_n \mu_p)(p - n) E \frac{\partial (\delta n)}{\partial x}
\]

\[
+ (\mu_n n + \mu_p p)(g - R) = (\mu_n + \mu_p) \frac{\partial (\delta n)}{\partial t}
\]

(6.38)

If we divide Equation (6.38) by the term \((\mu_n + \mu_p)\), this equation becomes

\[
D' \frac{\partial^2 (\delta n)}{\partial x^2} + \mu' E \frac{\partial (\delta n)}{\partial x} + g - R = \frac{\partial (\delta n)}{\partial t}
\]

(6.39)

# 6.3 Ambipolar Transport

where

\[
D' = \frac{\mu_n n D_n + \mu_p D_p}{\mu_n n + \mu_p p}
\]
(6.40)

and

\[
\mu' = \frac{\mu_n \mu_p (p - n)}{\mu_n n + \mu_p p}
\]
(6.41)

Equation (6.39) is called the **ambipolar transport equation** and describes the behavior of the excess electrons and holes in time and space. The parameter \( D' \) is called the **ambipolar diffusion coefficient** and \( \mu' \) is called the **ambipolar mobility**.

The Einstein relation relates the mobility and diffusion coefficient by

\[
\frac{\mu_n}{D_n} = \frac{\mu_p}{D_p} = \frac{e}{kT}
\]
(6.42)

Using these relations, the ambipolar diffusion coefficient may be written in the form

\[
D' = \frac{D_n D_p (n + p)}{D_n n + D_p p}
\]
(6.43)

The ambipolar diffusion coefficient, \( D' \), and the ambipolar mobility, \( \mu' \), are functions of the electron and hole concentrations, \( n \) and \( p \), respectively. Since both \( n \) and \( p \) contain the excess carrier concentration \( \delta n \), the coefficient in the ambipolar transport equation are not constants. The ambipolar transport equation, given by Equation (6.39), then, is a nonlinear differential equation.

## 6.3.2 Limits of Extrinsic Doping and Low Injection

The ambipolar transport equation may be simplified and linearized by considering an extrinsic semiconductor and by considering low-level injection. The ambipolar diffusion coefficient, from Equation (6.43), may be written as

\[
D' = \frac{D_n D_p [(n_0 + \delta n) + (p_0 + \delta n)]}{D_n (n_0 + \delta n) + D_p (p_0 + \delta n)}
\]
(6.44)

where \( n_0 \) and \( p_0 \) are the thermal-equilibrium electron and hole concentrations, respectively, and \( \delta n \) is the excess carrier concentration. If we consider a p-type semiconductor, we can assume that \( p_0 \gg n_0 \). The condition of low-level injection, or just low injection, means that the excess carrier concentration is much smaller than the thermal-equilibrium majority carrier concentration. For the p-type semiconductor, then, low injection implies that \( \delta n \ll p_0 \). Assuming that \( n_0 \ll p_0 \) and \( \delta n \ll p_0 \), and assuming that \( D_n \) and \( D_p \) are on the same order of magnitude, the ambipolar diffusion coefficient from Equation (6.44) reduces to

\[
D' = D_n
\]
(6.45)

# Chapter 6: Nonequilibrium Excess Carriers in Semiconductors

If we apply the conditions of an extrinsic p-type semiconductor and low injection to the ambipolar mobility, Equation (6.41) reduces to

\[
\mu' = \mu_n
\]

(6.46)

It is important to note that for an extrinsic p-type semiconductor under low injection, the ambipolar diffusion coefficient and the ambipolar mobility coefficient reduce to the minority carrier electron parameter values, which are constants. The ambipolar transport equation reduces to a linear differential equation with constant coefficients.

If we now consider an extrinsic n-type semiconductor under low injection, we may assume that \( p_0 \ll n_0 \) and \( \delta n \ll n_0 \). The ambipolar diffusion coefficient from Equation (6.43) reduces to

\[
D' = D_p
\]

(6.47)

and the ambipolar mobility from Equation (6.41) reduces to

\[
\mu' = -\mu_p
\]

(6.48)

The ambipolar parameters again reduce to the minority-carrier values, which are constants. Note that, for the n-type semiconductor, the ambipolar mobility is a negative value. The ambipolar mobility term is associated with carrier drift; therefore, the sign of the drift term depends on the charge of the particle. The equivalent ambipolar particle is negatively charged, as one can see by comparing Equations (6.30) and (6.39). If the ambipolar mobility reduces to that of a positively charged hole, a negative sign is introduced as shown in Equation (6.48).

The remaining terms we need to consider in the ambipolar transport equation are the generation rate and the recombination rate. Recall that the electron and hole recombination rates are equal and are given by Equation (6.35) as \( R_n = R_p = n/\tau_n = p/\tau_p = R \), where \( \tau_n \) and \( \tau_p \) are the mean electron and hole lifetimes, respectively. If we consider the inverse lifetime functions, then \( 1/\tau_n \) is the probability per unit time that an electron will encounter a hole and recombine. Likewise, \( 1/\tau_p \) is the probability per unit time that a hole will encounter an electron and recombine. If we again consider an extrinsic p-type semiconductor under low injection, the concentration of majority carrier holes will be essentially constant, even when excess carriers are present. Then, the probability per unit time of a minority carrier electron encountering a majority carrier hole will be essentially constant. Hence, the minority carrier electron lifetime, \( \tau_n = \tau_p \), will remain a constant for the extrinsic p-type semiconductor under low injection.

Similarly, if we consider an extrinsic n-type semiconductor under low injection, the minority carrier hole lifetime, \( \tau_p = \tau_n \), will remain constant. Even under the condition of low injection, the minority carrier hole concentration may increase by several orders of magnitude. The probability per unit time of a minority carrier electron encountering a hole may change drastically. The majority carrier lifetime, then, may change substantially when excess carriers are present.

Consider, again, the generation and recombination terms in the ambipolar transport equation. For electrons we may write

\[
g - R = g_n - R_n = (G_{n0} + g'_n) - (R_{n0} + R'_n)
\]

(6.49)

```markdown
where \( G_{n0} \) and \( g'_n \) are the thermal-equilibrium electron and excess electron generation rates, respectively. The terms \( R_{n0} \) and \( R'_n \) are the thermal-equilibrium electron and excess electron recombination rates, respectively. For thermal equilibrium, we have that

\[
G_{n0} = R_{n0}
\]

(6.50)

so Equation (6.49) reduces to

\[
g - R = g'_n - R'_n = g'_n - \frac{\delta n}{\tau_n}
\]

(6.51)

where \( \tau_n \) is the excess minority carrier electron lifetime.

For the case of holes, we may write

\[
g - R = g_p - R_p = (G_{p0} + g'_p) - (R_{p0} + R'_p)
\]

(6.52)

where \( G_{p0} \) and \( g'_p \) are the thermal-equilibrium hole and excess hole generation rates, respectively. The terms \( R_{p0} \) and \( R'_p \) are the thermal-equilibrium hole and excess hole recombination rates, respectively. Again, for thermal equilibrium, we have that

\[
G_{p0} = R_{p0}
\]

(6.53)

so that Equation (6.52) reduces to

\[
g - R = g'_p - R'_p = g'_p - \frac{\delta p}{\tau_p}
\]

(6.54)

where \( \tau_p \) is the excess minority carrier hole lifetime.

The generation rate for excess electrons must equal the generation rate for excess holes. We may then define a generation rate for excess carriers as \( g' \), so that \( g'_n = g'_p = g' \). We also determined that the minority carrier lifetime is essentially a constant for low injection. Then, the term \( g - R \) in the ambipolar transport equation may be written in terms of the minority carrier parameters.

The ambipolar transport equation, given by Equation (6.39), for a p-type semiconductor under low injection then becomes

\[
D_n \frac{\partial^2 (\delta n)}{\partial x^2} - \mu_n E \frac{\partial (\delta n)}{\partial x} + g' - \frac{\delta n}{\tau_{n0}} = \frac{\partial (\delta n)}{\partial t}
\]

(6.55)

The parameter \( \delta n \) is the excess minority carrier electron concentration, the parameter \( \tau_{n0} \) is the minority carrier lifetime under low injection, and the other parameters are the usual minority carrier electron parameters.

Similarly, for an extrinsic n-type semiconductor under low injection, the ambipolar transport equation becomes

\[
D_p \frac{\partial^2 (\delta p)}{\partial x^2} - \mu_p E \frac{\partial (\delta p)}{\partial x} + g' - \frac{\delta p}{\tau_{p0}} = \frac{\partial (\delta p)}{\partial t}
\]

(6.56)

The parameter \( \delta p \) is the excess minority carrier hole concentration, the parameter \( \tau_{p0} \) is the minority carrier hole lifetime under low injection, and the other parameters are the usual minority carrier hole parameters.
```

```markdown
# CHAPTER 6 Non-equilibrium Excess Carriers in Semiconductors

It is extremely important to note that the transport and recombination parameters in Equations (6.55) and (6.56) are those of the minority carrier. **Equations (6.55) and (6.56) describe the drift, diffusion, and recombination of excess minority carriers as a function of spatial coordinates and as a function of time.** Recall that we had imposed the condition of charge neutrality; the excess minority carrier concentration is equal to the excess majority carrier concentration. The excess majority carriers, then, diffuse and drift with the excess minority carriers; thus, the behavior of the excess majority carrier is determined by the minority carrier parameters. This ambipolar phenomenon is extremely important in semiconductor physics, and is the basis for describing the characteristics and behavior of semiconductor devices.

## 6.3.3 Applications of the Ambipolar Transport Equation

We solve the ambipolar transport equation for several problems. These examples help illustrate the behavior of excess carriers in a semiconductor material, and the results are used later in the discussion of the pn junction and the other semiconductor devices.

The following examples use several common simplifications in the solution of the ambipolar transport equation. Table 6.2 summarizes these simplifications and their effects.

### EXAMPLE 6.2

**Objective:** Determine the time behavior of excess carriers as a semiconductor returns to thermal equilibrium.

Consider an infinitely large, homogeneous n-type semiconductor with zero applied electric field. Assume that at time \( t = 0 \), a uniform concentration of excess carriers exists in the crystal, but assume that \( g' = 0 \) for \( t > 0 \). If we assume that the concentration of excess carriers is much smaller than the thermal-equilibrium electron concentration, then the low-injection condition applies. Calculate the excess carrier concentration as a function of time for \( t \geq 0 \).

### Table 6.2 | Common ambipolar transport equation simplifications

| Specification                                      | Effect                                                                 |
|----------------------------------------------------|------------------------------------------------------------------------|
| Steady state                                       | \(\frac{\partial (\delta n)}{\partial t} = 0, \quad \frac{\partial (\delta p)}{\partial t} = 0\) |
| Uniform distribution of excess carriers            | \(D_n \frac{\partial^2 (\delta n)}{\partial x^2} = 0, \quad D_p \frac{\partial^2 (\delta n)}{\partial x^2} = 0\) |
| (uniform generation rate)                          |                                                                        |
| Zero electric field                                | \(E \frac{\partial (\delta n)}{\partial x} = 0, \quad E \frac{\partial (\delta p)}{\partial x} = 0\) |
| No excess carrier generation                       | \(g' = 0\)                                                             |
| No excess carrier recombination                    | \(\frac{\delta n}{\tau_{n0}} = 0, \quad \frac{\delta p}{\tau_{p0}} = 0\) |
| (infinite lifetime)                                |                                                                        |
```

# Solution

For the n-type semiconductor, we need to consider the ambipolar transport equation for the minority carrier holes, which is given by Equation (6.56). The equation is

\[
D_p \frac{\partial^2 (\delta p)}{\partial x^2} - \mu_p E \frac{\partial (\delta p)}{\partial x} + g' - \frac{\delta p}{\tau_{p0}} = \frac{\partial (\delta p)}{\partial t}
\]

We are assuming a uniform concentration of excess holes so that \(\partial^2 (\delta p)/\partial x^2 = \partial (\delta p)/\partial x = 0\). For \(t > 0\), we are also assuming that \(g' = 0\). Equation (6.56) reduces to

\[
\frac{\partial (\delta p)}{\partial t} = -\frac{\delta p}{\tau_{p0}}
\]

(6.57)

Since there is no spatial variation, the total time derivative may be used. At low injection, the minority carrier hole lifetime, \(\tau_{p0}\), is a constant. The left side of Equation (6.57) is the time rate of change of \(\delta p\) and the right side of the equation is the recombination rate. The solution to Equation (6.57) is

\[
\delta p(t) = \delta p(0)e^{-t/\tau_{p0}}
\]

(6.58)

where \(\delta p(0)\) is the uniform concentration of excess carriers that exists at time \(t = 0\). The concentration of excess holes decays exponentially with time, with a time constant equal to the minority carrier hole lifetime.

From the charge-neutrality condition, we have that \(\delta n = \delta p\), so the excess electron concentration is given by

\[
\delta n(t) = \delta p(t) = \delta p(0)e^{-t/\tau_{p0}}
\]

(6.59)

# Comment

The excess electrons and holes recombine at the rate determined by the excess minority carrier hole lifetime in the n-type semiconductor.

# Exercise Problem

**Ex. 6.2** Consider n-type GaAs doped at \(N_d = 10^{16} \text{ cm}^{-3}\). Assume that \(10^{14}\) electron–hole pairs have been uniformly created per cm\(^3\) at \(t = 0\), and assume the minority carrier hole lifetime is \(\tau_{p0} = 50 \text{ ns}\). Determine the time at which the minority carrier hole concentration reaches (a) 1/10 of its initial value and (b) 10% of its initial value.

\[ \text{[SI 1 = (a) 50 ns = 1 (b) 50 ns]} \]

----

# Objective

Determine the time dependence of excess carriers in reaching a steady-state condition.

Again consider an infinitely large, homogeneous n-type semiconductor with a zero applied electric field. Assume that, for \(t < 0\), the semiconductor is in thermal equilibrium and that, for \(t \geq 0\), a uniform generation rate exists in the crystal. Calculate the excess carrier concentration as a function of time assuming the condition of low injection.

**Example 6.3**

# Chapter 6: Nonequilibrium Excess Carriers in Semiconductors

## Solution

The condition of a uniform generation rate and a homogeneous semiconductor again implies that \( \partial^2 (\delta p)/\partial x^2 = \delta p(x)/\partial x = 0 \) in Equation (6.56). The equation, for this case, reduces to

\[
g' - \frac{\delta p}{\tau_{p0}} = \frac{d(\delta p)}{dt}
\]

(6.60)

The solution to this differential equation is

\[
\delta p(t) = g' \tau_{p0} (1 - e^{-t/\tau_{p0}})
\]

(6.61)

## Comment

We may note that, as \( t \rightarrow \infty \), a steady-state excess hole and electron concentration of \( g' \tau_{p0} \) is reached. Equation (6.60) contains both a generation rate term and a recombination rate term for the excess carriers.

## Exercise Problem

**Ex 6.3** In Example 6.3, consider n-type silicon at \( T = 300 \, \text{K} \) doped to \( N_d = 5 \times 10^{16} \, \text{cm}^{-3} \). Assume that \( g' = 5 \times 10^{21} \, \text{cm}^{-3} \, \text{s}^{-1} \) and let \( \tau_{p0} = 10^{-7} \, \text{s} \). (a) Determine \( \delta p(t) \) at (i) \( t = 0 \), (ii) \( t = 10^{-7} \, \text{s} \), (iii) \( t = 5 \times 10^{-7} \, \text{s} \), and (iv) \( t \rightarrow \infty \). (b) Considering the results of part (a), is the low-injection condition maintained?

\[
\left[ \frac{5 \times 10^{10} \times (1 - e^{-t/\tau_{p0}})}{5 \times 10^{16}} \right] \ll 1 \quad \text{(yes)}
\]

The excess minority carrier hole concentration increases with time with the same time constant \( \tau_{p0} \), which is the excess minority carrier lifetime. The excess carrier concentration reaches a steady-state value as time goes to infinity, even though a steady-state generation of excess electrons and holes exists. This steady-state effect can be seen from Equation (6.60) by setting \( d(\delta p)/dt = 0 \). The remaining terms simply state that, in steady state, the generation rate is equal to the recombination rate.

## Test Your Understanding

**TYU 6.1** Silicon at \( T = 300 \, \text{K} \) has been doped with boron atoms to a concentration of \( N_a = 5 \times 10^{16} \, \text{cm}^{-3} \). Excess carriers have been generated in the uniformly doped material to a concentration of \( 10^{15} \, \text{cm}^{-3} \). The minority carrier lifetime is 5 μs.  
(a) What carrier type is the minority carrier?  
(b) Assuming \( g' = E = 0 \) for \( t > 0 \), determine the minority carrier concentration for \( t > 0 \).

\[
\left[ \frac{10^{15} \times e^{-t/\tau_{p0}}}{5 \times 10^{16}} \right] \ll 1 \quad \text{(yes)}
\]

**TYU 6.2** Consider silicon with the same parameters as given in TYU 6.1. The material is in thermal equilibrium for \( t < 0 \). At \( t = 0 \), a source generates excess carriers turned on, producing a generation rate of \( g' = 10^{20} \, \text{cm}^{-3} \, \text{s}^{-1} \).  
(a) What carrier type is the minority carrier?  
(b) Determine the minority carrier concentration for \( t > 0 \).  
(c) What is the minority carrier concentration as \( t \rightarrow \infty \)?

\[
\left[ \frac{10^{20} \times \tau_{p0} \times (1 - e^{-t/\tau_{p0}})}{5 \times 10^{16}} \right] \ll 1 \quad \text{(yes)}
\]

# Objective

Determine the steady-state spatial dependence of the excess carrier concentration.

Consider a p-type semiconductor that is homogeneous and infinite in extent. Assume a zero applied electric field. For a one-dimensional crystal, assume that excess carriers are being generated at \( x = 0 \) only, as indicated in Figure 6.6. The excess carriers being generated at \( x = 0 \) will begin diffusing in both the \( +x \) and \( -x \) directions. Calculate the steady-state excess carrier concentration as a function of \( x \).

## Solution

The ambipolar transport equation for excess minority carrier electrons is given by Equation (6.55), and is written as

\[
D_n \frac{d^2 (\delta n)}{dx^2} + \mu_n E \frac{d(\delta n)}{dx} + g' - \frac{\delta n}{\tau_0} = \frac{\partial (\delta n)}{\partial t}
\]

From our assumptions, we have \( E = 0 \), \( g' = 0 \) for \( x \neq 0 \), and \( \partial (\delta n)/\partial t = 0 \) for steady state. Assuming a one-dimensional crystal, Equation (6.55) reduces to

\[
D_n \frac{d^2 (\delta n)}{dx^2} - \frac{\delta n}{\tau_0} = 0
\]

(6.62)

Dividing by the diffusion coefficient, Equation (6.62) may be written as

\[
\frac{d^2 (\delta n)}{dx^2} - \frac{\delta n}{D_n \tau_0} = \frac{d^2 (\delta n)}{dx^2} - \frac{\delta n}{L_n^2} = 0
\]

(6.63)

where we have defined \( L_n^2 = D_n \tau_n \). The parameter \( L_n \) has the unit of length and is called the minority carrier electron diffusion length. The general solution to Equation (6.63) is

\[
\delta n(x) = Ae^{-x/L_n} + Be^{x/L_n}
\]

(6.64)

As the minority carrier electrons diffuse away from \( x = 0 \), they will recombine with the majority carrier holes. The minority carrier electron concentration will then decay toward zero at both \( x = +\infty \) and \( x = -\infty \). These boundary conditions mean that \( B = 0 \) for \( x > 0 \) and \( A = 0 \) for \( x < 0 \). The solution to Equation (6.63) may then be written as

\[
\delta n(x) = \delta n(0)e^{-x/L_n}, \quad x \geq 0
\]

(6.65a)

!Figure 6.6

**Figure 6.6** Steady-state generation rate at \( x = 0 \).

```markdown
# CHAPTER 6 Non-equilibrium Excess Carriers in Semiconductors

and

\[
\delta n(x) = \delta n(0)e^{x/L_n} \quad x \leq 0
\]
(6.65b)

where \(\delta n(0)\) is the value of the excess electron concentration at \(x = 0\). The steady-state excess electron concentration decays exponentially with distance away from the source at \(x = 0\).

## Comment

We may note that the steady-state excess concentration decays to \(1/e\) of its value at \(x = L_n\).

## EXERCISE PROBLEM

**Ex 6.4** In Example 6.4, consider p-type silicon at \(T = 300 \, \text{K}\) doped to \(N_a = 5 \times 10^{16} \, \text{cm}^{-3}\). Assume that \(\tau_n = 5 \times 10^{-7} \, \text{s}\), \(D_n = 25 \, \text{cm}^2/\text{s}\), and \(\delta n(0) = 10^{15} \, \text{cm}^{-3}\). (a) Calculate the value of diffusion length \(L_n\). (b) Determine \(\delta n\) at (i) \(x = 0\), (ii) \(x = +30 \, \mu\text{m}\), (iii) \(x = -50 \, \mu\text{m}\), (iv) \(x = +85 \, \mu\text{m}\), and (v) \(x = -120 \, \mu\text{m}\).

As before, we will assume charge neutrality; thus, the steady-state excess majority carrier hole concentration also decays exponentially with distance with the same characteristic minority carrier electron diffusion length \(L_n\). Figure 6.7 is a plot of the total electron and hole concentrations as a function of distance. We are assuming low injection, that is, \(\delta n(0) \ll p_0\) in the p-type semiconductor. The total concentration of

!Figure 6.7

**Figure 6.7** Steady-state electron and hole concentrations for the case when excess electrons and holes are generated at \(x = 0\).
```

```markdown
majority carrier holes barely changes. However, we may have \(\delta n(0) \gg n_0\) and still satisfy the low-injection condition. The minority carrier concentration may change by many orders of magnitude.

----

### TEST YOUR UNDERSTANDING

**TYU 6.3** Excess electrons and holes are generated at the end of a silicon bar (\(x = 0\)). The silicon is doped with phosphorus atoms to a concentration of \(N_d = 10^{17} \, \text{cm}^{-3}\). The minority carrier lifetime is 1 μs, the electron diffusion coefficient is \(D_n = 25 \, \text{cm}^2/\text{s}\), and the hole diffusion coefficient is \(D_p = 10 \, \text{cm}^2/\text{s}\). If \(\delta p(0) = \delta n(0) = 10^5 \, \text{cm}^{-3}\), determine the steady-state electron and hole concentrations in the silicon for \(x > 0\).

**TYU 6.4** Using the parameters given in TYU 6.3, calculate the electron and hole diffusion current densities at \(x = 10 \, \mu\text{m}\).

----

The three previous examples, which applied the ambipolar transport equation to specific situations, assumed either a homogeneous or a steady-state condition; only the time variation or the spatial variation was considered. Now consider an example in which both the time and spatial dependence are considered in the same problem.

----

### Objective: Determine both the time dependence and spatial dependence of the excess carrier concentration.

Assume that a finite number of electron-hole pairs is generated instantaneously at time \(t = 0\) and at \(x = 0\), but assume \(g' = 0\) for \(t > 0\). Assume we have an n-type semiconductor with a constant applied electric field equal to \(E_0\), which is applied in the \(+x\) direction. Calculate the excess carrier concentration as a function of \(x\) and \(t\).

#### Solution

The one-dimensional ambipolar transport equation for the minority carrier holes can be written from Equation (6.56) as

\[
D_p \frac{\partial^2 (\delta p)}{\partial x^2} - \mu_p E_0 \frac{\partial (\delta p)}{\partial x} - \frac{\delta p}{\tau_{p0}} = \frac{\partial (\delta p)}{\partial t}
\]

(6.66)

The solution to this partial differential equation is of the form

\[
\delta p(x, t) = p'(x, t)e^{-t/\tau_{p0}}
\]

(6.67)

By substituting Equation (6.67) into Equation (6.66), we are left with the partial differential equation

\[
D_p \frac{\partial^2 p'(x, t)}{\partial x^2} - \mu_p E_0 \frac{\partial p'(x, t)}{\partial x} = \frac{\partial p'(x, t)}{\partial t}
\]

(6.68)
```

# Chapter 6: Nonequilibrium Excess Carriers in Semiconductors

Equation (6.68) is normally solved using Laplace transform techniques. The solution, without going through the mathematical details, is

\[
p'(x, t) = \frac{1}{(4\pi D_p t)^{1/2}} \exp \left[ -\frac{(x - \mu_p E_0 t)^2}{4D_p t} \right]
\]

(6.69)

The total solution, from Equations (6.67) and (6.69), for the excess minority carrier hole concentration is

\[
\delta p(x, t) = \frac{e^{-t/\tau_p}}{(4\pi D_p t)^{1/2}} \exp \left[ -\frac{(x - \mu_p E_0 t)^2}{4D_p t} \right]
\]

(6.70)

**Comment**

We could show that Equation (6.70) is a solution to the partial differential equation, Equation (6.66), by direct substitution. We may also note that Equation (6.70) is not normalized.

## Exercise Problem

**Ex 6.5** Consider the result of Example 6.5. Let \( D_p = 10 \, \text{cm}^2/\text{s}, \tau_p = 10^{-7} \, \text{s}, \mu_p = 400 \, \text{cm}^2/\text{V}\cdot\text{s}, \) and \( E = 100 \, \text{V/cm}. \)

(a) Determine \( \delta p \) for \( t = 10^{-7} \, \text{s} \) at \( x = 20 \, \mu\text{m}, \) (ii) \( x = 40 \, \mu\text{m}, \) and (iii) \( x = 60 \, \mu\text{m}. \)

(b) Determine \( \delta p \) for \( x = 40 \, \mu\text{m} \) at (i) \( t = 5 \times 10^{-8} \, \text{s}, \) (ii) \( t = 10^{-7} \, \text{s}, \) and (iii) \( t = 2 \times 10^{-7} \, \text{s}. \) Compare the results to the curves shown in Figure 6.9.

----

Equation (6.70) can be plotted as a function of distance \( x \), for various times. Figure 6.8 shows such a plot for the case when the applied electric field is zero. For \( t > 0 \), the excess minority carrier holes diffuse in both the \( +x \) and \( -x \) directions. During this time, the excess majority carrier electrons, which were generated, diffuse at exactly the same rate as the holes. As time proceeds, the excess holes recombine with the excess electrons so that at \( t = \infty \) the excess hole concentration is zero. In this particular example, both diffusion and recombination processes are occurring at the same time.

Figure 6.9 shows a plot of Equation (6.70) as a function of distance \( x \) at various times for the case when the applied electric field is not zero. In this case, the pulse of excess minority carrier holes is drifting in the \( +x \) direction, which is the direction of the electric field. We still have the same diffusion and recombination processes as we had before. An important point to consider is that, with charge neutrality, \( \delta n = \delta p \) at any instant of time and at any point in space. The excess electron concentration is equal to the excess hole concentration. In this case, then, the excess electron pulse is moving in the same direction as the applied electric field even though the electrons have a negative charge. In the ambipolar transport process, the excess carriers are characterized by the minority carrier parameters. In this example, the excess carriers behave according to the minority carrier hole parameters, which include \( D_p, \mu_p, \) and \( \tau_p \). The excess majority carrier electrons are being pulled along by the excess minority carrier holes.

# 6.3 Ambipolar Transport

!Graph of excess hole concentration versus distance at various times for zero applied electric field.

**Figure 6.8** Excess hole concentration versus distance at various times for zero applied electric field.

- \( \delta p(x, t) \)
- \( t = t_1 > 0 \)
- \( t = t_2 > t_1 \)
- \( t = t_3 > t_2 \)
- \( E = 0 \)
- \( x = 0 \)
- Distance, \( x \)

----

!Graph of excess hole concentration versus distance at various times for a constant applied electric field.

**Figure 6.9** Excess hole concentration versus distance at various times for a constant applied electric field.

- \( \delta p(x, t) \)
- \( t = t_1 > 0 \)
- \( t = t_2 > t_1 \)
- \( t = t_3 > t_2 \)
- \( E_0 \)
- \( x = 0 \)
- Distance, \( x \)

# CHAPTER 6 Non-equilibrium Excess Carriers in Semiconductors

## TEST YOUR UNDERSTANDING

**TYU 6.5**  
As a good approximation, the peak value of a normalized excess carrier concentration, given by Equation (6.70), occurs at \( x = L_{n,p} \). Assume the following parameters: \( \tau_{p0} = 5 \, \mu s \), \( D_p = 10 \, \text{cm}^2/s \), \( \mu_p = 386 \, \text{cm}^2/V \cdot s \), and \( E_0 = 10 \, \text{V/cm} \). Calculate the peak value at times of (a) \( t = 1 \, \mu s \), (b) \( t = 5 \, \mu s \), (c) \( t = 15 \, \mu s \), and (d) \( t = 25 \, \mu s \). What are the corresponding values of \( x \) for parts (a) to (d)?

\[
\text{TYU 6.5:} \quad \text{Peak value} = x \cdot \text{Equation} \cdot \text{Value}
\]

**TYU 6.6**  
The excess carrier concentration, given by Equation (6.70), is to be calculated at distances of one diffusion length away from the peak value. Using the parameters given in TYU 6.5, calculate the values of \( \delta p \) for (i) \( t = 1 \, \mu s \) at (i) \( x = 1.093 \times 10^{-2} \, \text{cm} \) and (ii) \( x = -3.21 \times 10^{-2} \, \text{cm} \); (b) \( t = 5 \, \mu s \) at (i) \( x = 2.64 \times 10^{-2} \, \text{cm} \) and (ii) \( x = 1.22 \times 10^{-2} \, \text{cm} \); (c) \( t = 15 \, \mu s \) at (i) \( x = 6.50 \times 10^{-2} \, \text{cm} \) and (ii) \( x = 5.08 \times 10^{-2} \, \text{cm} \).

\[
\text{TYU 6.6:} \quad \text{Calculate} \, \delta p \, \text{using} \, x \, \text{and} \, t
\]

## 6.3.4 Dielectric Relaxation Time Constant

We have assumed in the previous analysis that a quasi-neutrality condition exists—that is, the concentration of excess holes is balanced by an equal concentration of excess electrons. Suppose that we have a situation as shown in Figure 6.10, in which a uniform concentration of holes \( \delta p \) is suddenly injected into a portion of the surface of a semiconductor. We will instantly have a concentration of excess holes and a net positive charge density that is not balanced by a concentration of excess electrons. How is charge neutrality achieved and how fast?

There are three defining equations to be considered. Poisson’s equation is

\[
\nabla \cdot \mathbf{E} = \frac{\rho}{\varepsilon} \tag{6.71}
\]

The current equation, Ohm’s law, is

\[
J = \sigma E \tag{6.72}
\]

The continuity equation, neglecting the effects of generation and recombination, is

\[
\nabla \cdot J = -\frac{\partial \rho}{\partial t} \tag{6.73}
\]

!Figure 6.10  
**Figure 6.10** | The injection of a concentration of holes into a small region at the surface of an n-type semiconductor.

## 6.3 Ambipolar Transport

The parameter \( \rho \) is the net charge density and the initial value is given by \( e\delta p \). We will assume that \( \delta p \) is uniform over a short distance at the surface. The parameter \( \epsilon \) is the permittivity of the semiconductor.

Taking the divergence of Ohm’s law and using Poisson’s equation, we find

\[
\nabla \cdot \mathbf{J} = \sigma \nabla \cdot \mathbf{E} = \frac{\sigma \rho}{\epsilon}
\tag{6.74}
\]

Substituting Equation (6.74) into the continuity equation, we have

\[
\frac{\sigma \rho}{\epsilon} = -\frac{\partial \rho}{\partial t} = -\frac{d \rho}{dt}
\tag{6.75}
\]

Since Equation (6.75) is a function of time only, we can write the equation as a total derivative. Equation (6.75) can be rearranged as

\[
\frac{d \rho}{dt} + \left( \frac{\sigma}{\epsilon} \right) \rho = 0
\tag{6.76}
\]

Equation (6.76) is a first-order differential equation whose solution is

\[
\rho(t) = \rho(0)e^{-t/\tau_d}
\tag{6.77}
\]

where

\[
\tau_d = \frac{\epsilon}{\sigma}
\tag{6.78}
\]

and is called the dielectric relaxation time constant.

----

### Objective: Calculate the dielectric relaxation time constant for a particular semiconductor.

**EXAMPLE 6.6**

Consider n-type silicon with a donor impurity concentration of \( N_d = 10^{16} \, \text{cm}^{-3} \).

#### Solution

The conductivity is found as

\[
\sigma \approx e \mu_n N_d = (1.6 \times 10^{-19})(1200)(10^{16}) = 1.92 \, (\Omega \cdot \text{cm})^{-1}
\]

where the value of mobility is the approximate value found from Figure 5.3. The permittivity of silicon is

\[
\epsilon = \epsilon_r \epsilon_0 = (11.7)(8.85 \times 10^{-14}) \, \text{F/cm}
\]

The dielectric relaxation time constant is then

\[
\tau_d = \frac{\epsilon}{\sigma} = \frac{(11.7)(8.85 \times 10^{-14})}{1.92} = 5.39 \times 10^{-13} \, \text{s}
\]

or

\[
\tau_d = 0.539 \, \text{ps}
\]

```markdown
# CHAPTER 6 Non-equilibrium Excess Carriers in Semiconductors

**Comment**

Equation (6.77) then predicts that in approximately four time constants, or in approximately 2 ps, the net charge density is essentially zero; that is, quasi-neutrality has been achieved. Since the continuity equation, Equation (6.73), does not contain any generation or recombination terms, the initial positive charge is then neutralized by pulling in electrons from the bulk n-type material to create excess electrons. This process occurs very quickly compared to the normal excess carrier lifetimes of approximately 0.1 μs. The condition of quasi-charge-neutrality is then justified.

**EXERCISE PROBLEM**

Ex 6.6 (a) Consider n-type GaAs doped to \( N_d = 5 \times 10^{15} \, \text{cm}^{-3} \). Determine the dielectric relaxation time. (b) Repeat part (a) for p-type silicon doped to \( N_a = 2 \times 10^{16} \, \text{cm}^{-3} \).

\[ [sd \, 6080 = f \, (p) \, sd \, 5610 = f \, (p) \, suv] \]

## *6.3.5 Haynes–Shockley Experiment

We have derived the mathematics describing the behavior of excess carriers in a semiconductor. The Haynes–Shockley experiment was one of the first experiments to actually measure excess carrier behavior.

Figure 6.11 shows the basic experimental arrangement. The voltage source \( V_1 \) establishes an applied electric field \( E_0 \) in the \( +x \) direction in the n-type semiconductor sample. Excess carriers are effectively injected into the semiconductor at contact A. Contact B is a rectifying contact that is under reverse bias by the voltage source \( V_2 \). The contact B will collect a fraction of the excess carriers as they drift through the semiconductor. The collected carriers will generate an output voltage, \( V_0 \).

This experiment corresponds to the problem we discussed in Example 6.5. Figure 6.12 shows the excess carrier concentrations at contacts A and B for two conditions. Figure 6.12a shows the idealized excess carrier pulse at contact A at time \( t = 0 \). For a given electric field \( E_0 \), the excess carriers will drift along the

!Figure 6.11

**Figure 6.11** The basic Haynes–Shockley experimental arrangement.
```

!Figure 6.12

**Figure 6.12** (a) The idealized excess carrier pulse at terminal A at \( t = 0 \). (b) The excess carrier pulse versus time at terminal B for a given applied electric field. (c) The excess carrier pulse versus time at terminal B for a smaller applied electric field.

----

The semiconductor producing an output voltage as a function of time is given in Figure 6.12b. The peak of the pulse will arrive at contact B at time \( t_0 \). If the applied electric field is reduced to a value \( E_{02}, \, E_{02} < E_{01} \), the output voltage response at contact B will look approximately as shown in Figure 6.12c. For the smaller electric field, the drift velocity of the pulse of excess carriers is smaller, and so it will take a longer time for the pulse to reach the contact B. During this longer time period, there is more diffusion and more recombination. The excess carrier pulse shapes shown in Figure 6.12b, c are different for the two electric field conditions.

# CHAPTER 6 Non-equilibrium Excess Carriers in Semiconductors

The minority carrier mobility, lifetime, and diffusion coefficient can be determined from this single experiment. As a good first approximation, the peak of the minority carrier pulse will arrive at contact B when the experiment involving distance and time in Equation (6.70) is zero, or

\[
x - \mu_n E_0 t = 0
\]

(6.79a)

In this case \( x = d \), where \( d \) is the distance between contacts A and B, and \( t = t_0 \), where \( t_0 \) is the time at which the peak of the pulse reaches contact B. The mobility may be calculated as

\[
\mu_n = \frac{d}{E_0 t_0}
\]

(6.79b)

Figure 6.13 again shows the output response as a function of time. At times \( t_1 \) and \( t_2 \), the magnitude of the excess concentration is \( e^{-1} \) of its peak value. If the time difference between \( t_1 \) and \( t_2 \) is not too large, \( e^{-t/\tau_p} \) and \( (4\pi D_p t)^{1/2} \) do not change appreciably during this time; then the equation

\[
(d - \mu_n E_0 t) = 4D_p t
\]

(6.80)

is satisfied at both \( t = t_1 \) and \( t = t_2 \). If we set \( t = t_1 \) and \( t = t_2 \) in Equation (6.80) and add the two resulting equations, we may show that the diffusion coefficient is given by

\[
D_p = \frac{(\mu_n E_0)^2 (\Delta t)^2}{16 t_0}
\]

(6.81)

where

\[
\Delta t = t_2 - t_1
\]

(6.82)

The area \( S \) under the curve shown in Figure 6.13 is proportional to the number of excess holes that have not recombined with majority carrier electrons. We may write

\[
S = K \exp\left(-\frac{t_0}{\tau_p}\right) = K \exp\left(-\frac{d}{\mu_n E_0 \tau_p}\right)
\]

(6.83)

where \( K \) is a constant. By varying the electric field, the area under the curve will change. A plot of \(\ln(S)\) as a function of \((d/\mu_n E_0)\) will yield a straight line whose slope is \((1/\tau_p)\), so the minority carrier lifetime can also be determined from this experiment.

!Figure 6.13

**Figure 6.13** | The output excess carrier pulse versus time to determine the diffusion coefficient.

# 6.4 Quasi-Fermi Energy Levels

The Haynes–Shockley experiment is elegant in the sense that the three basic processes of drift, diffusion, and recombination are all observed in a single experiment. The determination of mobility is straightforward and can yield accurate values. The determination of the diffusion coefficient and lifetime is more complicated and may lead to some inaccuracies.

## 6.4.1 Quasi-Fermi Energy Levels

The thermal-equilibrium electron and hole concentrations are functions of the Fermi energy level. We can write

\[
n_0 = n_i \exp \left( \frac{E_F - E_{Fi}}{kT} \right) \tag{6.84a}
\]

and

\[
p_0 = n_i \exp \left( \frac{E_{Fi} - E_F}{kT} \right) \tag{6.84b}
\]

where \( E_F \) and \( E_{Fi} \) are the Fermi energy and intrinsic Fermi energy, respectively, and \( n_i \) is the intrinsic carrier concentration. Figure 6.14a shows the energy-band diagram for an n-type semiconductor in which \( E_F > E_{Fi} \). For this case, we may note from Equations (6.84a) and (6.84b) that \( n_0 > n_i \) and \( p_0 < n_i \), as we would expect. Similarly, Figure 6.14b shows the energy-band diagram for a p-type semiconductor in which \( E_F < E_{Fi} \). Again we may note from Equations (6.84a) and (6.84b) that \( n_0 < n_i \) and \( p_0 > n_i \), as we would expect for the p-type material. These results are for thermal equilibrium.

If excess carriers are created in a semiconductor, we are no longer in thermal equilibrium and the Fermi energy is strictly no longer defined. However, we may define a quasi-Fermi level for electrons and a quasi-Fermi level for holes that apply for nonequilibrium. If \( \delta n \) and \( \delta p \) are the excess electron and hole concentrations, respectively, we may write

\[
n_0 + \delta n = n_i \exp \left( \frac{E_n - E_{Fi}}{kT} \right) \tag{6.85a}
\]

!Figure 6.14

**Figure 6.14** | Thermal-equilibrium energy-band diagrams for (a) n-type semiconductor and (b) p-type semiconductor.

# Chapter 6: Nonequilibrium Excess Carriers in Semiconductors

and

\[
p_0 + \delta p = n_i \exp \left( \frac{E_{Fi} - E_{Fp}}{kT} \right)
\]

(6.85b)

where \(E_{Fn}\) and \(E_{Fp}\) are the quasi-Fermi energy levels for electrons and holes, respectively. The total electron concentration and the total hole concentration are functions of the quasi-Fermi levels.

## Example 6.7

**Objective:** Calculate the quasi-Fermi energy levels.

Consider an n-type semiconductor at \(T = 300 \, \text{K}\) with carrier concentrations of \(n_0 = 10^5 \, \text{cm}^{-3}\), \(n_i = 10^{10} \, \text{cm}^{-3}\), and \(p_0 = 10^6 \, \text{cm}^{-3}\). In nonequilibrium, assume that the excess carrier concentrations are \(\delta n = \delta p = 10^{13} \, \text{cm}^{-3}\).

### Solution

The Fermi level for thermal equilibrium can be determined from Equation (6.84a). We have

\[
E_F - E_{Fi} = kT \ln \left( \frac{n_0}{n_i} \right) = 0.2982 \, \text{eV}
\]

We can use Equation (6.85a) to determine the quasi-Fermi level for electrons in nonequilibrium. We can write

\[
E_{Fn} - E_{Fi} = kT \ln \left( \frac{n_0 + \delta n}{n_i} \right) = 0.2984 \, \text{eV}
\]

Equation (6.85b) can be used to calculate the quasi-Fermi level for holes in nonequilibrium. We can write

\[
E_{Fi} - E_{Fp} = kT \ln \left( \frac{p_0 + \delta p}{n_i} \right) = 0.179 \, \text{eV}
\]

### Comment

We may note that the quasi-Fermi level for electrons is above \(E_{Fi}\) while the quasi-Fermi level for holes is below \(E_{Fi}\).

### Exercise Problem

**Ex 6.7** Impurity concentrations of \(N_d = 3 \times 10^{15} \, \text{cm}^{-3}\) and \(N_a = 10^{16} \, \text{cm}^{-3}\) are added to silicon at \(T = 300 \, \text{K}\). Excess carriers are generated in the semiconductor such that the steady-state excess carrier concentrations are \(\delta n = \delta p = 4 \times 10^{14} \, \text{cm}^{-3}\).

(a) Determine the thermal-equilibrium Fermi level with respect to the intrinsic Fermi level.  
(b) Find \(E_{Fn}\) and \(E_{Fp}\) with respect to \(E_g\).

Figure 6.15a shows the energy-band diagram with the Fermi energy level corresponding to thermal equilibrium. Figure 6.15b now shows the energy-band diagram under the nonequilibrium condition. Since the majority carrier electron concentration does not change significantly for this low-injection condition, the quasi-Fermi level for electrons is not much different from the thermal-equilibrium Fermi level.

```markdown
!Figure 6.15

**Figure 6.15** (a) Thermal-equilibrium energy-band diagram for \( N_A = 10^5 \, \text{cm}^{-3} \) and \( n_i = 10^{10} \, \text{cm}^{-3} \). (b) Quasi-Fermi levels for electrons and holes if \( 10^{13} \, \text{cm}^{-3} \) excess carriers are present.

The quasi-Fermi energy level for the minority carrier holes is significantly different from the Fermi level and illustrates the fact that we have deviated from thermal equilibrium significantly. Since the electron concentration has increased, the quasi-Fermi level for electrons has moved slightly closer to the conduction band. The hole concentration has increased significantly so that the quasi-Fermi level for holes has moved much closer to the valence band. We will consider the quasi-Fermi energy levels again when we discuss forward-biased pn junctions.

## 6.5 EXCESS CARRIER LIFETIME

The rate at which excess electrons and holes recombine is an important characteristic of the semiconductor and influences many of the device characteristics, as we will see in later chapters. We considered recombination briefly at the beginning of this chapter and argued that the recombination rate is inversely proportional to the mean carrier lifetime. We have assumed up to this point that the mean carrier lifetime is simply a parameter of the semiconductor material.

We have been considering an ideal semiconductor in which electronic energy states do not exist within the forbidden-energy bandgap. This ideal effect is present in a perfect single-crystal material with an ideal periodic-potential function. In a real semiconductor material, defects occur within the crystal and disrupt the perfect periodic-potential function. If the density of these defects is not too great, the defects will create discrete electronic energy states within the forbidden-energy band. These allowed energy states may be the dominant effect in determining the mean carrier lifetime. The mean carrier lifetime may be determined from the Shockley–Read–Hall theory of recombination.

### 6.5.1 Shockley–Read–Hall Theory of Recombination

An allowed energy state, also called a trap, within the forbidden bandgap may act as a recombination center, capturing both electrons and holes with almost equal probability. This equal probability of capture means that the capture cross sections for
```

```markdown
# CHAPTER 6 Non-equilibrium Excess Carriers in Semiconductors

!Figure 6.16

**Figure 6.16** | The four basic trapping and emission processes for the case of an acceptor-type trap.

- **Process 1**: Electron capture
- **Process 2**: Electron emission
- **Process 3**: Hole capture
- **Process 4**: Hole emission

The Shockley–Read–Hall theory of recombination assumes that a single recombination center, or trap, exists at an energy \( E_t \) within the bandgap. There are four basic processes, shown in Figure 6.16, that may occur at this single trap. We will assume that the trap is an acceptor-type trap; that is, it is negatively charged when it contains an electron and is neutral when it does not contain an electron.

The four basic processes are as follows:

- **Process 1**: The capture of an electron from the conduction band by an initially neutral empty trap.
- **Process 2**: The inverse of process 1—the emission of an electron that is initially occupying a trap level back into the conduction band.
- **Process 3**: The capture of a hole from the valence band by a trap containing an electron. (Or we may consider the process to be the emission of an electron from the trap into the valence band.)
- **Process 4**: The inverse of process 3—the emission of a hole from a neutral trap into the valence band. (Or we may consider this process to be the capture of an electron from the valence band.)

In process 1, the rate at which electrons from the conduction band are captured by the traps is proportional to the density of electrons in the conduction band and...
```

```markdown
proportional to the density of empty trap states. We can then write the electron capture rate as

\[
R_{cn} = C_n N_t [1 - f_t(E_t)] n
\]

(6.86)

where

- \( R_{cn} \) = capture rate (#/cm\(^3\)·s)
- \( C_n \) = constant proportional to electron-capture cross section
- \( N_t \) = total concentration of trapping centers
- \( n \) = electron concentration in the conduction band
- \( f_t(E_t) \) = Fermi function at the trap energy

The Fermi function at the trap energy is given by

\[
f_t(E) = \frac{1}{1 + \exp\left(\frac{E_t - E_f}{kT}\right)}
\]

(6.87)

which is the probability that a trap will contain an electron. The function \([1 - f_t(E_t)]\) is the probability that the trap is empty. In Equation (6.87), we have assumed that the degeneracy factor is 1, which is the usual approximation made in this analysis. However, if a degeneracy factor is included, it will eventually be absorbed in other constants later in the analysis.

For process 2, the rate at which electrons are emitted from filled traps back into the conduction band is proportional to the number of filled traps, so that

\[
R_{en} = E_n N_t f_t(E_t)
\]

(6.88)

where

- \( R_{en} \) = emission rate (#/cm\(^3\)·s)
- \( E_n \) = constant
- \( f_t(E_t) \) = probability that the trap is occupied

In thermal equilibrium, the rate of electron capture from the conduction band and the rate of electron emission back into the conduction band must be equal. Then

\[
R_{cn} = R_{en}
\]

(6.89)

so that

\[
E_n N_t f_t(E_t) = C_n N_t [1 - f_{t0}(E_t)] n_0
\]

(6.90)

where \( f_{t0} \) denotes the thermal-equilibrium Fermi function. Note that, in thermal equilibrium, the value of the electron concentration in the capture rate term is the equilibrium value \( n_0 \). Using the Boltzmann approximation for the Fermi function, we can find \( E_n \) in terms of \( C_n \) as

\[
E_n = n' C_n
\]

(6.91)
```

where \( n' \) is defined as

\[
n' = N_c \exp \left[ \frac{-(E_c - E_t)}{kT} \right]
\]

(6.92)

The parameter \( n' \) is equivalent to an electron concentration that would exist in the conduction band if the trap energy \( E_t \) coincided with the Fermi energy \( E_f \).

In nonequilibrium, excess electrons exist, so that the net rate at which electrons are captured from the conduction band is given by

\[
R_n = R_{cn} - R_{en}
\]

(6.93)

which is just the difference between the capture rate and the emission rate. Combining Equations (6.86) and (6.88) with (6.93) gives

\[
R_n = [C_n N_c (1 - f_r(E_t)) n] - [E_n N_t f_r(E_t)]
\]

(6.94)

We may note that, in this equation, the electron concentration \( n \) is the total concentration, which includes the excess electron concentration. The remaining constants and terms in Equation (6.94) are the same as defined previously and the Fermi energy in the Fermi probability function needs to be replaced by the quasi-Fermi energy for electrons. The constants \( E_n \) and \( C_n \) are related by Equation (6.91), so the net recombination rate can be written as

\[
R_n = C_n N_t [n(1 - f_r(E_t)) - n'f_r(E_t)]
\]

(6.95)

If we consider processes 3 and 4 in the recombination theory, the net rate at which holes are captured from the valence band is given by

\[
R_p = C_p N_v [pf_r(E_t) - p'(1 - f_r(E_t))]
\]

(6.96)

where \( C_p \) is a constant proportional to the hole capture rate, and \( p' \) is given by

\[
p' = N_v \exp \left[ \frac{-(E_t - E_v)}{kT} \right]
\]

(6.97)

In a semiconductor in which the trap density is not too large, the excess electron and hole concentrations are equal and the recombination rates of electrons and holes are equal. If we set Equation (6.95) equal to Equation (6.96) and solve for the Fermi function, we obtain

\[
f_r(E_t) = \frac{C_n n + C_p p'}{C_n(n + n') + C_p(p + p')}
\]

(6.98)

We may note that \( n' p' = n_i^2 \). Then, substituting Equation (6.98) back into either Equation (6.95) or (6.96) gives

\[
R_n = R_p = \frac{C_n C_p N_t (np - n_i^2)}{C_n(n + n') + C_p(p + p')} = R
\]

(6.99)

Equation (6.99) is the recombination rate of electrons and holes due to the recombination center at \( E = E_t \). If we consider thermal equilibrium, then \( np = n_0 p_0 = n_i^2 \), so

# 6.5 Excess Carrier Lifetime

that \( R_n = R_p = 0 \). Equation (6.99), then, is the recombination rate of excess electrons and holes.

Since \( R \) in Equation (6.99) is the recombination rate of the excess carriers, we may write

\[
R = \frac{\delta n}{\tau}
\]

(6.100)

where \(\delta n\) is the excess carrier concentration and \(\tau\) is the lifetime of the excess carriers.

## 6.5.2 Limits of Extrinsic Doping and Low Injection

We simplified the ambipolar transport equation, Equation (6.39), from a nonlinear differential equation to a linear differential equation by applying limits of extrinsic doping and low injection. We may apply these same limits to the recombination rate equation.

Consider an n-type semiconductor under low injection. Then

\[
n_0 \gg p_0, \quad n_0 \gg \delta p, \quad n_0 \gg n', \quad n_0 \gg p'
\]

where \(\delta p\) is the excess minority carrier hole concentration. The assumptions of \(n_0 \gg n'\) and \(n_0 \gg p'\) imply that the trap level energy is near midgap so that \(n'\) and \(p'\) are not too different from the intrinsic carrier concentration. With these assumptions, Equation (6.99) reduces to

\[
R = C_n N_i \delta p
\]

(6.101)

The recombination rate of excess carriers in the n-type semiconductor is a function of the parameter \(C_n\), which is related to the minority carrier hole capture cross section. The recombination rate, then, is a function of the minority carrier parameter in the same way that the ambipolar transport parameters reduced to their minority carrier values.

The recombination rate is related to the mean carrier lifetime. Comparing Equations (6.100) and (6.101), we may write

\[
R = \frac{\delta n}{\tau} = C_n N_i \delta p = \frac{\delta p}{\tau_{p0}}
\]

(6.102)

where

\[
\tau_{p0} = \frac{1}{C_n N_i}
\]

(6.103)

and where \(\tau_{p0}\) is defined as the excess minority carrier hole lifetime. If the trap concentration increases, the probability of excess carrier recombination increases; thus, the excess minority carrier lifetime decreases.

Similarly, if we have a strongly extrinsic p-type material under low injection, we can assume that

\[
p_0 \gg n_0, \quad p_0 \gg \delta n, \quad p_0 \gg n', \quad p_0 \gg p'
\]

The lifetime then becomes that of the excess minority carrier electron lifetime, or

\[
\tau_{n0} = \frac{1}{C_n N_t}
\]

(6.104)

Again note that for the n-type material, the lifetime is a function of \(C_p\), which is related to the capture rate of the minority carrier hole. And for the p-type material, the lifetime is a function of \(C_n\), which is related to the capture rate of the minority carrier electron. The excess carrier lifetime for an extrinsic material under low injection reduces to that of the minority carrier.

----

### EXAMPLE 6.8

**Objective:** Determine the excess carrier lifetime in an intrinsic semiconductor.

If we substitute the definitions of excess carrier lifetimes from Equations (6.103) and (6.104) into Equation (6.99), the recombination rate can be written as

\[
R = \frac{(np - n_i^2)}{\tau_{p0}(n + n_i^*) + \tau_{n0}(p + p_i^*)}
\]

(6.105)

Consider an intrinsic semiconductor containing excess carriers. Then \(n = n_i + \delta n\) and \(p = n_i + \delta n\). Also assume that \(n' = p' = n_i\).

#### Solution

Equation (6.105) now becomes

\[
R = \frac{2n_i \delta n + (\delta n)^2}{(2n_i + \delta n)\tau_p + \tau_{n0}}
\]

If we also assume very low injection, so that \(\delta n \ll 2n_i\), then we can write

\[
R = \frac{\delta n}{\tau_p + \tau_{n0}} = \frac{\delta n}{\tau}
\]

where \(\tau\) is the excess carrier lifetime. We see that \(\tau = \tau_p + \tau_{n0}\) in the intrinsic material.

#### Comment

The excess carrier lifetime increases as we change from an extrinsic to an intrinsic semiconductor.

#### EXERCISE PROBLEM

**Ex 6.8** Consider silicon at \(T = 300 \, K\) doped at concentrations of \(N_d = 10^{15} \, \text{cm}^{-3}\) and \(N_a = 0\). Assume that \(n' = p' = n_i\) in the excess carrier recombination rate equation and assume parameter values of \(\tau_{n0} = \tau_{p0} = 5 \times 10^{-7} \, s\). Calculate the recombination rate of excess carriers if \(\delta n = \delta p = 10^{14} \, \text{cm}^{-3}\).

----

Intuitively, we can see that the number of majority carriers that are available for recombining with excess minority carriers decreases as the extrinsic semiconductor becomes intrinsic. Since there are fewer carriers available for recombining in the intrinsic material, the mean lifetime of an excess carrier increases.

```markdown
# 6.6 Surface Effects

## 6.6.1 Surface States

In all previous discussions, we have implicitly assumed the semiconductors were infinite in extent; thus, we were not concerned with any boundary conditions at a semiconductor surface. In any real application of semiconductors, the material is not infinitely large and therefore surfaces do exist between the semiconductor and an adjacent medium.

When a semiconductor is abruptly terminated, the perfect periodic nature of the idealized single-crystal lattice ends abruptly at the surface. The disruption of the periodic-potential function results in allowed electronic energy states within the energy bandgap. In the previous section, we argued that simple defects in the semiconductor would create discrete energy states within the bandgap. The abrupt termination of the periodic potential at the surface results in a distribution of allowed energy states within the bandgap, shown schematically in Figure 6.17 along with the discrete energy states in the bulk semiconductor.

The Shockley–Read–Hall recombination theory shows that the excess minority carrier lifetime is inversely proportional to the density of trap states. We may argue that since the density of traps at the surface is larger than in the bulk, the excess minority carrier lifetime at the surface will be smaller than the corresponding lifetime in the bulk material. If we consider an extrinsic n-type semiconductor, for example, the recombination rate of excess carriers in the bulk, given by Equation (6.102), is

\[
R = \frac{\delta p}{\tau_{p0}} = \frac{\delta p_B}{\tau_{p0}}
\]

(6.106)

where \(\delta p_B\) is the concentration of excess minority carrier holes in the bulk material. We may write a similar expression for the recombination rate of excess carriers at the surface as

\[
R_s = \frac{\delta p_s}{\tau_{ps}}
\]

(6.107)

where \(\delta p_s\) is the excess minority carrier hole concentration at the surface and \(\tau_{ps}\) is the excess minority carrier hole lifetime at the surface.

!Figure 6.17

**Figure 6.17** | Distribution of surface states within the forbidden bandgap.
```

# Chapter 6: Nonequilibrium Excess Carriers in Semiconductors

!Figure 6.18  
**Figure 6.18** | Steady-state excess hole concentration versus distance from a semiconductor surface.

Assume that excess carriers are being generated at a constant rate throughout the entire semiconductor material. We showed that, in steady state, the generation rate is equal to the recombination rate for the case of a homogeneous, infinite semiconductor. Using this argument, the recombination rates at the surface and in the bulk material must be equal. Since \(\tau_{p0} < \tau_{pb}\), then the excess minority carrier concentration at the surface is smaller than the excess minority carrier concentration in the bulk region, or \(\delta p_s < \delta p_b\). Figure 6.18 shows an example of the excess carrier concentration plotted as a function of distance from the semiconductor surface.

## Example 6.9

**Objective:** Determine the steady-state excess carrier concentration as a function of distance from the surface of a semiconductor.

Consider Figure 6.18, in which the surface is at \(x = 0\). Assume that in the n-type semiconductor \(\delta p_b = 10^{14} \, \text{cm}^{-3}\) and \(\tau_p = 10^{-6} \, \text{s}\) in the bulk, and \(\tau_{p0} = 10^{-7} \, \text{s}\) at the surface. Assume zero applied electric field and let \(D_p = 10 \, \text{cm}^2/\text{s}\).

### Solution

From Equations (6.106) and (6.107), we have

\[
\frac{\delta p_b}{\tau_{p0}} = \frac{\delta p_f}{\tau_{p0b}}
\]

so that

\[
\delta p_f = \delta p_b \left( \frac{\tau_{p0b}}{\tau_{p0}} \right) = (10^{14}) \left( \frac{10^{-7}}{10^{-6}} \right) = 10^{13} \, \text{cm}^{-3}
\]

From Equation (6.56), we can write

\[
D_p \frac{d^2 (\delta p)}{dx^2} + g' - \frac{\delta p}{\tau_{p0}} = 0 \tag{6.108}
\]

The generation rate can be determined from the steady-state conditions in the bulk, or

\[
g' = \frac{\delta p_b}{\tau_p} = \frac{10^{14}}{10^{-6}} = 10^{20} \, \text{cm}^{-3} \cdot \text{s}^{-1}
\]

## 6.6 Surface Effects

The solution to Equation (6.107) is of the form

\[
\delta p(x) = g' \tau_{p0} + Ae^{x/L_p} + Be^{-x/L_p}
\]

(6.109)

As \( x \to +\infty \), \(\delta p(x) = \delta p = g' \tau_{p0} = 10^{14} \, \text{cm}^{-3}\), which implies that \( A = 0 \). At \( x = 0 \), we have

\[
\delta p(0) = \delta p_s = 10^{14} + B = 10^{13} \, \text{cm}^{-3}
\]

so that \( B = -9 \times 10^{13} \). The entire solution for the minority carrier hole concentration as a function of distance from the surface is

\[
\delta p(x) = 10^{14} (1 - 0.9 e^{-x/L_p})
\]

where

\[
L_p = \sqrt{D_p \tau_{p0}} = \sqrt{(10)(10^{-6})} = 31.6 \, \mu \text{m}
\]

### Comment

The excess carrier concentration is smaller at the surface than in the bulk.

### EXERCISE PROBLEM

**Ex 6.9** (a) Repeat Example 6.9 for the case when \(\tau_{p0} = 0\). (b) What is the excess hole concentration at \( x = 0 \)? (c) For this particular case, what is the recombination rate of excess carriers at the surface?

\[
[\omega = x] (y) 0 = (0)\delta p (q) \frac{q}{\tau_{p0}} - 1) \psi_{L_p} \beta = (\chi \delta p) \text{SU}
\]

## 6.6.2 Surface Recombination Velocity

A gradient in the excess carrier concentration exists near the surface as shown in Figure 6.18; excess carriers from the bulk region diffuse toward the surface where they recombine. This diffusion toward the surface can be described by

\[
-D_p \left[ \hat{n} \cdot \left( \frac{d \delta p}{d x} \right) \right]_{\text{surf}} = s \delta p_{\text{surf}}
\]

(6.110)

where each side of the equation is evaluated at the surface. The parameter \(\hat{n}\) is the unit outward vector normal to the surface. Using the geometry of Figure 6.18, \(d(\delta p)/dx\) is a positive quantity and \(\hat{n}\) is negative, so that the parameter \(s\) is a positive quantity.

A dimensional analysis of Equation (6.110) shows that the parameter \(s\) has units of cm/s, or velocity. The parameter \(s\) is called the **surface recombination velocity**. If the excess concentrations at the surface and in the bulk region were equal, then the gradient term would be zero and the surface recombination velocity would be zero. As the excess concentration at the surface becomes smaller, the gradient term becomes larger, and the surface recombination velocity increases. The surface recombination velocity gives some indication of the surface characteristics as compared with the bulk region.

## CHAPTER 6 Non-equilibrium Excess Carriers in Semiconductors

Equation (6.110) may be used as a boundary condition to the general solution given by Equation (6.109) in Example 6.8. Using Figure 6.18, we have that at \( \bar{t} = -1 \), and Equation (6.110) becomes

\[
D_p \frac{d(\delta p)}{dx} \bigg|_{\text{surf}} = s \delta p|_{\text{surf}}
\]

(6.111)

We have argued that the coefficient \( A \) is zero in Equation (6.109). Then, from Equation (6.109), we can write that

\[
\delta p_{\text{surf}} = \delta p(0) = g' \tau_{p0} + B
\]

(6.112a)

and

\[
\frac{d(\delta p)}{dx} \bigg|_{\text{surf}} = \frac{d(\delta p)}{dx} \bigg|_{x=0} = -\frac{B}{L_p}
\]

(6.112b)

Substituting Equations (6.112a) and (6.112b) into Equation (6.111) and solving for the coefficient \( B \), we obtain

\[
B = -\frac{s g' \tau_{p0}}{(D_p/L_p) + s}
\]

(6.113)

The excess minority carrier hole concentration can then be written as

\[
\delta p(x) = g' \tau_{p0} \left[ 1 - \frac{s L_p e^{-x/L_p}}{D_p + s L_p} \right]
\]

(6.114)

----

### EXAMPLE 6.10

**Objective:** Determine the value of surface recombination velocity corresponding to the parameters given in Example 6.9.

From Example 6.9, we have that \( g' \tau_{p0} = 10^{14} \, \text{cm}^{-3} \), \( D_p = 10 \, \text{cm}^2/\text{s} \), \( L_p = 31.6 \, \mu\text{m} \), and \( \delta p(0) = 10^{13} \, \text{cm}^{-3} \).

**Solution**

Writing Equation (6.114) at the surface, we have

\[
\delta p(0) = g' \tau_{p0} \left[ 1 - \frac{s}{(D_p/L_p) + s} \right]
\]

Solving for the surface recombination velocity, we find that

\[
s = \frac{D_p}{L_p} \left( \frac{g' \tau_{p0}}{\delta p(0)} - 1 \right)
\]

which becomes

\[
s = \frac{10}{31.6 \times 10^{-4}} \left[ \frac{10^{14}}{10^{13}} - 1 \right] = 2.85 \times 10^4 \, \text{cm/s}
\]

## Comment

This example shows that a surface recombination velocity of approximately \( s = 3 \times 10^4 \, \text{cm/s} \) could seriously degrade the performance of semiconductor devices, such as solar cells, since these devices tend to be fabricated close to a surface.

## EXERCISE PROBLEM

**Ex 6.10**  
(a) Using Equation (6.114), determine \(\delta p(x)\) for (i) \(s = \infty\) and (ii) \(s = 0\).  
(b) What does (i) an infinite surface recombination velocity \((s = \infty)\) and (ii) a zero surface recombination velocity \((s = 0)\) imply?

\[
\begin{align*}
\text{(i)} & \quad \delta p(x) \approx \delta p(0) \left( q \right) \cdot s_{0,L} = \left( x \delta q \right) \cdot \left( q \right) \cdot s_{0,L} = \left( x \delta q \right) \cdot \left( q \right) \cdot \left( \tau_{n,p} - 1 \right) \cdot s_{0,L} = \left( x \delta q \right) \cdot \left( q \right) \cdot \text{usv} \\
\end{align*}
\]

In the above example, the surface influences the excess carrier concentration to the extent that, even at a distance of \( L_p = 31.6 \, \mu \text{m} \) from the surface, the excess carrier concentration is only two-thirds of the value in the bulk. We will see in later chapters that device performance is dependent in large part on the properties of excess carriers.

## 6.7 | SUMMARY

- The processes of excess electron and hole generation and recombination were discussed. The excess carrier generation rate and recombination rate were defined.
- Excess electrons and holes do not move independently of each other, but move together. This common movement is called ambipolar transport.
- The ambipolar transport equation was derived and limits of low injection and extrinsic doping were applied to the coefficients. Under these conditions, the excess electrons and holes diffuse and drift together with the characteristics of the minority carrier, a result that is fundamental to the behavior of semiconductor devices.
- The concept of excess carrier lifetime was developed.
- Examples of excess carrier behavior as a function of time, as a function of space, and as a function of both time and space were examined.
- The quasi-Fermi level for electrons and the quasi-Fermi level for holes were defined. The degree of quasi-Fermi level splitting is a measure of departure from thermal equilibrium.
- The Shockley–Read–Hall theory of recombination was considered. Expressions for the excess minority carrier lifetime were developed. Generation and recombination of excess carriers increase as a result of traps in a semiconductor.
- The effect of a semiconductor surface influences the behavior of excess electrons and holes. The surface recombination velocity was defined.

## GLOSSARY OF IMPORTANT TERMS

- **ambipolar diffusion coefficient**  
  The effective diffusion coefficient of excess carriers.

- **ambipolar mobility**  
  The effective mobility of excess carriers.

- **ambipolar transport**  
  The process whereby excess electrons and holes diffuse, drift, and recombine with the same effective diffusion coefficient, mobility, and lifetime.

# Chapter 6: Nonequilibrium Excess Carriers in Semiconductors

### Key Terms

- **ambipolar transport equation**: The equation describing the behavior of excess carriers as a function of time and space coordinates.

- **carrier generation**: The process of elevating electrons from the valence band into the conduction band, creating an electron–hole pair.

- **carrier recombination**: The process whereby an electron “falls” into an empty state in the valence band (a hole) so that an electron–hole pair is annihilated.

- **excess carriers**: The term describing both excess electrons and excess holes.

- **excess electrons**: The concentration of electrons in the conduction band over and above the thermal-equilibrium concentration.

- **excess holes**: The concentration of holes in the valence band over and above the thermal-equilibrium concentration.

- **excess minority carrier lifetime**: The average time that an excess minority carrier exists before it recombines.

- **generation rate**: The rate (#/cm\(^3\)·s) at which electron–hole pairs are created.

- **low-level injection**: The condition in which the excess carrier concentration is much smaller than the thermal-equilibrium majority carrier concentration.

- **minority carrier diffusion length**: The average distance a minority carrier diffuses before recombining; a parameter equal to \(\sqrt{D \tau}\) where \(D\) and \(\tau\) are the minority carrier diffusion coefficient and lifetime, respectively.

- **quasi-Fermi level**: The quasi-Fermi level for electrons and the quasi-Fermi level for holes relate the nonequilibrium electron and hole concentrations, respectively, to the intrinsic carrier concentration and the intrinsic Fermi level.

- **recombination rate**: The rate (#/cm\(^3\)·s) at which electron–hole pairs recombine.

- **surface recombination velocity**: A parameter that relates the gradient of the excess carrier concentration at a surface to the surface concentration of excess carriers.

- **surface states**: The electronic energy states that exist within the bandgap at a semiconductor surface.

### CHECKPOINT

After studying this chapter, the reader should have the ability to:

- Describe the concept of excess carrier generation and recombination.
- Describe the concept of an excess carrier lifetime.
- Describe how the time-dependent diffusion equations for holes and electrons are derived.
- Describe how the ambipolar transport equation is derived.
- Understand the consequence of the coefficients in the ambipolar transport equation reducing to the minority carrier values under low injection and extrinsic semiconductors.
- Apply the ambipolar transport equation to various problems.
- Understand the concept of the dielectric relaxation time constant and what it means.
- Calculate the quasi-Fermi levels for electrons and holes.
- Calculate the excess carrier recombination rate for a given concentration of excess carriers.
- Understand the effect of a surface on the excess carrier concentrations.

# REVIEW QUESTIONS

1. Why are the electron generation rate and recombination rate equal in thermal equilibrium?

2. Define the excess carrier recombination rate in terms of excess carrier concentration and lifetime.

3. Explain how the density of holes, for example, can change as a result of a change in the flux of particles.

4. Why is the general ambipolar transport equation nonlinear?

5. Explain qualitatively why a pulse of excess electrons and holes would move together in the presence of an applied electric field.

6. Explain qualitatively why the excess carrier lifetime reduces to that of the minority carrier under low injection.

7. What is the time dependence of the density of excess carriers when the generation rate becomes zero?

8. In the presence of an external force, why doesn’t the density of excess carriers continue to increase with time?

9. When a concentration of one type of excess carrier is suddenly created in a semiconductor, what is the mechanism by which the net charge density quickly becomes zero?

10. State the definition of the quasi-Fermi level for electrons. Repeat for holes.

11. Explain why the presence of traps in a semiconductor increases the recombination rate of excess carriers.

12. Why, in general, is the concentration of excess carriers less at the surface of a semiconductor than in the bulk?

# PROBLEMS

*(Note: Use the semiconductor parameters listed in Appendix B if they are not specifically given in a problem. Assume T = 300 K.)*

## Section 6.1 Carrier Generation and Recombination

**6.1** Consider silicon at \( T = 300 \, \text{K} \) that is doped with donor impurity atoms to a concentration of \( N_d = 5 \times 10^{15} \, \text{cm}^{-3} \). The excess carrier lifetime is \( 2 \times 10^{-7} \, \text{s} \).  
(a) Determine the thermal equilibrium recombination rate of holes.  
(b) Excess carriers are generated such that \( \delta n = \delta p = 10^{14} \, \text{cm}^{-3} \). What is the recombination rate of holes for this condition?

**6.2** GaAs, at \( T = 300 \, \text{K} \), is uniformly doped with acceptor impurity atoms to a concentration of \( N_a = 2 \times 10^{16} \, \text{cm}^{-3} \). Assume an excess carrier lifetime of \( 5 \times 10^{-7} \, \text{s} \).  
(a) Determine the electron-hole recombination rate if the excess electron concentration is \( \delta n = 5 \times 10^{14} \, \text{cm}^{-3} \).  
(b) Using the results of part (a), what is the lifetime of holes?

**6.3** An n-type silicon sample contains a donor concentration of \( N_d = 10^{16} \, \text{cm}^{-3} \). The minority carrier hole lifetime is found to be \( \tau_p = 20 \, \mu\text{s} \).  
(a) What is the lifetime of the majority carrier electrons?  
(b) Determine the thermal-equilibrium generation rate for electrons and holes in this material.  
(c) Determine the thermal-equilibrium recombination rate for electrons and holes in this material.

