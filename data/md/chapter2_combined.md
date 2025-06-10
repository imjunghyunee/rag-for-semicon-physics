# Chapter 2: Introduction to Quantum Mechanics

The goal of this text is to help readers understand the operation and characteristics of semiconductor devices. Ideally, we would like to begin discussing these devices immediately. However, in order to understand the current-voltage characteristics, we need some knowledge of electron behavior in a semiconductor when the electron is subjected to various potential functions.

The motion of large objects, such as planets and satellites, can be predicted to a high degree of accuracy using classical theoretical physics based on Newton’s laws of motion. But certain experimental results, involving electrons and light waves, appear to be inconsistent with classical physics. However, these experimental results can be predicted using the principles of quantum mechanics. The behavior and characteristics of electrons in a semiconductor can be described by the formulation of quantum mechanics called wave mechanics. The essential elements of this wave mechanics, using Schrödinger’s wave equation, are presented in this chapter.

The goal of this chapter is to provide a brief introduction to quantum mechanics so that readers gain an understanding of and become comfortable with the analysis techniques. This introductory material forms the basis of semiconductor physics.

## 2.0 | Preview

In this chapter, we will:

- Discuss a few basic principles of quantum mechanics that apply to semiconductor device physics.
- State Schrödinger’s wave equation and discuss the physical meaning of the wave function.
- Consider the application of Schrödinger’s wave equation to various potential functions to determine some of the fundamental properties of electron behavior in a crystal.

## 2.1 Principles of Quantum Mechanics

Before we delve into the mathematics of quantum mechanics, there are three principles we need to consider: the principle of energy quanta, the wave–particle duality principle, and the uncertainty principle.

### 2.1.1 Energy Quanta

One experiment that demonstrates an inconsistency between experimental results and the classical theory of light is called the photoelectric effect. If monochromatic light is incident on a clean surface of a material, then under certain conditions, electrons (photoelectrons) are emitted from the surface. According to classical physics, if the intensity of the light is large enough, the work function of the material will be overcome and an electron will be emitted from the surface independent of the incident frequency. This result is not observed. The observed effect is that at a constant incident intensity, the maximum kinetic energy of the photoelectron varies linearly with frequency with a limiting frequency \( \nu = \nu_0 \), below which no photoelectron is produced. This result is shown in Figure 2.1. If the incident intensity varies at a constant frequency, the rate of photoelectron emission changes, but the maximum kinetic energy remains the same.

Planck postulated in 1900 that thermal radiation is emitted from a heated surface in discrete packets of energy called quanta. The energy of these quanta is given by \( E = h\nu \) where \( \nu \) is the frequency of the radiation and \( h \) is a constant now known as Planck’s constant (\( h = 6.625 \times 10^{-34} \, \text{J·s} \)). Then in 1905, Einstein interpreted the photoelectric results by suggesting that the energy in a light wave is also contained in quanta.

!Figure 2.1

**Figure 2.1** (a) The photoelectric effect and (b) the maximum kinetic energy of the photoelectron as a function of incident frequency.

- **(a)** Incident monochromatic light on a material results in photoelectron kinetic energy \( T \).
- **(b)** Graph showing maximum kinetic energy \( T_{\text{max}} \) versus frequency \( \nu \).

## 2.1 Principles of Quantum Mechanics

In discrete packets or bundles, the particle-like packet of energy is called a **photon**, whose energy is also given by \( E = h\nu \). A photon with sufficient energy, then, can knock an electron from the surface of the material. The minimum energy required to remove an electron is called the **work function** of the material and any excess photon energy goes into the kinetic energy of the photoelectron. This result was confirmed experimentally as demonstrated in Figure 2.1. The photoelectric effect shows the discrete nature of the photon and demonstrates the particle-like behavior of the photon.

The maximum kinetic energy of the photoelectron can be written as

\[
T = \frac{1}{2} mv^2 = h\nu - \Phi = h\nu - h\nu_0 \quad (\nu \geq \nu_0)
\]

\[
\Phi = h\nu_0
\]

where \( h\nu \) is the incident photon energy and \( \Phi = h\nu_0 \) is the minimum energy, or work function, required to remove an electron from the surface.

### Objective

Calculate the photon energy corresponding to a particular wavelength.

**Example 2.1**

Consider an x-ray with a wavelength of \( \lambda = 0.708 \times 10^{-8} \) cm.

#### Solution

The energy is

\[
E = h\nu = \frac{hc}{\lambda} = \frac{(6.625 \times 10^{-34})(3 \times 10^{10})}{0.708 \times 10^{-8}} = 2.81 \times 10^{-15} \, \text{J}
\]

This value of energy may be given in the more common unit of electron-volt (see Appendix D). We have

\[
E = \frac{2.81 \times 10^{-15}}{1.6 \times 10^{-19}} = 1.75 \times 10^4 \, \text{eV}
\]

#### Comment

The reciprocal relation between photon energy and wavelength is demonstrated: A large energy corresponds to a short wavelength.

#### Exercise Problem

**Ex 2.1** Determine the energy (in eV) of a photon having a wavelength of (a) \( \lambda = 100 \, \text{Å} \) and (b) \( \lambda = 4500 \, \text{Å} \).

----

### 2.1.2 Wave–Particle Duality

We have seen in the last section that light waves, in the photoelectric effect, behave as if they are particles. The particle-like behavior of electromagnetic waves was also instrumental in the explanation of the **Compton effect**. In this experiment, an x-ray beam was incident on a solid. A portion of the x-ray beam was deflected and the frequency of the deflected wave had shifted compared with the incident wave. The observed change in frequency and the deflected angle corresponded exactly to the expected results of a "billiard ball" collision between an x-ray quanta, or photon, and an electron in which both energy and momentum are conserved.

# Introduction to Quantum Mechanics

In 1924, de Broglie postulated the existence of matter waves. He suggested that since waves exhibit particle-like behavior, particles should be expected to show wave-like properties. The hypothesis of de Broglie was the existence of a **wave-particle duality principle**. The momentum of a photon is given by

\[
p = \frac{h}{\lambda}
\]

where \(\lambda\) is the wavelength of the light wave. Then, de Broglie hypothesized that the wavelength of a particle can be expressed as

\[
\lambda = \frac{h}{p}
\]

where \(p\) is the momentum of the particle and \(\lambda\) is known as the **de Broglie wavelength** of the matter wave.

The wave nature of electrons has been tested in several ways. In one experiment by Davisson and Germer in 1927, electrons from a heated filament were accelerated at normal incidence onto a single crystal of nickel. A detector measured the scattered electrons as a function of angle. Figure 2.2 shows the experimental setup and Figure 2.3 shows the results. The existence of a peak in the density of scattered electrons can be explained as a constructive interference of waves scattered by the periodic atoms in the planes of the nickel crystal. The angular distribution is very similar to an interference pattern produced by light diffracted from a grating.

In order to gain some appreciation of the frequencies and wavelengths involved in the wave-particle duality principle, Figure 2.4 shows the electromagnetic frequency spectrum. We see that a wavelength of 72.7 Å obtained in the next example is in the ultraviolet range. Typically, we will be considering wavelengths in the

!Exemplo do comportamento ondulatório dos elétrons: 1) MeV; 2) FIB.

----

!Figure 2.2

**Figure 2.2** | Experimental arrangement of the Davisson–Germer experiment.

----

!Figure 2.3

**Figure 2.3** | Scattered electron flux as a function of scattering angle for the Davisson–Germer experiment.

## 2.1 Principles of Quantum Mechanics

!Figure 2.4 The electromagnetic frequency spectrum.

| λ (μm) | 0.001 | 0.20 | 0.39 | 0.49 | 0.55 | 0.577 | 0.62 | 0.77 | 1.5 | 6.0 | 40 | 1000 |
|--------|-------|------|------|------|------|-------|------|------|-----|-----|----|------|
|        | Extreme | Far | Near | Violet | Blue | Yellow | Orange | Red | Near | Medium | Far | Extreme |
|        |         |     |      | 0.455 | 0.492 | 0.527 | 0.622 | 0.770 |     |     |    |      |

| Wavelength (m) | 1 fm | 1 pm | 1 Å | 1 nm | 1 μm | 1 mm | 1 m | 1 km | 1 Mm |
|----------------|------|------|-----|------|------|------|-----|------|------|
|                | 10^-15 | 10^-12 | 10^-10 | 10^-9 | 10^-6 | 10^-3 | 1 | 10^3 | 10^6 |

| Frequency (Hz) | 1 THz | 1 GHz | 1 MHz | 1 kHz | 1 Hz |
|----------------|-------|-------|-------|-------|-----|
|                | 10^12 | 10^9 | 10^6 | 10^3 | 1 |

ultraviolet and visible range. These wavelengths are very short compared to the usual radio spectrum range.¹

### Objective

Calculate the de Broglie wavelength of a particle.

Consider an electron traveling at a velocity of \(10^7 \, \text{cm/s} = 10^5 \, \text{m/s}\).

### Solution

The momentum is given by

\[
p = mv = (9.11 \times 10^{-31})(10^5) = 9.11 \times 10^{-26} \, \text{kg-m/s}
\]

Then, the de Broglie wavelength is

\[
\lambda = \frac{h}{p} = \frac{6.625 \times 10^{-34}}{9.11 \times 10^{-26}} = 7.27 \times 10^{-9} \, \text{m}
\]

or

\[
\lambda = 72.7 \, \text{Å}
\]

### Comment

This calculation shows the order of magnitude of the de Broglie wavelength for a "typical" electron.

### EXERCISE PROBLEM

**Ex 2.2** (a) An electron has a kinetic energy of 12 meV. Determine the de Broglie wavelength in Å. (b) A particle with mass \(2.2 \times 10^{-31} \, \text{kg}\) has a de Broglie wavelength of 112 Å. Determine the momentum and kinetic energy of the particle. \([λ^2.01 \times L6 \text{t} = 3 \text{su} \cdot 8 \text{q} \cdot 01 \leq 516} \text{s} = \text{d} (q \cdot \text{Y} \cdot 11} = \text{Y} \cdot \text{SUY}]\)

----

¹An electron microscope is a microscope that produces a magnified image of a specimen. The electron microscope has a magnification approximately 1000 times that of an optical microscope, because the electrons have wavelengths on the order of 100,000 times shorter than the light waves.

## 2.1.3 The Uncertainty Principle

The Heisenberg uncertainty principle, given in 1927, also applies primarily to very small particles and states that we cannot describe with absolute accuracy the behavior of these subatomic particles. The uncertainty principle describes a fundamental relationship between conjugate variables, including position and momentum and also energy and time.

The first statement of the uncertainty principle is that it is impossible to simultaneously describe with absolute accuracy the position and momentum of a particle. If the uncertainty in the momentum is \(\Delta p\) and the uncertainty in the position is \(\Delta x\), then the uncertainty principle is stated as:

\[
\Delta p \Delta x \geq \hbar
\]

(2.4)

where \(\hbar\) is defined as \(\hbar = h/2\pi = 1.054 \times 10^{-34} \, \text{J-s}\) and is called a modified Planck’s constant. This statement may be generalized to include angular position and angular momentum.

The second statement of the uncertainty principle is that it is impossible to simultaneously describe with absolute accuracy the energy of a particle and the instant of time the particle has this energy. Again, if the uncertainty in the energy is given by \(\Delta E\) and the uncertainty in the time is given by \(\Delta t\), then the uncertainty principle is stated as:

\[
\Delta E \Delta t \geq \hbar
\]

(2.5)

One way to visualize the uncertainty principle is to consider the simultaneous measurement of position and momentum, and the simultaneous measurement of energy and time. The uncertainty principle implies that these simultaneous measurements are in error to a certain extent. However, the modified Planck’s constant \(\hbar\) is very small; the uncertainty principle is only significant for subatomic particles. We must keep in mind nevertheless that the uncertainty principle is a fundamental statement and does not deal only with measurements.

One consequence of the uncertainty principle is that we cannot, for example, determine the exact position of an electron. We will, instead, determine the probability of finding an electron at a particular position. In later chapters, we will develop a

----

*In some texts, the uncertainty principle is stated as \(\Delta p \Delta x \approx \hbar/2\). We are interested here in the order of magnitude and will not be concerned with small differences.*

probability density function that will allow us to determine the probability that an electron has a particular energy. So in describing electron behavior, we will be dealing with probability functions.

----

### TEST YOUR UNDERSTANDING

**TYU 2.1**  
The uncertainty in position of an electron is \(\Delta x = 8 \, \text{Å}\).  
(a) Determine the minimum uncertainty in momentum.  
(b) If the nominal value of momentum is \(p = 1.2 \times 10^{-23} \, \text{kg·m/s}\), determine the corresponding uncertainty in kinetic energy. (The uncertainty in kinetic energy can be found from \(\Delta E = (dE/dp)\Delta p = (p/\Delta p)m [\Delta x \Delta p] = 3Y (q) s\)u-3\!<0! X 8!<1 = dy (0) s\)u\])

**TYU 2.2**  
(a) A proton’s energy is measured with an uncertainty of 0.8 eV. Determine the minimum uncertainty in time over which this energy is measured.  
(b) Repeat part (a) for an electron. [(0) u\)d s\) oues (q) s\) s\-0!X<7<8 = 7Y (0) s\)u\]

----

## 2.2.1 SCHRODINGER’S WAVE EQUATION

The various experimental results involving electromagnetic waves and particles, which could not be explained by classical laws of physics, showed that a revised formulation of mechanics was required. Schrödinger, in 1926, provided a formulation called wave mechanics, which incorporated the principles of quanta introduced by Planck, and the wave–particle duality principle introduced by de Broglie. On the basis of wave–particle duality principle, we will describe the motion of electrons in a crystal by wave theory. This wave theory is described by Schrödinger’s wave equation.

### 2.2.1 The Wave Equation

The one-dimensional, nonrelativistic Schrödinger’s wave equation is given by

\[
-\frac{\hbar^2}{2m} \frac{\partial^2 \Psi(x, t)}{\partial x^2} + V(x)\Psi(x, t) = j\hbar \frac{\partial \Psi(x, t)}{\partial t}
\]

(2.6)

where \(\Psi(x, t)\) is the wave function, \(V(x)\) is the potential function assumed to be independent of time, \(m\) is the mass of the particle, and \(j\) is the imaginary constant \(\sqrt{-1}\). There are theoretical arguments that justify the form of Schrödinger’s wave equation, but the equation is a basic postulate of quantum mechanics. The wave function \(\Psi(x, t)\) will be used to describe the behavior of the system and, mathematically, \(\Psi(x, t)\) can be a complex quantity.

We may determine the time-dependent portion of the wave function and the position-dependent, or time-independent, portion of the wave function by using the technique of separation of variables. Assume that the wave function can be written in the form

\[
\Psi(x, t) = \psi(x)\phi(t)
\]

(2.7)

# Chapter 2: Introduction to Quantum Mechanics

where \( \psi(x) \) is a function of the position \( x \) only and \( \Phi(t) \) is a function of time \( t \) only. Substituting this form of the solution into Schrödinger’s wave equation, we obtain

\[
-\frac{\hbar^2}{2m} \Phi(x) \frac{\partial^2 \psi(x)}{\partial x^2} + V(x) \psi(x) \Phi(t) = j\hbar \psi(x) \frac{\partial \Phi(t)}{\partial t}
\]

(2.8)

If we **divide by the total wave function**, Equation (2.8) becomes

\[
-\frac{\hbar^2}{2m} \frac{1}{\psi(x)} \frac{\partial^2 \psi(x)}{\partial x^2} + V(x) = j\hbar \cdot \frac{1}{\Phi(t)} \cdot \frac{\partial \Phi(t)}{\partial t}
\]

(2.9)

Since the left side of Equation (2.9) is a function of position \( x \) only and the right side of the equation is a function of time \( t \) only, each side of this equation must be equal to a constant. We will denote this separation of variables constant by \( \eta \).

The time-dependent portion of Equation (2.9) is then written as

\[
\eta = j\hbar \cdot \frac{1}{\Phi(t)} \cdot \frac{\partial \Phi(t)}{\partial t}
\]

(2.10)

where again the parameter \( \eta \) is called a separation constant. **The solution of Equation (2.10) can be written in the form**

\[
\Phi(t) = e^{-j\eta t/\hbar}
\]

(2.11a)

The form of this solution is the classical exponential form of a sinusoidal wave. We have that \( E = h\nu \) or \( E = h\omega/2\pi \). Then \( \omega = \eta/\hbar = E/\hbar \) so that the separation constant is equal to the total energy \( E \) of the particle.

We can then write

\[
\Phi(t) = e^{-jEt/\hbar} = e^{-j\omega t}
\]

(2.11b)

We see that \( \omega = E/\hbar \) and is the radian or angular frequency of the sinusoidal wave.

The time-independent portion of Schrödinger’s wave equation can now be written from Equation (2.9) as

\[
-\frac{\hbar^2}{2m} \cdot \frac{1}{\psi(x)} \cdot \frac{\partial^2 \psi(x)}{\partial x^2} + V(x) = E
\]

(2.12)

where the separation constant is the total energy \( E \) of the particle. Equation (2.12) may be written as

\[
\frac{\partial^2 \psi(x)}{\partial x^2} + \frac{2m}{\hbar^2} (E - V(x)) \psi(x) = 0
\]

(2.13)

where again \( m \) is the mass of the particle, \( V(x) \) is the potential experienced by the particle, and \( E \) is the total energy of the particle. **This time-independent Schrödinger’s wave equation can also be justified on the basis of the classical wave equation as shown in Appendix E.** The pseudo-derivation in the appendix is a simple approach but shows the plausibility of the time-independent Schrödinger’s equation.

## 2.2.2 Physical Meaning of the Wave Function

We are ultimately trying to use the wave function \( \Psi(x, t) \) to describe the behavior of an electron in a crystal. The function \( \Psi(x, t) \) is a wave function, so it is reasonable to...

## 2.2 Schrödinger's Wave Equation

ask what the relation is between the function and the electron. The total wave function is the product of the position-dependent, or time-independent, function and the time-dependent function. From Equation (2.7) we have

\[
\Psi(x, t) = \psi(x)\phi(t) = \psi(x)e^{-iEt/\hbar} = \psi(x)e^{-i\omega t}
\]

(2.14)

Since the total wave function \(\Psi(x, t)\) is a complex function, it cannot by itself represent a real physical quantity.

Max Born postulated in 1926 that the function \(|\Psi(x, t)|^2 \, dx\) is the probability of finding the particle between \(x\) and \(x + dx\) at a given time, or that \(|\Psi(x, t)|^2\) is a probability density function. We have that

\[
|\Psi(x, t)|^2 = \Psi(x, t) \cdot \Psi^*(x, t)
\]

(2.15)

where \(\Psi^*(x, t)\) is the complex conjugate function. Therefore

\[
\Psi^*(x, t) = \psi^*(x) \cdot e^{iEt/\hbar}
\]

Then the product of the total wave function and its complex conjugate is given by

\[
\Psi(x, t)\Psi^*(x, t) = [\psi(x)e^{-iEt/\hbar}] [\psi^*(x)e^{iEt/\hbar}] = \psi(x)\psi^*(x)
\]

(2.16)

Therefore, we have that

\[
|\Psi(x, t)|^2 = \psi(x)\psi^*(x) = |\psi(x)|^2
\]

(2.17)

is the probability density function and is independent of time. One major difference between classical and quantum mechanics is that in classical mechanics, the position of a particle or body can be determined precisely, whereas in quantum mechanics, the position of a particle is found in terms of a probability. We will determine the probability density function for several examples, and since this property is independent of time, we will, in general, only be concerned with the time-independent wave function.

### 2.2.3 Boundary Conditions

Since the function \(|\psi(x)|^2\) represents the probability density function, then for a single particle, we must have

\[
\int_{-\infty}^{\infty} |\psi(x)|^2 \, dx = 1
\]

(2.18)

The probability of finding the particle somewhere is certain. Equation (2.18) allows us to normalize the wave function and is one boundary condition that is used to determine some wave function coefficients.

The remaining boundary conditions imposed on the wave function and its derivative are postulates. However, we may state the boundary conditions and present arguments that justify why they must be imposed. The wave function and its first derivative must have the following properties if the total energy \(E\) and the potential \(V(x)\) are finite everywhere.

- **Condition 1.** \(\psi(x)\) must be finite, single-valued, and continuous.
- **Condition 2.** \(\frac{\partial \psi(x)}{\partial x}\) must be finite, single-valued, and continuous.

# Introduction to Quantum Mechanics

!Figure 2.5

**Figure 2.5** Potential functions and corresponding wave function solutions for the case (a) when the potential function is finite everywhere and (b) when the potential function is infinite in some regions.

Since \(|\psi(x)|^2\) is a probability density, then \(\psi(x)\) must be finite and single-valued. If the probability density were to become infinite at some point in space, then the probability of finding the particle at this position would be certain and the uncertainty principle would be violated. If the total energy \(E\) and the potential \(V(x)\) are finite everywhere, then from Equation (2.13), the second derivative must be finite, which implies that the first derivative must be continuous. The first derivative is related to the particle momentum, which must be finite and single-valued. Finally, a finite first derivative implies that the function itself must be continuous. In some of the specific examples that we will consider, the potential function will become infinite in particular regions of space. For these cases, the first derivative will not necessarily be continuous, but the remaining boundary conditions will still hold.

Figure 2.5 shows two possible examples of potential functions and the corresponding wave solutions. In Figure 2.5a, the potential function is finite everywhere. The wave function as well as its first derivative is continuous. In Figure 2.5b, the potential function is infinite for \(x < 0\) and for \(x > a\). The wave function is continuous at the boundaries, but the first derivative is discontinuous. We will actually determine the wave functions in the following sections and in end-of-chapter problems.

## 2.3 APPLICATIONS OF SCHRODINGER’S WAVE EQUATION

We will now apply Schrodinger’s wave equation in several examples using various potential functions. These examples will demonstrate the techniques used in the solution of Schrodinger’s differential equation and the results of these examples will provide an indication of the electron behavior under these various potentials.

## 2.3 Applications of Schrödinger's Wave Equation

We will utilize the resulting concepts later in the discussion of semiconductor properties.

### 2.3.1 Electron in Free Space

As a first example of applying the Schrödinger's wave equation, consider the motion of an electron in free space. If there is no force acting on the particle, then the potential function \( V(x) \) will be constant and we must have \( E > V(x) \). Assume, for simplicity, that the potential function \( V(x) = 0 \) for all \( x \). Then, the time-independent wave equation can be written from Equation (2.13) as

\[
\frac{\partial^2 \psi(x)}{\partial x^2} + \frac{2mE}{\hbar^2} \psi(x) = 0
\]

(2.19)

The solution to this differential equation can be written in the form

\[
\psi(x) = A \exp \left( j \frac{\sqrt{2mE}}{\hbar} x \right) + B \exp \left( -j \frac{\sqrt{2mE}}{\hbar} x \right)
\]

(2.20a)

or

\[
\psi(x) = A \exp(jkx) + B \exp(-jkx)
\]

(2.20b)

where

\[
k = \frac{\sqrt{2mE}}{\hbar}
\]

(2.21)

and is called a wave number.

Recall that the time-dependent portion of the solution is

\[
\phi(t) = e^{-j\omega t} = e^{-j\omega t}
\]

(2.22)

Then the total solution for the wave function is given by

\[
\Psi(x, t) = A \exp[j(kx - \omega t)] + B \exp[-j(kx + \omega t)]
\]

(2.23)

This wave function solution is a traveling wave, which means that a particle moving in free space is represented by a traveling wave. The first term, with the coefficient \( A \), is a wave traveling in the \( +x \) direction, while the second term, with the coefficient \( B \), is a wave traveling in the \( -x \) direction. The value of these coefficients will be determined from boundary conditions. We will again see the traveling-wave solution for an electron in a crystal or semiconductor material.

Assume, for a moment, that we have a particle traveling in the \( +x \) direction, which will be described by the \( +x \) traveling wave. The coefficient \( B = 0 \). We can write the traveling-wave solution in the form

\[
\Psi(x, t) = A \exp[j(kx - \omega t)]
\]

(2.24)

where \( k \) is the wave number given by

\[
k = \frac{\sqrt{2mE}}{\hbar} = \frac{\sqrt{p^2}}{\hbar} = \frac{p}{\hbar}
\]

(2.25a)

# Chapter 2: Introduction to Quantum Mechanics

or

\[
p = \hbar k
\]
(2.25b)

Also recall that the de Broglie wavelength was given by

\[
\lambda = \frac{h}{p} = \frac{2\pi \hbar}{p}
\]
(2.26)

Combining Equations (2.25a) and (2.26), the wavelength can also be written in terms of the wave number as

\[
\lambda = \frac{2\pi}{k}
\]
(2.27a)

or

\[
k = \frac{2\pi}{\lambda}
\]
(2.27b)

A free particle with a well-defined energy will also have a well-defined wavelength and momentum.

The probability density function is \(\Psi(x, t)\Psi^*(x, t) = A A^*\), which is a constant independent of position. A free particle with a well-defined momentum can be found anywhere with equal probability. This result is in agreement with the Heisenberg uncertainty principle in which a precise momentum implies an undefined position.

A localized free particle is defined by a wave packet, formed by a superposition of wave functions with different momentum or \(k\) values. We will not consider the wave packet here.

## 2.3.2 The Infinite Potential Well

The problem of a particle in the infinite potential well is a classic example of a bound particle. The potential \(V(x)\) as a function of position for this problem is shown in Figure 2.6. The particle is assumed to exist in region II, so the particle is contained within a finite region of space. The time-independent Schrödinger’s wave equation is again given by Equation (2.13) as

\[
\frac{\partial^2 \psi(x)}{\partial x^2} + \frac{2m}{\hbar^2} (E - V(x))\psi(x) = 0
\]
(2.13)

where \(E\) is the total energy of the particle. If \(E\) is finite, the wave function must be zero, or \(\psi(x) = 0\), in both regions I and III. A particle cannot penetrate these infinite potential barriers, so the probability of finding the particle in regions I and III is zero.

The time-independent Schrödinger’s wave equation in region II, where \(V = 0\), becomes

\[
\frac{\partial^2 \psi(x)}{\partial x^2} + \frac{2mE}{\hbar^2} \psi(x) = 0
\]
(2.28)

## 2.3 Applications of Schrödinger's Wave Equation

!Figure 2.6  
**Figure 2.6** Potential function of the infinite potential well.

A particular form of solution to this equation is given by

\[
\psi(x) = A_1 \cos kx + A_2 \sin kx
\]

(2.29)

where

\[
k = \sqrt{\frac{2mE}{\hbar^2}}
\]

(2.30)

One boundary condition is that the wave function \(\psi(x)\) must be continuous so that

\[
\psi(x = 0) = \psi(x = a) = 0
\]

(2.31)

Applying the boundary condition at \(x = 0\), we must have that \(A_1 = 0\). At \(x = a\), we have

\[
\psi(x = a) = 0 = A_2 \sin ka
\]

(2.32)

This equation is valid if \(ka = n\pi\), where the parameter \(n\) is a positive integer, or \(n = 1, 2, 3, \ldots\). The parameter \(n\) is referred to as a quantum number. We can write

\[
k = \frac{n\pi}{a}
\]

(2.33)

Negative values of \(n\) simply introduce a negative sign in the wave function and yield redundant solutions for the probability density function. We cannot physically distinguish any difference between \(+n\) and \(-n\) solutions. Because of this redundancy, negative values of \(n\) are not considered.

The coefficient \(A_2\) can be found from the normalization boundary condition that was given by Equation (2.18) as \(\int_0^a \psi^*(x) \psi(x) \, dx = 1\). If we assume that the wave

# Chapter 2: Introduction to Quantum Mechanics

If the function solution \(\psi(x)\) is a real function, then \(\psi(x) = \psi^*(x)\). Substituting the wave function into Equation (2.18), we have

\[
\int_0^a A_2^2 \sin^2 kx \, dx = 1
\]

(2.34)

Evaluating this integral gives

\[
A_2 = \sqrt{\frac{2}{a}}
\]

(2.35)

Finally, the time-independent wave solution is given by

\[
\psi(x) = \sqrt{\frac{2}{a}} \sin \left( \frac{n \pi x}{a} \right) \quad \text{where } n = 1, 2, 3, \ldots
\]

(2.36)

This solution represents the electron in the infinite potential well and is a standing wave solution. The free electron was represented by a traveling wave, and now the bound particle is represented by a standing wave.

The parameter \(k\) in the wave solution was defined by Equations (2.30) and (2.33). Equating these two expressions for \(k\), we obtain

\[
k^2 = \frac{2mE}{\hbar^2} = \frac{n^2 \pi^2}{a^2}
\]

(2.37)

The total energy can then be written as

\[
E = E_n = \frac{\hbar^2 n^2 \pi^2}{2ma^2} \quad \text{where } n = 1, 2, 3, \ldots
\]

(2.38)

For the particle in the infinite potential well, the wave function is now given by

\[
\psi(x) = \sqrt{\frac{2}{a}} \sin k_n x
\]

(2.39)

where the constant \(k_n\), from Equation (2.37), must have discrete values, implying that the total energy of the particle can only have discrete values. This result means that the energy of the particle is quantized. That is, the energy of the particle can only have particular discrete values. The quantization of the particle energy is contrary to results from classical physics, which would allow the particle to have continuous energy values. The discrete energies lead to quantum states that will be considered in more detail in this and later chapters. The quantization of the energy of a bound particle is an important result.

This quantization of electron energy will be observed again at the end of the chapter for an electron bound to an ion forming an atom.

----

^3 A more thorough analysis shows that \(|A_2| = \sqrt{2/a}\), so solutions for the coefficient \(A\) include \(+\sqrt{2/a}, -\sqrt{2/a}, +i\sqrt{2/a}, -i\sqrt{2/a}\), or any complex number whose magnitude is \(\sqrt{2/a}\). Since the wave function itself has no physical meaning, the choice of which coefficient to use is immaterial: They all produce the same probability density function.

## Objective

Calculate the first three energy levels of an electron in an infinite potential well.

Consider an electron in an infinite potential well of width 5 Å.

### Solution

From Equation (2.38) we have:

\[
E_n = \frac{h^2 n^2}{2m a^2} = \frac{n^2 (1.054 \times 10^{-34})^2 \pi^2}{2(9.11 \times 10^{-31})(5 \times 10^{-10})^2} = n^2 (2.41 \times 10^{-19}) \, \text{J}
\]

or

\[
E_n = \frac{n^2 (2.41 \times 10^{-19})}{1.6 \times 10^{-19}} = n^2 (1.51) \, \text{eV}
\]

Then,

\[
E_1 = 1.51 \, \text{eV}, \quad E_2 = 6.04 \, \text{eV}, \quad E_3 = 13.59 \, \text{eV}
\]

### Comment

This calculation shows the order of magnitude of the energy levels of a bound electron.

### EXERCISE PROBLEM

**Ex 2.3** (a) The width of an infinite potential well is 12 Å. Determine the first three allowed energy levels (in eV) for an electron. (b) Repeat part (a) for a proton.

----

Figure 2.7a shows the first four allowed energies for the particle in the infinite potential well, and Figure 2.7b,c shows the corresponding wave functions and probability functions. We may note that as the energy increases, the probability of finding the particle at any given value of \( x \) becomes more uniform.

## 2.3.3 The Step Potential Function

Consider now a **step potential function** as shown in Figure 2.8. In the previous section, we considered a particle being confined between two potential barriers. In this example, we will assume that a flux of particles is incident on the potential barrier. We will assume that the particles are traveling in the \( +x \) direction and that they originated at \( x = -\infty \). A particularly interesting result is obtained for the case when the total energy of the particle is less than the barrier height, or \( E < V_0 \).

We again need to consider the time-independent wave equation in each of the two regions. This general equation was given in Equation (2.13) as:

\[
\frac{\partial^2 \psi(x)}{\partial x^2} + \frac{2m}{\hbar^2} (E - V(x)) \psi(x) = 0
\]

The wave equation in region I, in which \( V = 0 \), is:

\[
\frac{\partial^2 \psi_I(x)}{\partial x^2} + \frac{2mE}{\hbar^2} \psi_I(x) = 0
\]

(2.40)

----

[^4]: See Appendix D for a discussion of the electron-volt (eV) as a unit of energy.

# Chapter 2: Introduction to Quantum Mechanics

!Figure 2.7

**Figure 2.7** | Particle in an infinite potential well: (a) four lowest discrete energy levels, (b) corresponding wave functions, and (c) corresponding probability functions. (From Pierret [10].)

The general solution to this equation can be written in the form

\[
\psi_I(x) = A e^{j k_1 x} + B e^{-j k_1 x} \quad (x \leq 0)
\]

(2.41)

where the constant \( k_1 \) is

\[
k_1 = \sqrt{\frac{2mE}{\hbar^2}}
\]

(2.42)

The first term in Equation (2.41) is a traveling wave in the \( +x \) direction that represents the incident wave, and the second term is a traveling wave in the \( -x \) direction.

!Figure 2.8

**Figure 2.8** | The step potential function.

## 2.3 Applications of Schrödinger's Wave Equation

For the incident wave, \( A_1 \cdot A_1^* \) is the probability density function of the incident particles. If we multiply this probability density function by the incident velocity, then \( v_1 \cdot A_1 \cdot A_1^* \) is the flux of incident particles in units of \#/cm\(^2\)-s. Likewise, the quantity \( v_1 \cdot B_1 \cdot B_1^* \) is the flux of the reflected particles, where \( v_1 \) is the velocity of the reflected wave. (The parameters \( v_1 \) and \( v_1 \) in these terms are actually the magnitudes of the velocity only.)

In region II, the potential is \( V = V_0 \). If we assume that \( E < V_0 \), then the differential equation describing the wave function in region II can be written as

\[
\frac{\partial^2 \psi_2(x)}{\partial x^2} - \frac{2m}{\hbar^2} (V_0 - E) \psi_2(x) = 0
\]

(2.43)

The general solution may then be written in the form

\[
\psi_2(x) = A_2 e^{-k_2 x} + B_2 e^{k_2 x} \quad (x \geq 0)
\]

(2.44)

where

\[
k_2 = \sqrt{\frac{2m(V_0 - E)}{\hbar^2}}
\]

(2.45)

One boundary condition is that the wave function \( \psi_2(x) \) must remain finite, which means that the coefficient \( B_2 = 0 \). The wave function is now given by

\[
\psi_2(x) = A_2 e^{-k_2 x} \quad (x \geq 0)
\]

(2.46)

The wave function at \( x = 0 \) must be continuous so that

\[
\psi_1(0) = \psi_2(0)
\]

(2.47)

Then from Equations (2.41), (2.46), and (2.47), we obtain

\[
A_1 + B_1 = A_2
\]

(2.48)

Since the potential function is everywhere finite, the first derivative of the wave function must also be continuous so that

\[
\left. \frac{\partial \psi_1}{\partial x} \right|_{x=0} = \left. \frac{\partial \psi_2}{\partial x} \right|_{x=0}
\]

(2.49)

Using Equations (2.41), (2.46), and (2.49), we obtain

\[
jk_1 A_1 - jk_1 B_1 = -k_2 A_2
\]

(2.50)

We can solve Equations (2.48) and (2.50) to determine the coefficients \( B_1 \) and \( A_2 \) in terms of the incident wave coefficient \( A_1 \). The results are

\[
B_1 = \left( \frac{-k_2^2 + 2jk_1 k_2 - k_1^2}{k_2^2 + k_1^2} \right) \cdot A_1
\]

(2.51a)

and

\[
A_2 = \frac{2k_1(k_1 - jk_2)}{k_2^2 + k_1^2} \cdot A_1
\]

(2.51b)

# Introduction to Quantum Mechanics

The reflected probability density function is given by

\[
B_i \cdot B_i^* = \frac{(k_2^2 - k_1^2 + 2ijk_1k_2)(k_2^2 - k_1^2 - 2jk_1k_2)}{(k_2^2 + k_1^2)^2} \cdot A_i \cdot A_i^*
\]

(2.52)

We can define a reflection coefficient, \( R \), as the ratio of the reflected flux to the incident flux, which is written as

\[
R = \frac{v_r \cdot B_i \cdot B_i^*}{v_i \cdot A_i \cdot A_i^*}
\]

(2.53)

where \( v_i \) and \( v_r \) are the incident and reflected velocities, respectively, of the particles. In region I, \( V = 0 \) so that \( E = T \), where \( T \) is the kinetic energy of the particle. The kinetic energy is given by

\[
T = \frac{1}{2} mv^2
\]

(2.54)

so that the constant \( k_i \), from Equation (2.42) may be written as

\[
k_i = \sqrt{\frac{2m}{\hbar^2} \left(\frac{1}{2} mv^2\right)} = \sqrt{m^2 \frac{v^2}{\hbar^2}} = \frac{mv}{\hbar}
\]

(2.55)

The incident velocity can then be written as

\[
v_i = \frac{\hbar}{m} \cdot k_i
\]

(2.56)

Since the reflected particle also exists in region I, the reflected velocity (magnitude) is given by

\[
v_r = \frac{\hbar}{m} \cdot k_i
\]

(2.57)

The incident and reflected velocities (magnitudes) are equal. The reflection coefficient is then

\[
R = \frac{v_r \cdot B_i \cdot B_i^*}{v_i \cdot A_i \cdot A_i^*} = \frac{B_i \cdot B_i^*}{A_i \cdot A_i^*}
\]

(2.58)

Substituting the expression from Equation (2.52) into Equation (2.58), we obtain

\[
R = \frac{B_i \cdot B_i^*}{A_i \cdot A_i^*} = \frac{k_2^2 - k_1^2}{k_2^2 + k_1^2} + \frac{4k_1^2k_2^2}{k_2^2 + k_1^2} = 1.0
\]

(2.59)

The result of \( R = 1 \) implies that all of the particles incident on the potential barrier for \( E < V_0 \) are eventually reflected. Particles are not absorbed or transmitted through the potential barrier. This result is entirely consistent with classical physics and one might ask why we should consider this problem in terms of quantum mechanics. The interesting result is in terms of what happens in region II.

The wave solution in region II was given by Equation (2.46) as \( \psi(x) = A_2 e^{-\kappa x} \). The coefficient \( A_2 \) from Equation (2.48) is \( A_2 = A_i + B_i \), which we derived from the boundary conditions. For the case of \( E < V_0 \), the coefficient \( A_2 \) is not zero. If \( A_2 \) is not zero, then the probability density function \( \psi_2(x) \cdot \psi_2^*(x) \) of the particle being found in region II is not equal to zero. This result implies that there is a finite probability of finding the particle in region II.

# 2.3 Applications of Schrödinger's Wave Equation

**Probability that the incident particle will penetrate the potential barrier and exist in region II. The probability of a particle penetrating the potential barrier is another difference between classical and quantum mechanics: The quantum mechanical penetration is classically not allowed.** Although there is a finite probability that the particle may penetrate the barrier, since the reflection coefficient in region I is unity, the particle in region II must eventually turn around and move back into region I.

## Objective

Calculate the penetration depth of a particle impinging on a potential barrier.

**Example 2.4**

Consider an incident electron that is traveling at a velocity of \(1 \times 10^6 \, \text{m/s}\) in region I.

### Solution

With \(V(x) = 0\), the total energy is also equal to the kinetic energy so that

\[
E = T = \frac{1}{2} mv^2 = 4.56 \times 10^{-21} \, \text{J} = 2.85 \times 10^{-2} \, \text{eV}
\]

Now, assume that the potential barrier at \(x = 0\) is twice as large as the total energy of the incident particle, or that \(V_0 = 2E\). The wave function solution in region II is \(\psi(x) = A_2 e^{-kx}\), where the constant \(k\) is given by \(k = \sqrt{2m(V_0 - E)}/\hbar\).

In this example, we want to determine the distance \(x = d\) at which the wave function magnitude has decayed to \(e^{-1}\) of its value at \(x = 0\). Then, for this case, we have \(kd = 1\) or

\[
1 = d \sqrt{\frac{2m(2E - E)}{\hbar^2}} = d \sqrt{\frac{2mE}{\hbar^2}}
\]

The distance is then given by

\[
d = \sqrt{\frac{\hbar^2}{2mE}} = \frac{1.054 \times 10^{-34}}{\sqrt{2(9.11 \times 10^{-31})(4.56 \times 10^{-21})}} = 11.6 \times 10^{-10} \, \text{m}
\]

or

\[
d = 11.6 \, \text{Å}
\]

### Comment

This penetration distance corresponds to approximately two lattice constants of silicon. The numbers used in this example are rather arbitrary. We used a distance at which the wave function decayed to \(e^{-1}\) of its initial value. We could have arbitrarily used \(e^{-2}\), for example, but the results give an indication of the magnitude of penetration depth.

### Exercise Problem

**Ex 2.4** The probability of finding a particle at a distance \(d\) in region II compared with that at \(x = 0\) is given by \(\exp(-2kd)\). Consider an electron traveling in region I at a velocity of \(10^6 \, \text{m/s}\) incident on a potential barrier whose height is three times the kinetic energy of the electron. Find the probability of finding the electron at a distance \(d\) compared with \(x = 0\) where \(d\) is (a) 10 Å and (b) 100 Å into the potential barrier.

The case when the total energy of a particle, which is incident on the potential barrier, is greater than the barrier height, or \(E > V_0\), is left as an exercise at the end of the chapter.

# 2.3.4 The Potential Barrier and Tunneling

We now want to consider the potential barrier function, which is shown in Figure 2.9. The more interesting problem, again, is in the case when the total energy of an incident particle is \( E < V_0 \). Again assume that we have a flux of incident particles originating on the negative x-axis traveling in the \( +x \) direction. As before, we need to solve Schrödinger's time-independent wave equation in each of the three regions. The solutions of the wave equation in regions I, II, and III are given, respectively, as

\[
\psi_1(x) = A_1 e^{jk_1x} + B_1 e^{-jk_1x} \tag{2.60a}
\]

\[
\psi_2(x) = A_2 e^{k_2x} + B_2 e^{-k_2x} \tag{2.60b}
\]

\[
\psi_3(x) = A_3 e^{jk_1x} + B_3 e^{-jk_1x} \tag{2.60c}
\]

where

\[
k_1 = \sqrt{\frac{2mE}{\hbar^2}} \tag{2.61a}
\]

and

\[
k_2 = \sqrt{\frac{2m}{\hbar^2} (V_0 - E)} \tag{2.61b}
\]

The coefficient \( B_3 \) in Equation (2.60c) represents a negative traveling wave in region III. However, once a particle gets into region III, there are no potential changes to cause a reflection; therefore, the coefficient \( B_3 \) must be zero. We must keep both exponential terms in Equation (2.60b) since the potential barrier width is finite; that is, neither term will become unbounded. We have four boundary relations for the boundaries at \( x = 0 \) and \( x = a \) corresponding to the wave function and its first derivative being continuous. We can solve for the four coefficients \( B_1, A_2, B_2, \) and \( A_3 \) in terms of \( A_1 \). The wave solutions in the three regions are shown in Figure 2.10.

!Figure 2.9  
*Figure 2.9 | The potential barrier function.*

## 2.3 Applications of Schrödinger's Wave Equation

!Wave Functions

**Figure 2.10** The wave functions through the potential barrier.

One particular parameter of interest is the transmission coefficient, in this case defined as the ratio of the transmitted flux in region III to the incident flux in region I. Then the transmission coefficient \( T \) is

\[
T = \frac{v_t \cdot A_3 \cdot A_1^*}{v_i \cdot A_1 \cdot A_3^*} = \frac{A_3 \cdot A_1^*}{A_1 \cdot A_3^*}
\]

(2.62)

where \( v_t \) and \( v_i \) are the velocities of the transmitted and incident particles, respectively. Since the potential \( V = 0 \) in both regions I and III, the incident and transmitted velocities are equal. The transmission coefficient may be determined by solving the boundary condition equations. For the special case when \( E \ll V_0 \), we find that

\[
T \approx 16 \left( \frac{E}{V_0} \right) \left( 1 - \frac{E}{V_0} \right) \exp(-2ka)
\]

(2.63)

Equation (2.63) implies that there is a finite probability that a particle impinging a potential barrier will penetrate the barrier and will appear in region III. This phenomenon is called tunneling and it, too, contradicts classical mechanics. We will see later how this quantum mechanical tunneling phenomenon can be applied to semiconductor device characteristics, such as in the tunnel diode.

----

**Objective:** Calculate the probability of an electron tunneling through a potential barrier.

**Example 2.5**

Consider an electron with an energy of 2 eV impinging on a potential barrier with \( V_0 = 20 \) eV and a width of 3 Å.

### Solution

Equation (2.63) is the tunneling probability. The factor \( k_2 \) is

\[
k_2 = \sqrt{\frac{2m(V_0 - E)}{\hbar^2}} = \sqrt{\frac{(2 \cdot 9.11 \times 10^{-31})(20 - 2)(1.6 \times 10^{-19})}{(1.054 \times 10^{-34})^2}}
\]

or

\[
k_2 = 2.17 \times 10^{10} \, \text{m}^{-1}
\]

Then

\[
T = 16(0.1)(1 - 0.1) \exp \left[ -2(2.17 \times 10^{10})(3 \times 10^{-10}) \right]
\]

# Chapter 2: Introduction to Quantum Mechanics

and finally

\[ T = 3.17 \times 10^{-6} \]

**Comment**

The tunneling probability may appear to be a small value, but the value is not zero. If a large number of particles impinge on a potential barrier, a significant number can penetrate the barrier.

## Exercise Problem

**Ex 2.5** (a) Estimate the probability of an electron with energy \( E = 0.12 \, \text{eV} \) tunneling through a rectangular potential barrier with a height of \( V_0 = 1.2 \, \text{eV} \) and a width of \( a = 5 \, \text{Å} \). (b) Repeat part (a) for a barrier width of \( a = 25 \, \text{Å} \).

\[ [\cdot 01 \times 6L = L (q) \cdot 01 \times 6L = L (q) \cdot 01 \times 20L = L (q) \cdot sUV] \]

Additional applications of Schrödinger’s wave equation with various one-dimensional potential functions are found in problems at the end of the chapter. Several of these potential functions represent quantum well structures that are found in modern semiconductor devices.

## Test Your Understanding

**TYU 2.3** (a) Estimate the probability of an electron with energy \( E = 0.10 \, \text{eV} \) tunneling through a rectangular potential barrier with a barrier height of \( V_0 = 0.8 \, \text{eV} \) and width \( a = 12 \, \text{Å} \). (b) Repeat part (a) for a barrier height of \( V_0 = 1.5 \, \text{eV} \).

\[ [\cdot 01 \times 6L = L (q) \cdot 01 \times 6L = L (q) \cdot sUV] \]

**TYU 2.4** A certain semiconductor device requires a tunneling probability of \( T = 5 \times 10^{-6} \) for an electron tunneling through a rectangular barrier with a barrier height of \( V_0 = 0.8 \, \text{eV} \). The electron energy is \( E = 0.08 \, \text{eV} \). Determine the maximum barrier width.

\[ (Y 9Y = L = D \cdot sUV) \]

## 2.4 Extensions of the Wave Theory to Atoms

So far in this chapter, we have considered several one-dimensional potential energy functions and solved Schrödinger’s time-independent wave equation to obtain the probability function of finding a particle at various positions. Consider now the one-electron, or hydrogen, atom potential function. We will only briefly consider the mathematical details and wave function solutions, but the results are interesting and important.

### 2.4.1 The One-Electron Atom

The nucleus is a heavy, positively charged proton and the electron is a light, negatively charged particle that, in the classical Bohr theory, is revolving around the nucleus. 

*The detailed mathematical analysis is beyond the scope of this text, but the results, which are emphasized in this section, are important in the following discussions of semiconductor physics.*

## 2.4 Extensions of the Wave Theory to Atoms

The potential function is due to the coulomb attraction between the proton and electron and is given by

\[
V(r) = -\frac{e^2}{4\pi \varepsilon_0 r}
\]

(2.64)

where \( e \) is the magnitude of the electronic charge and \( \varepsilon_0 \) is the permittivity of free space. This potential function, although spherically symmetric, leads to a three-dimensional problem in spherical coordinates.

We may generalize the time-independent Schrödinger’s wave equation to three dimensions by writing

\[
\nabla^2 \psi(r, \theta, \phi) + \frac{2m_0}{\hbar^2} (E - V(r)) \psi(r, \theta, \phi) = 0
\]

(2.65)

where \( \nabla^2 \) is the Laplacian operator and must be written in spherical coordinates for this case. The parameter \( m_0 \) is the rest mass of the electron. In spherical coordinates, Schrödinger’s wave equation may be written as

\[
\frac{1}{r^2} \frac{\partial}{\partial r} \left( r^2 \frac{\partial \psi}{\partial r} \right) + \frac{1}{r^2 \sin \theta} \frac{\partial}{\partial \theta} \left( \sin \theta \frac{\partial \psi}{\partial \theta} \right) + \frac{1}{r^2 \sin^2 \theta} \frac{\partial^2 \psi}{\partial \phi^2} + \frac{2m_0}{\hbar^2} (E - V(r)) \psi = 0
\]

(2.66)

The solution to Equation (2.66) can be determined by the separation-of-variables technique. We will assume that the solution to the time-independent wave equation can be written in the form

\[
\psi(r, \theta, \phi) = R(r) \cdot \Theta(\theta) \cdot \Phi(\phi)
\]

(2.67)

where \( R \), \( \Theta \), and \( \Phi \) are functions only of \( r \), \( \theta \), and \( \phi \), respectively. Substituting this form of solution into Equation (2.66), we will obtain

\[
\sin^2 \theta \left[ \frac{1}{R} \frac{\partial}{\partial r} \left( r^2 \frac{\partial R}{\partial r} \right) + \frac{1}{\Theta \sin \theta} \frac{\partial}{\partial \theta} \left( \sin \theta \frac{\partial \Theta}{\partial \theta} \right) + \frac{1}{\Phi \sin^2 \theta} \frac{\partial^2 \Phi}{\partial \phi^2} \right] + r^2 \sin^2 \theta \cdot \frac{2m_0}{\hbar^2} (E - V) = 0
\]

(2.68)

We may note that the second term in Equation (2.68) is a function of \( \theta \) only, whereas all the other terms are functions of either \( r \) or \( \phi \). We may then write that

\[
\frac{1}{\Phi} \frac{\partial^2 \Phi}{\partial \phi^2} = -m^2
\]

(2.69)

where \( m \) is a separation of variables constant. The solution to Equation (2.69) is of the form

\[
\Phi = e^{jm\phi}
\]

(2.70)

*The mass should be the rest mass of the two-particle system, but since the proton mass is much greater than the electron mass, the equivalent mass reduces to that of the electron.*

*Where \( m \) means the separation-of-variables constant developed historically. That meaning will be retained here even though there may be some confusion with the electron mass. In general, the mass parameter will be used in conjunction with a subscript.*

# Introduction to Quantum Mechanics

Since the wave function must be single-valued, we impose the condition that \( m \) is an integer, or

\[
m = 0, \pm 1, \pm 2, \pm 3, \ldots
\]

(2.71)

Incorporating the separation-of-variables constant, we can further separate the variables \(\theta\) and \(r\) and generate two additional separation-of-variables constants \(l\) and \(n\). The separation-of-variables constants \(n, l, m\) are known as quantum numbers. The parameter \(n\) is referred to as the principal quantum number, \(l\) is the azimuthal or angular quantum number, and \(m\) is the magnetic quantum number. The quantum numbers are related by

\[
n = 1, 2, 3, \ldots
\]

\[
l = n - 1, n - 2, n - 3, \ldots 0
\]

\[
|m| = l, (l - 1, \ldots, 0)
\]

(2.72)

Each set of quantum numbers corresponds to a quantum state that the electron may occupy. The electron energy may be written in the form

\[
E_n = -\frac{m_0 e^4}{(4\pi \varepsilon_0)^2 2\hbar^2 n^2}
\]

(2.73)

where again \(n\) is the principal quantum number. The negative energy indicates that the electron is bound to the nucleus and we again see that the energy of the bound electron is quantized. If the energy were to become positive, then the electron would no longer be a bound particle and the total energy would no longer be quantized. Since the parameter \(n\) in Equation (2.73) is an integer, the total energy of the electron can take on only discrete values. The quantized energy is again a result of the particle being bound in a finite region of space.

## Example 2.6

**Objective:** Calculate the first three energy levels of an electron for the one-electron atom.

### Solution

We have

\[
E_n = -\frac{m_0 e^4}{(4\pi \varepsilon_0)^2 2\hbar^2 n^2} = -\frac{(9.11 \times 10^{-31})(1.6 \times 10^{-19})^4}{[4\pi(8.85 \times 10^{-12})]^2(1.054 \times 10^{-34})^2 n^2}
\]

\[
= -21.726 \times 10^{-19} \frac{1}{n^2} \quad \text{or} \quad = -\frac{13.58}{n^2} \, \text{eV}
\]

For \(n = 1\): \(E_1 = -13.58 \, \text{eV}\)

For \(n = 2\): \(E_2 = -3.39 \, \text{eV}\)

For \(n = 3\): \(E_3 = -1.51 \, \text{eV}\)

### Comment

As the energy levels increase, the energy becomes less negative, which means that the electron is becoming less tightly bound to the atom.

### Exercise Problem

**Ex 2.6** In Example 2.6, assume the permittivity of free space, \(\varepsilon_0\), is replaced by the permittivity of a material where \(\varepsilon = \varepsilon_r \varepsilon_0\). Repeat the calculations in Example 2.6 if \(\varepsilon_r = 11.7\) (silicon).

## 2.4 Extensions of the Wave Theory to Atoms

The solution of the wave equation may be designated by \(\psi_{nlm}\), where \(n\), \(l\), and \(m\) are again the various quantum numbers. For the lowest energy state, \(n = 1\), \(l = 0\), and \(m = 0\), and the wave function is given by

\[
\psi_{100} = \frac{1}{\sqrt{\pi}} \left( \frac{1}{a_0} \right)^{3/2} e^{-r/a_0}
\]

(2.74)

This function is spherically symmetric, and the parameter \(a_0\) is given by

\[
a_0 = \frac{4\pi \varepsilon_0 \hbar^2}{m_0 e^2} = 0.529 \, \text{Å}
\]

(2.75)

and is equal to the Bohr radius.

The radial probability density function, or the probability of finding the electron at a particular distance from the nucleus, is proportional to the product \(\psi_{100}^* \psi_{100}\) and also to the differential volume of the shell around the nucleus. The probability density function for the lowest energy state is plotted in Figure 2.11a. The most probable distance from the nucleus is at \(r = a_0\), which is the same as the Bohr theory. Considering this spherically symmetric probability function, we may now begin to conceive the concept of an electron cloud, or energy shell, surrounding the nucleus rather than a discrete particle orbiting around the nucleus.

The radial probability density function for the next higher, spherically symmetric wave function, corresponding to \(n = 2\), \(l = 0\), and \(m = 0\), is shown in Figure 2.11b. This figure shows the idea of the next-higher energy shell of the electron. The second energy shell is at a greater radius from the nucleus than the first energy shell. As indicated in the figure, though, there is still a small probability that the electron will exist at the smaller radius. For the case of \(n = 2\) and \(l = 1\), there are three possible states corresponding to the three allowed values of the quantum number \(m\). These wave functions are no longer spherically symmetric.

Although we have not gone into a great deal of mathematical detail for the one-electron atom, three results are important for the further analysis of semiconductor materials.

!Figure 2.11

**Figure 2.11** The radial probability density function for the one-electron atom in the (a) lowest energy state and (b) next-higher energy state.  
*(From Eisberg and Resnick [5].)*

The first is the solution of Schrödinger's wave equation, which again yields electron probability functions, as it did for the simpler potential functions. In developing the physics of semiconductor materials in later chapters, we will also be considering electron probability functions. The second result is the quantization of allowed energy levels for the bound electron. The third is the concept of quantum numbers and quantum states, which evolved from the separation-of-variables technique. We will consider this concept again in the next section and in later chapters when we deal with the semiconductor material physics.

## 2.4.2 The Periodic Table

The initial portion of the periodic table of elements may be determined by using the results of the one-electron atom plus two additional concepts. The first concept needed is that of **electron spin**. The electron has an intrinsic angular momentum, or spin, which is quantized and may take on one of two possible values. The spin is designated by a quantum number \( s \), which has a value of \( s = +\frac{1}{2} \) or \( s = -\frac{1}{2} \). We now have four basic quantum numbers: \( n, l, m, \) and \( s \).

The second concept needed is the **Pauli exclusion principle**. The Pauli exclusion principle states that, in any given system (an atom, molecule, or crystal), no two electrons may occupy the same quantum state. In an atom, the exclusion principle means that no two electrons may have the same set of quantum numbers. We will see that the exclusion principle is also an important factor in determining the distribution of electrons among available energy states in a crystal.

Table 2.1 shows the first few elements of the periodic table. For the first element, hydrogen, we have one electron in the lowest energy state corresponding to \( n = 1 \). From Equation (2.72) both quantum numbers \( l \) and \( m \) must be zero. However, the electron can take on either spin factor \( +\frac{1}{2} \) or \( -\frac{1}{2} \). For helium, two electrons may exist in the lowest energy state. For this case, \( l = m = 0 \), so now both electron spin states are occupied and the lowest energy shell is full. The chemical activity of an element is determined primarily by the valence, or outermost, electrons. Since the valence energy shell of helium is full, helium does not react with other elements and is an inert element.

### Table 2.1 Initial portion of the periodic table

| Element  | Notation  | \( n \) | \( l \) | \( m \) | \( s \)            |
|----------|-----------|---------|---------|---------|--------------------|
| Hydrogen | 1s\(^1\)  | 1       | 0       | 0       | \( +\frac{1}{2} \) or \( -\frac{1}{2} \) |
| Helium   | 1s\(^2\)  | 1       | 0       | 0       | \( +\frac{1}{2} \) and \( -\frac{1}{2} \) |
| Lithium  | 1s\(^2\)2s\(^1\) | 2 | 0     | 0       | \( +\frac{1}{2} \) or \( -\frac{1}{2} \) |
| Beryllium| 1s\(^2\)2s\(^2\) | 2 | 0     | 0       |                    |
| Boron    | 1s\(^2\)2s\(^2\)2p\(^1\) | 2 | 1     | \( m = 0, -1, +1 \) | \( s = +\frac{1}{2}, -\frac{1}{2} \) |
| Carbon   | 1s\(^2\)2s\(^2\)2p\(^2\) | 2 | 1     |                     |                    |
| Nitrogen | 1s\(^2\)2s\(^2\)2p\(^3\) | 2 | 1     |                     |                    |
| Oxygen   | 1s\(^2\)2s\(^2\)2p\(^4\) | 2 | 1     |                     |                    |
| Fluorine | 1s\(^2\)2s\(^2\)2p\(^5\) | 2 | 1     |                     |                    |
| Neon     | 1s\(^2\)2s\(^2\)2p\(^6\) | 2 | 1     |                     |                    |


# 2.5.1 Summary

- A few basic concepts of quantum mechanics, which can be used to describe the behavior of electrons under various potential functions, were considered. The understanding of electron behavior is crucial in understanding semiconductor physics.
- The wave–particle duality principle is an important element in quantum mechanics. Particles can have wave-like behavior and waves can have particle-like behavior.
- Schrödinger’s wave equation forms the basis for describing and predicting the behavior of electrons.
- Max Born postulated that \(|\psi(x)|^2\) is a probability density function.
- A result of applying Schrödinger’s wave equation to a bound particle is that the energy of the bound particle is quantized.
- A result of applying Schrödinger’s wave equation to an electron incident on a potential barrier is that there is a finite probability of tunneling.
- The concept of quantum numbers was developed from the results of applying Schrödinger’s wave equation to the one-electron atom.
- The basic structure of the periodic table is predicted by applying Schrödinger’s wave equation to the one-electron atom and using the Pauli exclusion principle.

# Glossary of Important Terms

**de Broglie wavelength**  
The wavelength of a particle given as the ratio of Planck’s constant to momentum.

**Heisenberg uncertainty principle**  
The principle that states that we cannot describe with absolute accuracy the relationship between sets of conjugate variables that describe the behavior of particles, such as momentum and position.

**Pauli exclusion principle**  
The principle that states that no two electrons can occupy the same quantum state.

**photon**  
The particle-like packet of electromagnetic energy.

**quanta**  
The particle-like packet of thermal radiation.

**quantized energies**  
The allowed discrete energy levels that bound particles may occupy.

**quantum numbers**  
A set of numbers that describes the quantum state of a particle, such as an electron in an atom.

# Chapter 2: Introduction to Quantum Mechanics

**quantum state**  
A particular state of an electron that may be described, for example, by a set of quantum numbers.

**tunneling**  
The quantum mechanical phenomenon by which a particle may penetrate through a thin potential barrier.

**wave–particle duality**  
The characteristic by which electromagnetic waves sometimes exhibit particle-like behavior and particles sometimes exhibit wave-like behavior.

## CHECKPOINT

After studying this chapter, the reader should have the ability to:

- Discuss the principle of energy quanta, the wave–particle duality principle, and the uncertainty principle.
- Apply Schrödinger’s wave equation and boundary conditions to problems with various potential functions.
- Determine quantized energy levels of bound particles.
- Determine the approximate tunneling probability of a particle incident on a potential barrier.
- State Pauli exclusion principle.
- Discuss the results of the one-electron atom analysis, including quantum numbers and their interrelationships as well as the initial formation of the periodic table.

## REVIEW QUESTIONS

1. State the wave–particle duality principle and state the relationship between momentum and wavelength.
2. What is the physical meaning of Schrödinger’s wave function?
3. What is meant by a probability density function?
4. List the boundary conditions for solutions to Schrödinger’s wave equation.
5. What is meant by quantized energy levels? Can an electron contained in a potential well have an arbitrary energy?
6. Describe the concept of tunneling.
7. List the quantum numbers of the one-electron atom and discuss how they were developed.
8. State the interrelationship between the quantum numbers of the one-electron atom and how this result leads to, for example, the development of inert elements.

## PROBLEMS

2.1 The classical wave equation for a two-wire transmission line is given by 

\[
\partial^2 V(x, t) / \partial x^2 = LC \cdot \partial^2 V(x, t) / \partial t^2
\]

One possible solution is given by 

\[
V(x, t) = (\sin Kx) \cdot (\sin \omega t)
\]

where \( K = m\pi / a \) and \( \omega = K / \sqrt{LC} \). Sketch, on the same graph, the function \( V(x, t) \) as a function of \( x \) for \( 0 \leq x \leq a \) and \( n = 1 \) when (i) \( \omega t = 0 \), (ii) \( \omega t = \pi/2 \), (iii) \( \omega t = \pi \), (iv) \( \omega t = 3\pi/2 \), and (v) \( \omega t = 2\pi \).

2.2 The function \( V(x, t) = \cos (2\pi x/\lambda - \omega t) \) is also a solution to the classical wave equation. Sketch on the same graph the function \( V(x, t) \) as a function of \( x \) for \( 0 \leq x \leq \lambda \) when: (i) \( \omega t = 0 \), (ii) \( \omega t = 0.25\pi \), (iii) \( \omega t = 0.5\pi \), (iv) \( \omega t = 0.75\pi \), and (v) \( \omega t = \pi \).

2.3 Repeat Problem 2.2 for the function \( V(x, t) = \cos (2\pi x/\lambda + \omega t) \).

2.4 Determine the phase velocities of the traveling waves described in Problems 2.2 and 2.3.

