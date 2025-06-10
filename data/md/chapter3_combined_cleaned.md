# Chapter 3: Introduction to the Quantum Theory of Solids

In the last chapter, we applied quantum mechanics and Schrödinger’s wave equation to determine the behavior of electrons in the presence of various potential functions. We found one important characteristic of an electron bound to an atom or bound within a finite space to be that the electron can take on only discrete values of energy; that is, the energies are quantized. We also discussed the Pauli exclusion principle, which stated that only one electron is allowed to occupy any given quantum state. In this chapter, we will generalize these concepts to the electron in a crystal lattice.

One of our goals is to determine the electrical properties of a semiconductor material, which we will then use to develop the current–voltage characteristics of semiconductor devices. Toward this end, we have two tasks in this chapter: to determine the properties of electrons in a crystal lattice and to determine the statistical characteristics of the very large number of electrons in a crystal.

## 3.0 | Preview

In this chapter, we will:

- Develop the concept of allowed and forbidden electron energy bands in a single-crystal material, and describe conduction and valence energy bands in a semiconductor material.
- Discuss the concept of negatively charged electrons and positively charged holes as two distinct charge carriers in a semiconductor material.
- Develop electron energy versus momentum curves in a single-crystal material, which yields the concept of direct and indirect bandgap semiconductor materials.
- Discuss the concept of effective mass of an electron and a hole.
- Derive the density of quantum states in the allowed energy bands.
- Develop the Fermi-Dirac probability function, which describes the statistical distribution of electrons among the allowed energy levels, and define the Fermi energy level.

## 3.1 Allowed and Forbidden Energy Bands

In the last chapter, we considered the one-electron, or hydrogen, atom. That analysis showed that the energy of the bound electron is quantized: Only discrete values of electron energy are allowed. The radial probability density for the electron was also determined. This function gives the probability of finding the electron at a particular distance from the nucleus and shows that the electron is not localized at a given radius. We can extrapolate these single-atom results to a crystal and qualitatively derive the concepts of allowed and forbidden energy bands. We will then apply quantum mechanics and Schrödinger’s wave equation to the problem of an electron in a single crystal. We find that the electronic energy states occur in bands of allowed states that are separated by forbidden energy bands.

### 3.1.1 Formation of Energy Bands

Figure 3.1a shows the radial probability density function for the lowest electron energy state of the single, noninteracting hydrogen atom, and Figure 3.1b shows the same probability curves for two atoms that are in close proximity to each other. The wave functions of the electrons of the two atoms overlap, which means that the two electrons will interact. This interaction or perturbation results in the discrete quantized energy level splitting into two discrete energy levels, schematically shown in Figure 3.1c. The splitting of the discrete state into two states is consistent with the Pauli exclusion principle.

A simple analogy of the splitting of energy levels by interacting particles is the following. Two identical race cars and drivers are far apart on a race track. There is no interaction between the cars, so they both must provide the same power to achieve a given speed. However, if one car pulls up close behind the other car, there is an interaction called draft. The second car will be pulled to an extent by the lead car. The lead car will therefore require more power to achieve the same speed since it is pulling the second car, and the second car will require less power since it is being pulled by the lead car. So there is a “splitting” of power (energy) of the two interacting race cars. (Keep in mind not to take analogies too literally.)

**Figure 3.1** (a) Probability density function of an isolated hydrogen atom. (b) Overlapping probability density functions of two adjacent hydrogen atoms. (c) The splitting of the \( n = 1 \) state.

|   |   |   |
|---|---|---|
| (a) | (b) | (c) |
| \( p(r) \) | \( p(r) \) | \( p(r) \) |
| \( r \) | \( d_0 \) | \( d_0 \) |
|   |   | Electron energy |
|   |   | \( n = 1 \) |
|   |   | \( n = 1 \) |


**Figure 3.2** The splitting of an energy state into a band of allowed energies.

Now, if we somehow start with a regular periodic arrangement of hydrogen-type atoms that are initially very far apart, and begin pushing the atoms together, the initial quantized energy level will split into a band of discrete energy levels. This effect is shown schematically in Figure 3.2, where the parameter \( r_0 \) represents the equilibrium interatomic distance in the crystal. At the equilibrium interatomic distance, there is a band of allowed energies, but within the allowed band, the energies are at discrete levels. The Pauli exclusion principle states that the joining of atoms to form a system (crystal) does not alter the total number of quantum states regardless of size. However, since no two electrons can have the same quantum number, the discrete energy must split into a band of energies in order that each electron can occupy a distinct quantum state.

We have seen previously that, at any energy level, the number of allowed quantum states is relatively small. In order to accommodate all of the electrons in a crystal, we must have many energy levels within the allowed band. As an example, suppose that we have a system with \( 10^9 \) one-electron atoms and also suppose that, at the equilibrium interatomic distance, the width of the allowed energy band is 1 eV. For simplicity, we assume that each electron in the system occupies a different energy level and, if the discrete energy states are equidistant, then the energy levels are separated by \( 10^{-9} \) eV. This energy difference is extremely small, so that for all practical purposes, we have a quasi-continuous energy distribution through the allowed energy band. The fact that \( 10^{-9} \) eV is a very small difference between two energy states can be seen from the following example.

## Example 3.1

**Objective:** Calculate the change in kinetic energy of an electron when the velocity changes by a small amount.

Consider an electron traveling at a velocity of \( 10^7 \) cm/s. Assume that the velocity increases by a value of 1 cm/s. The increase in kinetic energy is given by

\[
\Delta E = \frac{1}{2} m v_2^2 - \frac{1}{2} m v_1^2 = \frac{1}{2} m (v_2^2 - v_1^2)
\]

Let \( v_2 = v_1 + \Delta v \). Then

\[
v_2^2 = (v_1 + \Delta v)^2 = v_1^2 + 2v_1 \Delta v + (\Delta v)^2
\]

But \(\Delta v \ll v_i\), so we have that

\[
\Delta E \approx \frac{1}{2} m (2v_i \Delta v) = mv_i \Delta v
\]

**Solution**

Substituting the number into this equation, we obtain

\[
\Delta E = (9.11 \times 10^{-31})(10^7)(0.01) = 9.11 \times 10^{-38} \, \text{J}
\]

which may be converted to units of electron volts as

\[
\Delta E = \frac{9.11 \times 10^{-38}}{1.6 \times 10^{-19}} = 5.7 \times 10^{-19} \, \text{eV}
\]

**Comment**

A change in velocity of 1 cm/s compared with \(10^7\) cm/s results in a change in energy of \(5.7 \times 10^{-19}\) eV, which is orders of magnitude larger than the change in energy of \(10^{-19}\) eV between energy states in the allowed energy band. This example serves to demonstrate that a difference in adjacent energy states of \(10^{-19}\) eV is indeed very small, so that the discrete energies within an allowed band may be treated as a quasi-continuous distribution.

**Exercise Problem**

**Ex 3.1** The initial velocity of an electron is \(10^7\) cm/s. If the kinetic energy of the electron increases by \(\Delta E = 10^{-12}\) eV, determine the increase in velocity.

\[
(s\mu) - .01 \, \mu \ll 1 = qv \, s\mu
\]


Consider again a regular periodic arrangement of atoms, in which each atom now contains more than one electron. Suppose the atom in this imaginary crystal contains electrons up through the \(n = 3\) energy level. If the atoms are initially very far apart, the electrons in adjacent atoms will not interact and will occupy the discrete energy levels. If these atoms are brought closer together, the outermost electrons in the \(n = 3\) energy shell will begin to interact initially, so that this discrete energy level will split into a band of allowed energies. If the atoms continue to move closer together, the electrons in the \(n = 2\) shell may begin to interact and will also split into a band of allowed energies. Finally, if the atoms become sufficiently close together, the innermost electrons in the \(n = 1\) level may interact, so that this energy level may also split into a band of allowed energies. The splitting of these discrete energy levels is qualitatively shown in Figure 3.3. If the equilibrium interatomic distance is \(r_0\), then we have bands of allowed energies that the electrons may occupy separated by bands of forbidden energies. This energy-band splitting and the formation of allowed and forbidden bands is the energy-band theory of single-crystal materials.

The actual band splitting in a crystal is much more complicated than indicated in Figure 3.3. A schematic representation of an isolated silicon atom is shown in Figure 3.4a. Ten of the 14 silicon atom electrons occupy deep-lying energy levels close to the nucleus. The four remaining valence electrons are relatively weakly bound and are the electrons involved in chemical reactions. Figure 3.4b shows the band splitting of silicon. We need only consider the \(n = 3\) level for the valence.

**Figure 3.3.1** Schematic showing the splitting of three energy states into allowed bands of energies.

**Figure 3.4.1** (a) Schematic of an isolated silicon atom. (b) The splitting of the 3s and 3p states of silicon into the allowed and forbidden energy bands.  
*(From Shockley [6].)*

Electrons, since the first two energy shells are completely full and are tightly bound to the nucleus. The 3s state corresponds to \( n = 3 \) and \( l = 0 \) and contains two quantum states per atom. This state will contain two electrons at \( T = 0 \) K. The 3p state corresponds to \( n = 3 \) and \( l = 1 \) and contains six quantum states per atom. This state will contain the remaining two electrons in the individual silicon atom.

As the interatomic distance decreases, the 3s and 3p states interact and overlap. At the equilibrium interatomic distance, the bands have again split, but now four quantum states per atom are in the lower band and four quantum states per atom are in the upper band. At absolute zero degrees, electrons are in the lowest energy state, so that all states in the lower band (the valence band) will be full and all states in the upper band (the conduction band) will be empty. The bandgap energy \( E_g \) between the top of the valence band and the bottom of the conduction band is the width of the forbidden energy band.

We have discussed qualitatively how and why bands of allowed and forbidden energies are formed in a crystal. The formation of these energy bands is directly related to the electrical characteristics of the crystal, as we will see later in our discussion.

### 3.1.2 The Kronig–Penney Model

In the previous section, we discussed qualitatively the splitting of allowed electron energies as atoms are brought together to form a crystal. The concept of allowed and forbidden energy bands can be developed more rigorously by considering quantum mechanics and Schrödinger’s wave equation. It may be easy for the reader to “get lost” in the following derivation, but the result forms the basis for the energy-band theory of semiconductors.

The potential function of a single, noninteracting, one-electron atom is shown in Figure 3.5a. Also indicated on the figure are the discrete energy levels allowed for the electron. Figure 3.5b shows the same type of potential function for the case when several atoms in close proximity are arranged in a one-dimensional array. The potential functions of adjacent atoms overlap, and the net potential function for this case is shown in Figure 3.5c. It is this potential function we would need to use in Schrödinger’s wave equation to model a one-dimensional single-crystal material.

The solution to Schrödinger’s wave equation, for this one-dimensional single-crystal lattice, is made more tractable by considering a simpler potential function. Figure 3.6 is the one-dimensional Kronig–Penney model of the periodic potential function, which is used to represent a one-dimensional single-crystal lattice. We need to solve Schrödinger’s wave equation in each region. As with previous quantum mechanical problems, the more interesting solution occurs for the case when \( E < V_0 \), which corresponds to a particle being bound within the crystal. The electrons are contained in the potential wells, but we have the possibility of tunneling between wells. The Kronig–Penney model is an idealized periodic potential representing a one-dimensional single crystal, but the results will illustrate many of the important features of the quantum behavior of electrons in a periodic lattice.

To obtain the solution to Schrödinger’s wave equation, we make use of a mathematical theorem by Bloch. The theorem states that all one-electron wave functions, for problems involving periodically varying potential energy functions, must be of the form

\[
\psi(x) = u(x)e^{j k x}
\]

(3.1)


*Indicates sections that will aid in the total summation of understanding of semiconductor devices, but may be skipped the first time through the text without loss of continuity.

\(^1\)Other techniques, such as the nearly free electron model, can be used to predict the energy-band theory of semiconductor materials. See, for example, Kittel [3] or Wolfe et al. [14].

**Figure 3.5** 
- (a) Potential function of a single isolated atom.
- (b) Overlapping potential functions of adjacent atoms.
- (c) Net potential function of a one-dimensional single crystal.


**Figure 3.6** 
The one-dimensional periodic potential function of the Kronig–Penney model.

- **(a)**: Shows the potential function \( V(x) \) for a single atom with energy levels \( E_1, E_2, E_3, E_4 \).
- **(b)**: Illustrates overlapping potential functions for multiple atoms.
- **(c)**: Depicts the net potential function for a one-dimensional crystal.

The potential function \( V(x) \) is shown with periodic barriers and wells, representing the Kronig–Penney model. The potential is periodic with intervals \( -a-b \) to \( a+b \), with potential \( V_0 \) in the barriers.

The parameter \( k \) is called a constant of motion and will be considered in more detail as we develop the theory. The function \( u(x) \) is a periodic function with period \( (a + b) \).

We stated in Chapter 2 that the total solution to the wave equation is the product of the time-independent solution and the time-dependent solution, or

\[
\Psi(x, t) = \psi(x)\phi(t) = u(x)e^{ikx} \cdot e^{-iEt/\hbar}
\]

(3.2)

which may be written as

\[
\Psi(x, t) = u(x)e^{i(kx-Et/\hbar)}
\]

(3.3)

This traveling-wave solution represents the motion of an electron in a single-crystal material. The amplitude of the traveling wave is a periodic function and the parameter \( k \) is also referred to as a wave number.

We can now begin to determine a relation between the parameter \( k \), the total energy \( E \), and the potential \( V_0 \). If we consider region I in Figure 3.6 (0 < x < a) in which \( V(x) = 0 \), take the second derivative of Equation (3.1), and substitute this result into the time-independent Schrödinger’s wave equation given by Equation (2.13), we obtain the relation

\[
\frac{d^2u(x)}{dx^2} + 2ijk \frac{du(x)}{dx} - (k^2 - \alpha^2)u(x) = 0
\]

(3.4)

The function \( u_1(x) \) is the amplitude of the wave function in region I and the parameter \( \alpha \) is defined as

\[
\alpha^2 = \frac{2mE}{\hbar^2}
\]

(3.5)

Consider now a specific region II, \(-b < x < 0\), in which \( V(x) = V_0 \), and apply Schrödinger’s wave equation. We obtain the relation

\[
\frac{d^2u(x)}{dx^2} + 2ijk \frac{du(x)}{dx} - \left[k^2 - \frac{2mV_0}{\hbar^2}\right]u(x) = 0
\]

(3.6)

where \( u_2(x) \) is the amplitude of the wave function in region II. We may define

\[
\frac{2m}{\hbar^2}(E - V_0) = \alpha^2 - \frac{2mV_0}{\hbar^2} = \beta^2
\]

(3.7)

so that Equation (3.6) may be written as

\[
\frac{d^2u(x)}{dx^2} + 2ijk \frac{du(x)}{dx} - (k^2 - \beta^2)u(x) = 0
\]

(3.8)

Note that from Equation (3.7), if \( E \geq V_0 \), the parameter \( \beta \) is real, whereas if \( E < V_0 \), then \( \beta \) is imaginary.

The solution to Equation (3.4), for region I, is of the form

\[
u_1(x) = Ae^{\alpha x} + Be^{-\alpha x} \quad \text{for} \quad (0 < x < a)
\]

(3.9)

and the solution to Equation (3.8), for region II, is of the form

\[
u_2(x) = Ce^{\beta x} + De^{-\beta x} \quad \text{for} \quad (-b < x < 0)
\]

(3.10)


Since the potential function \( V(x) \) is everywhere finite, both the wave function \( \psi(x) \) and its first derivative \( \partial \psi(x)/\partial x \) must be continuous. This continuity condition implies that the wave amplitude function \( u(x) \) and its first derivative \( \partial u(x)/\partial x \) must also be continuous.

If we consider the boundary at \( x = 0 \) and apply the continuity condition to the wave amplitude, we have

\[
u_1(0) = u_2(0)
\]

(3.11)

Substituting Equations (3.9) and (3.10) into Equation (3.11), we obtain

\[
A + B = C - D = 0
\]

(3.12)

Now applying the condition that

\[
\frac{du_1}{dx}\bigg|_{x=0} = \frac{du_2}{dx}\bigg|_{x=0}
\]

(3.13)

we obtain

\[
(\alpha - k)A - (\alpha + k)B = (\beta - K)C + (\beta + K)D = 0
\]

(3.14)

We have considered region I as \( 0 < x < a \) and region II as \(-b < x < 0\). The periodicity and the continuity condition mean that the function \( u_1 \), as \( x \to a \), is equal to the function \( u_2 \), as \( x \to -b \). This condition may be written as

\[
u_1(a) = u_2(-b)
\]

(3.15)

Applying the solutions for \( u_1(x) \) and \( u_2(x) \) to the boundary condition in Equation (3.15) yields

\[
Ae^{\alpha a} + Be^{-\alpha a} = Ce^{\beta b} - De^{-\beta b} = 0
\]

(3.16)

The last boundary condition is

\[
\frac{du_1}{dx}\bigg|_{x=a} = \frac{du_2}{dx}\bigg|_{x=-b}
\]

(3.17)

which gives

\[
(\alpha - k)Ae^{\alpha a} - (\alpha + k)Be^{-\alpha a} = (\beta - K)Ce^{\beta b} + (\beta + K)De^{-\beta b} = 0
\]

(3.18)

We now have four homogeneous equations, Equations (3.12), (3.14), (3.16), and (3.18), with four unknowns as a result of applying the four boundary conditions. In a set of simultaneous, linear, homogeneous equations, there is a nontrivial solution if, and only if, the determinant of the coefficients is zero. In our case, the coefficients in question are the coefficients of the parameters \( A, B, C, \) and \( D \).

The evaluation of this determinant is extremely laborious and will not be considered in detail. The result is

\[
-\frac{(\alpha^2 + \beta^2)}{2\alpha \beta} (\sin \alpha a \sin \beta b) + (\cos \alpha a \cos \beta b) = \cos k(a + b)
\]

(3.19)


Equation (3.19) relates the parameter \( k \) to the total energy \( E \) (through the parameter \( \alpha \)) and the potential function \( V_0 \) (through the parameter \( \beta \)).

As we mentioned, the more interesting solutions occur for \( E < V_0 \), which applies to the electron bound within the crystal. From Equation (3.7), the parameter \( \beta \) is then an imaginary quantity. We may define

\[
\beta = j\gamma
\]

where \( \gamma \) is a real quantity. Equation (3.19) can be written in terms of \( \gamma \) as

\[
\gamma^2 - \frac{\alpha^2}{\alpha\gamma} (\sin \alpha a)(\sinh \gamma b) + (\cos \alpha a)(\cosh \gamma b) = \cos k(a + b)
\]

Equation (3.21) does not lend itself to an analytical solution, but must be solved using numerical or graphical techniques to obtain the relation between \( k, E, \) and \( V_0 \).

The solution of Schrödinger’s wave equation for a single bound particle resulted in discrete allowed energies. The solution of Equation (3.21) will result in a band of allowed energies.

To obtain an equation that is more susceptible to a graphical solution and thus will illustrate the nature of the results, let the potential barrier width \( b \rightarrow 0 \) and the barrier height \( V_0 \rightarrow \infty \), but such that the product \( bV_0 \) remains finite. Equation (3.21) then reduces to

\[
\left( \frac{mV_0ba}{\hbar^2} \right) \frac{\sin \alpha a}{\alpha a} + \cos \alpha a = \cos ka
\]

We may define a parameter \( P' \) as

\[
P' = \frac{mV_0ba}{\hbar^2}
\]

Then, finally, we have the relation

\[
P' \frac{\sin \alpha a}{\alpha a} + \cos \alpha a = \cos ka
\]

Equation (3.24) again gives the relation between the parameter \( k \), total energy \( E \) (through the parameter \( \alpha \)), and the potential barrier \( bV_0 \). We may note that Equation (3.24) is not a solution of Schrödinger’s wave equation but gives the conditions for which Schrödinger’s wave equation will have a solution. If we assume that the crystal is infinitely large, then \( k \) in Equation (3.24) can assume a continuum of values and must be real.

### 3.1.3 The k-Space Diagram

To begin to understand the nature of the solution, initially consider the special case for which \( V_0 = 0 \). In this case \( P' = 0 \), which corresponds to a free particle since there are no potential barriers. From Equation (3.24), we have that

\[
\cos \alpha a = \cos ka
\]

or

\[
\alpha = k
\]


*Figure 3.7 | The parabolic \( E \) versus \( k \) curve for the free electron.*

Since the potential is equal to zero, the total energy \( E \) is equal to the kinetic energy, so that, from Equation (3.5), Equation (3.26) may be written as

\[
\alpha = \sqrt{\frac{2mE}{\hbar^2}} = \sqrt{\frac{2m\left(\frac{mv^2}{2}\right)}{\hbar^2}} = \frac{p}{\hbar} = k
\]

(3.27)

where \( p \) is the particle momentum. The constant of the motion parameter \( k \) is related to the particle momentum for the free electron. The parameter \( k \) is also referred to as a wave number.

We can also relate the energy and momentum as

\[
E = \frac{p^2}{2m} = \frac{k^2\hbar^2}{2m}
\]

(3.28)

Figure 3.7 shows the parabolic relation of Equation (3.28) between the energy \( E \) and momentum \( p \) for the free particle. Since the momentum and wave number are linearly related, Figure 3.7 is also the \( E \) versus \( k \) curve for the particle.

We now want to consider the relation between \( E \) and \( k \) from Equation (3.24) for the particle in the single-crystal lattice. As the parameter \( P' \) increases, the particle becomes more tightly bound to the potential well or atom. We may define the left side of Equation (3.24) to be a function \( f(\alpha a) \), so that

\[
f(\alpha a) = P' \frac{\sin \alpha a}{\alpha a} + \cos \alpha a
\]

(3.29)

Figure 3.8a is a plot of the first term of Equation (3.29) versus \( \alpha a \). Figure 3.8b shows a plot of the \( \cos \alpha a \) term and Figure 3.8c is the sum of the two terms, or \( f(\alpha a) \).

Now from Equation (3.24), we also have that

\[
f(\alpha a) = \cos ka
\]

(3.30)

For Equation (3.30) to be valid, the allowed values of the \( f(\alpha a) \) function must be bounded between \( +1 \) and \( -1 \). Figure 3.8c shows the allowed values of \( f(\alpha a) \) and the allowed values of \( \alpha a \) in the shaded areas. Also shown on the figure are the values of \( ka \) from the right side of Equation (3.30), which correspond to the allowed values of \( f(\alpha a) \).

**Figure 3.8** A plot of (a) the first term in Equation (3.29), (b) the second term in Equation (3.29), and (c) the entire \( f(\alpha a) \) function. The shaded areas show the allowed values of \( (\alpha a) \) corresponding to real values of \( k \).

The parameter \( \alpha \) is related to the total energy \( E \) of the particle through Equation (3.5), which is \( \alpha^2 = \frac{2mE}{\hbar^2} \). A plot of the energy \( E \) of the particle as a function of the wave number \( k \) can be generated from Figure 3.8c. **Figure 3.9 shows this plot and thus shows the concept of allowed energy bands for the particle propagating in the crystal lattice.** Since the energy \( E \) has discontinuities, we also have the concept of forbidden energies for the particles in the crystal.


## EXAMPLE 3.2

**Objective**:
Determine the width (in eV) of a forbidden energy band. Determine the width of the forbidden bandgap that exists at \( ka = \pi \) (see Figure 3.9). Assume that the coefficient \( p' = 8 \) and the potential width is \( a = 4.5 \, \text{Å} \).


**Figure 3.9** | The \( E \) versus \( k \) diagram generated from Figure 3.8. The allowed energy bands and forbidden energy bandgaps are indicated.


**Solution**
Combining Equations (3.29) and (3.30), we have

\[
\cos ka = p' \frac{\sin \alpha a}{\alpha a} + \cos \alpha a
\]

At \( ka = \pi \) and using \( P' = 8 \), we have

\[
-1 = g \frac{\sin \alpha a}{\alpha a} + \cos \alpha a
\]

We need to find the smallest values of \( \alpha a \) that satisfy this equation and then relate \( \alpha \) to the energy \( E \) to find the bandgap energy. From Figure 3.8, we see that, at one value of \( ka = \pi \), we have \( \alpha a = \pi = \alpha a \). Then

\[
\alpha a = \sqrt{\frac{2mE_i}{\hbar^2}} \cdot a = \pi
\]

or

\[
E_i = \frac{\pi^2 \hbar^2}{2ma^2} = \frac{\pi^2 (1.054 \times 10^{-34})^2}{2(9.11 \times 10^{-31})(4.5 \times 10^{-10})^2} = 2.972 \times 10^{-19} \, \text{J}
\]

From Figure 3.8, we see that, at the other value of \( ka = \pi \), \( \alpha a \) is in the range \( \pi < \alpha a < 2\pi \). By trial and error, we find \( \alpha a = 5.141 = \alpha a \). Then

\[
\alpha a = \sqrt{\frac{2mE_i}{\hbar^2}} \cdot a = 5.141
\]

or

\[
E_2 = \frac{(5.141)^2 \hbar^2}{2ma^2} = \frac{(5.141)(1.054 \times 10^{-34})^2}{2(9.11 \times 10^{-31})(4.5 \times 10^{-10})^2} = 7.958 \times 10^{-19} \, \text{J}
\]

The bandgap energy is then

\[
E_g = E_2 - E_i = 7.958 \times 10^{-19} - 2.972 \times 10^{-19} = 4.986 \times 10^{-19} \, \text{J}
\]

or

\[
E_g = \frac{4.986 \times 10^{-19}}{1.6 \times 10^{-19}} = 3.12 \, \text{eV}
\]

**Comment**

The results of this example give an order of magnitude of forbidden energy band widths.

**Exercise Problem**

**Ex 3.2** Using the parameters given in Example 3.2, determine the width of the allowed energy band in the range \(\pi < ka < 2\pi\).

Consider again the right side of Equation (3.24), which is the function \(\cos ka\). The cosine function is periodic so that

\[
\cos ka = \cos(ka + 2\pi n) = \cos(ka - 2\pi n)
\]

where \(n\) is a positive integer. We may consider Figure 3.9 and displace portions of the curve by \(2\pi\). Mathematically, Equation (3.24) is still satisfied. Figure 3.10 shows how various segments of the curve can be displaced by the \(2\pi\) factor. Figure 3.11 shows the case in which the entire \(E\) versus \(k\) plot is contained within \(-\pi/a < k < \pi/a\). This plot is referred to as a reduced \(k\)-space diagram, or a reduced-zone representation.

We noted in Equation (3.27) that for a free electron, the particle momentum and the wave number \(k\) are related by \(p = \hbar k\). Given the similarity between the free electron solution and the results of the single crystal shown in Figure 3.9, the parameter \(\hbar k\) in a single crystal is referred to as the crystal momentum. This parameter is not the actual momentum of the electron in the crystal but is a constant of the motion that includes the crystal interaction.


**Figure 3.10** The \(E\) versus \(k\) diagram showing \(2\pi\) displacements of several sections of allowed energy bands.


**Figure 3.11** The \(E\) versus \(k\) diagram in the reduced-zone representation.

We have been considering the Kronig–Penney model, which is a one-dimensional periodic potential function used to model a single-crystal lattice. The principal result of this analysis, so far, is that electrons in the crystal occupy certain allowed energy bands and are excluded from the forbidden energy bands. For real three-dimensional single-crystal materials, a similar energy-band theory exists. We will obtain additional electron properties from the Kronig–Penney model in the next sections.


## TEST YOUR UNDERSTANDING

**TYU 3.1** Using the parameters given in Example 3.2, determine the width (in eV) of the second forbidden energy band existing at \(ka = 2\pi\) (see Figure 3.8(c)).  
(\(\Delta E = ? \text{ eV}\))

**TYU 3.2** Using the parameters given in Example 3.2, determine the width (in eV) of the allowed energy band in the range \(0 < ka < \pi\) (see Figure 3.8(c)).  
(\(\Delta E = ? \text{ eV}\))


## 3.2 ELECTRICAL CONDUCTION IN SOLIDS

Again, we are eventually interested in determining the current–voltage characteristics of semiconductor devices. We will need to consider electrical conduction in solids as it relates to the band theory we have just developed. Let us begin by considering the motion of electrons in the various allowed energy bands.

### 3.2.1 The Energy Band and the Bond Model

In Chapter 1, we discussed the covalent bonding of silicon. Figure 3.12 shows a two-dimensional representation of the covalent bonding in a single-crystal silicon lattice. This figure represents silicon at \(T = 0 \, K\) in which each silicon atom is surrounded by eight valence electrons that are in their lowest energy state and are directly involved in the covalent bonding. Figure 3.4b represented the splitting of the discrete silicon energy states into bands of allowed energies as the silicon crystal is formed. At \(T = 0 \, K\), the \(4N\) states in the lower band, the valence band, are filled with the valence electrons. All of the valence electrons schematically shown in Figure 3.12 are in the valence band. The upper energy band, the conduction band, is completely empty at \(T = 0 \, K\).

As the temperature increases above \(0 \, K\), a few valence band electrons may gain enough thermal energy to break the covalent bond and jump into the conduction band. Figure 3.13a shows a two-dimensional representation of this bond-breaking process.

**Figure 3.12** | Two-dimensional representation of the covalent bonding in a semiconductor at \( T = 0 \, \text{K} \).

**Figure 3.13**  
(a) Two-dimensional representation of the breaking of a covalent bond.  
(b) Corresponding line representation of the energy band and the generation of a negative and positive charge with the breaking of a covalent bond.

The semiconductor is neutrally charged. This means that, as the negatively charged electron breaks away from its covalent bonding position, a positively charged "empty state" is created in the original covalent bonding position in the valence band. As the temperature further increases, more covalent bonds are broken, more electrons jump to the conduction band, and more positive "empty states" are created in the valence band.

We can also relate this bond breaking to the \( E \) versus \( k \) energy bands. **Figure 3.14a** shows the \( E \) versus \( k \) diagram of the conduction and valence bands at \( T = 0 \, \text{K} \). The energy states in the valence band are completely full and the states in the conduction band are empty. **Figure 3.14b** shows these same bands for \( T > 0 \, \text{K} \), in which some electrons have gained enough energy to jump to the conduction band and have left empty states in the valence band. We are assuming at this point that no external forces are applied so the electron and "empty state" distributions are symmetrical with \( k \).

**Figure 3.14** The \( E \) versus \( k \) diagram of the conduction and valence bands of a semiconductor at (a) \( T = 0 \, \text{K} \) and (b) \( T > 0 \, \text{K} \).

### 3.2.2 Drift Current

Current is due to the net flow of charge. If we had a collection of positively charged ions with a volume density \( N \, (\text{cm}^{-3}) \) and an average drift velocity \( v_d \, (\text{cm/s}) \), then the drift current density would be

\[
J = qNv_d \quad \text{A/cm}^2
\]

(3.32)

If, instead of considering the average drift velocity, we considered the individual ion velocities, then we could write the drift current density as

\[
J = q \frac{1}{V} \sum_{i=1}^{M} v_i
\]

(3.33)

where \( v_i \) is the velocity of the \( i \)th ion. The summation in Equation (3.33) is taken over a unit volume so that the current density \( J \) is still in units of A/cm\(^2\).

Since electrons are charged particles, a net drift of electrons in the conduction band will give rise to a current. The electron distribution in the conduction band, as shown in Figure 3.14b, is an even function of \( k \) when no external force is applied. Recall that \( k \) for a free electron is related to momentum so that, since there are as many electrons with a \( +|k| \) value as there are with a \( -|k| \) value, the net drift current density due to these electrons is zero. This result is certainly expected since there is no externally applied force.

If a force is applied to a particle and the particle moves, it must gain energy. This effect is expressed as

\[
dE = F \, dx = F \, v \, dt
\]

(3.34)

where \( F \) is the applied force, \( dx \) is the differential distance the particle moves, \( v \) is the velocity, and \( dE \) is the increase in energy. If an external force is applied to the electrons in the conduction band, there are empty energy states into which the electrons can move; therefore, because of the external force, electrons can gain energy and a

*Figure 3.15 | The asymmetric distribution of electrons in the E versus k diagram when an external force is applied.*

The electron distribution in the conduction band may look like that shown in Figure 3.15, which implies that the electrons have gained a net momentum.

We may write the drift current density due to the motion of electrons as

\[
J = -e \sum_{i} n_i v_i
\]

(3.35)

where \( e \) is the magnitude of the electronic charge and \( n \) is the number of electrons per unit volume in the conduction band. Again, the summation is taken over a unit volume so that the current density is still in units of A/cm\(^2\). We may note from Equation (3.35) that the current is directly related to the electron velocity; that is, the current is related to how well the electron can move in the crystal.

### 3.2.3 Electron Effective Mass

The movement of an electron in a lattice will, in general, be different from that of an electron in free space. In addition to an externally applied force, there are internal forces in the crystal due to positively charged ions or protons and negatively charged electrons, which will influence the motion of electrons in the lattice. We can write

\[
F_{\text{total}} = F_{\text{ext}} + F_{\text{int}} = ma
\]

(3.36)

where \( F_{\text{total}}, F_{\text{ext}}, \) and \( F_{\text{int}} \) are the total force, the externally applied force, and the internal forces, respectively, acting on a particle in a crystal. The parameter \( a \) is the acceleration and \( m \) is the rest mass of the particle.

Since it is difficult to take into account all of the internal forces, we will write the equation

\[
F_{\text{ext}} = m^* a
\]

(3.37)

where the acceleration \( a \) is now directly related to the external force. The parameter \( m^* \), called the effective mass, takes into account the particle mass and also takes into account the effect of the internal forces.

To use an analogy for the effective mass concept, consider the difference in motion between a glass marble in a container filled with water and in a container filled with oil. In general, the marble will drop through the water at a faster rate than through the oil. The external force in this example is the gravitational force and the internal forces are related to the viscosity of the liquids. Because of the difference in motion of the marble in these two cases, the mass of the marble would appear to be different in water than in oil. (As with any analogy, we must be careful not to be too literal.)

We can also relate the effective mass of an electron in a crystal to the \( E \) versus \( k \) curves, such as is shown in Figure 3.11. In a semiconductor material, we will be dealing with allowed energy bands that are almost empty of electrons and other energy bands that are almost full of electrons.

To begin, consider the case of a free electron whose \( E \) versus \( k \) curve is shown in Figure 3.7. Recalling Equation (3.28), the energy and momentum are related by

\[
E = p^2/2m = \hbar^2 k^2/2m
\]

where \( m \) is the mass of the electron. The momentum and wave number \( k \) are related by \( p = \hbar k \). If we take the derivative of Equation (3.28) with respect to \( k \), we obtain

\[
\frac{dE}{dk} = \frac{\hbar^2 k}{m} = \frac{\hbar p}{m}
\]

Relating momentum to velocity, Equation (3.38) can be written as

\[
\frac{1}{\hbar} \frac{dE}{dk} = \frac{p}{m} = v
\]

(3.39)

where \( v \) is the velocity of the particle. The first derivative of \( E \) with respect to \( k \) is related to the velocity of the particle.

If we now take the second derivative of \( E \) with respect to \( k \), we have

\[
\frac{d^2E}{dk^2} = \frac{\hbar^2}{m}
\]

(3.40)

We may rewrite Equation (3.40) as

\[
\frac{1}{\hbar} \frac{d^2E}{dk^2} = \frac{1}{m}
\]

(3.41)

The second derivative of \( E \) with respect to \( k \) is inversely proportional to the mass of the particle. For the case of a free electron, the mass is a constant (nonrelativistic effect), so the second derivative function is a constant. We may also note from Figure 3.7 that \( d^2E/dk^2 \) is a positive quantity, which implies that the mass of the electron is also a positive quantity.

If we apply an electric field to the free electron and use Newton’s classical equation of motion, we can write

\[
F = ma = -eE
\]

(3.42)

where \( a \) is the acceleration, \( E \) is the applied electric field, and \( e \) is the magnitude of the electronic charge. Solving for the acceleration, we have

\[
a = -\frac{eE}{m}
\]

(3.43)

The motion of the free electron is in the opposite direction to the applied electric field because of the negative charge.

We may now apply the results to the electron in the bottom of an allowed energy band. Consider the allowed energy band in Figure 3.16a. The energy near the bottom...

**Figure 3.16** (a) The conduction band in reduced \( k \) space, and the parabolic approximation. (b) The valence band in reduced \( k \) space, and the parabolic approximation.


The energy band may be approximated by a parabola, just as that of a free particle. We may write

\[
E - E_c = C_1 k^2
\]

(3.44)

The energy \( E_c \) is the energy at the bottom of the band. Since \( E > E_c \), the parameter \( C_1 \) is a positive quantity.

Taking the second derivative of \( E \) with respect to \( k \) from Equation (3.44), we obtain

\[
\frac{d^2E}{dk^2} = 2C_1
\]

(3.45)

We may put Equation (3.45) in the form

\[
\frac{1}{\hbar^2} \frac{d^2E}{dk^2} = \frac{2C_1}{\hbar^2}
\]

(3.46)

Comparing Equation (3.46) with Equation (3.41), we may equate \( \hbar^2/2C_1 \) to the mass of the particle. However, the curvature of the curve in Figure 3.16a will not, in general, be the same as the curvature of the free-particle curve. We may write

\[
\frac{1}{\hbar^2} \frac{d^2E}{dk^2} = \frac{2C_1}{\hbar^2} = \frac{1}{m^*}
\]

(3.47)

where \( m^* \) is called the effective mass. Since \( C_1 > 0 \), we have that \( m^* > 0 \) also.

The effective mass is a parameter that relates the quantum mechanical results to the classical force equations. In most instances, the electron in the bottom of the conduction band can be thought of as a classical particle whose motion can be modeled by Newtonian mechanics, provided that the internal forces and quantum mechanical properties are taken into account through the effective mass. If we apply an electric field to the electron in the bottom of the allowed energy band, we may write the acceleration as

\[
a = \frac{-eE}{m^*_e}
\]

(3.48)

where \( m^*_e \) is the effective mass of the electron. The effective mass \( m^*_e \) of the electron near the bottom of the conduction band is a constant.

### 3.2.4 Concept of the Hole

In considering the two-dimensional representation of the covalent bonding shown in Figure 3.13a, a positively charged “empty state” was created when a valence electron was elevated into the conduction band. For \( T > 0 \, \text{K} \), all valence electrons may gain thermal energy; if a valence electron gains a small amount of thermal energy, it may hop into the empty state. **The movement of a valence electron into the empty state is equivalent to the movement of the positively charged empty state itself.** Figure 3.17 shows the movement of valence electrons in the crystal, alternately filling one empty state and creating a new empty state—a motion equivalent to a positive charge moving in the valence band. The crystal now has a second equally important charge carrier that can give rise to a current. This charge carrier is called a **hole** and, as we will see, can also be thought of as a classical particle whose motion can be modeled using Newtonian mechanics.

The drift current density due to electrons in the valence band, such as shown in Figure 3.14b, can be written as

\[
J = -e \sum_{i \, ( \text{filled} )} v_i
\]

(3.49)

where the summation extends over all filled states. This summation is inconvenient since it extends over a nearly full valence band and takes into account a very large number of states. We may rewrite Equation (3.49) in the form

\[
J = -e \sum_{i \, ( \text{debit} )} v_i + e \sum_{i \, ( \text{empty} )} v_i
\]

(3.50)

If we consider a band that is totally full, all available states are occupied by electrons. The individual electrons can be thought of as moving with a velocity as given by Equation (3.39):

\[
v(E) = \left( \frac{1}{\hbar} \right) \frac{dE}{dk}
\]

(3.39)

**The band is symmetric in \( k \) and each state is occupied so that, for every electron with a velocity \( |v| \), there is a corresponding electron with a velocity \( -|v| \).** Since the band is full, the distribution of electrons with respect to \( k \) cannot be changed with an


**Figure 3.17** Visualization of the movement of a hole in a semiconductor.

**Figure 3.18** (a) Valence band with conventional electron-filled states and empty states. (b) Concept of positive charges occupying the original empty states.

The net drift current density generated from a completely full band, then, is zero, or

\[
-e \sum_{i \, (occ)} v_i = 0
\]

(3.51)

We can now write the drift current density from Equation (3.50) for an almost full band as

\[
J = +e \sum_{i \, (empty)} v_i
\]

(3.52)

where the \( v_i \) in the summation is the

\[
v(E) = \left( \frac{1}{\hbar} \right) \frac{dE}{dk}
\]

associated with the empty state. Equation (3.52) is entirely equivalent to placing a positively charged particle in the empty states and assuming all other states in the band are empty, or neutrally charged. This concept is shown in Figure 3.18. Figure 3.18a shows the valence band with the conventional electron-filled states and empty states, whereas Figure 3.18b shows the new concept of positive charges occupying the original empty states. This concept is consistent with the discussion of the positively charged "empty state" in the valence band, as shown in Figure 3.17.

The \( v_i \) in the summation of Equation (3.52) is related to how well this positively charged particle moves in the semiconductor. Now consider an electron near the top of the allowed energy band shown in Figure 3.16b. The energy near the top of the allowed energy band may again be approximated by a parabola so that we may write

\[
(E - E_s) = -C(k)^2
\]

(3.53)

The energy \( E_s \) is the energy at the top of the energy band. Since \( E < E_s \) for electrons in this band, the parameter \( C \) must be a positive quantity.

Taking the second derivative of \( E \) with respect to \( k \) from Equation (3.53), we obtain

\[
\frac{d^2E}{dk^2} = -2C
\]

(3.54)

We may rearrange this equation so that

\[
\frac{1}{\hbar^2} \frac{d^2E}{dk^2} = \frac{-2C}{\hbar^2}
\]

(3.55)

Comparing Equation (3.55) with Equation (3.41), we may write

\[
\frac{1}{\hbar^2} \frac{d^2 E}{dk^2} = -\frac{2C_2}{\hbar^2} = \frac{1}{m^*}
\]

(3.56)

where \( m^* \) is again an effective mass. We have argued that \( C_2 \) is a positive quantity, which now implies that \( m^* \) is a negative quantity. **An electron moving near the top of an allowed energy band behaves as if it has a negative mass.**

We must keep in mind that the effective mass parameter is used to relate quantum mechanics and classical mechanics. The attempt to relate these two theories leads to this strange result of a negative effective mass. However, we must recall that solutions to Schrödinger’s wave equation also led to results that contradicted classical mechanics. The negative effective mass is another such example.

In discussing the concept of effective mass in the previous section, we used an analogy of marbles moving through two liquids. Now consider placing an ice cube in the center of a container filled with water: the ice cube will move upward toward the surface in a direction opposite to the gravitational force. The ice cube appears to have a negative effective mass since its acceleration is opposite to the external force. The effective mass parameter takes into account all internal forces acting on the particle.

If we again consider an electron near the top of an allowed energy band and use Newton’s force equation for an applied electric field, we will have

\[
F = m^* a = -eE
\]

(3.57)

However, \( m^* \) is now a negative quantity, so we may write

\[
a = -\frac{eE}{|m^*|} = +\frac{eE}{|m^*|}
\]

(3.58)

**An electron moving near the top of an allowed energy band moves in the same direction as the applied electric field.**

The net motion of electrons in a nearly full band can be described by considering just the empty states, provided that a positive electronic charge is associated with each state and that the negative of \( m^* \) from Equation (3.56) is associated with each state. We now can model this band as having particles with a positive electronic charge and a positive effective mass. The density of these particles in the valence band is the same as the density of empty electronic energy states. This new particle is the hole. The hole, then, has a positive effective mass denoted by \( m^*_h \) and a positive electronic charge, so it will move in the same direction as an applied field.

### 3.2.5 Metals, Insulators, and Semiconductors

Each crystal has its own energy-band structure. We noted that the splitting of the energy states in silicon, for example, to form the valence and conduction bands, was complex. Complex band splitting occurs in other crystals, leading to large variations in band structures between various solids and to a wide range of electrical characteristics observed in these various materials. We can qualitatively begin to understand some basic differences in electrical characteristics caused by variations in band structure by considering some simplified energy bands.


**Figure 3.19** Allowed energy bands showing (a) an empty band, (b) a completely full band, and (c) the bandgap energy between the two allowed bands.

- **(a)** Allowed energy band (empty)
- **(b)** Allowed energy band (full)
- **(c)** Conduction band (empty), Valence band (full), \( E_g \)


**Figure 3.20** Allowed energy bands showing (a) an almost empty band, (b) an almost full band, and (c) the bandgap energy between the two allowed bands.

- **(a)** Allowed energy band (almost empty)
- **(b)** Allowed energy band (almost full)
- **(c)** Conduction band (almost empty), Valence band (almost full), Electrons, Empty electronic states, \( E_g \)

There are several possible energy-band conditions to consider. Figure 3.19a shows an allowed energy band that is completely empty of electrons. If an electric field is applied, there are no particles to move, so there will be no current. Figure 3.19b shows another allowed energy band whose energy states are completely full of electrons. We argued in the previous section that a completely full energy band will also not give rise to a current. A material that has energy bands either completely empty or completely full is an insulator. The resistivity of an insulator is very large or, conversely, the conductivity of an insulator is very small. There are essentially no charged particles that can contribute to a drift current. Figure 3.19c shows a simplified energy-band diagram of an insulator. The bandgap energy \( E_g \) of an insulator is usually on the order of 3.5 to 6 eV or larger, so that at room temperature, there are essentially no electrons in the conduction band and the valence band remains completely full. There are very few thermally generated electrons and holes in an insulator.

Figure 3.20a shows an energy band with relatively few electrons near the bottom of the band. Now, if an electric field is applied, the electrons can gain energy, move to higher energy states, and move through the crystal. The net flow of charge is a current. Figure 3.20b shows an allowed energy band that is almost full of electrons, which means that we can consider the holes in this band. If an electric field is applied, the holes can move and give rise to a current. Figure 3.20c shows the simplified energy-band diagram for this case. The bandgap energy may be on the order of 1 eV.

**Figure 3.21**

Two possible energy bands of a metal showing:

- **(a)** a partially filled band
- **(b)** overlapping allowed energy bands.


This energy-band diagram represents a semiconductor for \( T > 0 \, \text{K} \). The resistivity of a semiconductor, as we will see in the next chapter, can be controlled and varied over many orders of magnitude.

The characteristics of a metal include a very low resistivity. The energy-band diagram for a metal may be in one of two forms. Figure 3.21a shows the case of a partially full band in which there are many electrons available for conduction, so that the material can exhibit a large electrical conductivity. Figure 3.21b shows another possible energy-band diagram of a metal. The band splitting into allowed and forbidden energy bands is a complex phenomenon, and Figure 3.21b shows a case in which the conduction and valence bands overlap at the equilibrium interatomic distance. As in the case shown in Figure 3.21a, there are large numbers of electrons as well as large numbers of empty energy states into which the electrons can move, so this material can also exhibit a very high electrical conductivity.

## Test Your Understanding

**TYU 3.3**

A simplified \( E \) versus \( k \) curve for an electron in the conduction band is given. The value of \( a \) is 10 Å. Determine the relative effective mass \( m^*/m_0 \).  
(\( 5L1T = \frac{\partial u}{\partial u} \, s^2V \))

**TYU 3.4**

A simplified \( E \) versus \( k \) curve for a hole in the valence band is given. Assume a value of \( a = 12 \, \text{Å} \). Determine the relative effective mass \( m^*/m_0 \).  
(\( 586T0 = \frac{\partial u}{\partial u} \, s^2V \))


- **Figure 3.22**: Figure for Exercise TYU 3.3.


  \[
  E = E_c + 0.32 \, \text{eV}
  \]

- **Figure 3.23**: Figure for Exercise TYU 3.4.


  \[
  E = E_v - 0.875 \, \text{eV}
  \]

## 3.3 Extension to Three Dimensions

### 3.3.1 Extension to Three Dimensions

The basic concepts of allowed and forbidden energy bands and effective mass have been developed in the previous sections. In this section, we will extend these concepts to three dimensions and to real crystals. We will qualitatively consider particular characteristics of the three-dimensional crystal in terms of the \( E \) versus \( k \) plots, bandgap energy, and effective mass. We must emphasize that we will only briefly touch on the basic three-dimensional concepts; therefore, many details will not be considered.

One problem encountered in extending the potential function to a three-dimensional crystal is that the distance between atoms varies as the direction through the crystal changes. Figure 3.24 shows a face-centered cubic structure with the [100] and [110] directions indicated. Electrons traveling in different directions encounter different potential patterns and therefore different \( k \)-space boundaries. The \( E \) versus \( k \) diagrams are, in general, a function of the \( k \)-space direction in a crystal.

#### 3.3.1.1 The \( k \)-Space Diagrams of Si and GaAs

Figure 3.25 shows an \( E \) versus \( k \) diagram of gallium arsenide (GaAs) and of silicon (Si). These simplified diagrams show the basic properties considered in this text but do not show many of the details more appropriate for advanced-level courses.

Note that in place of the usual positive and negative \( k \) axes, we now show two different crystal directions. The \( E \) versus \( k \) diagram for the one-dimensional model was symmetric in \( k \) so that no new information is obtained by displaying the negative axis. It is normal practice to plot the [100] direction along the normal \( +k \) axis and to plot the [111] portion of the diagram so the \( +k \) points to the left. In the case of diamond or zincblende lattices, the maxima in the valence band energy and minima in the conduction band energy occur at \( k = 0 \) or along one of these two directions.

Figure 3.25a shows the \( E \) versus \( k \) diagram for GaAs. The valence band maximum and the conduction band minimum both occur at \( k = 0 \). The electrons in the


**Figure 3.24** | The (100) plane of a face-centered cubic crystal showing the [100] and [110] directions.

**Figure 3.25** Energy-band structures of (a) GaAs and (b) Si.  
*(From Sze [12].)*

The conduction band tends to settle at the minimum conduction band energy that is at \( k = 0 \). Similarly, holes in the valence band tend to congregate at the uppermost valence band energy. In GaAs, the minimum conduction band energy and maximum valence band energy occur at the same \( k \) value. A semiconductor with this property is said to be a **direct bandgap semiconductor**; transitions between the two allowed bands can take place with no change in crystal momentum. This direct nature has a significant effect on the optical properties of the material. GaAs and other direct bandgap materials are ideally suited for use in semiconductor lasers and other optical devices.

The \( E \) versus \( k \) diagram for silicon is shown in Figure 3.25b. The maximum in the valence band energy occurs at \( k = 0 \) as before. The minimum in the conduction band energy occurs not at \( k = 0 \), but along the [100] direction. The difference between the minimum conduction band energy and the maximum valence band energy is still defined as the bandgap energy \( E_g \). A semiconductor whose maximum valence band energy and minimum conduction band energy do not occur at the same \( k \) value is called an **indirect bandgap semiconductor**. When electrons make a transition between the conduction and valence bands, we must invoke the law of conservation of momentum. A transition in an indirect bandgap material must necessarily include an interaction with the crystal so that crystal momentum is conserved.

**Germanium is also an indirect bandgap material**, whose valence band maximum occurs at \( k = 0 \) and whose conduction band minimum occurs along the [111] direction. GaAs is a direct bandgap semiconductor, but other compound semiconductors, such as GaP and AlAs, have indirect bandgaps.

### 3.3.2 Additional Effective Mass Concepts

The curvature of the \( E \) versus \( k \) diagrams near the minimum of the conduction band energy is related to the effective mass of the electron. We may note from Figure 3.25 that the curvature of the conduction band at its minimum value for GaAs is larger than that of silicon, so the effective mass of an electron in the conduction band of GaAs will be smaller than that in silicon.

For the one-dimensional \( E \) versus \( k \) diagram, the effective mass was defined by Equation (3.41) as \( 1/m^* = 1/\hbar^2 \cdot d^2E/dk^2 \). A complication occurs in the effective mass concept in a real crystal. A three-dimensional crystal can be described by three \( k \) vectors. The curvature of the \( E \) versus \( k \) diagram at the conduction band minimum may not be the same in the three \( k \) directions. In later sections and chapters, the effective mass parameters used in calculations will be a kind of statistical average that is adequate for most device calculations.²

## 3.4 Density of States Function

As we have stated, we eventually wish to describe the current–voltage characteristics of semiconductor devices. Since current is due to the flow of charge, an important step in the process is to determine the number of electrons and holes in the semiconductor that will be available for conduction. The number of carriers that can contribute to the conduction process is a function of the number of available energy or quantum states since, by the Pauli exclusion principle, only one electron can occupy a given quantum state. When we discussed the splitting of energy levels into bands of allowed and forbidden energies, we indicated that the band of allowed energies was actually made up of discrete energy levels. We must determine the density of these allowed energy states as a function of energy in order to calculate the electron and hole concentrations.

### 3.4.1 Mathematical Derivation

To determine the density of allowed quantum states as a function of energy, we need to consider an appropriate mathematical model. Electrons are allowed to move relatively freely in the conduction band of a semiconductor but are confined to the crystal. As a first step, we will consider a free electron confined to a three-dimensional infinite potential well, where the potential well represents the crystal. The potential of the infinite potential well is defined as

\[
V(x, y, z) = 
\begin{cases} 
0 & \text{for } 0 < x < a \\ 
0 & \text{for } 0 < y < a \\ 
0 & \text{for } 0 < z < a 
\end{cases}
\]

\[
V(x, y, z) = \infty \quad \text{elsewhere}
\]

where the crystal is assumed to be a cube with length \( a \). Schrödinger’s wave equation in three dimensions can be solved by using the separation of variables technique.


²See Appendix F for further discussion of effective mass concepts.


**Figure 3.26** (a) A two-dimensional array of allowed quantum states in \( k \) space. (b) The positive one-eighth of the spherical \( k \) space.

Extrapolating the results from the one-dimensional infinite potential well, we can show (see Problem 3.23) that

\[
\frac{2mE}{\hbar^2} = k^2 = k_x^2 + k_y^2 + k_z^2 = \left( n_x^2 + n_y^2 + n_z^2 \right) \left( \frac{\pi}{a} \right)^2
\]

(3.60)

where \( n_x, n_y, \) and \( n_z \) are positive integers. (Negative values of \( n_x, n_y, \) and \( n_z \) yield the same wave function, except for the sign, as the positive integer values, resulting in the same probability function and energy; therefore the negative integers do not represent a different quantum state.)

We can schematically plot the allowed quantum states in \( k \) space. Figure 3.26a shows a two-dimensional plot as a function of \( k_x \) and \( k_z \). Each point represents an allowed quantum state corresponding to various integral values of \( n_x \) and \( n_z \). Positive and negative values of \( k_x, k_y, \) or \( k_z \) have the same energy and represent the same energy state. Since negative values of \( k_x, k_y, \) or \( k_z \) do not represent additional quantum states, the density of quantum states will be determined by considering only the positive one-eighth of the spherical \( k \) space as shown in Figure 3.26b.

The distance between two quantum states in the \( k_x \) direction, for example, is given by

\[
k_{x,n+1} - k_{x,n} = (n_x + 1) \left( \frac{\pi}{a} \right) - n_x \left( \frac{\pi}{a} \right) = \frac{\pi}{a}
\]

(3.61)

Generalizing this result to three dimensions, the volume \( V_k \) of a single quantum state is

\[
V_k = \left( \frac{\pi}{a} \right)^3
\]

(3.62)

We can now determine the density of quantum states in \( k \) space. A differential volume in \( k \) space is shown in Figure 3.26b and is given by \( 4\pi k^2 dk \) so the differential density of quantum states in \( k \) space can be written as

\[
g_T(k) \, dk = 2 \left( \frac{1}{8} \right) \frac{4\pi k^2 \, dk}{\left( \frac{\pi}{a} \right)^3}
\]

(3.63)

The first factor, 2, takes into account the two spin states allowed for each quantum state; the next factor, \(\frac{1}{8}\), takes into account that we are considering only the quantum states for positive values of \(k_x\), \(k_y\), and \(k_z\). The factor \(4\pi k^2 dk\) is again the differential volume and the factor \((\pi a^3)\) is the volume of one quantum state. Equation (3.63) may be simplified to

\[
g_r(k)dk = \frac{\pi k^2 dk}{a^3}
\]

(3.64)

Equation (3.64) gives the density of quantum states as a function of momentum, through the parameter \(k\). We can now determine the density of quantum states as a function of energy \(E\). For a free electron, the parameters \(E\) and \(k\) are related by

\[
k^2 = \frac{2mE}{\hbar^2}
\]

(3.65a)

or

\[
k = \frac{1}{\hbar} \sqrt{2mE}
\]

(3.65b)

The differential \(dk\) is

\[
dk = \frac{1}{\hbar \sqrt{2E}} \sqrt{m} \, dE
\]

(3.66)

Then, substituting the expressions for \(k^2\) and \(dk\) into Equation (3.64), the number of energy states between \(E\) and \(E + dE\) is given by

\[
g_r(E)dE = \frac{\pi a^3}{\pi^3} \left(\frac{2mE}{\hbar^2}\right) \cdot \frac{1}{\hbar} \cdot \frac{\sqrt{m}}{\sqrt{2E}} \, dE
\]

(3.67)

Since \(\hbar = \frac{h}{2\pi}\), Equation (3.67) becomes

\[
g_r(E)dE = \frac{4\pi a^3}{h^3} \cdot (2m)^{3/2} \cdot \sqrt{E} \, dE
\]

(3.68)

Equation (3.68) gives the total number of quantum states between the energy \(E\) and \(E + dE\) in the crystal space volume of \(a^3\). If we divide by the volume \(a^3\), then we will obtain the density of quantum states per unit volume of the crystal. Equation (3.68) then becomes

\[
g(E) = \frac{4\pi(2m)^{3/2}}{h^3} \sqrt{E}
\]

(3.69)

The density of quantum states is a function of energy \(E\). As the energy of this free electron becomes small, the number of available quantum states decreases. This density function is really a double density, in that the units are given in terms of states per unit energy per unit volume.


## EXAMPLE 3.3

**Objective:** Calculate the density of states per unit volume over a particular energy range.


Consider the density of states for a free electron given by Equation (3.69). Calculate the density of states per unit volume with energies between 0 and 1 eV.

**Solution**

The volume density of quantum states, from Equation (3.69), is given by:

\[
N = \int_{0}^{E} g(E) \, dE = \frac{4\pi(2m)^{3/2}}{h^3} \cdot \int_{0}^{E} \sqrt{E} \, dE
\]

or

\[
N = \frac{4\pi(2m)^{3/2}}{h^3} \cdot \frac{2}{3} \cdot E^{3/2}
\]

The density of states is now:

\[
N = \frac{4\pi(29.11 \times 10^{-31})^{3/2}}{(6.625 \times 10^{-34})^3} \cdot \frac{2}{3} \cdot (1.6 \times 10^{-19})^{3/2} = 4.5 \times 10^{21} \, \text{m}^{-3}
\]

or

\[
N = 4.5 \times 10^{21} \, \text{states/cm}^3
\]

**Comment**

The density of quantum states is typically a large number. An effective density of states in a semiconductor, as we will see in the following sections and in the next chapter, is also a large number but is usually less than the density of atoms in the semiconductor crystal.

**Exercise Problem**

**Ex 3.3** For a free electron, calculate the density of quantum states (#/cm³) over the energy range of (a) \(0 \leq E \leq 2.0 \, \text{eV}\) and (b) \(1 \leq E \leq 2 \, \text{eV}\).

### 3.4.2 Extension to Semiconductors

In the previous section, we derived a general expression for the density of allowed electron quantum states using the model of a free electron with mass \(m\) bounded in a three-dimensional infinite potential well. We can extend this same general model to a semiconductor to determine the density of quantum states in the conduction band and the density of quantum states in the valence band. Electrons and holes are confined within the semiconductor crystal, so we will again use the basic model of the infinite potential well.

The parabolic relationship between energy and momentum of a free electron is given in Equation (3.28) as \(E = p^2/2m = \hbar^2k^2/2m\). Figure 3.16a shows the conduction energy band in the reduced \(k\) space. The \(E\) versus \(k\) curve near \(k = 0\) at the bottom of the conduction band can be approximated as a parabola, so we may write:

\[
E = E_c + \frac{\hbar^2k^2}{2m^*}
\]

where \(E_c\) is the bottom edge of the conduction band and \(m^*\) is the electron density of states effective mass. Equation (3.70) may be rewritten to give:

\[
E = E_c + \frac{\hbar^2k^2}{2m^*}
\]


*Again, see Appendix F for further discussion of effective mass concepts.*

The general form of the \( E \) versus \( k \) relation for an electron in the bottom of a conduction band is the same as the free electron, except the mass is replaced by the effective mass. We can then think of the electron in the bottom of the conduction band as being a "free" electron with its own particular mass. The right side of Equation (3.71) is of the same form as the right side of Equation (3.28), which was used in the derivation of the density of states function. Because of this similarity, which yields the "free" conduction electron model, we may generalize the free electron results of Equation (3.69) and write the density of allowed electronic energy states in the conduction band as

\[
g_c(E) = \frac{4\pi(2m_n)^{3/2}}{h^3} \sqrt{E - E_c}
\]

(3.72)

Equation (3.72) is valid for \( E \geq E_c \). As the energy of the electron in the conduction band decreases, the number of available quantum states also decreases.

The density of quantum states in the valence band can be obtained by using the same infinite potential well model, since the hole is also confined in the semiconductor crystal and can be treated as a "free" particle. The density of states effective mass of the hole is \( m_p^* \). Figure 3.16b shows the valence energy band in the reduced \( k \) space. We may also approximate the \( E \) versus \( k \) curve near \( k = 0 \) by a parabola for a "free" hole, so that

\[
E = E_v - \frac{\hbar^2 k^2}{2m_p^*}
\]

(3.73)

Equation (3.73) may be rewritten to give

\[
E_v - E = \frac{\hbar^2 k^2}{2m_p^*}
\]

(3.74)

Again, the right side of Equation (3.74) is of the same form used in the general derivation of the density of states function. We may then generalize the density of states function from Equation (3.69) to apply to the valence band, so that

\[
g_v(E) = \frac{4\pi(2m_p^*)^{3/2}}{h^3} \sqrt{E_v - E}
\]

(3.75)

Equation (3.75) is valid for \( E \leq E_v \).

We have argued that quantum states do not exist within the forbidden energy band, so \( g(E) = 0 \) for \( E_c < E < E_v \). Figure 3.27 shows the plot of the density of quantum states as a function of energy. If the electron and hole effective masses were equal, then the functions \( g_c(E) \) and \( g_v(E) \) would be symmetrical about the energy midway between \( E_c \) and \( E_v \), or the midgap energy, \( E_{\text{midgap}} \).


**Figure 3.27** The density of energy states in the conduction band and the density of energy states in the valence band as a function of energy.

## EXAMPLE 3.4

**Objective:** Determine the number (#/cm³) of quantum states in silicon between \( E_c \) and \( E_c + kT \) at \( T = 300 \, K \).

**Solution**

Using Equation (3.72), we can write

\[
N = \int_{E_c}^{E_c + kT} \frac{4\pi (2m_n^*)^{3/2}}{h^3} \sqrt{E - E_c} \cdot dE
\]

\[
= \frac{4\pi (2m_n^*)^{3/2}}{h^3} \cdot \frac{2}{3} \cdot (E - E_c)^{3/2} \bigg|_{E_c}^{E_c + kT}
\]

\[
= \frac{4\pi (2 \times 1.08 \times 9.11 \times 10^{-31})^{3/2}}{(6.625 \times 10^{-34})^3} \cdot \frac{2}{3} \cdot [(0.0259) \times (1 \times 10^{-19})]^{3/2}
\]

\[
= 2.12 \times 10^3 \, \text{m}^{-3}
\]

or

\[
N = 2.12 \times 10^{19} \, \text{cm}^{-3}
\]

**Comment**

The result of this example shows the order of magnitude of the density of quantum states in a semiconductor.

**Exercise Problem**

Ex3.4 Determine the number (#/cm³) of quantum states in silicon between \( (E_v - kT) \) and \( E_v \) at \( T = 300 \, K \).

## 3.5 Statistical Mechanics
In dealing with large numbers of particles, we are interested only in the statistical behavior of the group as a whole rather than in the behavior of each individual particle. For example, gas within a container will exert an average pressure on the walls of the vessel. The pressure is actually due to the collisions of the individual gas molecules with the walls, but we do not follow each individual molecule as it collides with the wall. Likewise in a crystal, the electrical  characteristics will  be determined by the statistical behavior of a large number of electrons. 

### 3.5.1  Statistical Laws 
In determining the statistical behavior of particles, we must consider the laws that the particles obey. There are three distribution laws determining the distribution of particles among available energy states. 

One distribution law is the Maxwell–Boltzmann probability function. In this case, the particles are considered to be distinguishable by being numbered, for example, from 1 to N, with no limit to the number of particles allowed in each energy state. The behavior of gas molecules in a container at fairly low pressure is an example of this distribution. 
A second distribution law is the Bose–Einstein function. The particles in this case are indistinguishable and, again, there is no limit to the number of particles permitted in each quantum state. The behavior of photons, or black body radiation, is an example of this law. 
The third distribution law is the Fermi–Dirac probability function. In this case, the particles are again indistinguishable, but now only one particle is permitted in each quantum state. Electrons in a crystal obey this law. In each case, the particles are assumed to be noninteracting. 

### 3.5.2 The Fermi-Dirac Probability Function

Figure 3.28 shows the $i$th energy level with $g_i$ quantum states. A maximum of one particle is allowed in each quantum state by the Pauli exclusion principle. There are $g_i$ ways of choosing where to place the first particle, $(g_i - 1)$ ways of choosing where to place the second particle, $(g_i - 2)$ ways of choosing where to place the third particle, and so on. Then the total number of ways of arranging $N_i$ particles in the $i$th energy level (where $N_i \leq g_i$) is

\[
(g_i)(g_i - 1) \cdots (g_i - (N_i - 1)) = \frac{g_i!}{(g_i - N_i)!}
\]
(3.76)

This expression includes all permutations of the $N_i$ particles among themselves.

**Figure 3.28** | The $i$th energy level with $g_i$ quantum states. 

However, since the particles are indistinguishable, the \(N_i!\) number of permutations that the particles have among themselves in any given arrangement do not count as separate arrangements. The interchange of any two electrons, for example, does not produce a new arrangement. Therefore, the actual number of independent ways of realizing a distribution of \(N_i\) particles in the \(i\)th level is

\[
W_i = \frac{g_i!}{N_i!(g_i - N_i)!}
\]

(3.77)

## Example 3.5

**Objective:** Determine the possible number of ways of realizing a particular distribution for (a) \(g_i = N_i = 10\) and (b) \(g_i = 10, N_i = 9\).

**Solution**

(a) \(g_i = N_i = 10\): We may note that \((g_i - N_i)! = 0! = 1\). Then, from Equation (3.77), we find

\[
\frac{g_i!}{N_i!(g_i - N_i)!} = \frac{10!}{10! \cdot 1} = 1
\]

(b) \(g_i = 10, N_i = 9\): We may note that \((g_i - N_i)! = 1! = 1\). Then, we find

\[
\frac{g_i!}{N_i!(g_i - N_i)!} = \frac{10!}{9!(1!)} = \frac{(10)(9!)}{9!} = 10
\]

**Comment**

In part (a), we have 10 particles to be arranged in 10 quantum states. There is only one possible arrangement. Each quantum state contains one particle. In part (b), we have 9 particles to be arranged in 10 quantum states. There is one empty quantum state, and there are 10 possible positions in which that empty state may occur. Thus, there are 10 possible arrangements for this case.

**Exercise Problem**

**Ex3.5** Determine the possible number of ways of realizing a particular distribution if \(g_i = 10\) and \(N_i = 8\). (SPICE)

Equation (3.77) gives the number of independent ways of realizing a distribution of \(N_i\) particles in the \(i\)th level. The total number of ways of arranging \((N_1, N_2, N_3, \ldots, N_n)\) indistinguishable particles among \(n\) energy levels is the product of all distributions, or

\[
W = \prod_{i=1}^{n} \frac{g_i!}{N_i!(g_i - N_i)!}
\]

(3.78)

The parameter \(W\) is the total number of ways in which \(N\) electrons can be arranged in this system, where \(N = \sum_{i=1}^{n} N_i\) is the total number of electrons in the system. We want to find the most probable distribution, which means that we want to find the maximum \(W\). The maximum \(W\) is found by varying \(N_i\) among the \(E_i\) levels, which varies the distribution, but at the same time, we will keep the total number of particles and total energy constant.

We may write the most probable distribution function as

\[
\frac{N(E)}{g(E)} = f_F(E) = \frac{1}{1 + \exp\left(\frac{E - E_F}{kT}\right)}
\]

(3.79)

The number density \( N(E) \) is the number of particles per unit volume per unit energy and the function \( g(E) \) is the number of quantum states per unit volume per unit energy. The function \( f(E) \) is called the **Fermi–Dirac distribution** or probability function and gives the probability that a quantum state at the energy \( E \) will be occupied by an electron. The energy \( E_F \) is called the **Fermi energy**. Another interpretation of the distribution function is that \( f(E) \) is the ratio of filled to total quantum states at any energy \( E \).

### 3.5.3 The Distribution Function and the Fermi Energy

To begin to understand the meaning of the distribution function and the Fermi energy, we can plot the distribution function versus energy. Initially, let \( T = 0 \) K and consider the case when \( E < E_F \). The exponential term in Equation (3.79) becomes \(\exp[(E - E_F)/kT] = \exp(-\infty) = 0\). The resulting distribution function is \( f(E < E_F) = 1 \).

Again let \( T = 0 \) K and consider the case when \( E > E_F \). The exponential term in the distribution function becomes \(\exp[(E - E_F)/kT] = \exp(+\infty) = +\infty\). The resulting Fermi–Dirac distribution function now becomes \( f(E > E_F) = 0 \).

The Fermi–Dirac distribution function for \( T = 0 \) K is plotted in Figure 3.29. This result shows that, for \( T = 0 \) K, the electrons are in their lowest possible energy states. The probability of a quantum state being occupied is unity for \( E < E_F \) and the probability of a state being occupied is zero for \( E > E_F \). All electrons have energies below the Fermi energy at \( T = 0 \) K.

Figure 3.30 shows discrete energy levels of a particular system as well as the number of available quantum states at each energy. If we assume, for this case, that the system contains 13 electrons, then Figure 3.30 shows how these electrons are distributed among the various quantum states at \( T = 0 \) K. The electrons will be in the lowest possible energy state, so the probability of a quantum state being occupied in energy levels \( E_1 \) through \( E_3 \) is unity, and the probability of a quantum state being occupied in energy level \( E_4 \) is zero. The Fermi energy, for this case, must be above \( E_4 \) but less than \( E_5 \). The Fermi energy determines the statistical distribution of electrons and does not have to correspond to an allowed energy level.

Now consider a case in which the density of quantum states \( g(E) \) is a continuous function of energy, as shown in Figure 3.31. If we have \( N_0 \) electrons in this system,

**Figure 3.29** The Fermi probability function versus energy for \( T = 0 \) K.

**Figure 3.30** Discrete energy states and quantum states for a particular system at \( T = 0 \) K.

**Figure 3.31** Density of quantum states and electrons in a continuous energy system at \( T = 0 \, \text{K} \).

\[
N_s = \int_{0}^{E_F} g(E) \, dE
\]


**Figure 3.32** Discrete energy states and quantum states for the same system shown in Figure 3.30 for \( T > 0 \, \text{K} \).


When the distribution of these electrons among the quantum states at \( T = 0 \, \text{K} \) is shown by the dashed line, the electrons are in the lowest possible energy state so that all states below \( E_F \) are filled and all states above \( E_F \) are empty. If \( g(E) \) and \( N_s \) are known for this particular system, then the Fermi energy \( E_F \) can be determined.

Consider the situation when the temperature increases above \( T = 0 \, \text{K} \). Electrons gain a certain amount of thermal energy so that some electrons can jump to higher energy levels, which means that the distribution of electrons among the available energy states will change. Figure 3.32 shows the same discrete energy levels and quantum states as in Figure 3.30. The distribution of electrons among the quantum states has changed from the \( T = 0 \, \text{K} \) case. Two electrons from the \( E_4 \) level have gained enough energy to jump to \( E_5 \), and one electron from \( E_2 \) has jumped to \( E_4 \). As the temperature changes, the distribution of electrons versus energy changes.

The change in the electron distribution among energy levels for \( T > 0 \, \text{K} \) can be seen by plotting the Fermi–Dirac distribution function. If we let \( E = E_F \) and \( T > 0 \, \text{K} \), then Equation (3.79) becomes

\[
f_k(E = E_F) = \frac{1}{1 + \exp(0)} = \frac{1}{1 + 1} = \frac{1}{2}
\]

The probability of a state being occupied at \( E = E_F \) is \( \frac{1}{2} \). Figure 3.33 shows the Fermi–Dirac distribution function plotted for several temperatures, assuming that the Fermi energy is independent of temperature.

**Figure 3.33** The Fermi probability function versus energy for different temperatures.

## EXAMPLE 3.6

**Objective**: 
Calculate the probability that an energy state above \( E_F \) is occupied by an electron.

Let \( T = 300 \, \text{K} \). Determine the probability that an energy level \( 3kT \) above the Fermi energy is occupied by an electron.

**Solution**

From Equation (3.79), we can write

\[
f(E) = \frac{1}{1 + \exp\left(\frac{E - E_F}{kT}\right)} = \frac{1}{1 + \exp\left(\frac{3kT}{kT}\right)}
\]

which becomes

\[
f(E) = \frac{1}{1 + 20.09} = 0.0474 = 4.74\%
\]

**Comment**

At energies above \( E_h \), the probability of a state being occupied by an electron can become significantly less than unity, or the ratio of electrons to available quantum states can be quite small.

**Exercise Problem**

**Ex 3.6** Assume the Fermi energy level is 0.30 eV below the conduction band energy \( E_c \).

- Assume \( T = 300 \, \text{K} \). (a) Determine the probability of a state being occupied by an electron at \( E = E_F + kT/4 \). (b) Repeat part (a) for an energy state at \( E = E_F + kT \).

We can see from Figure 3.33 that the probability of an energy above \( E_F \) being occupied increases as the temperature increases and the probability of a state below \( E_F \) being empty increases as the temperature increases.

## Example 3.7

**Objective**: 
Determine the temperature at which there is 1 percent probability that an energy state is empty.


Assume that the Fermi energy level for a particular material is 6.25 eV and that the electrons in this material follow the Fermi–Dirac distribution function. Calculate the temperature at which there is a 1 percent probability that a state 0.30 eV below the Fermi energy level will not contain an electron.

**Solution**

The probability that a state is empty is

\[
1 - f(E) = 1 - \frac{1}{1 + \exp\left(\frac{E - E_F}{kT}\right)}
\]

Then

\[
0.01 = 1 - \frac{1}{1 + \exp\left(\frac{5.95 - 6.25}{kT}\right)}
\]

Solving for \(kT\), we find \(kT = 0.06529 \, \text{eV}\), so that the temperature is \(T = 756 \, \text{K}\).

**Comment**
The Fermi probability function is a strong function of temperature.

**Exercise Problem**
**Ex 3.7** Assume that \(E_F\) is 0.3 eV below \(E_c\). Determine the temperature at which the probability of an electron occupying an energy state at \(E = (E_c + 0.025) \, \text{eV}\) is \(8 \times 10^{-6}\).


We may note that the probability of a state a distance \(dE\) above \(E_F\) being occupied is the same as the probability of a state a distance \(dE\) below \(E_F\) being empty. The function \(f_F(E)\) is symmetrical with the function \(1 - f_F(E)\) about the Fermi energy, \(E_F\). This symmetry effect is shown in Figure 3.34 and will be used in the next chapter.

Consider the case when \(E - E_F \gg kT\) where the exponential term in the denominator of Equation (3.79) is much greater than unity. We may neglect the 1 in the denominator, so the Fermi–Dirac distribution function becomes

\[
f_F(E) \approx \exp\left(-\frac{E - E_F}{kT}\right)
\]

(3.80)

Equation (3.80) is known as the Maxwell–Boltzmann approximation, or simply the Boltzmann approximation, to the Fermi–Dirac distribution function. Figure 3.35 shows the Fermi–Dirac probability function and the Boltzmann approximation. This figure gives an indication of the range of energies over which the approximation is valid.


**Figure 3.34** The probability of a state being occupied, \(f_F(E)\), and the probability of a state being empty, \(1 - f_F(E)\).


**Figure 3.35** | The Fermi–Dirac probability function and the Maxwell–Boltzmann approximation.

## EXAMPLE 3.8
**Objective:** Determine the energy at which the Boltzmann approximation may be considered valid.

Calculate the energy, in terms of \(kT\) and \(E_F\), at which the difference between the Boltzmann approximation and the Fermi–Dirac function is 5 percent of the Fermi function.

**Solution**

We can write

\[
\exp\left[-\frac{(E - E_F)}{kT}\right] = \frac{1}{1 + \exp\left[\frac{E - E_F}{kT}\right]} = 0.05
\]

\[
\frac{1}{1 + \exp\left[\frac{E - E_F}{kT}\right]}
\]

If we multiply both numerator and denominator by the \(1 + \exp()\) function, we have

\[
\exp\left[-\frac{(E - E_F)}{kT}\right] \cdot \left\{1 + \exp\left[\frac{E - E_F}{kT}\right]\right\} - 1 = 0.05
\]

which becomes

\[
\exp\left[-\frac{(E - E_F)}{kT}\right] = 0.05
\]

or

\[
(E - E_F) = kT \ln\left(\frac{1}{0.05}\right) \approx 3kT
\]

**Comment**

As seen in this example and in Figure 3.35, the \(E - E_F \gg kT\) notation is somewhat misleading. The Maxwell–Boltzmann and Fermi–Dirac functions are within 5 percent of each other when \(E - E_F \approx 3kT\).

**Exercise Problem**

Ex 3.8 Repeat Example 3.8 for the case when the difference between the Boltzmann approximation and the Fermi–Dirac function is 2 percent of the Fermi function.

\[
\left(\frac{1}{0.02} = 4.7 \approx 5\right)
\]


The actual Boltzmann approximation is valid when \(\exp[(E - E_F)/kT] \gg 1\). However, it is still common practice to use the \(E - E_F \gg kT\) notation when applying the Boltzmann approximation. We will use this Boltzmann approximation in our discussion of semiconductors in the next chapter.


## TEST YOUR UNDERSTANDING

**TYU 3.5**  
Assume that the Fermi energy level is 0.35 eV above the valence band energy. Let \(T = 300 \, \text{K}\).  
(a) Determine the probability of a state being empty of an electron at \(E = E_F - kT/2\).  
(b) Repeat part (a) for an energy state at \(E = E_F - 3kT/2\).  
\[
[.01 \times 2.0 \times 10^{-2} \times 0.3 \times 9 \times \text{e} \times \text{su}]
\]

**TYU 3.6**  
Repeat Exercise Problem Ex. 3.6 for \(T = 400 \, \text{K}\).  
\[
[.01 \times 1.2 \times 9 \times 10^{-1} \times 1 \times \text{E} \times \text{e} \times \text{su}]
\]

**TYU 3.7**  
Repeat Exercise Problem TYU 3.5 for \(T = 400 \, \text{K}\).  
\[
[.01 \times 8.8 \times 9 \times 10^{-1} \times 1.7 \times \text{E} \times \text{e} \times \text{su}]
\]


## 3.6 | SUMMARY

- Discrete allowed electron energies split into a band of allowed energies as atoms are brought together to form a crystal.
- The concept of allowed and forbidden energy bands was developed more rigorously by considering quantum mechanics and Schrödinger’s wave equation using the Kronig–Penney model representing the potential function of a single-crystal material. This result forms the basis of the energy-band theory of semiconductors.
- The concept of effective mass was developed. Effective mass relates the motion of a particle in a crystal to an externally applied force and takes into account the effect of the crystal lattice on the motion of the particle.
- Two charged particles exist in a semiconductor. An electron is a negatively charged particle with a positive effective mass existing at the bottom of an allowed energy band. A hole is a positively charged particle with a positive effective mass existing at the top of an allowed energy band.
- The \(E\) versus \(k\) diagrams of silicon and gallium arsenide were given and the concept of direct and indirect bandgap semiconductors was discussed.
- Energies within an allowed energy band are actually at discrete levels and each contains a finite number of quantum states. The density per unit energy of quantum states was determined by using the three-dimensional infinite potential well as a model.
- In dealing with large numbers of electrons and holes, we must consider the statistical behavior of these particles. The Fermi–Dirac probability function was developed, which gives the probability of a quantum state at an energy \(E\) of being occupied by an electron. The Fermi energy was defined.


## Glossary of Important Terms

- **allowed energy band**  
  A band or range of energy levels that an electron in a crystal is allowed to occupy based on quantum mechanics.

- **density of states function**  
  The density of available quantum states as a function of energy, given in units of number per unit energy per unit volume.

- **electron effective mass**  
  The parameter that relates the acceleration of an electron in the conduction band of a crystal to an external force; a parameter that takes into account the effect of internal forces in the crystal.

- **Fermi–Dirac probability function**  
  The function describing the statistical distribution of electrons among available energy states and the probability that an allowed energy state is occupied by an electron.

- **fermi energy**  
  In the simplest definition, the energy below which all states are filled with electrons and above which all states are empty at \( T = 0 \, \text{K} \).

- **forbidden energy band**  
  A band or range of energy levels that an electron in a crystal is not allowed to occupy based on quantum mechanics.

- **hole**  
  The positively charged “particle” associated with an empty state in the top of the valence band.

- **hole effective mass**  
  The parameter that relates the acceleration of a hole in the valence band of a crystal to an applied external force (a positive quantity); a parameter that takes into account the effect of internal forces in a crystal.

- **k-space diagram**  
  The plot of electron energy in a crystal versus \( k \), where \( k \) is the momentum-related constant of the motion that incorporates the crystal interaction.

- **Kronig–Penney model**  
  The mathematical model of a periodic potential function representing a one-dimensional single-crystal lattice by a series of periodic step functions.

- **Maxwell–Boltzmann approximation**  
  The condition in which the energy is several \( kT \) above the Fermi energy or several \( kT \) below the Fermi energy so that the Fermi–Dirac probability function can be approximated by a simple exponential function.

## CHECKPOINT

After studying this chapter, the reader should have the ability to:

- Discuss the concept of allowed and forbidden energy bands in a single crystal both qualitatively and more rigorously from the results of using the Kronig–Penney model.
- Discuss the splitting of energy bands in silicon.
- State the definition of effective mass from the \( E \) versus \( k \) diagram and discuss its meaning in terms of the movement of a particle in a crystal.
- Discuss the concept of a hole.
- Discuss the characteristics of a direct and an indirect bandgap semiconductor.
- Qualitatively, in terms of energy bands, discuss the difference between a metal, an insulator, and semiconductor.
- What is meant by the density of states function?
- Understand the meaning of the Fermi–Dirac distribution function and the Fermi energy.

## REVIEW QUESTIONS

1. What is the Kronig–Penney model? What does it represent?
2. State two results of using the Kronig–Penney model with Schrödinger’s wave equation.
3. What is effective mass? How is effective mass defined in terms of the \( E \) versus \( k \) diagram?
4. What is a direct bandgap semiconductor? What is an indirect bandgap semiconductor?
5. What is the meaning of the density of states function?
6. What was the mathematical model used in deriving the density of states function?  
7. In general, what is the relation between density of states and energy?  
8. What is the meaning of the Fermi–Dirac probability function?  
9. What is the Fermi energy?    

