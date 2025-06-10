# 9.1 The Schottky Barrier Diode

The potential can then be found as

\[
-\phi(x) = + \int_{x}^{\infty} Edx' = + \int_{x}^{\infty} \frac{e}{4\pi \epsilon_s (4x')^2} dx' = \frac{-e}{16\pi \epsilon_s x}
\]

(9.11)

where \( x' \) is the integration variable and where we have assumed that the potential is zero at \( x = \infty \).

The potential energy of the electron is \(-e\phi(x)\). Figure 9.4b is a plot of the potential energy assuming that no other electric fields exist. With an electric field present in the dielectric, the potential is modified and can be written as

\[
-e\phi(x) = \frac{-e}{16\pi \epsilon_s x} - eEx
\]

(9.12)

The potential energy of the electron, including the effect of a constant electric field, is plotted in Figure 9.4c. The peak potential barrier is now lowered. This lowering of the potential barrier is the Schottky effect, or image force–induced lowering.

We can find the Schottky barrier lowering, \(\Delta \phi\), and the position of the maximum barrier, \( x_m \), from the condition that

\[
\frac{d[e\phi(x)]}{dx} = 0
\]

(9.13)

We find that

\[
x_m = \sqrt{\frac{e}{16\pi \epsilon_s E}}
\]

(9.14)

and

\[
\Delta \phi = \sqrt{\frac{eE}{4\pi \epsilon_s}}
\]

(9.15)

----

**Objective:** Calculate the Schottky barrier lowering and the position of the maximum barrier height.

**Example 9.3**

Consider a gallium arsenide metal–semiconductor contact in which the electric field in the semiconductor is assumed to be \( E = 6.8 \times 10^4 \) V/cm.

**Solution**

The Schottky barrier lowering is given by Equation (9.15), which in this case yields

\[
\Delta \phi = \sqrt{\frac{eE}{4\pi \epsilon_s}} = \sqrt{\frac{(1.6 \times 10^{-19})(6.8 \times 10^6)}{4\pi (13.1)(8.85 \times 10^{-14})}} = 0.0273 \, \text{V}
\]

The position of the maximum barrier height is

\[
x_m = \sqrt{\frac{e}{16\pi \epsilon_s E}} = \sqrt{\frac{(1.6 \times 10^{-19})}{16\pi (13.1)(8.85 \times 10^{-14})(6.8 \times 10^6)}}
\]

or

\[
x_m = 2 \times 10^{-7} \, \text{cm} = 20 \, \text{Å}
\]