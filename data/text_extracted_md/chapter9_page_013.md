# 9.1 The Schottky Barrier Diode

The current density \( J_{x,n} \) is a function of the concentration of electrons which have \( x \)-directed velocities sufficient to overcome the barrier. We may write

\[
J_{x,n} = e \int_{E_c}^{\infty} v_x \, dn
\]

(9.18)

where \( E_c \) is the minimum energy required for thermionic emission into the metal, \( v \) is the carrier velocity in the direction of transport, and \( e \) is the magnitude of the electronic charge. The incremental electron concentration is given by

\[
dn = g_c(E) f_f(E) \, dE
\]

(9.19)

where \( g_c(E) \) is the density of states in the conduction band and \( f_f(E) \) is the Fermi-Dirac probability function. Assuming that the Maxwell-Boltzmann approximation applies, we may write

\[
dn = \frac{4\pi(2m_n^*)^{3/2}}{h^3} \sqrt{E - E_c} \exp \left[ -\frac{(E - E_f)}{kT} \right] dE
\]

(9.20)

If all of the electron energy above \( E_c \) is assumed to be kinetic energy, then we have

\[
\frac{1}{2} m_n^* v^2 = E - E_c
\]

(9.21)

The net current density in the metal-to-semiconductor junction can be written as

\[
J = J_{x,n} - J_{x,-n}
\]

(9.22)

which is defined to be positive in the direction from the metal to the semiconductor. We find that

\[
J = \left[ A^* T^2 \exp \left( -\frac{e\phi_{bn}}{kT} \right) \right] \left[ \exp \left( \frac{eV_a}{kT} \right) - 1 \right]
\]

(9.23)

where

\[
A^* = \frac{4\pi m_n^* k^2}{h^3}
\]

(9.24)

The parameter \( A^* \) is called the effective Richardson constant for thermionic emission. Equation (9.23) can be written in the usual diode form as

\[
J = J_{sT} \left[ \exp \left( \frac{eV_a}{kT} \right) - 1 \right]
\]

(9.25)

where \( J_{sT} \) is the reverse-saturation current density and is given by

\[
J_{sT} = A^* T^2 \exp \left( -\frac{e\phi_{bn}}{kT} \right)
\]

(9.26)

We may recall that the Schottky barrier height \( \phi_{bn} \) changes because of the image-force lowering. We have that \( \phi_{bn} = \phi_{b0} - \Delta \phi \). Then we can write Equation (9.26) as