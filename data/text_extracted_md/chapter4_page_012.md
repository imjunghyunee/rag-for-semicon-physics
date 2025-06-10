# 4.1 Charge Carriers in Semiconductors

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

----

**Objective:** Calculate the position of the intrinsic Fermi level with respect to the center of the bandgap in silicon at \( T = 300 \, \text{K} \).

**EXAMPLE 4.4**

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