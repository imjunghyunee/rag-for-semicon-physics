# Chapter 7: The pn Junction

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