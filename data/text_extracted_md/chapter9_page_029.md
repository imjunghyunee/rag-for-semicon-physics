# 9.3 Heterojunctions

Equation (9.36), from Figure 9.17, can be written as

\[
eV_{bi} = [E_{cp} + E_{fp} - (E_{fp} - E_{fn})] - [e\chi_n + E_{fn} - (E_{fn} - E_{wn})]
\]

(9.37a)

or

\[
eV_{bi} = e(\chi_p - \chi_n) + (E_{cp} - E_{fp}) + (E_{fn} - E_{wn}) - (E_{fp} - E_{fn})
\]

(9.37b)

which can be expressed as

\[
eV_{bi} = -\Delta E_c + \Delta E_v + kT \ln \left( \frac{N_{vn}}{p_{no}} \right) - kT \ln \left( \frac{N_{vp}}{p_{po}} \right)
\]

(9.38)

Finally, we can write Equation (9.38) as

\[
eV_{bi} = \Delta E_c + kT \ln \left( \frac{p_{no}}{p_{po}} \cdot \frac{N_{vn}}{N_{vp}} \right)
\]

(9.39)

where \( p_{no} \) and \( p_{po} \) are the hole concentrations in the P and n materials, respectively, and \( N_{vn} \) and \( N_{vp} \) are the effective density of states functions in the n and P materials, respectively. We can also obtain an expression for the built-in potential barrier in terms of the conduction band shift as

\[
eV_{bi} = -\Delta E_c + kT \ln \left( \frac{n_{no} \cdot N_{cp}}{n_{po} \cdot N_{cn}} \right)
\]

(9.40)

----

## Objective

Determine \(\Delta E_c\), \(\Delta E_v\), and \(V_{bi}\) for an n-Ge to P-GaAs heterojunction using the electron affinity rule.

**Example 9.8**

Consider n-type Ge doped with \(N_d = 10^{16} \, \text{cm}^{-3}\) and P-type GaAs doped with \(N_a = 10^{16} \, \text{cm}^{-3}\). Let \(T = 300 \, \text{K}\) so that \(n_i = 2.4 \times 10^{13} \, \text{cm}^{-3}\) for Ge.

### Solution

From Equation (9.34a), we have

\[
\Delta E_c = e(\chi_n - \chi_p) = e(4.13 - 4.07) = 0.06 \, \text{eV}
\]

and from Equation (9.34b), we have

\[
\Delta E_v = \Delta E_g - \Delta E_c = (1.43 - 0.67) - 0.06 = 0.70 \, \text{eV}
\]

To determine \(V_{bi}\) using Equation (9.39), we need to determine \(p_{no}\) in Ge, or

\[
p_{no} = \frac{n_i^2}{N_d} = \frac{(2.4 \times 10^{13})^2}{10^{16}} = 5.76 \times 10^{10} \, \text{cm}^{-3}
\]

Then

\[
eV_{bi} = 0.70 + (0.0259) \ln \left( \frac{[10^{16} \times 6 \times 10^{18}]}{[5.76 \times 10^{10} \times 7 \times 10^{18}]} \right)
\]

or, finally,

\[
V_{bi} \approx 1.0 \, \text{V}
\]

### Comment

There is a nonsymmetry in the \(\Delta E_c\) and \(\Delta E_v\) values that will tend to make the potential barriers seen by electrons and holes different. This nonsymmetry does not occur in homojunctions.