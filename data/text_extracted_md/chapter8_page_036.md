## 8.3 Small-Signal Model of the pn Junction

where

\[
I_{0} = \frac{eAD_{n}n_{p0}}{L_{n}} \exp \left( \frac{eV_{0}}{kT} \right)
\]

(8.97)

The total ac current phasor is the sum of \(\hat{i}_{n}\) and \(\hat{i}_{p}\). The pn junction admittance is the total ac current phasor divided by the ac voltage phasor, or

\[
Y = \frac{\hat{i}}{\hat{V}_{i}} = \left( \frac{1}{\hat{V}_{i}} \right) [I_{p0} \sqrt{1 + j\omega\tau_{p0}} + I_{n0} \sqrt{1 + j\omega\tau_{n0}}]
\]

(8.98)

There is not a linear, lumped, finite, passive, bilateral network that can be synthesized to give this admittance function. However, we may make the following approximations. Assume that

\[
\omega\tau_{p0} \ll 1
\]

(8.99a)

and

\[
\omega\tau_{n0} \ll 1
\]

(8.99b)

These two assumptions imply that the frequency of the ac signal is not too large. Then we may write

\[
\sqrt{1 + j\omega\tau_{p0}} \approx 1 + \frac{j\omega\tau_{p0}}{2}
\]

(8.100a)

and

\[
\sqrt{1 + j\omega\tau_{n0}} \approx 1 + \frac{j\omega\tau_{n0}}{2}
\]

(8.100b)

Substituting Equations (8.100a) and (8.100b) into the admittance Equation (8.98), we obtain

\[
Y = \left( \frac{1}{\hat{V}_{i}} \right) \left[ I_{p0} \left( 1 + \frac{j\omega\tau_{p0}}{2} \right) + I_{n0} \left( 1 + \frac{j\omega\tau_{n0}}{2} \right) \right]
\]

(8.101)

If we combine the real and imaginary portions, we get

\[
Y = \left( \frac{1}{\hat{V}_{i}} \right) (I_{p0} + I_{n0}) + j\omega \left( \frac{1}{2\hat{V}_{i}} \right) (I_{p0}\tau_{p0} + I_{n0}\tau_{n0})
\]

(8.102)

Equation (8.102) may be written in the form

\[
Y = g_{d} + j\omega C_{d}
\]

(8.103)

The parameter \(g_{d}\) is called the **diffusion conductance** and is given by

\[
g_{d} = \left( \frac{1}{\hat{V}_{i}} \right) (I_{p0} + I_{n0}) = \frac{I_{DQ}}{\hat{V}_{i}}
\]

(8.104)

where \(I_{DQ}\) is the dc bias current. Equation (8.104) is exactly the same conductance as we obtained previously in Equation (8.67). The parameter \(C_{d}\) is called the **diffusion capacitance** and is given by

\[
C_{d} = \left( \frac{1}{2\hat{V}_{i}} \right) (I_{p0}\tau_{p0} + I_{n0}\tau_{n0})
\]

(8.105)