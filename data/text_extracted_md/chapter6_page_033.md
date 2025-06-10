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