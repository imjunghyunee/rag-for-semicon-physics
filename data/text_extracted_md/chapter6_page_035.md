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