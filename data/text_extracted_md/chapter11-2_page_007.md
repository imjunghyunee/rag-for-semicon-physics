# 11.4 Additional Electrical Characteristics

!I-V Characteristics

**Figure 11.26** | Typical \( I-V \) characteristics of a MOSFET exhibiting punch-through effects.

Condition has been reached. Figure 11.26 shows some typical characteristics of a short-channel device with a near punch-through condition.

## Objective

Calculate the theoretical punch-through voltage assuming the abrupt junction approximation.

**EXAMPLE 11.5**

Consider an n-channel MOSFET with source and drain doping concentrations of \( N_d = 10^{19} \, \text{cm}^{-3} \) and a channel region doping of \( N_a = 10^{16} \, \text{cm}^{-3} \). Assume a channel length of \( L = 1.2 \, \mu\text{m} \), and assume the source and body are at ground potential.

### Solution

The pn junction built-in potential barrier is given by

\[
V_{bi} = V_t \ln \left( \frac{N_d N_a}{n_i^2} \right) = (0.0259) \ln \left( \frac{(10^{16})(10^{19})}{(1.5 \times 10^{10})^2} \right) = 0.874 \, \text{V}
\]

The zero-biased source–substrate pn junction width is

\[
x_{s0} = \left[ \frac{2 \varepsilon_s V_{bi}}{e N_a} \right]^{1/2} = \left[ \frac{2(11.7)(8.85 \times 10^{-14})(0.874)}{(1.6 \times 10^{-19})(10^{16})} \right]^{1/2} = 0.336 \, \mu\text{m}
\]

The reverse-biased drain-substrate pn junction width is given by

\[
x_j = \left[ \frac{2 \varepsilon_s (V_{bi} + V_{DS})}{e N_a} \right]^{1/2}
\]

At punch-through, we will have

\[
x_{s0} + x_j = L \quad \text{or} \quad 0.336 + x_j = 1.2
\]

which gives \( x_j = 0.864 \, \mu\text{m} \) at the punch-through condition. We can then find

\[
V_{bi} + V_{DS} = \frac{x_j^2 e N_a}{2 \varepsilon_s} = \frac{(0.864 \times 10^{-4})^2 (1.6 \times 10^{-19})(10^{16})}{2(11.7)(8.85 \times 10^{-14})}
\]

\[
= 5.77 \, \text{V}
\]