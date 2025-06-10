# CHAPTER 12 The Bipolar Transistor

## Table 12.2.1 Taylor expansions of hyperbolic functions

| Function | Taylor expansion |
|----------|------------------|
| sinh \(x\) | \(x + \frac{x^3}{3!} + \frac{x^5}{5!} + \cdots\) |
| cosh \(x\) | \(1 + \frac{x^2}{2!} + \frac{x^4}{4!} + \cdots\) |
| tanh \(x\) | \(x - \frac{x^3}{3} + \frac{2x^5}{15} + \cdots\) |

### Emitter Region

Consider, now, the minority carrier hole concentration in the emitter. The steady-state excess hole concentration is determined from the equation

\[
D_E \frac{\partial^2 (\delta p_E(x'))}{\partial x'^2} - \frac{\delta p_E(x')}{\tau_{E0}} = 0
\]

(12.16)

where \(D_E\) and \(\tau_{E0}\) are the minority carrier diffusion coefficient and minority carrier lifetime, respectively, in the emitter. The excess hole concentration is given by

\[
\delta p_E(x') = p_E(x') - p_{E0}
\]

(12.17)

The general solution to Equation (12.16) can be written as

\[
\delta p_E(x') = C \exp \left( \frac{x'}{L_E} \right) + D \exp \left( -\frac{x'}{L_E} \right)
\]

(12.18)

where \(L_E = \sqrt{D_E \tau_{E0}}\). If we assume the neutral emitter length \(x_E\) is not necessarily long compared to \(L_E\), then both exponential terms in Equation (12.18) must be retained.

The excess minority carrier hole concentrations at the two boundaries are

\[
\delta p_E(x' = 0) = \delta p_E(0) = C + D
\]

(12.19a)

and

\[
\delta p_E(x' = x_E) = \delta p_E(x_E) = C \exp \left( \frac{x_E}{L_E} \right) + D \exp \left( -\frac{x_E}{L_E} \right)
\]

(12.19b)

Again, the B–E junction is forward biased, so

\[
\delta p_E(0) = p_E(x' = 0) - p_{E0} = p_{E0} \left[ \exp \left( \frac{eV_{BE}}{kT} \right) - 1 \right]
\]

(12.20a)

An infinite surface recombination velocity at \(x' = x_E\) implies that

\[
\delta p_E(x_E) = 0
\]

(12.20b)

Solving for \(C\) and \(D\) using Equations (12.19) and (12.20) yields the excess minority carrier hole concentration in Equation (12.18):

\[
\delta p_E(x') = p_{E0} \left[ \exp \left( \frac{eV_{BE}}{kT} \right) - 1 \right] \frac{\sinh \left( \frac{x_E - x'}{L_E} \right)}{\sinh \left( \frac{x_E}{L_E} \right)}
\]

(12.21a)