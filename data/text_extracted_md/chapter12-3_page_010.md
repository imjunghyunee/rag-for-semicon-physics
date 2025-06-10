# CHAPTER 12 The Bipolar Transistor

## EXERCISE PROBLEM

**Ex 12.12** Repeat Example 12.12 for transistor parameters of \( \alpha_T = 0.992, \alpha_R = 0.05, I_C = 0.5 \, \text{mA}, \) and \( I_B = 50 \, \mu\text{A}. \)

## 12.5.2 Gummel–Poon Model

The Gummel–Poon model of the BJT considers more physics of the transistor than the Ebers–Moll model. This model can be used if, for example, there is a nonuniform doping concentration in the base.

The electron current density in the base of an npn transistor can be written as

\[
J_n = e \mu_n n(x) E + e D_n \frac{dn(x)}{dx}
\]

(12.78)

An electric field will occur in the base if nonuniform doping exists in the base. This is discussed in Section 12.4.5. The electric field, from Equation (12.52), can be written in the form

\[
E = \frac{kT}{e} \cdot \frac{1}{p(x)} \cdot \frac{dp(x)}{dx}
\]

(12.79)

where \( p(x) \) is the majority carrier hole concentration in the base. Under low injection, the hole concentration is just the acceptor impurity concentration. With the doping profile shown in Figure 12.31, the electric field is negative (from the collector to the emitter). The direction of this electric field aids the flow of electrons across the base.

Substituting Equation (12.79) into Equation (12.78), we obtain

\[
J_n = e \mu_n n(x) \cdot \frac{kT}{e} \cdot \frac{1}{p(x)} \cdot \frac{dp(x)}{dx} + e D_n \frac{dn(x)}{dx}
\]

(12.80)

Using Einstein’s relation, we can write Equation (12.80) in the form

\[
J_n = e D_n \left[ \frac{n(x)}{p(x)} \frac{dp(x)}{dx} + p(x) \frac{dn(x)}{dx} \right] = \frac{e D_n}{p(x)} \cdot \frac{d(pn)}{dx}
\]

(12.81)

Equation (12.81) can be written in the form

\[
\frac{J_n p(x)}{e D_n} = \frac{d(pn)}{dx}
\]

(12.82)

Integrating Equation (12.82) through the base region while assuming that the electron current density is essentially a constant and the diffusion coefficient is a constant, we find

\[
\frac{J_n}{e D_n} \int_0^{x'} p(x) dx = \int_0^{x'} \frac{d(p(x)n(x))}{dx} dx = p(x_B)n(x_B) - p(0)n(0)
\]

(12.83)

Assuming that the B–E junction is forward biased and the B–C junction is reverse biased, we have \( n(0) = n_{B0} \exp(V_{BE}/V_T) \) and \( n(x_B) = 0 \). We may note that \( n_{B0} p_0 = n_i^2 \) so that Equation (12.83) can be written as

\[
J_n = -\frac{e D_n n_i^2 \exp(V_{BE}/V_T)}{\int_0^{x'} p(x) dx}
\]

(12.84)