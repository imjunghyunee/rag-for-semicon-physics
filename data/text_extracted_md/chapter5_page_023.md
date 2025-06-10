# CHAPTER 5 Carrier Transport Phenomena

## Solution

Taking the derivative of the donor concentration, we have

\[
\frac{dN_D(x)}{dx} = -10^9 \quad (\text{cm}^{-4})
\]

The electric field is given by Equation (5.42), so we have

\[
E_x = -\frac{(0.0259)(-10^9)}{(10^{16} - 10^9)x}
\]

At \( x = 0 \), for example, we find

\[
E_x = 25.9 \, \text{V/cm}
\]

## Comment

We may recall from our previous discussion of drift current that fairly small electric fields can produce significant drift current densities, so that an induced electric field from nonuniform doping can significantly influence semiconductor device characteristics.

## EXERCISE PROBLEM

**Ex 5.6** Assume the donor concentration in an n-type semiconductor at \( T = 300 \, \text{K} \) is given by \( N_D(x) = 10^{16} e^{-x/L} \) where \( L = 2 \times 10^{-2} \, \text{cm} \). Determine the induced electric field in the semiconductor at (a) \( x = 0 \) and (b) \( x = 10^{-4} \, \text{cm} \).

----

## 5.3.2 The Einstein Relation

If we consider the nonuniformly doped semiconductor represented by the energy-band diagram shown in Figure 5.12 and assume there are no electrical connections so that the semiconductor is in thermal equilibrium, then the individual electron and hole currents must be zero. We can write

\[
J_n = 0 = en\mu_n E_x + eD_n \frac{dn}{dx} \tag{5.43}
\]

If we assume quasi-neutrality so that \( n \approx N_d(x) \), then we can rewrite Equation (5.43) as

\[
J_n \equiv 0 = e\mu_n N_d(x)E_x + eD_n \frac{dN_d(x)}{dx} \tag{5.44}
\]

Substituting the expression for the electric field from Equation (5.42) into Equation (5.44), we obtain

\[
0 = -e\mu_n N_d(x) \left( \frac{kT}{e} \right) \frac{1}{N_d(x)} \frac{dN_d(x)}{dx} + eD_n \frac{dN_d(x)}{dx} \tag{5.45}
\]

Equation (5.45) is valid for the condition

\[
\frac{D_n}{\mu_n} = \frac{kT}{e} \tag{5.46a}
\]

The hole current must also be zero in the semiconductor. From this condition, we can show that

\[
\frac{D_p}{\mu_p} = \frac{kT}{e} \tag{5.46b}
\]