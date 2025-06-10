# Solution

Since \( N_D > N_A \), the semiconductor is n-type and the majority carrier electron concentration, from Chapter 4 is given by

\[
n = \frac{N_D - N_A}{2} + \sqrt{\left(\frac{N_D - N_A}{2}\right)^2 + n_i^2} \approx 10^{16} \, \text{cm}^{-3}
\]

The minority carrier hole concentration is

\[
p = \frac{n_i^2}{n} = \frac{(1.8 \times 10^{20})}{10^{16}} = 3.24 \times 10^{-4} \, \text{cm}^{-3}
\]

For this extrinsic n-type semiconductor, the drift current density is

\[
J_{\text{drift}} = e(\mu_n n + \mu_p p)E \approx e\mu_n N_D E
\]

Then

\[
J_{\text{drift}} = (1.6 \times 10^{-19})(8500)(10^{16})(10) = 136 \, \text{A/cm}^2
\]

# Comment

Significant drift current densities can be obtained in a semiconductor applying relatively small electric fields. We may note from this example that the drift current will usually be due primarily to the majority carrier in an extrinsic semiconductor.

# EXERCISE PROBLEM

**Ex 5.1**  
A drift current density of \( J_{\text{drift}} = 75 \, \text{A/cm}^2 \) is required in a device using p-type silicon when an electric field of \( E = 120 \, \text{V/cm} \) is applied. Determine the required impurity doping concentration to achieve this specification. Assume that electron and hole mobilities given in Table 5.1 apply. (\( \mu_n = 0.1 \times 10^3 = \text{N s/V} \))

## 5.1.2 Mobility Effects

In the previous section, we defined mobility, which relates the average drift velocity of a carrier to the electric field. Electron and hole mobilities are important semiconductor parameters in the characterization of carrier drift, as seen in Equation (5.9).

Equation (5.3) related the acceleration of a hole to a force such as an electric field. We may write this equation as

\[
F = m_n \frac{dv}{dt} = eE
\]

(5.10)

where \( v \) is the velocity of the particle due to the electric field and does not include the random thermal velocity. If we assume that the conductivity effective mass and electric field are constants, then we may integrate Equation (5.10) and obtain

\[
v = \frac{eEt}{m_n^*}
\]

(5.11)

where we have assumed the initial drift velocity to be zero.

Figure 5.1a shows a schematic model of the random thermal velocity and motion of a hole in a semiconductor with zero electric field. There is a mean time