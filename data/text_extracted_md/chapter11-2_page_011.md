### 11.4 Additional Electrical Characteristics

A second type of implant approximation is the step junction, shown in Figure 11.29b. If the induced space charge width at the threshold inversion point is less than \( x_t \), then the threshold voltage is determined on the basis of a semiconductor with a uniform doping concentration of \( N_t \) atoms per cm\(^3\). On the other hand, if the induced space charge width is greater than \( x_t \) at the threshold inversion point, then a new expression for \( x_{dT} \) must be derived. We can apply Poisson’s equation and show that the maximum induced space charge width after the step implant is

\[
x_{dT} = \sqrt{\frac{2 \varepsilon_s}{e N_t} \left[ 2 \phi_{fp} - \frac{\varepsilon_{ox} t_{ox}^2}{2 \varepsilon_{s}} (N_t - N_a) \right]^{1/2}}
\]

(11.42)

The threshold voltage after a step implant for the case when \( x_{dT} > x_t \) can be written as

\[
V_T = V_{T0} + \frac{e D_I}{C_{ox}}
\]

(11.43)

where \( V_{T0} \) is the preimplant threshold voltage. The parameter \( D_I \) is given by

\[
D_I = (N_t - N_a)x_{dT}
\]

(11.44)

which is the number per cm\(^2\) of implanted ions. The preimplant threshold voltage is

\[
V_{T0} = V_{FB0} + 2 \phi_{fp0} + \frac{e N_a x_{dT0}}{C_{ox}}
\]

(11.45)

where the subscript 0 indicates the preimplant values.

----

**Objective:** Design the ion implant dose required to adjust the threshold voltage to a specified value.

**Example 11.6**

Consider an n-channel silicon MOSFET with a doping of \( N_a = 5 \times 10^{15} \) cm\(^{-3}\), an oxide thickness of \( t_{ox} = 180 \) Å, and an initial flat-band voltage of \( V_{FB0} = -1.25 \) V. Determine the ion implantation dose such that a threshold voltage of \( V_T = +0.4 \) V is obtained.

**Solution**

We find that

\[
\phi_{fp0} = V_t \ln \left( \frac{N_a}{n_i} \right) = (0.0259) \ln \left( \frac{5 \times 10^{15}}{1.5 \times 10^{10}} \right) = 0.3294 \, \text{V}
\]

\[
x_{dT0} = \left[ \frac{4 \varepsilon_s \phi_{fp0}}{e N_a} \right]^{1/2} = \left[ \frac{4(11.7)(8.85 \times 10^{-14})(0.3294)}{(1.6 \times 10^{-19})(5 \times 10^{15})} \right]^{1/2}
\]

\[
= 0.4130 \times 10^{-4} \, \text{cm}
\]

and

\[
C_{ox} = \frac{\varepsilon_{ox}}{t_{ox}} = \frac{(3.9)(8.85 \times 10^{-14})}{180 \times 10^{-8}} = 1.9175 \times 10^{-7} \, \text{F/cm}^2
\]

The initial pre-implant threshold voltage is

\[
V_{T0} = V_{FB0} + 2 \phi_{fp0} + \frac{e N_a x_{dT0}}{C_{ox}}
\]

\[
= -1.25 + 2(0.3294) + \frac{(1.6 \times 10^{-19})(5 \times 10^{15})(0.4130 \times 10^{-4})}{1.9175 \times 10^{-7}}
\]

\[
= -0.419 \, \text{V}
\]