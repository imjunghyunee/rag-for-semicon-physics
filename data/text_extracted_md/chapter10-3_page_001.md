# 10.3 The Basic MOSFET Operation

!Geometry of a MOSFET

**Figure 10.43** | Geometry of a MOSFET for \( I_D \) versus \( V_{DS} \) derivation.

The total channel current is found by integrating \( J_x \) over the cross-sectional area in the \( y \) and \( z \) directions. Then

\[
I_s = \int \int J_x \, dy \, dz
\]

(10.47)

We may write that

\[
Q'_i = - \int en(y) \, dy
\]

(10.48)

where \( Q'_i \) is the inversion layer charge per unit area and is a negative quantity for this case.

Equation (10.47) then becomes

\[
I_s = -W \mu_n Q'_i E_x
\]

(10.49)

where \( W \) is the channel width, the result of integrating over \( z \).

Two concepts we use in the current–voltage derivation are charge neutrality and Gauss’s law. Figure 10.44 shows the charge densities through the device for \( V_{GS} > V_T \). The charges are all given in terms of charge per unit area. Using the concept of charge neutrality, we can write

\[
Q'_i + Q'_d + Q'_s + Q_{DS}(max) = 0
\]

(10.50)

The inversion layer charge and induced space charge are negative for this n-channel device.

Gauss’s law can be written as

\[
\oint \varepsilon E_N \, dS = Q_f
\]

(10.51)