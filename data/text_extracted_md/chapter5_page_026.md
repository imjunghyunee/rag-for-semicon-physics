# 5.4 The Hall Effect

The magnetic field force will be exactly balanced by the induced electric field force. This balance may be written as

\[
F = q[E + v \times B] = 0
\]

which becomes

\[
qE_y = qv_xB_z
\]

The induced electric field in the y direction is called the **Hall field**. The Hall field produces a voltage across the semiconductor which is called the **Hall voltage**. We can write

\[
V_{H} = +E_yW
\]

where \(E_y\) is assumed positive in the +y direction and \(V_H\) is positive with the polarity shown.

In a p-type semiconductor, in which holes are the majority carrier, the Hall voltage will be positive as defined in Figure 5.13. In an n-type semiconductor, in which electrons are the majority carrier, the Hall voltage will have the opposite polarity. The polarity of the Hall voltage is used to determine whether an extrinsic semiconductor is n type or p type.

Substituting Equation (5.50) into Equation (5.49) gives

\[
V_H = v_xWB_z
\]

For a p-type semiconductor, the drift velocity of holes can be written as

\[
v_x = \frac{J_x}{-ep} = \frac{I}{(ep)Wd}
\]

where \(e\) is the magnitude of the electronic charge. Combining Equations (5.52) and (5.59), we have

\[
V_H = \frac{I B_z}{epd}
\]

or, solving for the hole concentration, we obtain

\[
p = \frac{I B_z}{edV_H}
\]

The majority carrier hole concentration is determined from the current, magnetic field, and Hall voltage.

For an n-type semiconductor, the Hall voltage is given by

\[
V_H = -\frac{I B_z}{ned}
\]

so that the electron concentration is

\[
n = -\frac{I B_z}{edV_H}
\]

Note that the Hall voltage is negative for the n-type semiconductor; therefore, the electron concentration determined from Equation (5.56) is actually a positive quantity.