# 10.3 The Basic MOSFET Operation

!MOSFET Diagrams

**Figure 10.36** | Cross section and circuit symbol for (a) a p-channel enhancement mode MOSFET and (b) a p-channel depletion mode MOSFET.

Figure 10.37b shows the same MOSFET with an applied gate voltage such that \( V_{GS} > V_T \). An electron inversion layer has been created so that when a small drain voltage is applied, the electrons in the inversion layer will flow from the source to the positive drain terminal. The conventional current enters the drain terminal and leaves the source terminal. In this ideal case, there is no current through the oxide to the gate terminal.

For small \( V_{DS} \) values, the channel region has the characteristics of a resistor, so we can write

\[
I_D = g_d V_{DS}
\]

(10.41)

where \( g_d \) is defined as the channel conductance in the limit as \( V_{DS} \to 0 \). The channel conductance is given by

\[
g_d = \frac{W}{L} \cdot \mu_n |Q_I'|
\]

(10.42)