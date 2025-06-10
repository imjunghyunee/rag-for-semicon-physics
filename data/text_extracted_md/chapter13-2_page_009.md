# 13.4 Equivalent Circuit and Frequency Limitations

!Equivalent Circuit

**Figure 13.20** (a) Ideal low-frequency small-signal equivalent circuit. (b) Ideal equivalent circuit including \( r_s \).

Capacitances become open circuits. The small-signal drain current is now

\[
I_{ds} = g_m V_{gs}
\]

(13.57)

which is a function only of the transconductance and the input-signal voltage.

The effect of the source series resistance can be determined using Figure 13.20b. We have

\[
I_{ds} = g_m V_{gs}'
\]

(13.58)

The relation between \( V_{gs} \) and \( V_{gs}' \) can be found from

\[
V_{gs} = V_{gs}' + (g_m V_{gs}') r_s = (1 + g_m r_s) V_{gs}'
\]

(13.59)

Equation (13.58) can then be written as

\[
I_{ds} = \left( \frac{g_m}{1 + g_m r_s} \right) V_{gs} = g_m' V_{gs}
\]

(13.60)

The effect of the source resistance is to reduce the effective transconductance or transistor gain.

Recall that \( g_m \) is a function of the dc gate-to-source voltage, so \( g_m' \) will also be a function of \( V_{GS} \). Equation (13.41b) is the relation between \( g_m \) and \( V_{GS} \) when the...