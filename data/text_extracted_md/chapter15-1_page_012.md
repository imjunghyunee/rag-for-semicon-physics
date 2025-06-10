# 15.4 Power Bipolar Transistors

## Objective

Determine the required current, voltage, and power rating of a power BJT.

Consider the common-emitter circuit in Figure 15.15. The parameters are \( R_L = 10 \, \Omega \) and \( V_{CC} = 35 \, \text{V} \).

## Solution

For \( V_{CE} = 0 \), the maximum collector current is

\[
I_C (\text{max}) = \frac{V_{CC}}{R_L} = \frac{35}{10} = 3.5 \, \text{A}
\]

For \( I_C = 0 \), the maximum collector-emitter voltage is

\[
V_{CE} (\text{max}) = V_{CC} = 35 \, \text{V}
\]

The load line is given by

\[
V_{CE} = V_{CC} - I_C R_L
\]

and must remain within the SOA, as shown in Figure 15.16.

The transistor power dissipation is

\[
P_T = V_{CE} I_C = (V_{CC} - I_C R_L) I_C = V_{CC} I_C - I_C^2 R_L
\]

The current at which the maximum power occurs is found by setting the derivative of this equation equal to zero as follows:

\[
\frac{dP_T}{dI_C} = 0 = V_{CC} - 2I_C R_L
\]

which yields

\[
I_C = \frac{V_{CC}}{2R_L} = \frac{35}{2(10)} = 1.75 \, \text{A}
\]

!Figure 15.15

**Figure 15.15** | Bipolar common-emitter circuit.

!Figure 15.16

**Figure 15.16** | Load line and maximum power curve for Example 15.1.

### Graph Description

- **\( I_C \) (A)**: Y-axis ranging from 0 to 4.
- **\( V_{CE} \) (V)**: X-axis ranging from 0 to 40.
- **Load Line**: Starts at \( I_C (\text{max}) = 3.5 \, \text{A} \) and intersects \( V_{CE} = 35 \, \text{V} \).
- **Maximum Power Point**: Occurs at \( I_C = 1.75 \, \text{A} \).