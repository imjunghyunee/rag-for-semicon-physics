# Semiconductor Microwave and Power Devices

!Figure 15.32  
**Figure 15.32** | The current–voltage characteristics of the pnpn device.

!Figure 15.33  
**Figure 15.33** (a) The three-terminal SCR. (b) The two-transistor equivalent circuit of the three-terminal SCR.

----

### Explanation

- **Forward Conducting**: The graph shows the transition from reverse blocking to forward conducting.
- **Reverse Blocking**: The device blocks current in the reverse direction.

### Equations

- **Equation (15.13)**: Describes the regenerative positive feedback situation, leading to a rapid increase in \( I_A \).

### Description

As the anode current \( I_A \) increases and \( \alpha_1 + \alpha_2 \) increases, the two equivalent bipolar transistors are driven into saturation, and the junction \( J_2 \) becomes forward biased. The total voltage across the device decreases and is approximately equal to one diode drop. The current in the device is limited by the external circuit. If the current is allowed to increase, ohmic losses may become important, slightly increasing the voltage drop across the device with current. The \( I_A \) versus \( V_A \) characteristic is shown in Figure 15.32.

## 15.6.2 Triggering the SCR

In the last section, we considered the case when the four-layer pnpn device is turned on by the avalanche breakdown process in the center junction. The turn-on condition can also be initiated by other means. Figure 15.33a shows three-terminal SCR in which the third terminal is the gate control. We can determine the effect of the gate current by reconsidering Equations (15.11a) and (15.11b).

Figure 15.33b again shows the two-transistor equivalent circuit including the gate current. We can write:

\[
I_{C1} = \alpha_1 I_A + I_{CO1} \tag{15.14a}
\]

and

\[
I_{C2} = \alpha_2 I_K + I_{CO2} \tag{15.14b}
\]

We now have \( I_K = I_A + I_G \), and we can still write \( I_{C1} + I_{C2} = I_A \). Adding Equations (15.14a) and (15.14b), we find that

\[
I_{C1} + I_{C2} = I_A = (\alpha_1 + \alpha_2) I_A + \alpha_2 I_G + I_{CO1} + I_{CO2} \tag{15.15}
\]