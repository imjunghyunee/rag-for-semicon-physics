# 15.4 Power Bipolar Transistors

## Table 15.1.1 Comparison of the characteristics and maximum ratings of small-signal and power BJTs

| Parameter                  | Small-signal BJT (2N2222A) | Power BJT (2N3055) | Power BJT (2N6078) |
|----------------------------|----------------------------|--------------------|--------------------|
| \( V_{CE} \) (max) (V)     | 40                         | 60                 | 250                |
| \( I_C \) (max) (A)        | 0.8                        | 15                 | 7                  |
| \( P_D \) (max) (W)        | 1.2                        | 115                | 45                 |
| (at \( T = 25^\circ C \))  |                            |                    |                    |
| \( \beta \)                | 35–100                     | 5–20               | 12–70              |
| \( f_T \) (MHz)            | 300                        | 0.8                | 1                  |

The table compares the characteristics of a general-purpose small-signal BJT to those of two power BJTs. The current gain is generally smaller in the power transistors, typically in the range of 20 to 100, and may be a strong function of collector current and temperature. Figure 15.12 shows typical current gain versus collector current characteristics for the 2N3055 power BJT at various temperatures.

The **maximum rated collector current** \( I_{C,max} \) may be related to the maximum current that the wires connecting the semiconductor to the external terminals can handle, the collector current at which the current gain falls below a minimum specified value, or the current that leads to the maximum power dissipation when the transistor is biased in saturation.

The **maximum rated voltage** in a BJT is generally associated with avalanche breakdown in the reverse-biased base–collector junction. In the common-emitter configuration, the breakdown voltage mechanism also involves the transistor gain, as well as the breakdown phenomenon in the pn junction. This is discussed in Section 12.4.6. Typical \( I_C \) versus \( V_{CE} \) characteristics are shown in Figure 15.13.

!Figure 15.12

**Figure 15.12** Typical dc beta characteristics (\( h_{FE} \) versus \( I_C \)) for 2N3055.

!Figure 15.13

**Figure 15.13** Typical collector current versus collector–emitter voltage characteristics of a bipolar transistor, showing breakdown effects.