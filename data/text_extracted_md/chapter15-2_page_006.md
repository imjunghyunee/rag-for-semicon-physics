Solving for \( I_a \), we find

\[
I_a = \frac{\alpha_2 I_c + (I_{co1} + I_{co2})}{1 - (\alpha_1 + \alpha_2)}
\]

(15.16)

We can think of the gate current as the flow of holes into the \( p_2 \) region. The additional holes increase the potential of this region, which increases the forward-biased B–E voltage of the npn bipolar transistor, and the transistor action. The transistor action of the npn increases the collector current \( I_{c2} \), which starts the transistor action of the pnp bipolar transistor, and the entire pnnp device can be turned on into its low-impedance state. The gate current required to switch the SCR into its on condition is typically in the milliamp range. SCR can be turned on with a small gate current, which can control hundreds of amperes of anode current. The gate current can be turned off and the SCR will remain in its conducting state. The gate loses control of the device once the SCR is triggered into its conducting state. The current–voltage characteristics of the SCR as a function of gate current is shown in Figure 15.34.

A simple application of an SCR in a half-wave control circuit is shown in Figure 15.35a. The input signal is an ac voltage and a trigger pulse will control the turn-on of the SCR. We assume that the trigger pulse occurs at time \( t_1 \) during the ac voltage cycle. Prior to \( t_1 \), the SCR is off so that the current in the load is zero; thus, there is a zero output voltage. At \( t = t_1 \), the SCR is triggered on and the input voltage appears across the load (neglecting the voltage drop across the SCR). The SCR turns off when the anode-to-cathode voltage becomes zero even though the trigger pulse has been turned off prior to this time. The time at which the SCR is triggered during the voltage cycle can be varied, changing the amount of power delivered to the load. Full-wave control circuits can be designed to increase efficiency and degree of control.

The gate allows control of the turn-on of the SCR. However, the four-layer pnnp structure can also be triggered on by other means. In many integrated circuits, parasitic pnnp structures exist. One such example is the CMOS structure that we considered in Chapter 10. A transient ionizing radiation pulse can trigger the parasitic structure.

!Figure 15.34 | Current–voltage characteristics of an SCR.

**Figure 15.34 | Current–voltage characteristics of an SCR.**

| \( I_a \) | \( I_a = 0 \) | \( I_{g1} \) | \( I_{g2} \) | \( I_{g3} \) | \( I_{g4} = 0 \) |
|-----------|--------------|--------------|--------------|--------------|------------------|
|           | \( I_{g1} \) | \( I_{g2} \) | \( I_{g3} \) |              | \( V_{AK} \)     |
|           | \( 0 < I_{g1} < I_{g2} < I_{g3} \) | | | | |
