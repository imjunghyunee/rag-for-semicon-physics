```markdown
### 15.6 The Thyristor

If \( dV/dt \) is large, the rate of removal of these carriers is rapid, which leads to a large transient current that is equivalent to a gate current and can trigger the device into a low-impedance conducting state. In SCR devices, a \( dV/dt \) rating is usually specified. However, in parasitic pnpn structures, the \( dV/dt \) triggering mechanism is a potential problem.

#### 15.6.3 SCR Turn-Off

Switching the four-layer pnpn structure from its conducting state to its blocking state can be accomplished if the current \( I_A \) is reduced below the value creating the \( \alpha_1 + \alpha_2 = 1 \) condition. This critical \( I_A \) current is called the holding current. If a parasitic four-layer structure is triggered into the conducting state, the effective anode current in the device must be reduced below the corresponding holding current in order to turn off the device. This requirement essentially implies that all power supplies must be turned off in order to bring the parasitic device back into its blocking state.

The SCR can be triggered on by supplying holes to the \( p_2 \) region of the device. The SCR can perhaps be turned off by removing holes from this same region. If the reverse gate current is large enough to bring the npn bipolar transistor out of saturation, then the SCR can be switched from the conducting state into the blocking state. However, the lateral dimensions of the device may be large enough so that nonuniform biasing in the \( J_2 \) and \( J_3 \) junctions occurs during a negative gate current and the device will remain in the low-impedance conducting state. The four-layer pnpn device must be specifically designed for a turn-off capability.

#### 15.6.4 Device Structures

Many thyristor structures have been fabricated with specific characteristics for specific applications. We consider a few of these types of device to gain an appreciation for the variety of structures.

**Basic SCR**  
There are many variations of diffusion, implantation, and epitaxial growth that can be used in the fabrication of the SCR device. The basic structure is shown in Figure 15.36. The \( p_1 \) and \( p_2 \) regions are diffused into a fairly high resistivity \( n \) material. The \( n^+ \) cathode is formed and the \( p^+ \) gate contact is made. High thermal conductivity materials can be used for the anode and cathode ohmic contacts to aid in heat dissipation for high-power devices. The \( n_i \) region width may be on the order of 250 μm in order to support very large reverse-biased voltages across the \( J_2 \) junction. The \( p_1 \) and \( p_2 \) regions may be on the order of 75 μm wide, while the \( n^+ \) and \( p^+ \) regions are normally quite thin.

**Bilateral Thyristor**  
Since thyristors are often used in ac power applications, it may be useful to have a device that switches symmetrically in the positive and negative cycles of the ac voltage. There are a number of such devices, but the basic concept is to connect two conventional thyristors in antiparallel as shown in Figure 15.37a. The integration of this concept into a single device is shown in Figure 15.37b. Symmetrical \( n \) regions can be diffused into a pnp structure. Figure 15.37c shows the current–voltage characteristics in which the triggering into the conduction mode
```