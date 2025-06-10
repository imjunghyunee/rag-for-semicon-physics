# CHAPTER 15 Semiconductor Microwave and Power Devices

!Figure 15.40  
**Figure 15.40** The V groove MOS gated thyristor.  
*(From Baliga [11])*

## MOS Gated Thyristor

The operation of a MOS gated thyristor is based upon controlling the gain of the npn bipolar transistor. Figure 15.40 shows a V-groove MOS gated thyristor. The MOS gate structure must extend into the n-drift region. If the gate voltage is zero, the depletion edge in the p-base remains essentially flat and parallel to the junction \( J_2 \); the gain of the npn transistor is low. This effect is shown in the figure by the dashed line. When a positive gate voltage is applied, the surface of the p base becomes depleted—the depletion region in the p base adjacent to the gate is shown by the dotted line. The undepleted base width \( W_b \) of the npn bipolar device narrows and the gain of the device increases.

At a gate voltage approximately equal to the threshold voltage, electrons from the n\(^+\) emitter are injected through the depletion region into the n-drift region. The potential of the n-drift region is lowered, which further forward biases the p\(^+\) anode to n-drift junction voltage, and the regenerative process is initiated. The gate voltage required to initiate turn-on is approximately the threshold voltage of the MOS device. One advantage of this device is that the input impedance to the control terminal is very high; relatively large currents can be switched with very small capacity coupled gate currents.

## MOS Turn-Off Thyristor

The MOS turn-off thyristor can both turn on and turn off the anode current by applying a signal to a MOS gate terminal. The basic device structure is shown in Figure 15.41. By applying a positive gate voltage, the n\(^+\)p bipolar transistor can be turned on as just discussed. Once the thyristor is turned on, the device can be turned off by applying a negative gate voltage: the negative gate voltage turns on the p-channel MOS transistor that effectively short circuits the B–E junction of the n\(^+\)pn bipolar transistor. Holes that now enter the p-base have an alternative path to the cathode. If the resistance of the p-channel MOSFET becomes low enough, all current will be diverted away from the n\(^+\)p emitter and the n\(^+\)pn device will effectively be turned off.