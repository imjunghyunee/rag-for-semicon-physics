# 14.4 Photoluminescence and Electroluminescence

into the p-type base, making the base positive with respect to the emitter. Since the B–E becomes forward-biased, electrons will be injected from the emitter back into the base, leading to the normal transistor action.

From Figure 14.20b, we see that

\[
I_E = \alpha I_C + I_L
\]

(14.44)

where \( I_L \) is the photon-generated current and \( \alpha \) is the common base current gain. Since the base is an open circuit, we have \( I_C = I_E \), so Equation (14.44) can be written as

\[
I_C = \alpha I_C + I_L
\]

(14.45)

Solving for \( I_C \), we find

\[
I_C = \frac{I_L}{1 - \alpha}
\]

(14.46)

Relating \( \alpha \) to \( \beta \), the dc common emitter current gain, Equation (14.46) becomes

\[
I_C = (1 + \beta) I_L
\]

(14.47)

Equation (14.47) shows that the basic B–C photocurrent is multiplied by the factor \((1 + \beta)\). The phototransistor, then, amplifies the basic photocurrent.

With the relatively large B–C junction area, the frequency response of the phototransistor is limited by the B–C junction capacitance. Since the base is essentially the input to the device, the large B–C capacitance is multiplied by the Miller effect, so the frequency response of the phototransistor is further reduced. The phototransistor, however, is a lower-noise device than the avalanche photodiode.

Phototransistors can also be fabricated in heterostructures. The injection efficiency is increased as a result of the bandgap differences, as we discussed in Chapter 12. With the bandgap difference, the lightly doped base restriction no longer applies. A fairly heavily doped, narrow-base device can be fabricated with a high blocking voltage and a high gain.

----

**TEST YOUR UNDERSTANDING**

**TYU 14.5** Consider a long silicon pn junction photodiode with the parameters given in Example 14.5. The cross-sectional area is \( A = 10^{-3} \, \text{cm}^2 \). Assume the photodiode is reverse biased by a 5-volt battery in series with a 5 k\(\Omega\) load resistor. An optical signal at a wavelength of \( \lambda = 1 \, \mu\text{m} \) is incident on the photodiode producing a uniform generation rate of excess carriers throughout the entire device. Determine the incident intensity such that the voltage across the load resistor is 0.5 V. \((\mu\text{W}/\lambda 9920 = \gamma \, \mu\text{V})\)

# 14.4 | PHOTOLUMINESCENCE AND ELECTROLUMINESCENCE

In the first section of this chapter, we have discussed the creation of excess electron–hole pairs by photon absorption. Eventually, excess electrons and holes recombine, and in direct bandgap materials the recombination process may result in the emission of a photon. The general property of light emission is referred to as luminescence.