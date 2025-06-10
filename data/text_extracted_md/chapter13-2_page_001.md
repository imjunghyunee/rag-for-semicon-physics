### 13.2 The Device Characteristics

then formed between the source and drain contacts. The forward-bias gate voltage must not be too large or an undesirable gate current will be present in the device.

#### EXERCISE PROBLEM

**Ex 13.6** An n-channel GaAs MESFET has a gate barrier height of \(\phi_{bn} = 0.89 \, \text{V}\). The channel doping concentration is \(N_D = 10^{16} \, \text{cm}^{-3}\). What channel thickness is required to yield a threshold voltage of \(V_T = 0.25 \, \text{V}\)?

*(unit 08C0 = D - SUY)*

Ideally, the I-V characteristics of the enhancement mode device are the same as the depletion mode device—the only real difference is the relative values of the internal pinchoff voltage. The current in the saturation region is given by Equation (13.35) as

\[
I_{D1} = I_{D1}(\text{sat}) = I_{p1} \left\{ 1 - 3 \left[ 1 - \frac{(V_{GS} - V_T)}{V_{p0}} \right] + 2 \left[ 1 - \frac{(V_{GS} - V_T)}{V_{p0}} \right]^{\frac{3}{2}} \right\}
\]

The threshold voltage for the n-channel device is defined in Equation (13.42) as \(V_T = V_{bi} - V_{p0}\), so we can also write

\[
V_{bi} = V_T + V_{p0}
\]

(13.43)

Substituting this expression for \(V_{bi}\) into Equation (13.35), we obtain

\[
I_{D1}(\text{sat}) = I_{p1} \left\{ 1 - 3 \left[ 1 - \frac{(V_{GS} - V_T)}{V_{p0}} \right] + 2 \left[ 1 - \frac{(V_{GS} - V_T)}{V_{p0}} \right]^{\frac{3}{2}} \right\}
\]

(13.44)

Equation (13.44) is valid for \(V_{GS} \geq V_T\).

When the transistor first turns on, we have \((V_{GS} - V_T) \ll V_{p0}\). Equation (13.44) can then be expanded into a Taylor series and we obtain

\[
I_{D1}(\text{sat}) \approx I_{p1} \left[ \frac{3}{4} \left( \frac{V_{GS} - V_T}{V_{p0}} \right)^2 \right]
\]

(13.45)

Substituting the expressions for \(I_{p1}\) and \(V_{p0}\), Equation (13.45) becomes

\[
I_{D1}(\text{sat}) = \frac{\mu_n \epsilon_s W}{2aL} (V_{GS} - V_T)^2 \quad \text{for} \quad V_{GS} \geq V_T
\]

(13.46)

We can now write Equation (13.46) as

\[
I_{D1}(\text{sat}) = k_n (V_{GS} - V_T)^2
\]

(13.47)

where

\[
k_n = \frac{\mu_n \epsilon_s W}{2aL}
\]

(13.48)

The factor \(k_n\) is called a conduction parameter. The form of Equation (13.47) is the same as for a MOSFET.