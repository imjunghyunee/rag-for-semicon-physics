# Chapter 14: Optical Devices

## Solution

We have that

\[
J_s = \frac{I_s}{A} = \left( \frac{eD_n n_{0e}}{L_n} + \frac{eD_p p_{0e}}{L_p} \right) = en_i^2 \left( \frac{D_n}{L_n N_D} + \frac{D_p}{L_p N_A} \right)
\]

We may calculate

\[
L_n = \sqrt{D_n \tau_{n0}} = \sqrt{25 \times (5 \times 10^{-7})} = 35.4 \, \mu m
\]

and

\[
L_p = \sqrt{D_p \tau_{p0}} = \sqrt{10 \times 10^{-7}} = 10.0 \, \mu m
\]

Then

\[
J_s = (1.6 \times 10^{-19})(1.5 \times 10^{19}) \times \left( \frac{25}{(35.4 \times 10^{-4} \times 5 \times 10^{15})} + \frac{10}{(10 \times 10^{-4} \times 10^{15})} \right)
\]

\[
= 3.6 \times 10^{-11} \, \text{A/cm}^2
\]

Then from Equation (14.10), we can find

\[
V_{oc} = V_t \ln \left( 1 + \frac{I_L}{I_s} \right) = V_t \ln \left( 1 + \frac{15 \times 10^{-3}}{3.6 \times 10^{-11}} \right) = (0.0259) \ln \left( 1 + \frac{15 \times 10^{-3}}{3.6 \times 10^{-11}} \right) = 0.514 \, \text{V}
\]

## Comment

We may determine the built-in potential barrier of this junction to be \( V_{bi} = 0.8556 \, \text{V} \). Taking the ratio of the open-circuit voltage to the built-in potential barrier, we find that \( V_{oc}/V_{bi} = 0.60 \). The open-circuit voltage will always be less than the built-in potential barrier.

## Exercise Problem

**Ex 14.3** Consider a GaAs pn junction solar cell with the following parameters:  
\( N_D = 10^{17} \, \text{cm}^{-3}, N_A = 2 \times 10^{16} \, \text{cm}^{-3}, D_n = 190 \, \text{cm}^2/\text{s}, D_p = 10 \, \text{cm}^2/\text{s}, \tau_{n0} = 10^{-7} \, \text{s}, \tau_{p0} = 10^{-8} \, \text{s} \). Assume a photocurrent density of \( J_L = 20 \, \text{mA/cm}^2 \) is generated in the solar cell. (a) Calculate the open-circuit voltage and (b) determine the ratio of open-circuit voltage to built-in potential barrier.  
\[ \text{[8GU 0 = "M/\lambda ^{(} \lambda 1.60 = ^{A} (b) 'SUVI} \]

The power delivered to the load is

\[
P = I \cdot V = I_L \cdot V - I_s \left[ \exp \left( \frac{eV}{kT} \right) - 1 \right] \cdot V \tag{14.11}
\]

We may find the current and voltage which will deliver the maximum power to the load by setting the derivative equal to zero, or \( dP/dV = 0 \). Using Equation (14.11), we find

\[
\frac{dP}{dV} = 0 = I_L - I_s \left[ \exp \left( \frac{eV_m}{kT} \right) - 1 \right] - I_s V_m \left( \frac{e}{kT} \right) \exp \left( \frac{eV_m}{kT} \right) \tag{14.12}
\]

where \( V_m \) is the voltage that produces the maximum power. We may rewrite Equation (14.12) in the form

\[
\left( 1 + \frac{V_m}{V_t} \right) \exp \left( \frac{eV_m}{kT} \right) = 1 + \frac{I_L}{I_s} \tag{14.13}
\]

The value of \( V_m \) may be determined by trial and error. Figure 14.8 shows the maximum power rectangle where \( I_m \) is the current when \( V = V_m \).