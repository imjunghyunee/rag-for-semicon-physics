```markdown
# CHAPTER 13 The Junction Field-Effect Transistor

## Comment
For this enhancement mode n-channel MESFET, the internal pinchoff voltage is less than the built-in potential barrier. A smaller channel thickness would result in a larger threshold voltage.

## EXERCISE PROBLEM

**Ex 13.5** Consider an n-channel GaAs MESFET with a gate barrier height of \( \phi_b = 0.85 \, \text{V} \).  
The channel doping concentration is \( N_d = 5 \times 10^{15} \, \text{cm}^{-3} \) and the channel thickness is \( a = 0.40 \, \mu\text{m} \). Calculate the internal pinchoff voltage and the threshold voltage.

The design of enhancement mode JFETs implies the use of narrow channel thicknesses and low channel doping concentrations to achieve this condition. The precise control of the channel thickness and doping concentration necessary to achieve internal pinchoff voltages of a few tenths of a volt makes the fabrication of enhancement mode MESFETs difficult.

## EXAMPLE 13.6

**Objective:** Calculate the forward-bias gate voltage required in an n-channel GaAs enhancement mode pn JFET to open up a channel.

Consider a GaAs n-channel pn JFET at \( T = 300 \, \text{K} \) with \( N_a = 10^{18} \, \text{cm}^{-3} \), \( N_d = 3 \times 10^{15} \, \text{cm}^{-3} \), and \( a = 0.70 \, \mu\text{m} \). Determine the forward-bias gate voltage required to open a channel region that is 0.10 \(\mu\text{m}\) thick with zero drain voltage.

### Solution

The built-in potential barrier is

\[
V_{bi} = V_t \left( \frac{N_a N_d}{n_i^2} \right) = (0.0259) \ln \left[ \frac{(10^{18} \times 3 \times 10^{15})}{(1.8 \times 10^6)^2} \right] = 1.25 \, \text{V}
\]

The internal pinchoff voltage is

\[
V_{po} = \frac{ea^2 N_d}{2\epsilon_s} = \frac{(1.6 \times 10^{-19})(0.7 \times 10^{-4})^2(3 \times 10^{15})}{2(13.1)(8.85 \times 10^{-14})} = 1.01 \, \text{V}
\]

which gives a threshold voltage of

\[
V_t = V_{bi} - V_{po} = 0.24 \, \text{V}
\]

The channel depletion width is given by Equation (13.1). Setting \( h = 0.60 \, \mu\text{m} \) will yield an undepleted channel thickness of 0.1 \(\mu\text{m}\). Solving for \( V_{GS} \), we obtain

\[
V_{GS} = V_{bi} - \frac{eh^2 N_d}{2\epsilon_s} = 1.25 - \frac{(1.6 \times 10^{-19})(0.6 \times 10^{-4})^2(3 \times 10^{15})}{2(13.1)(8.85 \times 10^{-14})}
\]

\[
= 1.25 - 0.745 = 0.50 \, \text{V}
\]

## Comment
An applied gate voltage of 0.50 V is greater than the threshold voltage, so the induced depletion region will be smaller than the metallurgical channel thickness. An n-channel region is
```