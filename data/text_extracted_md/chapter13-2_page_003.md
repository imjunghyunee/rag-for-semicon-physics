or

\[
1.109 \times 10^{-3} = \frac{(8000)(13.1)(8.85 \times 10^{-14})(W)}{2(0.70 \times 10^{-4})(1.2 \times 10^{-4})}
\]

The required channel width is then

\[
W = 20.1 \, \mu m
\]

### Comment
The saturation current will obviously increase if \( V_{GS} \) is increased or if the width of the transistor is increased.

### EXERCISE PROBLEM

**Ex 13.7** Consider the GaAs MESFET described in Exercise Problem Ex 13.5. In addition, assume \( \mu_n = 7000 \, \text{cm}^2/\text{V}\cdot\text{s}, L = 0.8 \, \mu m, \) and \( W = 25 \, \mu m \). Calculate the conduction parameter \( k \) and the current \( I_{D(sat)} \) for \( V_{DS} = 0.50 \, V \).

\[ \text{[ \text{VU 5ZC 0 = ( \text{V8S} \text{U4} \text{T4/VU} \text{L1'E = \text{Y} \text{SUV} ]} \] 

The transconductance of the enhancement mode device operating in the saturation region can also be derived. Using Equation (13.47), we can write

\[
g_{em} = \frac{\partial I_{D(sat)}}{\partial V_{GS}} = 2k_{n}(V_{GS} - V_{T})
\]

(13.49)

The transconductance increases as \( V_{GS} \) increases for the enhancement mode device as it did for the depletion mode device.

----

### TEST YOUR UNDERSTANDING

**TYU 13.1** Consider a GaAs pn junction n-channel FET. The \( p^+ \) gate doping concentration is \( N_a = 5 \times 10^{18} \, \text{cm}^{-3} \) and the n-channel doping concentration is \( N_d = 5 \times 10^{15} \, \text{cm}^{-3} \). The zero-bias depletion width is to be 1.2α; that is, the channel is completely depleted at zero bias. Determine the value of α and the pinchoff voltage.

\[ \text{( \text{A L6C 0 = + A \text{U7F} \text{E1'S 0 = D \text{SUV} )} \]

**TYU 13.2** The pinchoff current \( I_{p1} \) given by Equation (13.28) and the pinchoff voltage given by Equation (13.26c) also apply to a p-channel JFET in which \( \mu_n \) is replaced by \( \mu_p \) and \( N_d \) is replaced by \( N_a \). Assume a p-channel silicon JFET has the following parameters: \( N = 5 \times 10^{16} \, \text{cm}^{-3}, N_a = 2 \times 10^{16} \, \text{cm}^{-3}, a = 0.50 \, \mu m, L = 5 \, \mu m, W = 40 \, \mu m, \) and \( p_s = 400 \, \text{cm}^2/\text{V}\cdot\text{s} \). Calculate the pinchoff current \( I_{p1} \) and the maximum drain current \( I_{D(sat)} \) for \( V_{GS} = 0 \).

\[ \text{[ \text{VU 9SC 0 = ( \text{V8S} \text{U4} \text{G9O = 14'} \text{SUV} ]} \]

----

### *13.3.1 NONIDEAL EFFECTS*

As with any semiconductor device, there are nonideal effects that will change the ideal device characteristics. In all of the previous discussions, we have considered an ideal transistor with a constant channel length and constant mobility; we have also...