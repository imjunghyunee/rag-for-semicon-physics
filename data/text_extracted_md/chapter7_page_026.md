# Chapter 7 The pn Junction

!Generalized doping profiles of a one-sided p\(^+\)n junction.

*Figure 7.19* Generalized doping profiles of a one-sided p\(^+\)n junction.  
(From Sze [14].)

----

On a much more heavily doped n\(^+\) substrate layer. When the value of \(m\) is negative, we have what is referred to as a **hyperabrupt junction**. In this case, the n-type doping is larger near the metallurgical junction than in the bulk semiconductor. Equation (7.74) is used to approximate the n-type doping over a small region near \(x = x_0\) and does not hold at \(x = 0\) when \(m\) is negative.

The junction capacitance can be derived using the same analysis method as before and is given by

\[
C' = \left( \frac{eB\epsilon (m+1)}{(m + 2)\sqrt{V_{bi} + V}} \right)^{1/(m+2)}
\]

(7.75)

When \(m\) is negative, the capacitance becomes a very strong function of reverse-biased voltage, a desired characteristic in **varactor diodes**. The term varactor comes from the words **variable reactor** and means a device whose reactance can be varied in a controlled manner with bias voltage.

If a varactor diode and an inductance are in parallel, the resonant frequency of the LC circuit is

\[
f_r = \frac{1}{2\pi\sqrt{LC}}
\]

(7.76)

The capacitance of the diode, from Equation (7.75), can be written in the form

\[
C = C_0(V_{bi} + V_R)^{-1/(m+2)}
\]

(7.77)