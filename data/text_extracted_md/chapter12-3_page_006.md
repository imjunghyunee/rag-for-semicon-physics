# Chapter 12: The Bipolar Transistor

## Solution

From Equation (12.63), the minimum open-emitter junction breakdown voltage must be

\[
BV_{ceo} = \sqrt{B} BV_{cbo}
\]

Assuming the empirical constant \( n \) is 3, we find

\[
BV_{ceo} = \sqrt{100}(15) = 69.6 \, \text{V}
\]

From Figure 7.15, the maximum collector doping concentration should be approximately \( 7 \times 10^{15} \, \text{cm}^{-3} \) to achieve this breakdown voltage.

## Comment

In a transistor circuit, the transistor must be designed to operate under a worst-case situation. In this example, the transistor must be able to operate in an open-base configuration without going into breakdown. As we have determined previously, an increase in breakdown voltage can be achieved by decreasing the collector doping concentration.

## Exercise Problem

**Ex 12.11** A uniformly doped silicon bipolar transistor has base and collector doping concentrations of \( N_B = 7 \times 10^{16} \, \text{cm}^{-3} \) and \( N_C = 3 \times 10^{15} \, \text{cm}^{-3} \), respectively. The common-emitter current gain is \( \beta = 125 \). Assuming an empirical constant of \( n = 3 \), determine (a) \( BV_{ceo} \) and (b) \( BV_{cbo} \).

----

## Test Your Understanding

**TVU 12.8** A particular transistor has an output resistance of 200 kΩ and an Early voltage of \( V_A = 125 \, \text{V} \). Determine the change in collector current when \( V_{CE} \) increases from 2 V to 8 V.

**TVU 12.9** 
(a) If, because of fabrication tolerances, the neutral base width for a set of transistors varies over the range of 0.800 ≤ \( x_B = 1.00 \, \mu\text{m} \), determine the variation in the base transport factor \( \alpha_T \). Assume \( L_B = 1.44 \times 10^{-2} \, \text{cm} \). 
(b) Using the results of part (a) and assuming \( \gamma = 0.9967 \), what is the variation in common-emitter current gain?

**TVU 12.10** The base impurity doping concentration is \( N_B = 3 \times 10^{16} \, \text{cm}^{-3} \) and the metallurgical base width is \( x_B = 0.70 \, \mu\text{m} \). The minimum required punch-through breakdown voltage is specified to be \( V_{PT} = 70 \, \text{V} \). What is the maximum allowed collector doping concentration?

----

## 12.5 Equivalent Circuit Models

In order to analyze a transistor circuit either by hand calculations or using computer codes, one needs a mathematical model, or equivalent circuit, of the transistor. There are several possible models, each one having certain advantages and disadvantages.