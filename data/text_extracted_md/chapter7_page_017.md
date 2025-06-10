```markdown
### 7.3 Reverse Applied Bias

!Figure 7.10  
**Figure 7.10** | Space charge density of a one-sided p<sup>+</sup>n junction.

!Figure 7.11  
**Figure 7.11** | \( (1/C^2) \) versus \( V_R \) of a uniformly doped pn junction.

Figure 7.11 shows a plot of Equation (7.48). The built-in potential of the junction can be determined by extrapolating the curve to the point where \( (1/C^2) = 0 \). The slope of the curve is inversely proportional to the doping concentration of the low-doped region in the junction; thus, this doping concentration can be experimentally determined. The assumptions used in the derivation of this capacitance include uniform doping in both semiconductor regions, the abrupt junction approximation, and a planar junction.

----

**Objective:** Determine the impurity doping concentrations in a p<sup>+</sup>n junction given the parameters from Figure 7.11.

Assume that the intercept and the slope of the curve in Figure 7.11 are \( V_{bi} = 0.725 \, \text{V} \) and \( 6.15 \times 10^{15} \, (\text{F/cm}^2)^{-2} \, (\text{V}^{-1}) \), respectively, for a silicon p<sup>+</sup>n junction at \( T = 300 \, \text{K} \).

#### Solution

The slope of the curve in Figure 7.11 is given by \( 2/\varepsilon_s N_d \), so we may write

\[
N_d = \frac{2}{\varepsilon_s \cdot \text{slope}} = \frac{2}{(1.6 \times 10^{-19})(11.7)(8.85 \times 10^{-14})(6.15 \times 10^{15})}
\]

or

\[
N_d = 1.96 \times 10^{15} \, \text{cm}^{-3}
\]

From the expression for \( V_{bi} \), which is

\[
V_{bi} = V_t \ln \left( \frac{N_a N_d}{n_i^2} \right)
\]

we can solve for \( N_a \) as

\[
N_a = \frac{n_i^2}{N_d} \exp \left( \frac{V_{bi}}{V_t} \right) = \frac{(1.5 \times 10^{10})^2}{1.963 \times 10^{15}} \exp \left( \frac{0.725}{0.0259} \right)
\]
```