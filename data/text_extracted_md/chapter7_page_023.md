# 7.5.1 Linearly Graded Junctions

If we start with a uniformly doped n-type semiconductor, for example, and diffuse acceptor atoms through the surface, the impurity concentrations will tend to be like those shown in Figure 7.16. The point at \( x = x' \) on the figure corresponds to the metallurgical junction. The depletion region extends into the p and n regions from the metallurgical junction as we have discussed previously. The net p-type doping concentration near the metallurgical junction may be approximated as a linear function of distance from the metallurgical junction. Likewise, as a first approximation, the net n-type doping concentration is also a linear function of distance extending into the n region from the metallurgical junction. This effective doping profile is referred to as a linearly graded junction.

Figure 7.17 shows the space charge density in the depletion region of the linearly graded junction. For convenience, the metallurgical junction is placed at \( x = 0 \). The space charge density can be written as

\[
\rho(x) = e\alpha x
\]

(7.62)

where \( \alpha \) is the gradient of the net impurity concentration.

The electric field and potential in the space charge region can be determined from Poisson’s equation. We can write

\[
\frac{dE}{dx} = \frac{\rho(x)}{\varepsilon_s} = \frac{e\alpha x}{\varepsilon_s}
\]

(7.63)

so that the electric field can be found by integration as

\[
E = \int \frac{e\alpha x}{\varepsilon_s} \, dx = \frac{e\alpha}{2\varepsilon_s} (x^2 - x_0^2)
\]

(7.64)

The electric field in the linearly graded junction is a quadratic function of distance rather than the linear function found in the uniformly doped junction. The maximum electric field again occurs at the metallurgical junction. We may note that the electric field is zero at both \( x = +x_0 \) and at \( x = -x_0 \). The electric field in a nonuniformly doped junction...

!Figure 7.16  
**Figure 7.16** Impurity concentrations of a pn junction with a nonuniformly doped p region.

!Figure 7.17  
**Figure 7.17** Space charge density in a linearly graded pn junction.