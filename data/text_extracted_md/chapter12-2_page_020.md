```markdown
!Figure 12.30 | Figure for TYU 12.7.

----

### TEST YOUR UNDERSTANDING

**TYU 12.7** Consider the geometry shown in Figure 12.30. The base doping concentration is \( N_B = 10^{16} \, \text{cm}^{-3} \), the neutral base width is \( x_B = 0.80 \, \mu\text{m} \), the emitter width is \( S = 10 \, \mu\text{m} \), and the emitter length is \( L = 10 \, \mu\text{m} \).

(a) Determine the resistance of the base between \( x = 0 \) and \( x = S/2 \). Assume a hole mobility of \( \mu_p = 400 \, \text{cm}^2/\text{V}\cdot\text{s} \).

(b) If the base current in this region is uniform and given by \( I_B/2 = 5 \, \mu\text{A} \), determine the potential difference between \( x = 0 \) and \( x = S/2 \).

(c) Using the results of part (b), what is the ratio of emitter current density at \( x = 0 \) to that at \( x = S/2 \)?

----

### *12.4.5 Nonuniform Base Doping

In the analysis of the bipolar transistor, we have assumed uniformly doped regions. However, uniform doping rarely occurs. Figure 12.31 shows a doping profile in a doubly diffused npn transistor. We can start with a uniformly doped n-type substrate, diffuse acceptor atoms from the surface to form a compensated p-type base, and then diffuse donor atoms from the surface to form a doubly compensated n-type emitter. The diffusion process results in a nonuniform doping profile.

We determined in Chapter 5 that a graded impurity concentration leads to an induced electric field. For the p-type base region in thermal equilibrium, we can write

\[
J_p = e\mu_p N_a E - eD_p \frac{dN_a}{dx} = 0 \tag{12.51}
\]

Then

\[
E = + \left( \frac{kT}{e} \right) \frac{1}{N_a} \frac{dN_a}{dx} \tag{12.52}
\]

According to the example of Figure 12.31, \( dN_a/dx \) is negative; hence, the induced electric field is in the negative \( x \) direction.
```