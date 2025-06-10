# Chapter 12: The Bipolar Transistor

where \( L_B \) is the minority carrier diffusion length in the base, given by \( L_B = \sqrt{D_B \tau_{B0}} \). The base is of finite width so both exponential terms in Equation (12.11) must be retained.

The excess minority carrier electron concentrations at the two boundaries become

\[
\delta n_{B}(x = 0) = \delta n_{B}(0) = A + B
\]

(12.12a)

and

\[
\delta n_{B}(x = x_B) = \delta n_{B}(x_B) = A \exp\left(\frac{+x_B}{L_B}\right) + B \exp\left(\frac{-x_B}{L_B}\right)
\]

(12.12b)

The B–E junction is forward biased, so the boundary condition at \( x = 0 \) is

\[
\delta n_{B}(0) = n_{B}(x = 0) - n_{B0} = n_{B0} \left[\exp\left(\frac{eV_{BE}}{kT}\right) - 1\right]
\]

(12.13a)

The B–C junction is reverse biased, so the second boundary condition at \( x = x_B \) is

\[
\delta n_{B}(x_B) = n_{B}(x = x_B) - n_{B0} = 0 - n_{B0} = -n_{B0}
\]

(12.13b)

From the boundary conditions given by Equations (12.13a) and (12.13b), the coefficients \( A \) and \( B \) from Equations (12.12a) and (12.12b) can be determined. The results are

\[
A = -n_{B0} - n_{B0} \left[\exp\left(\frac{eV_{BE}}{kT}\right) - 1\right] \exp\left(\frac{-x_B}{L_B}\right) \bigg/ 2 \sinh\left(\frac{x_B}{L_B}\right)
\]

(12.14a)

and

\[
B = n_{B0} \left[\exp\left(\frac{eV_{BE}}{kT}\right) - 1\right] \exp\left(\frac{x_B}{L_B}\right) + n_{B0} \bigg/ 2 \sinh\left(\frac{x_B}{L_B}\right)
\]

(12.14b)

Then, substituting Equations (12.14a) and (12.14b) into Equation (12.9), we can write the excess minority carrier electron concentration in the base region as

\[
\delta n_{B}(x) = \frac{n_{B0} \left[\exp\left(\frac{eV_{BE}}{kT}\right) - 1\right] \sinh\left(\frac{x_B - x}{L_B}\right) - \sinh\left(\frac{x}{L_B}\right)}{\sinh\left(\frac{x_B}{L_B}\right)}
\]

(12.15a)

Equation (12.15a) may look formidable with the sinh functions. We have stressed that we want the base width \( x_B \) to be small compared to the minority carrier diffusion length \( L_B \). This condition may seem somewhat arbitrary at this point, but the reason becomes clear as we proceed through all of the calculations. Since we want \( x_B < L_B \), the argument in the sinh functions is always less than unity and in most cases will be much less than unity. Figure 12.15 shows a plot of sinh \( y \) for \( 0 \leq y \leq 1 \) and also shows the linear approximation for small values of \( y \). If \( y < 0.4 \), the sinh \( y \) function differs from its linear approximation by less than 3 percent. All of this leads to the conclusion that the excess electron concentration \( \delta n_{B} \) in Equation (12.15a) is approximately a linear function of \( x \) through the neutral base region.