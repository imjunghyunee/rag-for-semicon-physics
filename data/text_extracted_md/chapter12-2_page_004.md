# CHAPTER 12 The Bipolar Transistor

and

\[
J_{xE} = (-e)D_{B} \left. \frac{d\delta n_{B}(x)}{dx} \right|_{x=0}
\]

(12.36b)

Using the expression for \(\delta n_{B}(x)\) given in Equation (12.15), we find that

\[
J_{nC} = \frac{eD_{B}n_{B0}}{L_{B}} \left[ \exp \left( \frac{eV_{BE}}{kT} \right) - 1 \right] \frac{1}{\sinh(x_{B}/L_{B})} = \frac{1}{\tanh(x_{B}/L_{B})}
\]

(12.37)

The expression for \(J_{xE}\) is given in Equation (12.34a).

If we again assume that the B–E junction is biased sufficiently far in the forward bias so that \(V_{BE} \gg kT/e\), then \(\exp(eV_{BE}/kT) \gg 1\). Substituting Equations (12.37) and (12.34b) into Equation (12.31b), we have

\[
\alpha_{T} = \frac{J_{nC}}{J_{xE}} = \frac{\exp(eV_{BE}/kT) + \cosh(x_{B}/L_{B})}{1 + \exp(eV_{BE}/kT) \cosh(x_{B}/L_{B})}
\]

(12.38)

In order for \(\alpha_{T}\) to be close to unity, the neutral base width \(x_{B}\) must be much smaller than the minority carrier diffusion length in the base \(L_{B}\). If \(x_{B} \ll L_{B}\), then \(\cosh(x_{B}/L_{B})\) will be just slightly greater than unity. In addition, if \(\exp(eV_{BE}/kT) \gg 1\), then the base transport factor is approximately

\[
\alpha_{T} \approx \frac{1}{\cosh(x_{B}/L_{B})}
\]

(12.39a)

For \(x_{B} \ll L_{B}\), we may expand the cosh function in a Taylor series, so that

\[
\alpha_{T} \approx \frac{1}{\cosh(x_{B}/L_{B})} \approx \frac{1}{1 + \frac{1}{2}(x_{B}/L_{B})^{2}} \approx 1 - \frac{1}{2}(x_{B}/L_{B})^{2}
\]

(12.39b)

The base transport factor \(\alpha_{T}\) will be close to one if \(x_{B} \ll L_{B}\). We can now see why we indicated earlier that the neutral base width \(x_{B}\) would be less than \(L_{B}\).

----

## EXAMPLE 12.2

**Objective:** Calculate the base transport factor.

Assume transistor parameters of \(x_{B} = 0.80 \, \mu m\) and \(L_{B} = 10.0 \, \mu m\).

### Solution

From Equation (12.39a), we find

\[
\alpha_{T} \approx \frac{1}{\cosh \left( \frac{x_{B}}{L_{B}} \right)} = \frac{1}{\cosh \left( \frac{0.80}{10.0} \right)} = 0.9968
\]

### Comment

This simple example shows a typical magnitude of the base transport factor.

### EXERCISE PROBLEM

**Ex 12.2** Repeat Example 12.2 for \(x_{B} = 1.2 \, \mu m\) and \(L_{B} = 10.0 \, \mu m\). (8%60 = 40 %V)