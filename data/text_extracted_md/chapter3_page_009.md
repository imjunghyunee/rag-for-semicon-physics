# Chapter 3: Introduction to the Quantum Theory of Solids

Since the potential function \( V(x) \) is everywhere finite, both the wave function \( \psi(x) \) and its first derivative \( \partial \psi(x)/\partial x \) must be continuous. This continuity condition implies that the wave amplitude function \( u(x) \) and its first derivative \( \partial u(x)/\partial x \) must also be continuous.

If we consider the boundary at \( x = 0 \) and apply the continuity condition to the wave amplitude, we have

\[
u_1(0) = u_2(0)
\]

(3.11)

Substituting Equations (3.9) and (3.10) into Equation (3.11), we obtain

\[
A + B = C - D = 0
\]

(3.12)

Now applying the condition that

\[
\frac{du_1}{dx}\bigg|_{x=0} = \frac{du_2}{dx}\bigg|_{x=0}
\]

(3.13)

we obtain

\[
(\alpha - k)A - (\alpha + k)B = (\beta - K)C + (\beta + K)D = 0
\]

(3.14)

We have considered region I as \( 0 < x < a \) and region II as \(-b < x < 0\). The periodicity and the continuity condition mean that the function \( u_1 \), as \( x \to a \), is equal to the function \( u_2 \), as \( x \to -b \). This condition may be written as

\[
u_1(a) = u_2(-b)
\]

(3.15)

Applying the solutions for \( u_1(x) \) and \( u_2(x) \) to the boundary condition in Equation (3.15) yields

\[
Ae^{\alpha a} + Be^{-\alpha a} = Ce^{\beta b} - De^{-\beta b} = 0
\]

(3.16)

The last boundary condition is

\[
\frac{du_1}{dx}\bigg|_{x=a} = \frac{du_2}{dx}\bigg|_{x=-b}
\]

(3.17)

which gives

\[
(\alpha - k)Ae^{\alpha a} - (\alpha + k)Be^{-\alpha a} = (\beta - K)Ce^{\beta b} + (\beta + K)De^{-\beta b} = 0
\]

(3.18)

We now have four homogeneous equations, Equations (3.12), (3.14), (3.16), and (3.18), with four unknowns as a result of applying the four boundary conditions. In a set of simultaneous, linear, homogeneous equations, there is a nontrivial solution if, and only if, the determinant of the coefficients is zero. In our case, the coefficients in question are the coefficients of the parameters \( A, B, C, \) and \( D \).

The evaluation of this determinant is extremely laborious and will not be considered in detail. The result is

\[
-\frac{(\alpha^2 + \beta^2)}{2\alpha \beta} (\sin \alpha a \sin \beta b) + (\cos \alpha a \cos \beta b) = \cos k(a + b)
\]

(3.19)