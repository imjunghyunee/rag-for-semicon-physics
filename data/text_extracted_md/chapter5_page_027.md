```markdown
# CHAPTER 5 Carrier Transport Phenomena

Once the majority carrier concentration has been determined, we can calculate the low-field majority carrier mobility. For a p-type semiconductor, we can write

\[
J_p = e p \mu_p E_x
\]

(5.57)

The current density and electric field can be converted to current and voltage so that Equation (5.57) becomes

\[
\frac{I_x}{Wd} = \frac{ep\mu_p V_t}{L}
\]

(5.58)

The hole mobility is then given by

\[
\mu_p = \frac{I_x L}{epV_t Wd}
\]

(5.59)

Similarly for an n-type semiconductor, the low-field electron mobility is determined from

\[
\mu_n = \frac{I_x L}{enV_t Wd}
\]

(5.60)

## EXAMPLE 5.8

**Objective:** Determine the majority carrier concentration and mobility, given Hall effect parameters.

Consider the geometry shown in Figure 5.13. Let \( L = 10^{-1} \) cm, \( W = 10^{-2} \) cm, and \( d = 10^{-3} \) cm. Also assume that \( I_x = 1.0 \) mA, \( V_t = 12.5 \) V, \( B_z = 500 \) gauss \( = 5 \times 10^{-2} \) tesla, and \( V_H = -6.25 \) mV.

### Solution

A negative Hall voltage for this geometry implies that we have an n-type semiconductor. Using Equation (5.56), we can calculate the electron concentration as

\[
n = \frac{- (10^{-3}) \times 5 \times 10^{-2}}{(1.6 \times 10^{-19})(10^{-3})(-6.25 \times 10^{-3})} = 5 \times 10^{21} \, \text{m}^{-3} = 5 \times 10^{15} \, \text{cm}^{-3}
\]

The electron mobility is then determined from Equation (5.60) as

\[
\mu_n = \frac{(10^{-3})(10^{-3})}{(1.6 \times 10^{-19})(5 \times 10^{21})(12.5 \times 10^{-3})} = 0.10 \, \text{m}^2/\text{V}\cdot\text{s}
\]

or

\[
\mu_n = 1000 \, \text{cm}^2/\text{V}\cdot\text{s}
\]

### Comment

It is important to note that the MKS units must be used consistently in the Hall effect equations to yield correct results.

### EXERCISE PROBLEM

**Ex 5.8** A p-type silicon sample with the geometry shown in Figure 5.13 has parameters \( L = 0.2 \) cm, \( W = 10^{-2} \) cm, and \( d = 8 \times 10^{-4} \) cm. The semiconductor parameters are \( p = 10^{16} \, \text{cm}^{-3} \) and \( \mu_p = 320 \, \text{cm}^2/\text{V}\cdot\text{s} \). For \( V_t = 10 \) V and \( B_z = 500 \) gauss \( = 5 \times 10^{-2} \) tesla, determine \( I_x \) and \( V_H \). (AU\# 080 = "A \# \nu 8f0 0 = 7 \text{suV}")
```