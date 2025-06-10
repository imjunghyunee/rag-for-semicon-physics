# CHAPTER 10 Fundamentals of the Metal–Oxide–Semiconductor Field-Effect Transistor

!Figure 10.48

**Figure 10.48** (a) \( I_D \) versus \( V_{GS} \) (for small \( V_{DS} \)) for enhancement mode MOSFET.  
(b) Ideal \( \sqrt{I_D} \) versus \( V_{GS} \) in saturation region for enhancement mode (curve A) and depletion mode (curve B) n-channel MOSFETs.

is due to mobility being a function of gate voltage. Both of these effects will be considered in the next chapter. The extrapolation of the straight line to zero current gives the threshold voltage, and the slope is proportional to the inversion carrier mobility.

Now consider the case when the transistor is biased in the saturation region. If we take the square root of Equation (10.66), we obtain

\[
\sqrt{I_{D(sat)}} = \sqrt{\frac{W \mu_n C_{ox}}{2L} (V_{GS} - V_T)}
\]

(10.69)

Figure 10.48b is a plot of Equation (10.69). In the ideal case, we can obtain the same information from both curves. However, as we will see in the next chapter, the threshold voltage may be a function of \( V_{DS} \) in short-channel devices. Since Equation (10.69) applies to devices biased in the saturation region, the \( V_T \) parameter in this equation may differ from the extrapolated value determined in Figure 10.48a. In general, the nonsaturation current–voltage characteristics will produce the more reliable data.

## EXAMPLE 10.8

**Objective:** Determine the inversion carrier mobility from experimental results.

Consider an n-channel MOSFET with \( W = 15 \, \mu m, L = 2 \, \mu m, \) and \( C_{ox} = 6.9 \times 10^{-8} \, F/cm^2 \).  
Assume that the drain current in the nonsaturation region for \( V_{GS} = 0.10 \, V \) is \( I_D = 35 \, \mu A \) at \( V_{GS} = 1.5 \, V \) and \( I_D = 75 \, \mu A \) at \( V_{GS} = 2.5 \, V \).

### Solution

From Equation (10.68), we can write

\[
I_{D2} - I_{D1} = \frac{W \mu_n C_{ox}}{L} (V_{GS2} - V_{GS1}) V_{DS}
\]

so that

\[
75 \times 10^{-6} - 35 \times 10^{-6} = \left( \frac{15}{2} \right) \mu_n (6.9 \times 10^{-8}) (2.5 - 1.5) (0.10)
\]