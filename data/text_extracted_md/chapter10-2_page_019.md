# 10.3 The Basic MOSFET Operation

!MOSFET Diagram

**Figure 10.40** Family of \( I_D \) versus \( V_{DS} \) curves for an n-channel enhancement mode MOSFET.

**Figure 10.41** Cross section of an n-channel depletion mode MOSFET.

Thickness \( t_c \) must be less than the maximum induced space charge width in order to be able to turn the device off. The general \( I_D \) versus \( V_{DS} \) family of curves for an n-channel depletion mode MOSFET is shown in Figure 10.42.

In the next section, we derive the ideal current–voltage relation for the n-channel MOSFET. In the nonsaturation region, we obtain

\[
I_D = \frac{W \mu_n C_{ox}}{2L} \left[ 2(V_{GS} - V_T)V_{DS} - V_{DS}^2 \right]
\]

(10.44a)

which can be written as

\[
I_D = \frac{k'}{2} \cdot \frac{W}{L} \left[ 2(V_{GS} - V_T)V_{DS} - V_{DS}^2 \right]
\]

(10.44b)

or

\[
I_D = K_n \left[ 2(V_{GS} - V_T)V_{DS} - V_{DS}^2 \right]
\]

(10.44c)

!MOSFET Curves

**Figure 10.42** Family of \( I_D \) versus \( V_{DS} \) curves for an n-channel depletion mode MOSFET.