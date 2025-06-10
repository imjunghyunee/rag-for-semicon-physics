## 2.3 Applications of Schrödinger's Wave Equation

!Wave Functions

**Figure 2.10** The wave functions through the potential barrier.

One particular parameter of interest is the transmission coefficient, in this case defined as the ratio of the transmitted flux in region III to the incident flux in region I. Then the transmission coefficient \( T \) is

\[
T = \frac{v_t \cdot A_3 \cdot A_1^*}{v_i \cdot A_1 \cdot A_3^*} = \frac{A_3 \cdot A_1^*}{A_1 \cdot A_3^*}
\]

(2.62)

where \( v_t \) and \( v_i \) are the velocities of the transmitted and incident particles, respectively. Since the potential \( V = 0 \) in both regions I and III, the incident and transmitted velocities are equal. The transmission coefficient may be determined by solving the boundary condition equations. For the special case when \( E \ll V_0 \), we find that

\[
T \approx 16 \left( \frac{E}{V_0} \right) \left( 1 - \frac{E}{V_0} \right) \exp(-2ka)
\]

(2.63)

Equation (2.63) implies that there is a finite probability that a particle impinging a potential barrier will penetrate the barrier and will appear in region III. This phenomenon is called tunneling and it, too, contradicts classical mechanics. We will see later how this quantum mechanical tunneling phenomenon can be applied to semiconductor device characteristics, such as in the tunnel diode.

----

**Objective:** Calculate the probability of an electron tunneling through a potential barrier.

**Example 2.5**

Consider an electron with an energy of 2 eV impinging on a potential barrier with \( V_0 = 20 \) eV and a width of 3 Å.

### Solution

Equation (2.63) is the tunneling probability. The factor \( k_2 \) is

\[
k_2 = \sqrt{\frac{2m(V_0 - E)}{\hbar^2}} = \sqrt{\frac{(2 \cdot 9.11 \times 10^{-31})(20 - 2)(1.6 \times 10^{-19})}{(1.054 \times 10^{-34})^2}}
\]

or

\[
k_2 = 2.17 \times 10^{10} \, \text{m}^{-1}
\]

Then

\[
T = 16(0.1)(1 - 0.1) \exp \left[ -2(2.17 \times 10^{10})(3 \times 10^{-10}) \right]
\]