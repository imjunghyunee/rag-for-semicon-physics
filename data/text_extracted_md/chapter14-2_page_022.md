# 14.6 Laser Diodes

At the onset of lasing, which is known as threshold, the optical loss of one round trip through the cavity is just offset by the optical gain. The threshold condition is then expressed as

\[
\Gamma \Gamma_1 \Gamma_2 \exp\left[ (2\gamma(\nu) - 2\alpha(\nu))L \right] = 1
\]

(14.69)

where \(\Gamma_1\) and \(\Gamma_2\) are the reflectivity coefficients of the two end mirrors. For the case when the optical mirrors are cleaved (110) surfaces of gallium arsenide, the reflectivity coefficients are given approximately by

\[
\Gamma_1 = \Gamma_2 = \left( \frac{\bar{n} - \bar{n}_i}{\bar{n} + \bar{n}_i} \right)^2
\]

(14.70)

where \(\bar{n}\) and \(\bar{n}_i\) are the index of refraction parameters for the semiconductor and air, respectively. The parameter \(\gamma(\nu)\) is the optical gain at threshold.

The optical gain at threshold, \(\gamma(\nu)\), may be determined from Equation (14.69) as

\[
\gamma(\nu) = \alpha + \frac{1}{2L} \ln \left( \frac{1}{\Gamma_1 \Gamma_2} \right)
\]

(14.71)

Since the optical gain is a function of the pn junction current, we can define a threshold current density as

\[
J_{th} = \frac{1}{\beta} \left[ \alpha + \frac{1}{2L} \ln \left( \frac{1}{\Gamma_1 \Gamma_2} \right) \right]
\]

(14.72)

where \(\beta\) can be determined theoretically or experimentally. Figure 14.36 shows the threshold current density as a function of the mirror losses. We may note the relatively high threshold current density for a pn junction laser diode.

!Threshold current density of a laser diode as a function of Fabry-Perot cavity end losses.

**Figure 14.36** | Threshold current density of a laser diode as a function of Fabry-Perot cavity end losses.  
*(After Yang [22].)*

### Table: Threshold Current Density

| \( \frac{1}{2L} \ln \frac{1}{\Gamma_1 \Gamma_2} \) (cm\(^{-1}\)) | \( J_{th} \) (A/cm\(^2\)) |
|:---------------------------------------------------------------:|:-------------------------:|
| 0                                                               | 0                         |
| 10                                                              | 500                       |
| 20                                                              | 1000                      |
| 30                                                              | 1500                      |
| 40                                                              | 2000                      |
| 50                                                              | 2500                      |
| 60                                                              | 3000                      |
| 70                                                              | 3500                      |

- \(\bar{\alpha} = 15 \, \text{cm}^{-1}\)
- \(\bar{\beta} = 2.1 \times 10^{-2} \, \text{cm/A}\)