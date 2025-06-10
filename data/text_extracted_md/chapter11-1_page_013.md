# 11.2 MOSFET Scaling

**TVU 11.2** Consider an NMOS transistor with the following parameters: 

- \( \mu_n = 1000 \, \text{cm}^2/\text{V} \cdot \text{s} \)
- \( C_{\text{ox}} = 10^{-8} \, \text{F/cm}^2 \)
- \( W = 10 \, \mu\text{m} \)
- \( L = 1 \, \mu\text{m} \)
- \( V_T = 0.4 \, \text{V} \)
- \( v_{\text{sat}} = 5 \times 10^6 \, \text{cm/s} \)

Plot on the same graph \( I_D \) (sat) versus \( V_{DS} \) over the range 0 to \( V_{GS} \leq 4 \, \text{V} \) for the case (a) of an ideal transistor (Equation (10.45a)) and (b) when velocity saturation occurs (Equation (11.17)).

\[ 
I_{\text{sat}} = \frac{(V_{GS} - V_T)^2}{2} \quad \text{and} \quad I_{\text{sat}} = \frac{(V_{GS} - V_T)q}{(V_{GS} - V_T)q + v_{\text{sat}}}
\]

## 11.2.1 MOSFET Scaling

As we noted in the previous chapter, the frequency response of MOSFETs increases as the channel length decreases. The driving force in CMOS technology evolution in the last couple of decades has been reduced channel lengths. Channel lengths of 0.13 μm or less are now the norm. One question that must be considered is what other device parameters must be scaled as the channel length is scaled down.

### 11.2.1 Constant-Field Scaling

The principle of constant-field scaling is that device dimensions and device voltages be scaled such that electric fields (both horizontal and vertical) remain essentially constant. To ensure that the reliability of the scaled device is not compromised, the electric fields in the scaled device must not increase.

Figure 11.12a shows the cross section and parameters of an original NMOS device and Figure 11.12b shows the scaled device, where the scaling parameter is \( k \). Typically, \( k \approx 0.7 \) per generation of a given technology.

As seen in the figure, the channel length is scaled from \( L \) to \( kL \). To maintain a constant horizontal electric field, the drain voltage must also be scaled from \( V_D \) to \( kV_D \). The maximum gate voltage will also be scaled from \( V_G \) to \( kV_G \) so that the gate and drain voltages remain compatible. To maintain a constant vertical electric field, the oxide thickness then must also be scaled from \( t_{\text{ox}} \) to \( kt_{\text{ox}} \).

The maximum depletion width at the drain terminal, for a one-sided pn junction, is

\[
x_D = \sqrt{\frac{2\varepsilon (V_{dn} + V_{bi})}{eN_a}}
\]

(11.21)

!Figure 11.12

**Figure 11.12** | Cross section of (a) original NMOS transistor and (b) scaled NMOS transistor.

- **(a)** Original NMOS transistor
  - \( V_G \)
  - \( V_D \)
  - \( t_{\text{ox}} \)
  - \( L \)
  - \( N_a \) doping

- **(b)** Scaled NMOS transistor
  - \( kV_G \)
  - \( kV_D \)
  - \( kt_{\text{ox}} \)
  - \( kL \)
  - \( \frac{N_a}{k} \) doping