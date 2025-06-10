### 12.1 The Bipolar Transistor Action

!Figure 12.9  
**Figure 12.9** | Bipolar transistor common-emitter current–voltage characteristics with load line superimposed.

enough that the combination of \(V_R\) and \(V_{CC}\) produces 0 V across the B–C junction. A slight increase in \(I_C\) beyond this point will cause a slight increase in \(V_R\) and the B–C junction will become forward biased (\(V_{CB} < 0\)). This condition is called **saturation**. In the saturation mode of operation, both B–E and B–C junctions are forward biased and the collector current is no longer controlled by the B–E voltage.

Figure 12.9 shows the transistor current characteristics, \(I_C\) versus \(V_{CE}\), for constant base currents when the transistor is connected in the common-emitter configuration (Figure 12.8). When the collector–emitter voltage is large enough so that the base–collector junction is reverse biased, the collector current is a constant in this first-order theory. For small values of C–E voltage, the base–collector junction becomes forward biased and the collector current decreases to zero for a constant base current.

Writing a Kirchhoff’s voltage equation around the C–E loop, we find

\[
V_{CE} = V_{CC} - I_C R_C
\]

(12.8)

Equation (12.8) shows a linear relation between collector current and collector–emitter voltage. This linear relation is called a **load line** and is plotted in Figure 12.9. The load line, superimposed on the transistor characteristics, can be used to visualize the bias condition and operating mode of the transistor. The cutoff mode occurs when \(I_C = 0\), saturation occurs when there is no longer a change in collector current for a change in base current, and the forward-active mode occurs when the relation \(I_C = \beta I_B\) is valid. These three operating modes are indicated on the figure.

A fourth mode of operation for the bipolar transistor is possible, although not with the circuit configuration shown in Figure 12.8. This fourth mode, known as

----

*The concept of “saturation” for the bipolar transistor is not the same as the principle of the “saturation region” for the MOSFET described in Chapter 10. The term “saturation” as applied to the BJT means that the output current and output voltage do not change as the base–emitter voltage changes. The term “saturation region” as applied to the MOSFET means that the output current does not change (ideally) with a change in the drain-to-source voltage.*