# Chapter 15: Semiconductor Microwave and Power Devices

The collector–emitter voltage at this maximum power point is

\[
V_{CE} = V_{CC} - I_C R_C = 35 - (1.75)(10) = 17.5 \, \text{V}
\]

The maximum power dissipated in the transistor occurs at the center of the load line. The maximum transistor power dissipation is therefore

\[
P_T = V_{CE} I_C = (17.5)(1.75) = 30.6 \, \text{W}
\]

**Comment**

To find a transistor for a given application, safety factors are normally used. For this example, a transistor with a current rating greater than 3.5 A, a voltage rating greater than 35 V, and a power rating greater than 30.6 W would be required for the application just described.

**Exercise Problem**

**Ex 15.1** Assume the BJT in the common-emitter circuit shown in Figure 15.15 has limiting factors of \( I_{C, \text{max}} = 5 \, \text{A}, V_{CE, \text{max}} = 75 \, \text{V}, \) and \( P_T = 30 \, \text{W} \). Neglecting second breakdown effects, determine the minimum value of \( R_C \) such that the Q-point of the transistor always stays within the safe operating area for (a) \( V_{CC} = 60 \, \text{V}, \) (b) \( V_{CC} = 40 \, \text{V}, \) and (c) \( V_{CC} = 20 \, \text{V} \). In each case, determine the maximum collector current and maximum transistor power dissipation.

\[
\begin{align*}
\text{[M} & \, \text{C} = (\exists u)(d \, \forall s = (\exists u))^u \, \forall s = \gamma y' (s) \land \text{C} = (\exists u)(d \, \forall s = (\exists u))^u \\
& \, \forall s = \gamma y' (q) : \land \, 0 \text{E} = (\exists u)(d \, \forall x = (\exists u))^u \, 0 \text{E} = \gamma y' (q) \, \text{SU} \, \text{V}]
\end{align*}
\]

## 15.4.3 Darlington Pair Configuration

As mentioned, the base width of a power BJT is relatively wide so that the current gain is then relatively small. One method that is used to increase the effective current gain is to use a Darlington pair such as shown in Figure 15.17. Considering the currents, we see that

\[
i_C = i_{CA} + i_{CB} = \beta_A i_B + \beta_B i_{EA} = \beta_A i_B + \beta_B (\beta_A i_B) = \beta_A i_B (1 + \beta_B) = \beta i_B
\]

\[
(15.7)
\]

!Figure 15.17

**Figure 15.17** | An npn Darlington pair configuration.