# CHAPTER 10 Fundamentals of the Metal–Oxide–Semiconductor Field-Effect Transistor

!Threshold voltage of a p-channel MOSFET

**Figure 10.22** | Threshold voltage of a p-channel MOSFET versus the n-type substrate doping concentration for various values of oxide trapped charge (\(t_{\text{ox}} = 500 \, \text{Å}\), aluminum gate).

and

\[
\phi_n = V_T \ln \left( \frac{N_d}{n_i} \right)
\]

(10.33d)

We may note that \(x_{\text{ox}}\) and \(\phi_n\) are defined as positive quantities. We may also note that the notation of \(V_{TP}\) is the threshold voltage that will induce an inversion layer of holes. We will later drop the N and P subscript notation on the threshold voltage, but, for the moment, the notation may be useful for clarity.

Figure 10.22 is a plot of \(V_{TP}\) versus doping concentration for several values of \(Q_i'\). We may note that, for all values of positive oxide charge, this MOS capacitor is always an enhancement mode device. As the \(Q_i'\) charge increases, the threshold voltage becomes more negative, which means that it takes a larger applied gate voltage to create the inversion layer of holes at the oxide–semiconductor interface.

## DESIGN EXAMPLE 10.5

**Objective:** Determine the gate material and design the semiconductor doping concentration to yield a specified threshold voltage.

Consider a MOS device with silicon dioxide and an n-type silicon substrate. The oxide thickness is \(t_{\text{ox}} = 12 \, \text{nm} = 120 \, \text{Å}\) and the oxide charge is \(Q_i' = 2 \times 10^{10} \, \text{cm}^{-2}\). The threshold voltage is to be approximately \(V_{TP} = -0.3 \, \text{V}\).

**Solution**

The solution to this design problem is not straightforward, since the doping concentration parameter appears in the terms \(\phi_n\), \(x_{\text{ox}}\), \(Q_i' (\text{max})\), and \(\phi_n\). The threshold voltage, then, is a nonlinear function of \(N_d\). We resort to trial and error to obtain a solution.

Figure 10.22 shows the threshold voltage for an aluminum gate system. Since the required threshold voltage in this problem is less negative than the values shown in Figure 10.22,