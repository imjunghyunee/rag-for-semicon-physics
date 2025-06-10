# Chapter 12: The Bipolar Transistor

[see Equation (6.13)]. Therefore, the current \(i_{BE}\) is also proportional to \(\exp(V_{BE}/V_T)\). The total base current is the sum of \(i_{BE}\) and \(i_{BC}\) and is proportional to \(\exp(V_{BE}/V_T)\).

The ratio of collector current to base current is a constant since both currents are directly proportional to \(\exp(V_{BE}/V_T)\). We can then write

\[
\frac{i_C}{i_B} = \beta
\]

(12.6)

where \(\beta\) is called the **common-emitter current gain**. Normally, the base current will be relatively small so that, in general, the common-emitter current gain is much larger than unity (on the order of 100 or larger).

## 12.1.3 The Modes of Operation

Figure 12.8 shows the npn transistor in a simple circuit. In this configuration, the transistor may be biased in one of three modes of operation. If the B–E voltage is zero or reverse biased (\(V_{BE} \leq 0\)), then majority carrier electrons from the emitter will not be injected into the base. The B–C junction is also reverse biased; thus, the emitter and collector currents will be zero for this case. This condition is referred to as **cutoff**—all currents in the transistor are zero.

When the B–E junction becomes forward biased, an emitter current will be generated as we have discussed, and the injection of electrons into the base results in a collector current. We may write the KVL equations around the collector–emitter loop as

\[
V_{CC} = I_C R_C + V_{CB} + V_{BE} = V_R + V_{CE}
\]

(12.7)

If \(V_{CC}\) is large enough and if \(V_R\) is small enough, then \(V_{CB} > 0\), which means that the B–C junction is reverse biased for this npn transistor. Again, this condition is the forward-active region of operation.

As the forward-biased B–E voltage increases, the collector current and hence \(V_R\) will also increase. The increase in \(V_R\) means that the reverse-biased C–B voltage decreases, or \(|V_{CB}|\) decreases. At some point, the collector current may become large.

!Figure 12.8

**Figure 12.8** | An npn bipolar transistor in a common-emitter circuit configuration.