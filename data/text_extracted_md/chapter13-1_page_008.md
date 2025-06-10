```markdown
# CHAPTER 13 The Junction Field-Effect Transistor

applied forward-bias gate voltage is limited to a few tenths of a volt before there is significant gate current. This device is known as an n-channel enhancement mode MESFET. Enhancement mode p-channel MESFETs and enhancement mode pn junction FETs have also been fabricated. The advantage of enhancement mode MESFETs is that circuits can be designed in which the voltage polarity on the gate and drain is the same. However, the output voltage swing will be quite small with these devices.

## 13.2 THE DEVICE CHARACTERISTICS

To describe the basic electrical characteristics of the JFET, we initially consider a uniformly doped depletion mode pn JFET and then later discuss the enhancement mode device. The pinchoff voltage and drain-to-source saturation voltage are defined and expressions for these parameters derived in terms of geometry and electrical properties. The ideal current–voltage relationship is developed, and then the transconductance, or transistor gain is determined.

Figure 13.9a shows a symmetrical, two-sided pn JFET and Figure 13.9b shows a MESFET with the semi-insulating substrate. One can derive the ideal DC current–voltage relationship for both devices by simply considering the two-sided device to be two JFETs in parallel. We derive the I–V characteristics in terms of \(I_{D1}\) so that the drain current in the two-sided device becomes \(I_{D2} = 2I_{D1}\). We ignore any depletion region at the substrate of the one-sided device in the ideal case.

### 13.2.1 Internal Pinchoff Voltage, Pinchoff Voltage, and Drain-to-Source Saturation Voltage

**n-channel pn JFET** Figure 13.10a shows a simplified one-sided n-channel pn JFET. The metallurgical channel thickness between the p\(^+\) gate region and the substrate is \(a\), and the induced depletion region width for the one-sided p\(^+\)n junction is \(h\). Assume the drain-to-source voltage is zero. If we assume the abrupt depletion approximation, then the space charge width is given by

\[
h = \left[\frac{2\varepsilon_s(V_{bi} - V_{GS})}{eN_d}\right]^{1/2}
\]

(13.1)

where \(V_{GS}\) is the gate-to-source voltage and \(V_{bi}\) is the built-in potential barrier. For a reverse-biased p\(^+\)n junction, \(V_{GS}\) must be a negative voltage.

!Figure 13.9

**Figure 13.9** | Drain currents of (a) a symmetrical, two-sided pn JFET, and (b) a one-sided MESFET.
```