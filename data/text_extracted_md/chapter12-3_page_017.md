### 12.6 Frequency Limitations

where \( \alpha_0 \) is the low-frequency common-base current gain and \( f_{\alpha} \) is defined as the **alpha cutoff frequency**. The frequency \( f_{\alpha} \) is related to the emitter-to-collector time delay \( \tau_{ec} \) as

\[
f_{\alpha} = \frac{1}{2\pi \tau_{ec}}
\]

(12.98)

When the frequency is equal to the alpha cutoff frequency, the magnitude of the common-base current gain is \( 1/\sqrt{2} \) of its low-frequency value.

We can relate the alpha cutoff frequency to the common-emitter current gain by considering

\[
\beta = \frac{\alpha}{1-\alpha}
\]

(12.99)

We may replace \( \alpha \) in Equation (12.99) with the expression given by Equation (12.97). When the frequency \( f \) is of the same order of magnitude as \( f_{\alpha} \), then

\[
|\beta| = \left| \frac{\alpha}{1-\alpha} \right| \approx \frac{f_{\alpha}}{f}
\]

(12.100)

where we have assumed that \( \alpha_0 \approx 1 \). When the signal frequency is equal to the alpha cutoff frequency, the magnitude of the common-emitter current gain is equal to unity. The usual notation is to define this **cutoff frequency** as \( f_T \), so we have

\[
f_T = \frac{1}{2\pi \tau_{ec}}
\]

(12.101)

From the analysis in Example 12.13, we may also write the common-emitter current gain as

\[
\beta = \frac{\beta_0}{1 + j(f/f_{\beta})}
\]

(12.102)

where \( f_{\beta} \) is called the **beta cutoff frequency** and is the frequency at which the magnitude of the common-emitter current gain \( \beta \) drops to \( 1/\sqrt{2} \) of its low-frequency value.

Combining Equations (12.99) and (12.97), we can write

\[
\beta = \frac{\alpha}{1-\alpha} = \frac{\alpha_0}{1 + j(f/f_{\beta})} = \frac{\alpha_0}{1 - \alpha_0 + j(f/f_T)}
\]

(12.103)

or

\[
\beta = \frac{\alpha_0}{(1-\alpha_0)\left[1 + j\frac{f}{(1-\alpha_0)f_T}\right]} \approx \frac{\beta_0}{1 + j\frac{\beta_0 f}{f_T}}
\]

(12.104)

where

\[
\beta_0 = \frac{\alpha_0}{1-\alpha_0} \approx \frac{1}{1-\alpha_0}
\]