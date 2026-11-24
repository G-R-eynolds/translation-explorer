# Summary: Qwen/Qwen2.5-0.5B-Instruct

24 layers, d = 896. Every number is inside this one model.

## A. English detour
- English peaks at layer 21/24; Italian overtakes at 24.
- Peak mean P(English) = 0.051 over 28 words.

## B. Round trip (masked)
- Tolerance ε = 0.439; hot-word AUROC = 0.605.
- untranslatable: U = 1.07 doublings, borrowing buys 0.063 (masked: 0.047).
- construction: U = 2.49 doublings, borrowing buys 0.141 (masked: -0.069).
- idiom: U = 0.00 doublings, borrowing buys -0.013 (masked: -0.013).
- control: U = 0.61 doublings, borrowing buys 0.058 (masked: -0.009).
- *Me gusta mucho el duende* → 'I like much the spirit.'; leftover duende 1.00, gusta 0.00, mucho 0.00.
- Sense recovered for *duende* (standard / keep / gloss / footnote / thick): -0.54 / -0.54 / -0.76 / -0.65 / -0.30

## C. Spectrum es–en (layer 18)
- 894 shared directions above the null (of 896); fit on 25191 word pairs.
- English predicted from es: 0.452; es predicted from English: 0.454; bias -0.003 (standardised -0.038).
- AUROC hot vs other words: U 0.73, U_raw 0.60, #tokens baseline 0.45; agreement with round trip 0.2771929824561403.

## C. Spectrum hi–en (layer 6)
- 427 shared directions above the null (of 896); fit on 15769 word pairs.
- English predicted from hi: 0.274; hi predicted from English: 0.381; bias -0.107 (standardised -0.172).
- AUROC hot vs other words: U 0.62, U_raw 0.48, #tokens baseline 0.62; agreement with round trip 0.08571428571428573.

## D. Mediation (layer 18)
- Hot words, private-part share of harm: translation -0.32, explanation 0.17.
- Flip fractions (private / shared / full): cura +0.00/+0.25/+0.24; cometa -0.12/+1.62/+1.46; pendiente +0.01/+0.46/+0.43; corte -0.01/+0.55/+0.54; frente +0.01/+0.39/+0.39; papa +0.03/+0.59/+0.58; aburrido +0.01/+0.25/+0.26; duende_ctx +0.03/-0.30/-0.26

## D. Mediation (layer 21)
- Hot words, private-part share of harm: translation -0.08, explanation -0.01.
- Flip fractions (private / shared / full): cura +0.00/-0.02/-0.01; cometa +0.04/+0.12/+0.17; pendiente -0.05/+0.03/+0.04; corte +0.01/+0.36/+0.35; frente -0.00/+0.03/+0.02; papa -0.01/+0.08/+0.07; aburrido -0.01/+0.08/+0.08; duende_ctx +0.01/-0.09/-0.07
