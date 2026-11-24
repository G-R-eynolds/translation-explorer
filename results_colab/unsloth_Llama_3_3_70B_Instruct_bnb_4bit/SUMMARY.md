# Summary: unsloth/Llama-3.3-70B-Instruct-bnb-4bit

80 layers, d = 8192. Every number is inside this one model.

## A. English detour
- English peaks at layer 60/80; Italian overtakes at 69.
- Peak mean P(English) = 0.039 over 28 words.

## B. Round trip (masked)
- Tolerance ε = 0.342; hot-word AUROC = 0.611.
- untranslatable: U = 0.00 doublings, borrowing buys 0.101 (masked: 0.013).
- construction: U = 0.33 doublings, borrowing buys 0.120 (masked: -0.046).
- idiom: U = 0.00 doublings, borrowing buys 0.039 (masked: 0.038).
- control: U = 0.00 doublings, borrowing buys 0.079 (masked: 0.076).
- *Me gusta mucho el duende* → 'I love the elf.'; leftover duende 0.07, gusta 1.00, mucho 1.00.
- Sense recovered for *duende* (standard / keep / gloss / footnote / thick): 0.30 / -0.04 / 0.67 / -0.72 / -0.46

## C. Spectrum es–en (layer 40)
- 8190 shared directions above the null (of 8192); fit on 384593 word pairs.
- English predicted from es: 0.408; es predicted from English: 0.431; bias -0.023 (standardised -0.023).
- AUROC hot vs other words: U 0.78, U_raw 0.70, #tokens baseline 0.45; agreement with round trip 0.13508771929824562.

## C. Spectrum hi–en (layer 40)
- 7288 shared directions above the null (of 8192); fit on 211905 word pairs.
- English predicted from hi: 0.237; hi predicted from English: 0.335; bias -0.098 (standardised -0.100).
- AUROC hot vs other words: U 0.83, U_raw 1.00, #tokens baseline 0.42; agreement with round trip 0.942857142857143.

## D. Mediation (layer 40)
- Hot words, private-part share of harm: translation 82.04, explanation 0.15.
- Flip fractions (private / shared / full): cura +0.00/+0.13/+0.13; cometa +0.01/+0.07/+0.04; pendiente +0.00/+0.17/+0.17; corte +0.00/+0.20/+0.20; frente +0.00/+0.24/+0.24; papa +0.01/+0.04/+0.05; aburrido +0.00/+0.08/+0.09; duende_ctx -0.00/+0.05/+0.05

## D. Mediation (layer 60)
- Hot words, private-part share of harm: translation -1.55, explanation 0.32.
- Flip fractions (private / shared / full): cura -0.00/-0.04/-0.04; cometa +0.00/+0.09/+0.06; pendiente -0.00/+0.01/+0.01; corte -0.00/-0.01/-0.01; frente +0.00/-0.02/-0.02; papa +0.00/+0.02/+0.02; aburrido -0.00/+0.00/+0.00; duende_ctx +0.00/+0.00/+0.00
