# Step 4B.4 EAPF Pi Visual Batch ¡ª Analysis Summary
Generated: 2026-06-17T19:10:43.303389
Trials included: 15

## Results by Condition
- **1.5 Hz**: sync=1.00, TTS=8.67¡À1.17s, MAE=0.0484s, LeaderFCR=0.947, Freq=2.006Hz, Loop=11.7Hz
- **1.8 Hz**: sync=1.00, TTS=6.52¡À2.52s, MAE=0.0504s, LeaderFCR=0.947, Freq=1.997Hz, Loop=10.7Hz
- **2.3 Hz**: sync=1.00, TTS=11.55¡À7.89s, MAE=0.0737s, LeaderFCR=0.950, Freq=2.008Hz, Loop=11.6Hz

**All conditions achieved 100% sync success: True**

## Key Interpretation
- EAPF Consensus successfully transfers to Pi visual HIL: 15 trials, 100% sync success.
- Steady-state MAE range: 0.0484¨C0.0737 s.
- Leader detection reliability: 0.947¨C0.950.
- Effective loop rate: 11¨C12 Hz.
- Step 4B.4: **PASSED**
- Ready for formal Step 5 Kuramoto vs EAPF comparison.