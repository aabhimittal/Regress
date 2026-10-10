---
title: REGRESS
emoji: 🧪
colorFrom: indigo
colorTo: red
sdk: gradio
sdk_version: 6.30.0
app_file: app.py
pinned: false
license: mit
short_description: Regression testing for self-evolving agents
---

# REGRESS

Regression testing for self-evolving agents. Re-run a frozen probe suite after every
learning cycle; alarm when memory homogenises while confidence rises.

* **Collapse Lab**: the interactive demo (four scenarios, three seeds, default vs calibrated thresholds).
* **Run your own scenario**: choose the collapse onset, strength, seed and length; REGRESS runs live.
* **Analyze your history**: upload `history.json` from `RegressionMonitor.save()` for a CI verdict,
  suspect memories, probe-suite health and an interactive report.

Source: https://github.com/aabhimittal/Regress
