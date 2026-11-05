# Hardik Pajiala — portfolio

Live at **https://pajiala99.github.io** · [LinkedIn](https://linkedin.com/in/hardik-pajiala)

A single-file interactive portfolio (`index.html`, no build step): projects written up as
problem → what I built → results → what I learned, a skills list that links each area to the
projects that use it, and a guide character who answers questions from my record. Themed as a
fan homage to Hunter × Hunter.

Projects featured: SafeTrail (first place, BIOS 2026 hackathon), Cricket
Win-Probability Lakehouse, FoodBridge, PRAMAAN (AI agent), Smriti (Smart India Hackathon),
Indian Sign Language Recognition, Academic Timetable Generator, E-Commerce Data Lake Pipeline,
Malaria Cell Classification, Lost & Found System.

Two models run live in the visitor's browser, with no server behind them:

- **[Sign-language recognition](https://pajiala99.github.io/#demo-isl)**: the BiLSTM from
  [isl-recognition](https://github.com/pajiala99/isl-recognition), exported to ONNX and run with onnxruntime-web, with
  MediaPipe's JavaScript tasks tracking the body and hands, on held-out test clips or the webcam. `assets/isl/isl.js` is a
  copy of that repo's `web/isl.js`, where the parity checks against the Python pipeline live.
- **[Cricket win probability](https://pajiala99.github.io/#demo-cricket)**: the real-data model from
  [cricket-winprob-lakehouse](https://github.com/pajiala99/cricket-winprob-lakehouse). Spark precomputed its predictions
  for 39,920 chase states into `assets/cricket/winprob_grid.json`, and the sliders look them up.

Run locally: `python3 -m http.server` in this folder. Opening `index.html` directly works for everything except the live
demos, which fetch their model files.
