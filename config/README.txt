config/ — Configuration Settings Guide
=======================================

This directory contains configuration files for the Digital Dharma OS
(Alaya V5) autonomous features.

Files
-----

semantic_emotions.json
  Defines four emotional dimensions used by PinealVessel for real-time
  emotional wavelength analysis:

    love     (愛/慈悲)  — Compassion, warmth, gratitude keywords
    logic    (論理/因明) — Analysis, reasoning, proof keywords
    fear     (恐怖/苦)  — Anxiety, doubt, suffering keywords
    creation (創造)     — Innovation, imagination, art keywords

  Each keyword maps to a weight in [0.0, 1.0]:
    1.0 = strong signal
    0.8 = moderate signal
    0.5 = mild signal
    0.3 = weak signal

  Keywords are provided in both Japanese and English.

Customization
-------------

To tune the system's emotional sensitivity:
  1. Open semantic_emotions.json in a text editor
  2. Add/remove keywords or adjust weights
  3. Restart the server (stop-dharma-server.bat → Start-DharmaServer.ps1)

To add support for a new language:
  Add keywords to each of the four categories with appropriate weights.

Security Note
-------------
Do NOT place API keys, tokens, or credentials in this directory.
Use environment variables (CLAUDE_API_KEY, GEMINI_API_KEY) instead.
See SECURITY.md for details.
