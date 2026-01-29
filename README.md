# Personal Metrics Dashboard

A **local-first personal metrics dashboard** that aggregates activity, performance, and status across multiple platforms into a single unified interface.

---

## Overview

This app allows you to:
- Add apps/accounts (e.g., GitHub, Wattpad, Cashew)
- Authenticate locally per platform
- View key metrics and activity summaries
- Receive AI-condensed notifications and daily insights using a **local LLM (Ollama)**

All data is stored **locally**. No API keys, no cloud database, no Supabase, no `.env` files.

---

## Key Features

- **Unified Dashboard** – All metrics in one place  
- **Local-First Architecture** – SQLite-only storage  
- **AI Summarization** – Ollama for clean, concise insights  
- **Modular Connectors** – Easily add new platforms  
- **Daily Knowledge Feeds**
  - Health (professional summaries)
  - Philippine law updates
  - Philosophy quotes
  - Daily Bible verse
- **Secure by Design**
  - No API keys
  - No `.env` files
  - Local authentication only
- **Customizable UI**
  - Night mode (default)
  - Day / system toggle

---

## Supported Platforms

### Currently Implemented
- GitHub (reference connector)

### Designed to Support
- YouTube, TikTok, Spotify  
- Wattpad, Pinterest, Pixiv  
- Google Classroom, MS Teams  
- Cashew, GCash, MariBank, Western Union  
- Shopee, Lazada, TikTok Affiliate  
- MyFitnessPal  
- Chess.com, Lichess, csTimer  
- Studypool, LinkedIn  
- BandLab, Talkpal  

…and more via modular connectors.

---

## Project Structure (STRICT)

```text
personal-dashboard/
├── app.py
├── Dockerfile
├── requirements.txt
├── templates/
│   ├── base.html
│   ├── dashboard.html
│   ├── add_app.html
│   └── auth.html
├── static/
│   ├── css/
│   ├── js/
│   └── icons/
└── data/        # SQLite DB (created at runtime)
