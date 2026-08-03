# 💸 GreenWallet AI

> AI-powered personal finance platform that helps users build better financial habits through intelligent expense tracking, savings forecasting, ethical investment insights, and gamified challenges.

---

## 📖 Overview

GreenWallet AI is a modern fintech application designed to make managing money simple, engaging, and educational.

Instead of only tracking expenses, GreenWallet AI analyzes spending behavior, predicts future savings, recommends smarter financial decisions, and encourages consistent saving through gamification.

The goal is to improve financial literacy for students and young professionals while promoting sustainable and ethical investing.

---

## ✨ Features

### 📊 Smart Expense Tracking

- Categorize expenses automatically
- Monthly spending analytics
- Interactive dashboards
- Budget monitoring

### 🤖 AI Financial Insights

- Spending pattern analysis
- Personalized saving suggestions
- Financial health score
- Smart recommendations

### 📈 Savings Forecasting

Predict future savings based on:

- Income
- Spending habits
- Monthly trends
- Historical data

### 🌱 Ethical Investing

Discover investments aligned with personal values.

Examples:

- Renewable Energy
- ESG Funds
- Clean Technology
- Sustainable Companies

### 🎮 Gamification

- Daily saving challenges
- Achievement badges
- Saving streaks
- XP & Level system
- Financial milestones

### 📉 Analytics Dashboard

Visualize

- Monthly expenses
- Category breakdown
- Income vs Expenses
- Savings growth
- Budget utilization

---

## 🛠 Tech Stack

### Frontend

- React
- TypeScript
- Tailwind CSS

### Backend

- Node.js
- Express.js

### Database

- PostgreSQL

### AI / ML

- Python
- Scikit-learn
- Pandas
- NumPy

### Authentication

- JWT
- bcrypt

### Charts

- Recharts / Chart.js

---

## 🏗 Architecture

```
React Frontend
        │
 REST APIs
        │
Node.js + Express
        │
 PostgreSQL Database
        │
 AI Recommendation Engine
        │
 Forecasting Models
```

---

## 📂 Project Structure

```
greenWallet/
│
├── client/
│   ├── components/
│   ├── pages/
│   ├── hooks/
│   └── utils/
│
├── server/
│   ├── routes/
│   ├── controllers/
│   ├── middleware/
│   ├── models/
│   └── services/
│
├── ml/
│   ├── forecasting/
│   ├── recommendation/
│   └── expense-classifier/
│
└── README.md
```

---

## 🚀 Getting Started

### Clone

```bash
git clone https://github.com/Senku24/greenWallet.git
```

```bash
cd greenWallet
```

### Install

Frontend

```bash
cd client
npm install
```

Backend

```bash
cd server
npm install
```

### Environment Variables

Create a `.env` file

```env
DATABASE_URL=

JWT_SECRET=

OPENAI_API_KEY=

PORT=5000
```

### Run

Backend

```bash
npm run dev
```

Frontend

```bash
npm run dev
```

---

## 📸 Screenshots

> Add screenshots or GIFs here.

```
Landing Page

Dashboard

Expense Analytics

Savings Forecast

Investment Recommendations
```

---

## 🎯 Future Roadmap

- AI Financial Chatbot
- OCR Receipt Scanner
- Bank Account Integration
- UPI Expense Sync
- Voice Expense Logging
- Goal-based Saving Plans
- Investment Portfolio Tracker
- Family Finance Dashboard
- Mobile Application
- AI Budget Planner
