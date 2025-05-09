# 🚗 DQN-based CarRacing Agent

This project trains a Deep Q-Network (DQN) agent to play the `CarRacing-v2` environment from OpenAI Gym using image-based inputs.

---

## 🛠️ Setup

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/dqn-carracing-agent.git
cd dqn-carracing-agent
```

### 2. Create and Activate Virtual Environment

**For Linux/macOS:**

```bash
python3 -m venv venv
source venv/bin/activate
```

**For Windows:**

```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🧠 Features

* 🧠 DQN with CNN for processing image inputs
* 💾 Experience Replay Buffer
* 🎯 Target Network for stability
* 🎲 Epsilon-Greedy Strategy for exploration
* 🎮 Discretized Actions and Reward Shaping for better learning

---

## 🚀 Training

To train the agent:

```bash
python train.py
```

You can modify training parameters directly in the script or add argparse options if implemented.

---

## 🎥 Evaluation

To evaluate a saved model:

```bash
python evaluate.py --model-path saved_models/dqn_agent.pth
```

Make sure the model file path is correct.

---

## 📁 Project Structure

```
dqn-carracing-agent/
│
├── train.py               # Training loop
├── evaluate.py            # Evaluation script
├── dqn_agent.py           # DQN model and logic
├── utils.py               # Helper functions
├── requirements.txt       # Dependencies
└── saved_models/          # Folder to save trained models
```

---

## 📌 Notes

* Make sure you have a GPU (or adjust batch size for CPU training).
* The environment is from `gymnasium`, so ensure proper setup of `CarRacing-v2`.
