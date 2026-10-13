import random
import numpy as np
import streamlit as st
from traffic import train, evaluate

st.title("Smart Traffic Light - Q-Learning MVP")
episodes = st.slider("Training episodes", 200, 5000, 2000, 200)
alpha = st.slider("Learning rate (alpha)", 0.01, 1.0, 0.1)
gamma = st.slider("Discount factor (gamma)", 0.1, 0.99, 0.9)

if st.button("Train and compare"):
    random.seed(0)
    Q, rewards = train(episodes=episodes, alpha=alpha, gamma=gamma)
    st.line_chart(rewards)
    st.table({
        "Method": ["Q-Learning", "Fixed timer", "Random"],
        "Avg reward (higher is better)": [
            evaluate(lambda s, t: int(np.argmax(Q[s]))),
            evaluate(lambda s, t: (t // 5) % 2),
            evaluate(lambda s, t: random.randint(0, 1)),
        ],
    })