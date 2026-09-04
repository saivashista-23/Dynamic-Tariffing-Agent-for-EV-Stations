# Pricing Agent & Simulation Architecture

This document outlines the theoretical and architectural plan for building the **Pricing Agent** and the **Customer Reaction Simulator**. 

## The Core Challenge
The Demand Prediction Agent gave us the *baseline* future (what will happen if prices stay normal). 
However, when the Pricing Agent changes the price, human behavior changes. To predict how demand, occupancy, and congestion will react to a new price, we must build an **Economic Simulation Engine** based on Price Elasticity.

## 1. The Concept: Price Elasticity of Demand (PED)
In microeconomics, Elasticity measures how sensitive customers are to price changes. Based on our earlier Exploratory Data Analysis (EDA) of the Shenzhen dataset, we discovered a massive structural difference:
*   **CBD (Central Business District): Highly ELASTIC.** Our EDA plots proved that CBD areas are severely *over-provisioned* with chargers, leading to very low utilization per pile. Because supply vastly outweighs demand, customers have endless alternatives. If you raise the price, they will simply drive to the cheaper station next door. They are highly sensitive to price (High Elasticity, e.g., -0.8).
*   **Non-CBD (Workplace/Residential): Highly INELASTIC.** Our EDA plots proved these areas have much higher utilization per pile and massive commuter peaks. During a morning rush at a workplace, customers *must* charge their cars. They have no alternatives and are locked into their location. If you raise the price, they will grit their teeth and pay it. They are insensitive to price (Low Elasticity, e.g., -0.2).

## 2. The Parameterized Heuristic Pricing Agent (The Optimizer)
Based on your clarification, the Pricing Agent is NOT a blind tabular RL agent. It is a highly intelligent **Parameterized Heuristic Function**. 

### A. The Pricing Function
The Agent does not need to look at the clock (`hour_of_week`) because our **LightGBM Demand Prediction Agent** already handles all the complex contextual math (weather, time, holidays) and outputs the ultimate reality. The Pricing Function simply takes those LightGBM predictions:
1.  `predicted_utilization`
2.  `predicted_congestion`
3.  `predicted_load`

It runs them through a dynamic pricing equation:
*   `IF util_rate > surge_threshold`: Apply Surge Formula (scaled by an aggressiveness parameter, $\alpha$).
*   `IF util_rate < discount_threshold`: Apply Discount Formula (scaled by an aggressiveness parameter, $\delta$).

If it's a public holiday, LightGBM will correctly predict a low `util_rate`, and the Pricing Function will automatically trigger a discount. The system is inherently context-aware!

### B. The Learning Loop (Context-Specific Parameter Matrix)
You are exactly right: tuning a single, global $\alpha$ or $\delta$ for the entire city will cause massive problems. The optimal surge aggressiveness for a CBD grid at 9 AM is fundamentally different than a Residential grid at 9 PM.

To solve this, we combine the heuristic formula with a **Parameter Matrix**. Instead of tuning one global knob, the Optimizer tunes thousands of independent knobs based on the context:
*   $\alpha[Grid\_ID][Hour\_of\_Week]$ (Surge Aggressiveness Matrix)
*   $\delta[Grid\_ID][Hour\_of\_Week]$ (Discount Aggressiveness Matrix)

**How the Hybrid Loop Works:**
1. The Agent looks at the current row in the dataset (e.g., Grid 5 on Tuesday 9 AM).
2. It pulls the LightGBM predictions for that row (`util_rate`, `cong_prob`, `load`).
3. It fetches the specific aggressiveness multipliers from the matrix for that exact grid and time ($\alpha_{5, Tue9AM}$).
4. It calculates the price using the heuristic formula: $Price = Base + \alpha_{5, Tue9AM} \times f(util, cong)$.
5. The Simulator calculates the Final Reality and Unified Reward.
6. The Optimizer tweaks *only* $\alpha_{5, Tue9AM}$ and saves it back to the matrix.

By looping through the 4 weeks of data hundreds of times, the Optimizer safely converges on the mathematically perfect aggressiveness multipliers for every single station and hour in the city!
2.  **Environment Simulator:** The Simulator will take the Base State (LightGBM predictions), apply the Price Shock using the newly learned Q-Learning elasticity proxy, and calculate the Final Simulated State.

### D. The Unified Reward Function ($R_t$)
To solve the gradient discontinuity problem, we will use a **Unified Reward Equation** that tracks all variables simultaneously but dynamically shifts their relative importance (weights) based on whether the action is a surge or a discount. 

$R_t = (w_{rev} \times \Delta Revenue) + (w_{util} \times \Delta Utilization) + (w_{uplift} \times Uplift) - (w_{queue} \times \Delta Queue)$

**Dynamic Weighting Logic:**
*   **If Action < 0 (Discounting):**
    *   $w_{rev} = 1.0$ (Still want revenue).
    *   $w_{util} = +2.0$ (Heavily reward *increasing* utilization).
    *   $w_{uplift} = +1.5$ (Reward filling idle piles during off-peak).
    *   $w_{queue} = 0.0$ (Queue length is irrelevant during off-peak).
*   **If Action > 0 (Surging):**
    *   $w_{rev} = 1.5$ (Primary goal is monetizing the peak).
    *   $w_{util} = -2.0$ (Heavily reward *decreasing* utilization to free up piles).
    *   $w_{uplift} = 0.0$ (Uplift is irrelevant).
    *   $w_{queue} = +2.0$ (Heavily reward shrinking the queue length).

*(Note: To perfectly stabilize the RL gradients, we can pass the Action through a `tanh()` function to smoothly blend these weights at the $0.0$ boundary instead of a hard if/else switch).*

## User Review Required & Critical Drawbacks
> [!CAUTION]  
> **Drawback: Q-Learning for Elasticity requires active pricing data**
> Using Q-learning to extract elasticity from offline datasets (ACN/UrbanEV) is highly dangerous. Q-Learning requires states, actions, and rewards. If the historical data has static or non-volatile pricing, the Q-Learning algorithm will just learn the natural daily demand curve, not elasticity. Offline RL suffers heavily from Out-of-Distribution errors. A simple multivariate regression ($\text{Demand} \sim \text{Price} + \text{Time}$) on historical data is mathematically far more accurate for extracting pure elasticity proxies than Q-Learning.
