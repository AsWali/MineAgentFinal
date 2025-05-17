# 🪓 MineAgent (Implementation from "MineDojo")

This repository contains the **open-source implementation** of **MineAgent** from the [MineDojo paper (Fan et al., 2022)](https://arxiv.org/abs/2206.08853). The authors described the agent’s architecture but did not release complete code—this repo fills that gap.

## 📌 What is MineAgent?

MineAgent is a hierarchical agent designed for open-ended tasks in Minecraft. It integrates:
- A **language-conditioned policy**
- A **skill selector** trained on human data
- A **low-level controller** that executes atomic actions

The agent can solve diverse tasks ranging from crafting tools to navigating terrain, using natural language instructions as input.

## 🧠 Paper Reference

> Linxi Fan, Guanzhi Wang, Yunfan Jiang, et al.  
> **"MineDojo: Building Open-Ended Embodied Agents with Internet-Scale Knowledge"**  
> *NeurIPS 2022*. [[arXiv](https://arxiv.org/abs/2206.08853)]

