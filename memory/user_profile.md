---
name: User profile — Lorenzo, thesis student
description: Who the user is, their compute constraints, and how to frame recommendations
type: user
---

Lorenzo Fresca — master's thesis student at KU Leuven (Belgium). Thesis topic: real-time visual defect detection on industrial images.

Local hardware: Intel Core i7 8th gen + 16 GB RAM, no GPU. Not usable for model training or inference on full Real-IAD / Deceuninck datasets — all serious compute must go to remote GPU (HPC cluster, Colab, or rented cloud).

Compute access:
- KU Leuven Genius HPC cluster (`/scratch/leuven/381/vsc38124/`) — primary, but queue times can be long.
- Google AI Pro (student subscription) — includes Colab Pro with priority T4/V100 and longer sessions.
- Willing to rent paid GPU (RunPod / Vast) for ~1 week when HPC queue stalls.

How to tailor recommendations:
- Always route heavy compute off the local machine.
- Treat HPC as preferred for long runs, Colab Pro as the fallback for time-bounded work, paid GPU for burst weeks.
- When proposing benchmarks, keep the "fits in one Colab Pro session" constraint in mind for preview / iteration work.
