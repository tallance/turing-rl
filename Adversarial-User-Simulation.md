---
title: "Adversarial User Simulation: Learning Human-Like Users with a Trainable Judge"
authors: Tal Lancewicki, Daniel Jiang
status: draft
---

# Adversarial User Simulation: Learning Human-Like Users with a Trainable Judge

**Proposal.** We propose to train a user simulator *adversarially*: a **generator** produces the next user turn in a conversation, and a **trained discriminator** tries to tell whether a turn came from a real human or from the generator. The generator is rewarded for fooling the discriminator. Because the discriminator keeps adapting, the simulator has to be human-like everywhere the discriminator can probe—not just in the ways it has already been caught.

**Motivation.** Realistic user simulators are needed for offline evaluation and as RL environments for conversational and recommendation agents—you cannot test new policies on real users cheaply, and logged data only covers the policy that collected it. Recent work trains LLMs to behave like users from real human data: [[UserLM]] flips the dialogue and learns the user's turn, [[HumanLM]] rewards alignment with the human response, and [[Turing-RL]] rewards turns that an LLM judge cannot distinguish from a real human's. Turing-RL is the closest to us and gives the cleanest signal—*be indistinguishable from a human*—but its judge is **frozen**. A fixed judge is a fixed target, so the generator can drift toward *fooling that one judge* rather than being genuinely human-like. Turing-RL even caps its reward to fight "more human than human" hacking, which is a symptom of optimizing against a static critic. 

**Question.** *Does adversarially co-training the discriminator with the user-simulator generator produce more human-like simulators than optimizing against a fixed judge?*

## Setup

We consider conversational turns. Let $h$ be the dialogue context (history so far). For each context, the data gives the real human next turn $u^*$, and the **generator** $G_\theta$ produces a candidate turn $\hat u$ given $h$. Following [[Turing-RL]]'s judge, the **discriminator** $D_\phi$ is shown $h$ and the two turns in *random order* and decides which one is the human. Writing $D_\phi(h, a, b)$ for the probability the judge assigns to the *first* turn $a$ being the human one, the two are trained against each other:

$$
\min_\theta \max_\phi \; \mathbb{E}_{h,\; u^* \sim \text{data},\; \hat u \sim G_\theta(\cdot\mid h)} \Big[ \tfrac{1}{2}\log D_\phi(h, u^*, \hat u) + \tfrac{1}{2}\log\big(1 - D_\phi(h, \hat u, u^*)\big) \Big]. \tag{1}
$$

The two terms average over the two presentation orders. The discriminator (max) is trained to correctly identify which slot holds the human turn; the generator (min) is trained so the judge instead picks its turn $\hat u$ as the human one—i.e. to fool the judge. This is similar to Turing-RL's setup, except the judge is **trained** rather than frozen.

**Two special cases.** Let $C(u)$ be the discriminator's scalar *humanness logit* for a single turn, so $\sigma\big(C(u)\big)$ is the probability that turn is human. A judge that sees the pair nests the two classic adversarial objectives: scoring each turn independently with cross-entropy, $\log \sigma\big(C(u^*)\big) + \log\big(1 - \sigma(C(\hat u))\big)$, recovers the standard GAN (given which turn is real); reading the decision off the logit difference, $\sigma\big(C(u^*) - C(\hat u)\big)$, recovers the relativistic GAN [[RelativisticGAN]]. Turing-RL's single-model "which of these two is human?" judge is the general comparator both specialize from; we adopt that general form.

## Approach

**Generator update.** Text is non-differentiable, so we would optimize $G_\theta$ with policy gradient (GRPO). The reward that descends Eq. (1) is

$$
r(\hat u) = -\tfrac{1}{2}\log D_\phi(h, u^*, \hat u) - \tfrac{1}{2}\log\big(1 - D_\phi(h, \hat u, u^*)\big),
$$

which is large when the judge is fooled into calling $\hat u$ the human in either order (note $\hat u$ enters *both* terms, since every comparison is against the fixed real turn $u^*$). In practice one usually uses the *non-saturating* variant

$$
r_{\text{ns}}(\hat u) = \tfrac{1}{2}\log D_\phi(h, \hat u, u^*) + \tfrac{1}{2}\log\big(1 - D_\phi(h, u^*, \hat u)\big),
$$

i.e. directly rewarding $\log$ of the judge's probability that $\hat u$ is human. This is the standard choice across GAN/GAIL variants because it gives strong gradients early in training, when the generator is easily caught and the saturating form above has near-zero gradient. Both are *log-based / unbounded*, unlike Turing-RL, whose reward is effectively *bounded* (the judge score, capped). Since any monotone function of the judge's scores shares the same equilibrium (the generator matching the human distribution), the reward shape is a knob we would ablate: the log-based rewards above versus a bounded one, e.g. $\tfrac{1}{2}D_\phi(h, \hat u, u^*) + \tfrac{1}{2}\big(1 - D_\phi(h, u^*, \hat u)\big)$ (the judge's probability that $\hat u$ is human, closest to Turing-RL). Turing-RL capped its reward to curb hacking of a *frozen* judge; co-training the discriminator removes that need. 

**Discriminator update.** This is the one change relative to Turing-RL: its judge is a frozen frontier model, whereas $D_\phi$ is a classifier trained alongside the generator. It is trained on pairs of a real and a generated turn, shown in random order **without telling the model which is which in the prompt**—if every sample is labeled "generated," the discriminator collapses to always predicting "generated." 

**Alternating loop.** Generator and discriminator updates would alternate on a GAN-style schedule.

**Smaller trainable model.** The judge in Turing-RL is a frontier model we cannot train. We propose to use a smaller trainable LLM (e.g. Qwen3-8B, Turing-RL's base) for both roles. A first experiment is simply to measure how much human-likeness we lose from this swap, before adding the adversarial loop.

**How we differ.**
- *vs. [[Turing-RL]]*: same pipeline and metrics, but the judge goes from frozen to a trainable adversary.
- *vs. [[HumanLM]]*: HumanLM rewards similarity to one ground-truth response (plus state alignment); we reward indistinguishability from the human distribution, with no single response to copy.

## First Experiment

Before building the full adversarial loop, we would test the premise that motivates it: **a frozen judge can be gamed.** We take a *fixed* judge—the paired Turing-style comparator from [[Turing-RL]], which is shown a real human turn and a candidate turn for the same context and decides which is human—and train *only* the generator (GRPO) to maximize the judge's belief that its turn is the human one.

*Question.* With the judge held fixed, can the generator drive the judge to rate its **fake** turn as *more human than the actual human* turn (win rate against the real turn well above 50%)?

If yes, the generator has found a way to win the judge's vote without being more human—exactly the loophole co-training is meant to close, and the same "more human than human" hacking that [[Turing-RL]] suppresses by capping its reward. 

## Evaluation

We would reuse [[Turing-RL]]'s metrics to stay aligned with the literature: a held-out Turing-test win rate (an **independent** judge, never the trained discriminator, to avoid circularity), plus content-similarity checks to confirm we do not lose coverage while chasing realism.

- **Main comparison.** Adversarial (trainable $D$) vs. frozen-judge ([[Turing-RL]]) vs. similarity reward ([[HumanLM]]) vs. SFT user-LM ([[UserLM]]), all from the same base model.
- **Control.** Trainable vs. frozen discriminator with everything else held fixed—this isolates whether co-training is what buys the human-likeness, rather than some other difference.
- **Detectability test.** Following [[ConvApparel]], train a fresh classifier to separate our simulated turns from real ones. ConvApparel found *every* simulator is easily detected; lowering this detectability is a concrete target.
- **Datasets.** Reddit / WildChat / PRISM, and the [[ConvApparel]] domains.
- *(Optional, not the headline)* a gold-user / gold-environment experiment on top.

## Related Work

**[[Turing-RL]]** is the closest: it rewards a user simulator for producing turns a frozen LLM judge cannot tell from a human's. We keep the goal and metrics but make the judge a trainable adversary, removing the fixed-critic hacking surface.

**[[HumanLM]]** trains an LLM user simulator with GRPO, using six psychological state dimensions plus similarity of the generated response to the ground-truth human response. We replace similarity-to-a-single-response with distributional indistinguishability from a trained discriminator.

**[[UserLM]]** builds a purpose-made user LM by "flipping the dialogue": it first extracts a high-level *intent* for each conversation, then fine-tunes the model to generate the next user turn conditioned on that intent (with the intent optional at inference). We share the user-role framing but condition only on the dialogue history—no pre-extracted intent—and shape the simulator with an adversarial signal rather than next-turn likelihood alone.

**[[ConvApparel]]** is a benchmark and validation framework for user simulators in conversational recommenders. It releases real shopping dialogues in which people chatted with either a helpful or a deliberately unhelpful assistant and self-reported their emotional state (happy / frustrated / confused) at each turn. It proposes three tests for a simulator: whether its statistics match real users, whether a discriminator can tell its conversations from real ones (their *Human-Likeness Score*—an LLM classifier fine-tuned to output the probability a conversation is human), and whether it reacts sensibly to an unseen assistant. Their finding is that data-driven simulators are more realistic than prompt-only ones, but all remain easily detectable.

**[[CtrlSim]]** studies controllable simulation and warns that controls extracted post-hoc from full trajectories leak the future and couple the simulator to the data-collection policy. Our generator conditions only on past context and introduces no future-dependent control, so this bias does not arise.

**[[RelativisticGAN]]** trains the discriminator to judge whether a real sample is *more* realistic than a fake one, rather than scoring a single sample as real/fake in isolation. This is the intuition for our pairwise judge, and—together with the standard GAN—a special case of the general "which of these two is human?" comparator we adopt (see Setup).

## References

1. [[Turing-RL]] — *Learning User Simulators with Turing Rewards*, arXiv:2606.19336.
2. [[HumanLM]] — *Simulating Users with State Alignment Beats Response Imitation*, arXiv:2603.03303.
3. [[UserLM]] — *Flipping the Dialogue: Training and Evaluating User Language Models*, arXiv:2510.06552.
4. [[ConvApparel]] — *A Benchmark Dataset and Validation Framework for User Simulators in Conversational Recommenders*, arXiv:2602.16938.
5. [[CtrlSim]] — *Controllable User Simulation*, arXiv:2605.11519.
6. I. Goodfellow, J. Pouget-Abadie, M. Mirza, et al., *Generative Adversarial Networks*, NeurIPS 2014.
7. [[GAIL]] — J. Ho and S. Ermon, *Generative Adversarial Imitation Learning*, NeurIPS 2016.
