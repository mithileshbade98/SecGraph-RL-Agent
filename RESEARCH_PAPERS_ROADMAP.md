# 📚 Research Papers Learning Roadmap

**Purpose:** Master the research behind SecGraph-RL Agent through a structured learning path from fundamentals to advanced topics.

---

## 🗺️ **Learning Path Mind Map**

```
FUNDAMENTALS (Week 1)
│
├─ Level 0: Math Prerequisites
│   ├─ Linear Algebra (matrices, vectors, norms)
│   ├─ Probability (distributions, expectations)
│   ├─ Calculus (gradients, chain rule)
│   └─ Basic Neural Networks
│
├─ Level 1: Core ML Concepts
│   ├─ [Paper 1] Neural Networks Basics
│   ├─ [Paper 2] Backpropagation
│   └─ [Paper 3] Optimization (SGD, Adam)
│
└─ Level 2: Transformers Foundation
    └─ [Paper 4] "Attention is All You Need" ⭐

INTERMEDIATE (Week 2-3)
│
├─ Level 3: Embeddings & Retrieval
│   ├─ [Paper 5] Word2Vec
│   ├─ [Paper 6] BERT
│   └─ [Paper 7] Sentence-BERT ⭐
│
├─ Level 4: Graph Neural Networks
│   ├─ [Paper 8] Graph Convolutions (GCN)
│   ├─ [Paper 9] GraphSAGE ⭐
│   └─ [Paper 10] Graph Attention Networks (GAT) ⭐
│
└─ Level 5: Temporal Graphs
    ├─ [Paper 11] Temporal Networks Survey
    └─ [Paper 12] TGAT ⭐

ADVANCED (Week 4-5)
│
├─ Level 6: Reinforcement Learning
│   ├─ [Paper 13] Policy Gradients Intro
│   ├─ [Paper 14] PPO ⭐
│   └─ [Paper 15] Trust Region Methods (TRPO)
│
├─ Level 7: Preference Learning
│   ├─ [Paper 16] RLHF (InstructGPT)
│   ├─ [Paper 17] DPO ⭐
│   └─ [Paper 18] Enhanced DPO (ACL 2024) ⭐
│
└─ Level 8: Reasoning & Agents
    ├─ [Paper 19] Chain-of-Thought
    ├─ [Paper 20] ReAct ⭐
    └─ [Paper 21] Process Supervision ⭐

SPECIALIZED (Week 6)
│
├─ Level 9: Efficient Fine-Tuning
│   ├─ [Paper 22] Parameter-Efficient Transfer Learning
│   ├─ [Paper 23] LoRA ⭐
│   └─ [Paper 24] QLoRA
│
└─ Level 10: Production ML
    ├─ [Paper 25] Concept Drift Detection
    └─ [Paper 26] Online Learning

⭐ = Critical for SecGraph-RL Agent
```

---

## 📖 **Complete Paper List with Prerequisites**

---

### **WEEK 1: FUNDAMENTALS**

---

#### **Level 0: Math Prerequisites (No papers, just concepts)**

**Before you read ANY papers, make sure you understand:**

**Linear Algebra:**
- Vectors and matrices
- Matrix multiplication
- Dot products and norms (L1, L2, Frobenius)
- Eigenvalues and eigenvectors (for spectral GNNs)

**Resources:**
- 3Blue1Brown "Essence of Linear Algebra" (YouTube)
- Khan Academy Linear Algebra

**Probability & Statistics:**
- Probability distributions (Gaussian, Bernoulli)
- Expected value and variance
- Conditional probability
- Maximum likelihood estimation

**Resources:**
- Khan Academy Probability
- StatQuest (YouTube)

**Calculus:**
- Derivatives and gradients
- Partial derivatives
- Chain rule (critical for backprop)

**Resources:**
- 3Blue1Brown "Essence of Calculus" (YouTube)

**Estimated Time:** 3-5 days (if reviewing)

---

#### **Level 1: Core ML Concepts**

**PREREQUISITE READING** (Not research papers, but foundational):

**1. Neural Networks Basics**
- Resource: "Neural Networks and Deep Learning" by Michael Nielsen (free online book)
- Chapters 1-2
- **Learn:** Perceptrons, activation functions, forward pass

**2. Backpropagation**
- Resource: Same book, Chapter 2
- **Learn:** How gradients flow backwards, chain rule application

**3. Optimization**
- Resource: "An Overview of Gradient Descent Optimization Algorithms" - Sebastian Ruder (blog post)
- Link: https://ruder.io/optimizing-gradient-descent/
- **Learn:** SGD, Momentum, Adam

**Estimated Time:** 2-3 days

---

#### **Level 2: Transformers Foundation**

---

**PAPER 1: "Attention is All You Need" ⭐⭐⭐**

**Citation:**
- Vaswani et al., NeurIPS 2017
- arXiv: https://arxiv.org/abs/1706.03762

**Prerequisites:**
✅ Understand neural networks
✅ Know what sequence-to-sequence models do (e.g., translation)
✅ Basic understanding of RNNs (optional but helpful)

**Topics to Study First:**
1. **Sequence-to-sequence models** - Read "Sequence to Sequence Learning with Neural Networks" (Sutskever et al., 2014) - just skim
2. **Attention mechanism basics** - Read "Neural Machine Translation by Jointly Learning to Align and Translate" (Bahdanau et al., 2015) - just the attention section

**Key Concepts in This Paper:**
- Self-attention mechanism
- Multi-head attention
- Positional encoding
- Encoder-decoder architecture

**Why You Need This:**
- Foundation for all modern LLMs
- Positional encoding used in temporal graph encoders
- Attention mechanism appears in GAT (Graph Attention Networks)

**Reading Strategy:**
1. Read Section 1 (Introduction) - understand the motivation
2. Read Section 3.2 (Attention) - this is the core innovation
3. Skip Section 4 (Why Self-Attention) on first read
4. Look at Figure 1 and Figure 2 - understand the architecture
5. Don't worry about implementation details initially

**Estimated Time:** 4-6 hours (spread over 2 days)

**How It's Used in SecGraph:**
- `reason_agent/embeddings/graph_encoder.py:154-219` - Temporal encoding uses same positional encoding technique

---

### **WEEK 2: EMBEDDINGS & GRAPHS**

---

#### **Level 3: Embeddings & Retrieval**

---

**PAPER 2: "Efficient Estimation of Word Representations in Vector Space" (Word2Vec)**

**Citation:**
- Mikolov et al., 2013
- arXiv: https://arxiv.org/abs/1301.3781

**Prerequisites:**
✅ Basic neural networks
✅ Understand the concept of "similarity"

**Topics to Study First:**
1. **Distributional semantics** - "You shall know a word by the company it keeps" (5 min read)
2. **Cosine similarity** - How to measure vector similarity

**Key Concepts:**
- Skip-gram model
- Continuous bag-of-words (CBOW)
- Negative sampling
- Word embeddings

**Why You Need This:**
- Foundation for understanding embeddings
- Concept of "semantic similarity in vector space"

**Reading Strategy:**
1. Read Section 1-2 (Introduction and Model Architectures)
2. Skim Section 3 (Results) - just see that it works
3. Skip Section 4 (Examples) on first read

**Estimated Time:** 2-3 hours

**How It's Used in SecGraph:**
- Conceptual foundation for semantic tool routing

---

**PAPER 3: "BERT: Pre-training of Deep Bidirectional Transformers"**

**Citation:**
- Devlin et al., 2019
- arXiv: https://arxiv.org/abs/1810.04805

**Prerequisites:**
✅ Read "Attention is All You Need" first
✅ Understand Word2Vec

**Topics to Study First:**
1. **Masked language modeling** - What it means to predict masked words
2. **Transfer learning** - Pre-train on large data, fine-tune on small data

**Key Concepts:**
- Bidirectional encoding
- Masked language model (MLM)
- Next sentence prediction (NSP)
- Fine-tuning paradigm

**Why You Need This:**
- Foundation for Sentence-BERT
- Understanding of contextual embeddings

**Reading Strategy:**
1. Read Section 1 (Introduction)
2. Read Section 3.1 (Pre-training BERT)
3. Skim Section 4 (Experiments) - just see the results
4. Skip Section 3.2 (Fine-tuning) on first read

**Estimated Time:** 3-4 hours

---

**PAPER 4: "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks" ⭐⭐⭐**

**Citation:**
- Reimers & Gurevych, EMNLP 2019
- arXiv: https://arxiv.org/abs/1908.10084

**Prerequisites:**
✅ Read BERT paper
✅ Understand cosine similarity

**Topics to Study First:**
1. **Siamese networks** - Two networks sharing weights
2. **Contrastive learning** - Learning by comparing similar/dissimilar pairs
3. **Semantic textual similarity** - Measuring sentence similarity

**Key Concepts:**
- Siamese network architecture
- Pooling strategies (CLS, mean, max)
- Contrastive loss
- Sentence embeddings

**Why You Need This:**
- **DIRECTLY USED** in SecGraph for semantic tool routing
- Generates embeddings for queries and tool descriptions

**Reading Strategy:**
1. Read Section 1 (Introduction) - understand why BERT alone isn't enough
2. Read Section 2 (Siamese Networks) - this is the key contribution
3. Read Section 3 (Pooling) - understand CLS vs mean pooling
4. Look at Table 1 - see the performance gains

**Estimated Time:** 3-4 hours

**How It's Used in SecGraph:**
- `reason_agent/embeddings/text_embedder.py:25-60` - Uses sentence-transformers/all-MiniLM-L6-v2 (based on Sentence-BERT)
- `reason_agent/tools/router.py:45-95` - Semantic routing with Sentence-BERT embeddings

---

#### **Level 4: Graph Neural Networks**

---

**PAPER 5: "Semi-Supervised Classification with Graph Convolutional Networks" (GCN)**

**Citation:**
- Kipf & Welling, ICLR 2017
- arXiv: https://arxiv.org/abs/1609.02907

**Prerequisites:**
✅ Linear algebra (matrix multiplication, eigenvalues)
✅ Understand convolution in CNNs (conceptually)

**Topics to Study First:**
1. **Graph theory basics** - Nodes, edges, adjacency matrix
2. **Spectral graph theory** - Graph Laplacian (just the concept)
3. **Message passing** - How information flows on graphs

**Resources:**
- "A Gentle Introduction to Graph Neural Networks" (Google Distill) - https://distill.pub/2021/gnn-intro/

**Key Concepts:**
- Graph convolution operation
- Localized spectral filters
- Neighborhood aggregation
- Semi-supervised node classification

**Why You Need This:**
- Foundation for understanding all GNNs
- Conceptual basis for GraphSAGE

**Reading Strategy:**
1. Read Section 1 (Introduction)
2. Read Section 2.1 (Spectral Graph Convolutions) - don't get stuck on math
3. Focus on Section 2.2 (Layer-wise propagation) - this is the simplified version
4. Look at Equation 8 - this is the actual formula used

**Estimated Time:** 4-5 hours

**How It's Used in SecGraph:**
- Conceptual foundation for graph encoders

---

**PAPER 6: "Inductive Representation Learning on Large Graphs" (GraphSAGE) ⭐⭐⭐**

**Citation:**
- Hamilton et al., NeurIPS 2017
- arXiv: https://arxiv.org/abs/1706.02216

**Prerequisites:**
✅ Read GCN paper
✅ Understand aggregation functions (mean, max, LSTM)

**Topics to Study First:**
1. **Transductive vs Inductive learning**
   - Transductive: Train and test on same graph (GCN)
   - Inductive: Generalize to new nodes (GraphSAGE)
2. **Sampling** - How to sample neighbors efficiently

**Key Concepts:**
- Neighborhood sampling
- Aggregator functions (mean, LSTM, pooling)
- Inductive learning on graphs
- Mini-batch training on graphs

**Why You Need This:**
- **DIRECTLY IMPLEMENTED** in SecGraph
- Used for generating node embeddings from graph structure

**Reading Strategy:**
1. Read Section 1 (Introduction) - understand transductive vs inductive
2. Read Section 3 (Proposed Method) - this is the core algorithm
3. Study Algorithm 1 - this is the actual GraphSAGE algorithm
4. Read Section 3.3 (Aggregator Functions) - understand mean, LSTM, pooling
5. Skim Section 4 (Experiments) - just see that it works

**Estimated Time:** 5-6 hours

**How It's Used in SecGraph:**
- `reason_agent/embeddings/graph_encoder.py:19-77` - GraphSAGEEncoder implementation
- `reason_agent/embeddings/graph_encoder.py:259-276` - TemporalGNNEncoder uses GraphSAGE

---

**PAPER 7: "Graph Attention Networks" (GAT) ⭐⭐⭐**

**Citation:**
- Veličković et al., ICLR 2018
- arXiv: https://arxiv.org/abs/1710.10903

**Prerequisites:**
✅ Read "Attention is All You Need"
✅ Read GraphSAGE

**Topics to Study First:**
1. **Attention mechanism review** - Go back to Transformer attention
2. **Graph neighborhood** - Understanding which nodes attend to which

**Key Concepts:**
- Attention on graphs
- Multi-head attention for graphs
- Masked attention (only attend to neighbors)
- Learnable attention weights

**Why You Need This:**
- **DIRECTLY IMPLEMENTED** in SecGraph
- Allows model to learn which graph edges are important

**Reading Strategy:**
1. Read Section 1 (Introduction)
2. Read Section 2.1 (Graph Attentional Layer) - this is the key innovation
3. Study Equation 1-3 - these define the attention mechanism
4. Read Section 2.2 (Multi-head Attention)
5. Look at Figure 1 - visual understanding

**Estimated Time:** 4-5 hours

**How It's Used in SecGraph:**
- `reason_agent/embeddings/graph_encoder.py:80-151` - GATEncoder implementation

---

#### **Level 5: Temporal Graphs**

---

**PAPER 8: "Temporal Networks" (Survey Paper)**

**Citation:**
- Holme & Saramäki, Physics Reports, 2012
- arXiv: https://arxiv.org/abs/1108.1780

**Prerequisites:**
✅ Understand basic graph theory
✅ Read GCN and GraphSAGE

**Topics to Study First:**
1. **Time-stamped events** - Events with temporal information
2. **Dynamic graphs** - Graphs that change over time
3. **Temporal paths** - Paths that respect time ordering

**Key Concepts:**
- Temporal networks representation
- Contact sequences
- Time-respecting paths
- Temporal motifs

**Why You Need This:**
- Understand why time matters in graphs
- Foundation for temporal GNNs

**Reading Strategy:**
1. This is a SURVEY - don't read cover-to-cover
2. Read Section 1 (Introduction)
3. Read Section 2.1 (Representations)
4. Skim Section 3 (Measures) - just get the concepts
5. Skip the physics applications

**Estimated Time:** 3-4 hours (selective reading)

**How It's Used in SecGraph:**
- `reason_agent/ingest/graph_loader.py:130-170` - Bitemporal graph loading
- Conceptual foundation for temporal feature extraction

---

**PAPER 9: "Inductive Representation Learning on Temporal Graphs" (TGAT) ⭐⭐**

**Citation:**
- Xu et al., ICLR 2020
- arXiv: https://arxiv.org/abs/2002.07962

**Prerequisites:**
✅ Read GAT (Graph Attention Networks)
✅ Read Temporal Networks survey
✅ Understand positional encoding from Transformers

**Topics to Study First:**
1. **Temporal point processes** - Events in continuous time
2. **Time encoding** - How to represent time in neural networks

**Key Concepts:**
- Temporal attention
- Time encoding function (harmonic functions)
- Inductive learning on temporal graphs
- Self-attention on temporal neighborhoods

**Why You Need This:**
- **DIRECTLY USED** in SecGraph for temporal encoding
- Combines graph structure + time

**Reading Strategy:**
1. Read Section 1 (Introduction)
2. Read Section 3.2 (Time Encoding) - this is the key innovation
3. Study Equation 2 - this is the temporal encoding formula
4. Read Section 3.3 (Temporal Attention)
5. Skim Section 4 (Experiments)

**Estimated Time:** 5-6 hours

**How It's Used in SecGraph:**
- `reason_agent/embeddings/graph_encoder.py:154-219` - TemporalEncoder implementation
- `reason_agent/embeddings/graph_encoder.py:221-305` - TemporalGNNEncoder

---

### **WEEK 3-4: REINFORCEMENT LEARNING**

---

#### **Level 6: Reinforcement Learning Foundations**

---

**PREREQUISITE READING:**

**Sutton & Barto "Reinforcement Learning: An Introduction" (Textbook)**
- Free online: http://incompleteideas.net/book/the-book-2nd.html
- **Read Chapters 1-3** before any RL papers
- **Learn:** MDP, value functions, policy, Q-learning

**Estimated Time:** 5-7 days

---

**PAPER 10: "Simple Statistical Gradient-Following Algorithms" (REINFORCE / Policy Gradients)**

**Citation:**
- Williams, Machine Learning, 1992
- DOI: 10.1007/BF00992696

**Prerequisites:**
✅ Sutton & Barto Chapters 1-3
✅ Understand calculus (gradients)
✅ Understand expected value

**Topics to Study First:**
1. **Markov Decision Process (MDP)** - States, actions, rewards, transitions
2. **Policy** - Mapping from states to actions
3. **Value function** - Expected cumulative reward
4. **Policy gradient theorem** - How to compute gradient of policy

**Key Concepts:**
- REINFORCE algorithm
- Monte Carlo policy gradient
- Baseline for variance reduction
- Score function estimator

**Why You Need This:**
- Foundation for all policy gradient methods
- Understanding for PPO

**Reading Strategy:**
1. Read Section 1-3 (Introduction and Background)
2. Read Section 4 (REINFORCE algorithm)
3. Study Algorithm 1
4. Don't get stuck on proofs - focus on intuition

**Estimated Time:** 4-5 hours

---

**PAPER 11: "Proximal Policy Optimization Algorithms" (PPO) ⭐⭐⭐**

**Citation:**
- Schulman et al., 2017
- arXiv: https://arxiv.org/abs/1707.06347

**Prerequisites:**
✅ Read REINFORCE paper
✅ Understand advantage function A(s,a) = Q(s,a) - V(s)
✅ Know what KL divergence is (measure of distribution difference)

**Topics to Study First:**
1. **Trust region methods** - Limit how much policy can change
2. **TRPO (Trust Region Policy Optimization)** - Just the concept, don't read full paper
3. **Clipping** - Constraining values to a range

**Key Concepts:**
- Clipped surrogate objective
- Importance sampling ratio
- KL penalty
- Generalized Advantage Estimation (GAE)

**Why You Need This:**
- **DIRECTLY USED** in SecGraph for RL training
- Industry standard for policy optimization

**Reading Strategy:**
1. Read Section 1 (Introduction)
2. Read Section 2 (Background) - review policy gradients
3. **FOCUS on Section 3** - this is the PPO algorithm
4. Study Equation 7 - this is the clipped objective
5. Read Section 3.1 (Algorithm) - understand the training loop
6. Skim Section 4 (Experiments) - just see that it works

**Estimated Time:** 6-7 hours

**How It's Used in SecGraph:**
- `reason_agent/rl/ppo.py:14-86` - PPO trainer implementation
- `configs/rl/ppo.yaml:19-35` - PPO hyperparameters (clip_range: 0.2, etc.)

---

**PAPER 12: "Trust Region Policy Optimization" (TRPO) - OPTIONAL**

**Citation:**
- Schulman et al., ICML 2015
- arXiv: https://arxiv.org/abs/1502.05477

**Prerequisites:**
✅ Read REINFORCE
✅ Read PPO (yes, read PPO first - it's simpler!)

**Why Read This:**
- Understand what PPO simplified
- Deeper understanding of trust regions

**Reading Strategy:**
- Skim only - PPO is the practical version
- Understand the concept of constraining policy updates
- Don't get stuck on the math

**Estimated Time:** 2-3 hours (optional, skim)

---

#### **Level 7: Preference Learning & Alignment**

---

**PAPER 13: "Training Language Models to Follow Instructions with Human Feedback" (InstructGPT / RLHF)**

**Citation:**
- Ouyang et al., 2022
- arXiv: https://arxiv.org/abs/2203.02155

**Prerequisites:**
✅ Read PPO paper
✅ Understand language models (GPT basics)

**Topics to Study First:**
1. **Reward modeling** - Training a model to predict human preferences
2. **Three-stage training** - SFT → Reward Model → RL
3. **Human feedback** - Collecting preference data

**Key Concepts:**
- Supervised fine-tuning (SFT)
- Reward model training from comparisons
- RLHF (Reinforcement Learning from Human Feedback)
- Three-stage pipeline

**Why You Need This:**
- Understand the standard approach (RLHF)
- See what DPO improves upon

**Reading Strategy:**
1. Read Section 1 (Introduction)
2. Read Section 3 (Methods) - understand the three stages
3. Study Figure 2 - this shows the full pipeline
4. Skim Section 4 (Results)

**Estimated Time:** 4-5 hours

**How It's Used in SecGraph:**
- Conceptual background for understanding DPO
- Comparison point for "why not RLHF?"

---

**PAPER 14: "Direct Preference Optimization: Your Language Model is Secretly a Reward Model" (DPO) ⭐⭐⭐**

**Citation:**
- Rafailov et al., NeurIPS 2023
- arXiv: https://arxiv.org/abs/2305.18290

**Prerequisites:**
✅ Read InstructGPT/RLHF paper
✅ Read PPO paper
✅ Understand KL divergence
✅ Understand sigmoid function

**Topics to Study First:**
1. **Bradley-Terry model** - Modeling pairwise preferences
2. **Implicit reward modeling** - Extracting rewards from preferences directly
3. **Reference model** - Frozen copy of the original model

**Key Concepts:**
- Direct optimization of preferences
- Bypass explicit reward model
- Bradley-Terry preference model
- Beta parameter (inverse temperature)

**Why You Need This:**
- **DIRECTLY USED** in SecGraph
- Simpler than RLHF, better results

**Reading Strategy:**
1. Read Section 1 (Introduction) - understand the motivation
2. **FOCUS on Section 3** - this is the core contribution
3. Study Equation 5 - this is the DPO loss function
4. Read Section 3.1 (Deriving DPO) - understand the intuition
5. Skim Section 4 (Experiments)

**Estimated Time:** 6-7 hours

**How It's Used in SecGraph:**
- `reason_agent/rl/dpo.py:14-90` - DPO trainer implementation
- `configs/rl/dpo.yaml:41-67` - DPO parameters (beta, dynamic_beta, etc.)

---

**PAPER 15: "Improving Direct Preference Optimization with Dynamic Beta and Offset" ⭐⭐**

**Citation:**
- ACL 2024 Findings
- URL: https://aclanthology.org/2024.findings-acl.216/

**Prerequisites:**
✅ Read DPO paper thoroughly

**Topics to Study First:**
1. **Beta scheduling** - Adjusting hyperparameters during training
2. **Preference strength** - Not all preferences are equally strong
3. **Margin-based losses** - Adding margins to preferences

**Key Concepts:**
- Dynamic beta (adjust based on confidence)
- Offset-DPO (handle preference magnitude)
- Adaptive scheduling

**Why You Need This:**
- **DIRECTLY USED** in SecGraph (enhanced DPO)
- 2024 improvements over vanilla DPO

**Reading Strategy:**
1. Read Section 1 (Introduction)
2. Read Section 3 (Method) - dynamic beta and offset
3. Study Algorithm 1 - see how beta adapts
4. Skim Section 4 (Experiments)

**Estimated Time:** 3-4 hours

**How It's Used in SecGraph:**
- `configs/rl/dpo.yaml:48-57` - Dynamic beta and offset parameters

---

#### **Level 8: Reasoning & Agents**

---

**PAPER 16: "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models"**

**Citation:**
- Wei et al., NeurIPS 2022
- arXiv: https://arxiv.org/abs/2201.11903

**Prerequisites:**
✅ Understand language models
✅ Know what few-shot learning is

**Topics to Study First:**
1. **Prompting** - How to query LLMs
2. **Few-shot learning** - Learning from examples in the prompt
3. **Reasoning tasks** - Math, logic, commonsense

**Key Concepts:**
- Chain-of-thought (step-by-step reasoning)
- Intermediate steps
- Few-shot examples with reasoning
- Emergent ability in large models

**Why You Need This:**
- Foundation for understanding ReAct
- Concept of explicit reasoning steps

**Reading Strategy:**
1. Read Section 1 (Introduction)
2. Look at Figure 1 - this shows the key idea visually
3. Read Section 2 (Chain-of-Thought Prompting)
4. Skim Section 3 (Experiments) - just see the performance gains

**Estimated Time:** 3-4 hours

---

**PAPER 17: "ReAct: Synergizing Reasoning and Acting in Language Models" ⭐⭐⭐**

**Citation:**
- Yao et al., ICLR 2023
- arXiv: https://arxiv.org/abs/2210.03629

**Prerequisites:**
✅ Read Chain-of-Thought paper
✅ Understand the concept of "tool use" by LLMs

**Topics to Study First:**
1. **Interleaving reasoning and actions** - Think, Act, Observe loop
2. **Tool-augmented LLMs** - LLMs that can call external functions
3. **Trajectory** - Sequence of thoughts and actions

**Key Concepts:**
- ReAct prompting format
- Thought → Action → Observation cycle
- Grounded reasoning (using external tools)
- Trajectory comparison

**Why You Need This:**
- **DIRECTLY USED** in SecGraph
- Foundation for the reasoning planner

**Reading Strategy:**
1. Read Section 1 (Introduction)
2. **FOCUS on Section 3** (ReAct Prompting)
3. Study Figure 2 - this shows the ReAct format
4. Read Section 3.1 (Knowledge Tasks)
5. Skim Section 4 (Experiments)

**Estimated Time:** 4-5 hours

**How It's Used in SecGraph:**
- `reason_agent/reasoning/planner.py:90-150` - ReAct-style planning
- `reason_agent/reasoning/trace_recorder.py:25-80` - Thought-Action-Observation traces

---

**PAPER 18: "Let's Verify Step by Step" (Process Supervision) ⭐⭐⭐**

**Citation:**
- Lightman et al., OpenAI, 2023
- arXiv: https://arxiv.org/abs/2305.20050

**Prerequisites:**
✅ Read Chain-of-Thought
✅ Understand reinforcement learning basics

**Topics to Study First:**
1. **Process rewards vs outcome rewards** - Rewarding steps vs final answer
2. **Supervision** - Different levels of feedback
3. **Math verification** - How to check if math is correct

**Key Concepts:**
- Process supervision (step-level feedback)
- Outcome supervision (final answer only)
- PRM (Process Reward Model)
- Active learning for step labeling

**Why You Need This:**
- **DIRECTLY USED** in SecGraph (verifiable rewards)
- Foundation for process-level rewards

**Reading Strategy:**
1. Read Section 1 (Introduction) - understand process vs outcome
2. Study Figure 1 - this shows the key difference
3. Read Section 2 (Method)
4. Read Section 3 (Results) - see that process > outcome
5. Skim Section 4 (Analysis)

**Estimated Time:** 4-5 hours

**How It's Used in SecGraph:**
- `reason_agent/rl/rewards.py:30-85` - Process rewards implementation
- `reason_agent/reasoning/verifiers/` - Step-level verification

---

**PAPER 19: "Training Language Models with Verifiable Rewards" ⭐⭐**

**Citation:**
- arXiv: https://arxiv.org/abs/2410.15246 (2024)

**Prerequisites:**
✅ Read "Let's Verify Step by Step"
✅ Read PPO paper

**Topics to Study First:**
1. **Verifiable tasks** - Tasks with objective ground truth
2. **Reward hacking** - When models game the reward function
3. **Objective verification** - Automated checking without humans

**Key Concepts:**
- Verifiable reward functions
- Programmatic verification
- Avoiding reward modeling pitfalls
- Scalable reward supervision

**Why You Need This:**
- **DIRECTLY CITED** in SecGraph
- Justification for verifiable rewards over RLHF

**Reading Strategy:**
1. Read Section 1 (Introduction)
2. Read Section 2 (Verifiable Rewards)
3. Study examples of verifiable tasks
4. Skim Section 3 (Experiments)

**Estimated Time:** 3-4 hours

**How It's Used in SecGraph:**
- `configs/rl/ppo.yaml:47-76` - Verifiable reward configuration
- Policy verifier, math verifier, unit test verifier

---

### **WEEK 5: EFFICIENT FINE-TUNING**

---

#### **Level 9: Parameter-Efficient Methods**

---

**PAPER 20: "Parameter-Efficient Transfer Learning for NLP"**

**Citation:**
- Houlsby et al., ICML 2019
- arXiv: https://arxiv.org/abs/1902.00751

**Prerequisites:**
✅ Understand neural network layers
✅ Know what transfer learning is

**Topics to Study First:**
1. **Adapters** - Small modules inserted into pre-trained models
2. **Parameter efficiency** - Training fewer parameters
3. **Bottleneck architecture** - Down-project, transform, up-project

**Key Concepts:**
- Adapter layers
- Bottleneck design
- Sequential vs parallel adapters
- Parameter count comparison

**Why You Need This:**
- Background for understanding LoRA
- Concept of efficient fine-tuning

**Reading Strategy:**
1. Read Section 1 (Introduction)
2. Read Section 2 (Adapter Modules)
3. Study Figure 1 - adapter architecture
4. Skim Section 3 (Experiments)

**Estimated Time:** 2-3 hours

---

**PAPER 21: "LoRA: Low-Rank Adaptation of Large Language Models" ⭐⭐⭐**

**Citation:**
- Hu et al., ICLR 2022
- arXiv: https://arxiv.org/abs/2106.09685

**Prerequisites:**
✅ Linear algebra (matrix decomposition, rank)
✅ Read adapter paper above
✅ Understand the concept of low-rank approximation

**Topics to Study First:**
1. **Matrix rank** - Number of linearly independent rows/columns
2. **Low-rank decomposition** - Approximating matrix with product of smaller matrices
3. **SVD (Singular Value Decomposition)** - Not needed in detail, just the concept

**Key Concepts:**
- Low-rank adaptation
- LoRA decomposition: ΔW = BA
- Rank r as a hyperparameter
- LoRA alpha (scaling factor)
- Merging LoRA weights

**Why You Need This:**
- **DIRECTLY USED** in SecGraph
- Standard for efficient LLM fine-tuning

**Reading Strategy:**
1. Read Section 1 (Introduction) - understand the problem
2. **FOCUS on Section 4** (LoRA Method)
3. Study Equation 1 - this is the key formula: h = W₀x + BAx
4. Read Section 4.2 (LoRA for Transformers) - where to apply LoRA
5. Study Figure 1 - visual understanding
6. Skim Section 5 (Experiments)

**Estimated Time:** 5-6 hours

**How It's Used in SecGraph:**
- `configs/rl/ppo.yaml:9-16` - LoRA configuration (r=16, alpha=32)
- `configs/rl/dpo.yaml:11-18` - LoRA for DPO training

---

**PAPER 22: "QLoRA: Efficient Finetuning of Quantized LLMs" - OPTIONAL**

**Citation:**
- Dettmers et al., NeurIPS 2023
- arXiv: https://arxiv.org/abs/2305.14314

**Prerequisites:**
✅ Read LoRA paper
✅ Understand quantization (concept of using fewer bits)

**Why Read This:**
- Even more efficient than LoRA (combines LoRA + quantization)
- Useful for understanding memory optimization

**Reading Strategy:**
- Skim only - understand the combination of LoRA + 4-bit quantization
- Not directly used in SecGraph but good to know

**Estimated Time:** 2-3 hours (optional)

---

### **WEEK 6: PRODUCTION ML & SPECIALIZATION**

---

#### **Level 10: Production ML Systems**

---

**PAPER 23: "Learning under Concept Drift: A Review"**

**Citation:**
- Gama et al., IEEE TKDE, 2014
- DOI: 10.1109/TKDE.2013.109

**Prerequisites:**
✅ Basic machine learning
✅ Understand distribution shift

**Topics to Study First:**
1. **Distribution shift** - When data distribution changes over time
2. **Covariate shift** - When P(X) changes but P(Y|X) stays same
3. **Concept drift** - When P(Y|X) changes

**Key Concepts:**
- Types of drift (sudden, gradual, incremental, recurring)
- Detection methods (statistical tests, error-based)
- Adaptation strategies (window-based, ensemble)

**Why You Need This:**
- **DIRECTLY USED** in SecGraph for drift detection
- Understanding why models degrade over time

**Reading Strategy:**
1. This is a SURVEY - selective reading
2. Read Section 1 (Introduction)
3. Read Section 2 (Concept Drift)
4. Read Section 3.1 (Detection Methods) - focus on statistical approaches
5. Skim Section 4 (Adaptation)

**Estimated Time:** 4-5 hours

**How It's Used in SecGraph:**
- `reason_agent/monitoring/drift.py:39-106` - Drift detection implementation

---

**PAPER 24: "Covariate Shift Adaptation by Importance Weighted Cross Validation"**

**Citation:**
- Shimodaira, JMLR, 2000
- URL: https://www.jmlr.org/papers/v1/shimodaira00a.html

**Prerequisites:**
✅ Probability and statistics
✅ Understand importance sampling

**Topics to Study First:**
1. **Importance sampling** - Reweighting samples from one distribution to match another
2. **Cross validation** - Model selection technique

**Key Concepts:**
- Covariate shift
- Importance weighting
- Kernel density estimation
- Adaptive learning

**Why You Need This:**
- Statistical foundation for drift detection
- Understanding distribution shift

**Reading Strategy:**
1. Read Section 1 (Introduction)
2. Read Section 2 (Covariate Shift)
3. Skim Section 3 (Method) - don't get stuck on proofs
4. Understand the concept more than the math

**Estimated Time:** 3-4 hours

**How It's Used in SecGraph:**
- `reason_agent/monitoring/drift.py:39-71` - Embedding drift detection (centroid + covariance shift)

---

**PAPER 25: "Online Learning: A Comprehensive Survey"**

**Citation:**
- Hoi et al., Neurocomputing, 2021
- arXiv: https://arxiv.org/abs/1802.02871

**Prerequisites:**
✅ Basic machine learning
✅ Understand gradient descent

**Topics to Study First:**
1. **Online learning** - Learning from streaming data
2. **Regret** - Difference between online learner and best offline strategy
3. **Stochastic gradient descent** - Online optimization

**Key Concepts:**
- Online convex optimization
- Regret bounds
- Second-order methods (AdaGrad, Adam)
- Online-to-batch conversion

**Why You Need This:**
- Background for continuous learning
- Understanding online adaptation

**Reading Strategy:**
1. This is a SURVEY - very selective reading
2. Read Section 1 (Introduction)
3. Read Section 2.1 (Online Learning Framework)
4. Skim Section 3 (Algorithms) - just get the taxonomy
5. Skip most of the mathematical proofs

**Estimated Time:** 3-4 hours (selective)

**How It's Used in SecGraph:**
- `configs/rl/dpo.yaml:134-143` - Online learning configuration
- Continuous model updates from expert audits

---

### **ADDITIONAL REFERENCES (Not Full Papers)**

---

**FAISS (Vector Search):**
- "Billion-scale similarity search with GPUs" - Johnson et al., IEEE 2019
- arXiv: https://arxiv.org/abs/1702.08734
- **Just skim** - understand it's a fast vector search library
- **How it's used:** `reason_agent/embeddings/faiss_index.py`

**Burstiness in Human Behavior:**
- "The origin of bursts and heavy tails in human dynamics" - Barabási, Nature 2005
- **Just read the abstract** - understand burstiness coefficient
- **How it's used:** `reason_agent/ingest/temporal_preprocess.py:87-120`

**Statistical Process Control (EWMA):**
- "Introduction to Statistical Quality Control" - Montgomery, textbook
- **Just understand EWMA concept** - no need to read full book
- **How it's used:** `reason_agent/monitoring/drift.py:73-106`

**Kubernetes Autoscaling:**
- Kubernetes Documentation (not a research paper)
- **Just read the HPA docs** - understand the algorithm
- **How it's used:** `reason_agent/serving/k8s/hpa.yaml`

---

## 🎯 **Reading Order Summary**

**Total Time Estimate: 6 weeks** (part-time study, ~3-4 hours/day)

### **Week 1: Foundations**
- Math prerequisites (3-5 days)
- Neural networks basics (2-3 days)
- ⭐ Paper 1: Attention is All You Need (2 days)

### **Week 2: Embeddings & Graphs**
- Paper 2: Word2Vec (1 day)
- Paper 3: BERT (1 day)
- ⭐ Paper 4: Sentence-BERT (1 day)
- Paper 5: GCN (1 day)
- ⭐ Paper 6: GraphSAGE (1.5 days)
- ⭐ Paper 7: GAT (1 day)

### **Week 3: Temporal Graphs**
- Paper 8: Temporal Networks Survey (1 day, selective)
- ⭐ Paper 9: TGAT (1.5 days)
- RL Prerequisites: Sutton & Barto Ch 1-3 (3 days)

### **Week 4: Reinforcement Learning**
- Paper 10: REINFORCE (1 day)
- ⭐ Paper 11: PPO (2 days)
- Paper 12: TRPO (optional, skim)
- Paper 13: InstructGPT/RLHF (1 day)
- ⭐ Paper 14: DPO (2 days)

### **Week 5: Advanced RL & Reasoning**
- ⭐ Paper 15: Enhanced DPO (1 day)
- Paper 16: Chain-of-Thought (1 day)
- ⭐ Paper 17: ReAct (1 day)
- ⭐ Paper 18: Process Supervision (1 day)
- Paper 19: Verifiable Rewards (1 day)
- Paper 20: Adapters (0.5 day)
- ⭐ Paper 21: LoRA (1.5 days)

### **Week 6: Production ML**
- Paper 23: Concept Drift (1 day)
- Paper 24: Covariate Shift (1 day)
- Paper 25: Online Learning (1 day, selective)
- Review & connect all concepts (2 days)

---

## 📊 **Critical Path (Minimum Reading)**

If you only have 2 weeks, read these **8 critical papers** marked with ⭐⭐⭐:

1. Attention is All You Need (Transformers)
2. Sentence-BERT (Embeddings)
3. GraphSAGE (GNNs)
4. GAT (Graph Attention)
5. PPO (Reinforcement Learning)
6. DPO (Preference Learning)
7. ReAct (Reasoning Agents)
8. LoRA (Efficient Fine-tuning)

Plus:
- Sutton & Barto RL Ch 1-3
- Process Supervision (Let's Verify Step by Step)

**Total: ~50-60 hours over 2 weeks**

---

## 💡 **Reading Tips**

1. **Don't read linearly** - Jump to key sections
2. **Focus on intuition first** - Skip proofs initially
3. **Study figures** - Often more valuable than text
4. **Implement simple versions** - Coding solidifies understanding
5. **Take notes** - Summarize each paper in 3-4 bullet points
6. **Connect concepts** - How does Paper X relate to Paper Y?

---

This roadmap gives you a clear path from fundamentals to advanced topics, with prerequisites and reading strategies for each paper!
