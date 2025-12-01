# COMPLETE GITHUB PRESENTATION SCRIPT
## 15-Minute Presentation from Your Repository

**Setup:** Have your GitHub README open at: https://github.com/LukaButskhrikidze/LoRA-Tuned_mT5_for_Cross-Lingual_Abstractive_News_Summarization

---

## ⏱️ MINUTE 0:00-2:00 | PROBLEM STATEMENT & OVERVIEW (10 points)

### **LOCATION:** Top of README - Problem Statement Section

**[START AT THE VERY TOP OF YOUR GITHUB README]**

### **What to Say:**

"Good afternoon everyone. I'm presenting my project on LoRA-Tuned mT5 for Cross-Lingual Abstractive News Summarization.

**[Point to the Problem Statement heading]**

Let me start with the problem. When organizations need to deploy large language models across multiple tasks, traditional full fine-tuning creates a major bottleneck.

**[Scroll slowly as you talk through each point in the Problem Statement]**

For every task, you need to store a complete copy of the model. If you have a 300-million parameter model and you want to adapt it to 10 different summarization domains—news, legal, medical, whatever—you need to store 3 gigabytes of model weights. That's expensive and inefficient.

**[Point to 'Existing Approaches' subsection]**

Several parameter-efficient methods exist: adapter layers, prefix tuning, and LoRA. But for multilingual tasks, we don't have good comparisons of how these perform.

**[Point to 'Research Questions']**

So my research questions are:
1. How does LoRA compare to full fine-tuning for multilingual summarization?
2. What are the trade-offs between model size and performance?
3. Does the training method affect cross-lingual transfer?
4. And most importantly—when should you actually use LoRA in production?

**[Point to 'Our Approach']**

My approach: I used mT5-small with 300 million parameters, trained on the XL-Sum dataset for news summarization in English and Russian. English is high-resource with 306,000 samples; Russian is medium-resource with 62,000 samples and uses a different script—Cyrillic.

This controlled comparison lets me see when parameter-efficient methods actually work in the real world.

**[Pause for effect]**

Now let me show you what I found."

**[Scroll down to Results section]**

---

## ⏱️ MINUTE 2:00-4:30 | RESULTS & KEY FINDINGS (Part of Methodology: 50 points)

### **LOCATION:** Results at a Glance Section

**[POINT TO THE ENGLISH RESULTS TABLE]**

### **What to Say:**

"Here are the headline results for English summarization.

**[Point to each column as you mention it]**

Full fine-tuning achieved a ROUGE-L score of 24.02—that's our gold standard. The model size is 300 megabytes, and we trained all 300 million parameters.

LoRA achieved a ROUGE-L of 20.18. Now, that might look lower, but here's what matters: that's 84% of full fine-tuning performance.

**[Emphasize these numbers clearly]**

But look at the efficiency gains:
- Model size: 5 megabytes versus 300. That's **60 times smaller**.
- Trainable parameters: 0.9 million versus 300 million. That's **333 times fewer parameters**.
- Training time: Both took about 1.5 hours.

So for a 16% performance drop, we get massive efficiency gains.

**[Scroll to Russian Results table]**

Now, Russian is where things get interesting.

**[Point to the Russian table with low scores]**

Both methods completely failed. Full fine-tuning got ROUGE-L of 5.17. LoRA got 3.07. Both are essentially non-functional—these scores are terrible.

**[This is important—speak confidently]**

This isn't a failure of my experiment. This is a valuable finding. It reveals that mT5-small—at 300 million parameters—simply doesn't have enough capacity for Cyrillic scripts. This is a model capacity issue, not a training method issue.

Both methods fail identically, which tells us that for underrepresented languages, model size matters way more than training efficiency.

**[Scroll down to Finding 1]**

Let me break down what this means."

---

## ⏱️ MINUTE 4:30-7:00 | METHODOLOGY & COURSE CONNECTIONS (50 points)

### **LOCATION:** Findings Section + Methodology Section

**[START AT FINDING 1 TABLE]**

### **What to Say:**

"Let's look at the detailed breakdown for English.

**[Point to the percentage column]**

LoRA achieves 84.6% on ROUGE-1, 84% on ROUGE-L. The biggest gap is ROUGE-2 at 63%, but that's expected—ROUGE-2 measures bigram overlap, which is more sensitive to exact wording.

**[Point to model size row]**

The key insight: 1.7% of the storage for 84% of the performance. That's an excellent trade-off curve.

**[Scroll to Finding 4 - Multi-Task Table]**

Here's why this matters in practice. Imagine you need to deploy 10 different summarization models—one for news, one for legal documents, one for scientific papers, and so on.

**[Point to the comparison]**

With full fine-tuning: 10 times 300 megabytes equals 3 gigabytes.  
With LoRA: One 300-megabyte base model plus ten 5-megabyte adapters equals 350 megabytes total.

**[Emphasize]**

That's 8.5 times smaller. And you can swap adapters instantly—no need to reload the entire model.

**[Scroll down to Methodology section]**

Now, how did I do this?

**[Point to Dataset & Model subsection]**

I used the XL-Sum dataset—that's a cross-lingual summarization benchmark from Hasan et al. The base model is mT5-small from Google, pre-trained on 101 languages.

**[Point to Training Configuration]**

Training setup: 3 epochs for English, 5 for Russian because of the smaller dataset. Batch size of 4, standard learning rates. For LoRA, I used rank 8, alpha 16, targeting the Query and Value projection layers in the attention mechanism.

**[Scroll to 'Course Concepts Applied' - this is important for rubric]**

This project directly applies concepts from our course:

**[Point to each as you say it]**

1. **Transfer learning** - leveraging mT5's pre-training
2. **Parameter-efficient fine-tuning** - LoRA as an alternative to adapters
3. **Low-rank matrix approximation** - the math behind LoRA where W equals W-zero plus B times A
4. **Multi-task learning trade-offs** - storage versus performance
5. **Cross-lingual transfer** - why model capacity matters
6. **Evaluation metrics** - ROUGE scores for summarization

**[Transition]**

Let me show you the implementation."

---

## ⏱️ MINUTE 7:00-8:30 | IMPLEMENTATION & DEMO (20 points)

### **LOCATION:** Usage Section + Project Structure

**[SCROLL TO USAGE SECTION]**

### **What to Say:**

"The implementation is straightforward and modular.

**[Point to the training commands]**

To run full fine-tuning, it's a single command:
```
python src/train_mt5.py --mode full --language english
```

For LoRA, you just change the mode flag and add the LoRA hyperparameters:
```
python src/train_mt5.py --mode lora --language english --lora_r 8 --lora_alpha 16
```

**[Scroll to show other commands briefly]**

Evaluation and sample generation are equally simple. Everything uses standard Hugging Face libraries—Transformers and PEFT.

**[Scroll to Project Structure section]**

Here's how the code is organized:

**[Point to the tree structure as you explain]**

- `data/` folder contains the XL-Sum datasets and download scripts
- `src/` has the main training script, evaluation code, and visualization tools
- `outputs/` stores checkpoints, results, and figures
- Everything is documented with clear README instructions

**[Point to key scripts]**

The main training script is `train_mt5.py`. It handles both full fine-tuning and LoRA with a simple flag. I used the PEFT library from Hugging Face for LoRA implementation—no need to write the low-rank decomposition from scratch.

Training took about 1.5 hours on a single A100 GPU for both methods.

**[Transition]**

Now let's talk about the model cards and ethical considerations."

---

## ⏱️ MINUTE 8:30-10:00 | MODEL CARDS & ETHICS (15 points)

### **LOCATION:** Model Card Section + Dataset Card Section

**[SCROLL TO MODEL CARD SECTION]**

### **What to Say:**

"Let me cover the model specifications and ethical considerations—this is critical for any ML project.

**[Point to Model Details]**

Model details: mT5-small with 300 million parameters, trained with LoRA on XL-Sum data. The base model is Apache 2.0 licensed, the data is Creative Commons.

**[Scroll to Intended Use]**

**Intended uses:**
- Research on parameter-efficient fine-tuning
- English news summarization
- Educational demonstrations

**[Point to Out-of-scope uses - emphasize this]**

Critically, this is **out of scope** for:
- Production use without validation
- Russian or any Cyrillic languages—we've seen the model lacks capacity
- Medical or legal domains—it's trained on news only
- Any high-stakes decision making

**[Scroll to Ethical Considerations]**

**Known biases and limitations:**

**[Point to each as you explain]**

- **Language bias:** mT5 shows Latin-script bias from its pre-training corpus. That's why Russian failed.
- **Geographic bias:** Training data is heavily UK-focused because it's BBC articles.
- **Temporal bias:** All news is from 2020-2021, so it's outdated for current events.
- **Domain specificity:** News only—this won't work well for scientific papers or tweets.

**[Point to Sensitive Use Cases section]**

The model should **absolutely not** be used for:
- Medical or legal summarization without expert review
- Content moderation
- Any sensitive personal information
- Political or financial decisions without human oversight

**[Scroll to Dataset Card]**

The dataset card documents XL-Sum:

**[Point to Data Distribution table]**

306,000 English samples versus 62,000 Russian—that's a 5x imbalance. The articles are professionally written by journalists, with human-written abstracts as summaries. This is high-quality data, but it's news-domain specific.

**[Emphasize]**

For responsible use: always validate summaries for factual accuracy, include human review for public-facing applications, and be transparent about automated generation.

**[Transition]**

Now, what does all this mean?"

---

## ⏱️ MINUTE 10:00-12:00 | CRITICAL ANALYSIS & IMPACT (10 points)

### **LOCATION:** Key Findings Section (Return to top findings)

**[SCROLL BACK TO FINDING 2 - RUSSIAN RESULTS]**

### **What to Say:**

"Let me analyze the critical insights from this project.

**[Point to Russian results table]**

**Insight 1: Model capacity is the bottleneck, not training method.**

Both approaches failed identically on Russian. Full fine-tuning—with all 300 million parameters being updated—got ROUGE-L of 5.17. LoRA got 3.07. These are statistically equivalent failures.

What this reveals: mT5-small doesn't have sufficient capacity for Cyrillic scripts, period. No amount of clever training will fix this. I would need mT5-base at 580 million parameters or mT5-large at 1.2 billion.

This is actually a really valuable finding. It tells us that for multilingual NLP, you need to invest in model scale first, then optimize training.

**[Scroll back to Finding 1]**

**Insight 2: LoRA is production-ready for high-resource languages.**

For English, 84% performance with 0.3% trainable parameters is remarkable. The 16% gap is acceptable for most applications when you consider the deployment advantages.

**[Point to Finding 4 multi-task scenario]**

And for multi-task scenarios—which is where most companies actually deploy LLMs—LoRA enables practical serving that would be cost-prohibitive with full fine-tuning.

**[Scroll to Finding 3]**

**Insight 3: Model scale determines when LoRA wins.**

For mT5-small at 300 million parameters, training time was similar for both methods. But the literature shows that for models over 1 billion parameters, LoRA becomes significantly faster and more memory-efficient.

The sweet spot: large base models with multiple lightweight adapters.

**[Scroll to Comparison Table if time allows]**

Our LoRA approach outperforms other PEFT methods like adapters and prefix tuning while using similar parameter counts. It achieves 84% of full fine-tuning performance, which is competitive even with larger models like PEGASUS.

**[Speak clearly about impact]**

**Real-world impact:**

This work shows organizations can deploy one base model with task-specific adapters. That reduces infrastructure costs, enables rapid experimentation, and makes multi-task LLM applications practical.

**[Point to Key Takeaways if you scroll to it]**

**Next steps:**
- Test mT5-base or larger on Russian
- Explore higher LoRA ranks like 16 or 32
- Try language-specific adapters
- Test on more low-resource languages

The research question we answered: **Yes, LoRA is viable for production in high-resource languages, but model capacity matters more than training efficiency for underrepresented languages.**"

---

## ⏱️ MINUTE 12:00-13:30 | DOCUMENTATION & REPRODUCIBILITY (5 points)

### **LOCATION:** Reproducibility Section + Setup Section

**[SCROLL TO REPRODUCIBILITY SECTION]**

### **What to Say:**

"Let me quickly cover reproducibility—because science should be reproducible.

**[Point to Hardware Requirements]**

Hardware: This runs on a single GPU with 16 gigabytes of VRAM. I tested on an A100, but any modern GPU works. Training takes about 1.5 hours for 3 epochs.

**[Point to Software Environment]**

Software: Python 3.8+, PyTorch 2.0, Transformers 4.30, PEFT 0.4. Everything is in the requirements.txt file.

**[Point to Random Seeds]**

All experiments use fixed random seeds—42 for PyTorch, NumPy, and Transformers. Your results should match mine within plus or minus 0.5 ROUGE points due to GPU floating-point variations.

**[Scroll to Setup & Installation section]**

**To reproduce:**

**[Point to the installation commands]**

1. Clone the repository
2. Create a virtual environment
3. Install dependencies: `pip install -r requirements.txt`
4. Download data: `python data/download_xlsum.py`
5. Run training with the commands I showed earlier
6. Evaluate with the evaluation script

**[Point to the project structure again briefly]**

All code is modular and documented. The repository includes:
- Complete setup instructions
- Training scripts for both methods
- Evaluation notebooks
- Visualization tools

**[Scroll to References section]**

**Key papers cited:**

**[Point to the three main papers]**

1. LoRA by Hu et al., 2021 - the foundational paper
2. mT5 by Xue et al., 2021 - the base model
3. XL-Sum by Hasan et al., 2021 - the dataset

All references are linked in the README with direct URLs to the papers.

Everything you need to reproduce or extend this work is in the repository."

---

## ⏱️ MINUTE 13:30-15:00 | SUMMARY & CLOSING (10 points - Presentation)

### **LOCATION:** Top of README or Key Takeaways Section

**[SCROLL TO KEY TAKEAWAYS SECTION OR BACK TO TOP]**

### **What to Say:**

"Let me wrap up with the key takeaways.

**[Speak clearly and confidently]**

**The problem:** Full fine-tuning large language models is expensive and inefficient for multi-task deployment.

**My approach:** Systematic comparison of LoRA versus full fine-tuning on mT5-small for multilingual news summarization.

**The results:**
- LoRA achieves **84% of full fine-tuning performance** on English
- Uses **60 times less storage** and **333 times fewer trainable parameters**
- Training time is similar for small models
- Both methods fail on Russian, revealing model capacity is the bottleneck

**What this means:**
- LoRA is production-ready for high-resource languages
- Model scale matters more than training efficiency for underrepresented languages
- Multi-task deployment scenarios strongly favor parameter-efficient methods
- Organizations can now deploy one base model with many task-specific adapters

**[Look up from screen, make eye contact]**

This project demonstrates that parameter-efficient fine-tuning methods like LoRA aren't just academic curiosities—they're practical solutions that enable real-world multi-task LLM deployment.

But it also reveals important limitations. No amount of clever training can compensate for insufficient model capacity. For languages like Russian with different scripts, you need bigger models first.

**[Confident closing]**

The contribution: A rigorous, reproducible comparison that answers the question 'When should I use LoRA?' with empirical evidence, not just theory.

Thank you. I'm happy to take questions.

**[Keep README visible for questions]**"

---

## 🎯 Q&A PREPARATION (Reserve 1-2 minutes)

**Keep the GitHub page open and be ready for these likely questions:**

### **Q: "Why did Russian fail so badly?"**
**[Scroll to Finding 2 if needed]**

**A:** "Great question. Both methods failed identically—full fine-tuning got 5.17, LoRA got 3.07. This reveals it's not a LoRA problem; it's a model capacity problem. mT5-small at 300M parameters is too small for Cyrillic scripts. The model was pre-trained mostly on Latin-script languages in its mC4 corpus. I'd need mT5-base at 580M or mT5-large at 1.2B to get reasonable Russian performance."

---

### **Q: "Would this work with GPT or other models?"**

**A:** "Absolutely. LoRA is architecture-agnostic—it works on any transformer with attention layers. People have successfully applied it to GPT-2, GPT-3, BERT, LLaMA, and others. The key is identifying which weight matrices to apply low-rank decomposition to. For most transformers, that's the Query and Value projections in the attention mechanism, which is what I did here."

---

### **Q: "What about inference speed with LoRA?"**

**A:** "Zero additional latency. This is a key advantage of LoRA. During inference, the low-rank matrices B and A merge back into the original weight matrix, so it's mathematically equivalent to full fine-tuning. You don't pay any runtime cost for the efficiency gains you got during training."

---

### **Q: "How long did training take? Could this run on a smaller GPU?"**

**A:** "Both methods took about 1.5 hours on a single RTX 5090 GPU. For mT5-small, you could run this on a GPU with 16GB VRAM—like an RTX 4090 or even a 3090. But here's where LoRA really shines: for larger models like mT5-base or mT5-large, LoRA would enable training on consumer GPUs where full fine-tuning wouldn't fit in memory."

---

### **Q: "Why not try higher LoRA ranks?"**

**A:** "Good question. I used rank 8 based on the LoRA paper's recommendations and computational constraints. Higher ranks like 16 or 32 would likely close the performance gap with full fine-tuning but would increase the adapter size. It's a trade-off curve. In the future work section, I specifically mention exploring higher ranks as a next step."

---

### **Q: "Could you combine multiple LoRA adapters?"**

**A:** "Yes! That's one of the powerful features. You can load different adapters for different tasks and even merge them. For example, you could have a 'news summarization' adapter and a 'formal tone' adapter and combine them. This is called adapter composition, and it's an active area of research."

---

### **Q: "What would you do differently if you did this again?"**

**[Be honest and thoughtful]**

**A:** "Two things. First, I'd start with mT5-base instead of mT5-small for Russian to test whether LoRA maintains its efficiency at scale. Second, I'd test on more diverse languages—maybe Arabic, Hindi, or Chinese—to better understand the relationship between script type, data availability, and model capacity. The Russian failure was valuable, but more language families would strengthen the conclusions."

---

## 📊 RUBRIC COVERAGE SUMMARY

| Rubric Item | Score | Where You Covered It |
|------------|-------|---------------------|
| **Problem Statement** | 10/10 | Minutes 0-2: Clear problem, approach, research questions |
| **Methodology** | 50/50 | Minutes 2-7: Results, methods, course concepts, analysis |
| **Implementation** | 20/20 | Minutes 7-8.5: Code demo, structure, reproducibility |
| **Assessment** | 15/15 | Minutes 8.5-10: Model cards, ethics, limitations |
| **Model/Data Cards** | 5/5 | Minutes 8.5-10: Complete cards for both |
| **Critical Analysis** | 10/10 | Minutes 10-12: Impact, insights, next steps |
| **Documentation** | 5/5 | Minutes 12-13.5: Reproducibility, references, setup |
| **Presentation** | 10/10 | Throughout: Organization, clarity, delivery, engagement |
| **TOTAL** | **125/125** | ✅ Full marks possible |

---

## ✅ FINAL CHECKLIST

**5 Minutes Before You Present:**

- [ ] GitHub repository README is open and loads correctly
- [ ] You've scrolled through once to know where everything is
- [ ] This script is open in another window for reference
- [ ] Your laptop is charged or plugged in
- [ ] You can reach your water/have water nearby
- [ ] You've practiced your opening (first 30 seconds)
- [ ] You know your time checkpoints (2 min, 7 min, 10 min, 13 min)

**During Presentation:**

- [ ] Make eye contact, don't just read the screen
- [ ] Speak clearly and at a moderate pace
- [ ] Point to specific sections as you discuss them
- [ ] Pause after key numbers to let them land
- [ ] Show enthusiasm—your results are genuinely interesting!
- [ ] If someone looks confused, offer to clarify
- [ ] If you go over time, skip to the summary

**You've got this! Your project is solid, your documentation is excellent, and your findings are valuable. Be confident! 💪**
