# DecodeLabs Industrial Training Kit — Batch 2026
## Artificial Intelligence Track

Internship-style project series under DecodeLabs, progressing from deterministic
rule-based logic to basic supervised machine learning. Each project unlocks the
next, per the official Qualification Criteria.

---

## 📋 Project Index

| # | Project | Status | Stack |
|---|---------|--------|-------|
| 1 | Rule-Based AI Chatbot (ARIA) 
| 2 | Preference-Based Recommendation System 
| 3 | Basic Classification Model 
**
## Project 1: Rule-Based AI Chatbot — "ARIA" 🤖
**
### Goal
Create a simple rule-based chatbot that responds to predefined user inputs.

### Key Requirements
- [x] Handle greetings and exit commands
- [x] Use if-else / dictionary logic for responses
- [x] Run in a continuous loop

### Key Skills
Control flow, decision-making logic, basic AI concepts.

### Architecture
Follows the **IPO Model**: `Input → Process → Output`
- **Input & Sanitization** — lowercase, whitespace trim, punctuation strip
- **Process** — knowledge base lookup (dictionary in Python / struct array in C) with fallback
- **Output** — response printed via continuous `while` loop, broken by exit command

### Files
- `rule_based_chatbot.jsx` — React implementation with typing animation, quick-intent buttons, session status
- `aria_chatbot.c` — C implementation, compiled clean with `-Wall -Wextra -Werror`

### Run it
```bash
# C version
gcc -Wall -Wextra -o aria_chatbot aria_chatbot.c
./aria_chatbot
```

### Sample interaction
You: hello

ARIA: Hello! I'm ARIA, your rule-based AI assistant. How can I help?   







---

**## Project 2: Preference-Based Recommendation System 🎯**

### Goal
Create a simple recommendation system based on user preferences.

### Key Requirements
- [ ] Take user input (choices or interests)
- [ ] Match preferences using logic or similarity
- [ ] Display recommended items

### Key Skills
Logic building, pattern matching, recommendation concepts.

### Planned Approach
- Build a small static dataset (e.g. movies/books/courses tagged by genre/category)
- Take user's preferred tags/genres as input
- Score each item using a simple similarity measure (e.g. tag overlap count or
  Jaccard similarity) — no ML model required at this stage, this is *logic-based*
  recommendation, consistent with Project 1's rule-based philosophy
- Sort and display top-N matches

### Stack
Python (`difflib` or manual set-overlap logic; optionally `pandas` if the dataset
grows beyond a small in-memory list)

### File (planned)
- `recommender.py`

---

## Project 3: Basic Classification Model 📊

### Goal
Build a basic classification model using a small dataset.

### Key Requirements
- [ ] Load and understand a dataset
- [ ] Split data into training and testing sets
- [ ] Apply a simple classification algorithm

### Key Skills
Data handling, supervised learning basics, model training.

### Planned Approach
- Load a small, well-known dataset (e.g. Iris, or a small CSV of labeled examples)
- Explore it: check shape, class balance, feature ranges
- Split into train/test sets (commonly 80/20) using `train_test_split`
- Train a simple classifier (e.g. `KNeighborsClassifier` or `DecisionTreeClassifier`)
- Evaluate using accuracy score / confusion matrix

### Stack
Python — `pandas`, `scikit-learn`

### File (planned)
- `classifier.py`

### Note on continuity
This is the first project in the track that uses **probabilistic / learned**
logic rather than hard-coded rules — the conceptual shift from "Project 1:
Discrete Mapping (Exact Match)" to "Project 2/3: Pattern Matching & Learned
Generalization" described in the training material's "Two Minds of AI" framing.

---

You: exit

ARIA: Goodbye! Keep building...
