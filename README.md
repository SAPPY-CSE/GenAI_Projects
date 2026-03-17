# 🤖 Generative AI Projects — Google Gemini + LangChain

A progressive series of four hands-on Generative AI projects built with **Google Gemini 2.5 Flash** and **LangChain**, designed to teach core GenAI engineering concepts from prompt design all the way to structured data querying. Each notebook is self-contained and runnable on **Google Colab**.

---

## 📁 Project Index

| # | Notebook | Topic | Difficulty |
|---|----------|-------|------------|
| 01 | `GenAI_Project_01.ipynb` | System Prompting & Response Post-Processing | 🟢 Easy |
| 02 | `GenAI_Project_02.ipynb` | Multi-Tool RAG Student Assistant | 🟡 Intermediate |
| 03 | `GenAI_Project_03.ipynb` | Hybrid Memory Management (Context Rot Fix) | 🟡 Intermediate |
| 04 | `GenAI_Project_04.ipynb` | Natural Language to SQL (Text-to-SQL Pipeline) | 🟠 Advanced |

---

## 🔑 Prerequisites

### API Keys Required

| Project | Key | Where to Get |
|---------|-----|--------------|
| All Projects | `GOOGLE_API_KEY` or `GEMINI_API_KEY` | [Google AI Studio](https://aistudio.google.com/) |
| Project 02 only | `TAVILY_API_KEY` | [Tavily](https://tavily.com/) |

> All notebooks support loading keys from **environment variables** or **Google Colab Secrets** automatically.

### Python Dependencies (auto-installed in each notebook)
- `google-genai`
- `langchain`, `langchain-core`, `langchain-google-genai`, `langchain-community`
- `langchain-chroma`, `langchain-huggingface`, `sentence-transformers`, `chromadb`
- `pypdf`, `tavily-python`, `sqlalchemy`

---

## 📓 Project Details

---

### 📌 Project 01 — System Prompting & Response Post-Processing
**File:** `GenAI_Project_01.ipynb`
**Difficulty:** 🟢 Easy
**Model:** `gemini-2.5-flash` via `google-genai` SDK

#### 🎯 What This Project Teaches
The single most impactful thing you can do to improve LLM output — **write a strong system prompt** — and why you should add a **post-processing layer** as a safety net.

#### 🧩 Concept Breakdown

**The Problem — Bad Chatbot (No System Prompt)**
- Sends raw user questions with zero instructions to the model
- Results in wildly inconsistent response lengths, tones, and formats
- Word counts vary from 20 to 300+ words for the same type of question

**The Fix — Good Chatbot (Strong System Prompt)**
A carefully engineered `STRONG_SYSTEM_PROMPT` enforces:
- Responses capped at **3–5 sentences**
- A strict **3-part format**: direct answer → supporting details → practical example
- No filler openers like *"Great question!"* or *"Certainly!"*
- Consistent tone regardless of topic

**The Extra Layer — Post-Processing**
Even with a perfect system prompt, models occasionally drift. The `post_process()` function acts as a rule-enforcer on top:
- Strips filler openers using regex patterns
- Enforces a hard **120-word limit** (trims at sentence boundaries)
- Ensures responses end with proper punctuation

#### 🔄 Pipeline Flow
```
User Question
     │
     ▼
[Bad Chatbot]  ──── No rules ──────────────────────► Inconsistent Response
     │
[Good Chatbot] ──── Strong System Prompt ──────────► Consistent Response
     │
[Good + Post-Processing] ──── System Prompt + Regex ► Cleaned & Capped Response
```

#### 📊 What You'll See
A **side-by-side comparison** output across 3 test questions showing word counts and response quality before vs. after the system prompt. The notebook ends with an **interactive chat loop** using the improved chatbot.

#### 🛠️ Key Functions
| Function | Purpose |
|----------|---------|
| `bad_chatbot(question)` | Raw Gemini call, no instructions |
| `good_chatbot(question)` | Gemini call with `STRONG_SYSTEM_PROMPT` |
| `post_process(response, max_words)` | Cleans filler, trims length, fixes punctuation |
| `good_chatbot_with_postprocessing(question)` | Combined best-practice pipeline |

---

### 📌 Project 02 — Multi-Tool RAG Student Assistant
**File:** `GenAI_Project_02.ipynb`
**Difficulty:** 🟡 Intermediate
**Model:** `gemini-2.5-flash` via LangChain
**Extra APIs:** Tavily (web search), Open-Meteo (weather, free & keyless)

#### 🎯 What This Project Teaches
How to build an **intelligent query router** that decides which tool to use — live weather, web search, PDF-based RAG, or direct LLM knowledge — based solely on the user's question.

#### 🧩 Architecture Overview

The assistant has **4 tools** and an LLM-powered **router** that classifies every query before sending it to the right handler.

```
User Query
    │
    ▼
[LLM Router] ── classifies into one of four routes ──►
    │
    ├── weather    ──► Open-Meteo API (free, no key) ──► Live weather data
    ├── web_search ──► Tavily Search + LLM synthesis ──► Current web info
    ├── rag        ──► ChromaDB + HuggingFace Embeds  ──► PDF document answers
    └── direct     ──► LLM general knowledge          ──► Concept explanations
```

#### 🔧 Setup Steps (in order inside the notebook)
1. **Install dependencies** and load API keys (Google + Tavily)
2. **Upload PDFs** → loaded with `PyPDFLoader` → chunked with `RecursiveCharacterTextSplitter` (chunk size: 1000, overlap: 200) → embedded with `sentence-transformers/all-MiniLM-L6-v2` → stored in **ChromaDB**
3. **Define 4 tools**: weather, web search, RAG retriever, direct LLM
4. **Build the router**: LLM classifies query → fallback keyword logic if classification fails
5. **Run test queries** covering all 4 routes

#### 🗂️ The Four Handlers

| Route | Trigger Keywords | Handler Logic |
|-------|-----------------|---------------|
| `weather` | weather, temperature, rain, forecast, umbrella | Binds `get_current_weather` tool to LLM, calls Open-Meteo geocoding + forecast API |
| `web_search` | latest, news, recent, today, update, 2025 | Binds `web_search` tool (Tavily), parses JSON results, asks LLM to synthesize with source citations |
| `rag` | document, pdf, uploaded, notes, summarize | Retrieves top-4 chunks from ChromaDB, injects as context, LLM answers grounded only in those chunks |
| `direct` | everything else | Straight LLM call with a helpful assistant system prompt |

#### 📄 RAG Vector Store Details
- **Embedding model:** `sentence-transformers/all-MiniLM-L6-v2` (runs on CPU)
- **Vector store:** ChromaDB (persisted to `/content/student_assistant_chroma`)
- **Retrieval:** top-4 most similar chunks per query
- **Source citation:** each retrieved chunk is labeled with filename and page number

#### 🛠️ Key Functions
| Function | Purpose |
|----------|---------|
| `classify_route(query)` | LLM + keyword fallback router |
| `handle_weather(query)` | Tool-bound LLM → Open-Meteo live call |
| `handle_web_search(query)` | Tool-bound LLM → Tavily → synthesis |
| `handle_rag(query)` | ChromaDB retrieval → grounded LLM answer |
| `handle_direct(query)` | Pure LLM knowledge answer |
| `student_assistant(query)` | Main entry point — routes and dispatches |

---

### 📌 Project 03 — Hybrid Memory Management (Context Rot Fix)
**File:** `GenAI_Project_03.ipynb`
**Difficulty:** 🟡 Intermediate
**Model:** `gemini-2.5-flash` via LangChain

#### 🎯 What This Project Teaches
Why LLM conversations **degrade over time** (context rot) and how to fix it with a **hybrid memory system** that combines a sliding window of recent messages with a compressed rolling summary of older ones.

#### 🧩 The Problem — Context Rot

In a plain chat system, the full message history is sent to the LLM on every turn. As conversations grow longer:
- The context window fills up and early messages get pushed out
- The LLM "forgets" facts mentioned at the start (e.g., user's name, preferences)
- Token costs balloon unnecessarily
- Response quality degrades for anything requiring long-term recall

The notebook **demonstrates this live**: a user shares their name and course early in the conversation, asks several follow-up questions, and then asks *"Do you remember my name and what I'm studying?"* — the plain chatbot fails to recall.

#### 🧩 The Solution — Hybrid Memory

```
Every Turn:
┌──────────────────────────────────────────────────────┐
│  Messages sent to LLM:                               │
│  1. System instruction                               │
│  2. [SUMMARY] Compressed older conversation          │
│  3. [WINDOW]  Last N raw message pairs (recent)      │
│  4. Current user message                             │
└──────────────────────────────────────────────────────┘

When window overflows:
  Old messages ──► summarize_history() ──► rolling summary (updated)
  Recent window ──► keeps last WINDOW_SIZE messages raw
```

#### ⚙️ Configuration
| Parameter | Default | Purpose |
|-----------|---------|---------|
| `WINDOW_SIZE` | `4` | Number of most recent messages to keep as raw text |
| `SLEEP_BETWEEN_CALLS` | `15s` | Rate limit protection for Gemini free tier (5 req/min) |

#### 🔄 Memory Compression (`summarize_history`)
When the recent window exceeds `WINDOW_SIZE`, the overflow messages are passed to the LLM with a dedicated **memory manager prompt** that:
- Preserves all important facts (names, preferences, decisions)
- Removes filler exchanges
- Outputs a concise summary under 200 words
- Merges with any existing summary (incremental, not from scratch)

#### 📊 Side-by-Side Comparison
The notebook runs the **exact same conversation** through both plain chat and hybrid memory, then prints a comparison showing:
- Plain chat: N raw messages growing unbounded, early context lost
- Hybrid: 4 raw messages + compact summary, early context preserved

#### 🛠️ Key Functions
| Function | Purpose |
|----------|---------|
| `plain_chat(history, user_text, system)` | No memory management — raw unbounded history |
| `summarize_history(old_messages, existing_summary)` | LLM-based compression of old messages |
| `hybrid_chat(recent_history, running_summary, user_text, system)` | Sliding window + rolling summary approach |

---

### 📌 Project 04 — Natural Language to SQL (Text-to-SQL Pipeline)
**File:** `GenAI_Project_04.ipynb`
**Difficulty:** 🟠 Advanced
**Model:** `gemini-2.5-flash` via LangChain
**Database:** SQLite (3 tables, 8 students, 24 score rows, 16 attendance rows)

#### 🎯 What This Project Teaches
Why **RAG fails on structured/tabular data** and how **Text-to-SQL** solves it by letting the LLM generate precise SQL queries that execute against a real database — delivering exact, aggregated, filterable answers instead of guesses.

#### 🗃️ Database Schema

```sql
students       (student_id, name, department, year)
scores         (score_id, student_id, subject, marks)
attendance     (attendance_id, student_id, subject, attended, total)
```

8 students across 3 departments: Computer Science, Electronics, Mathematics.

#### 🧩 Why RAG Fails on Structured Data

The notebook first **demonstrates the RAG failure mode**:
- Converts database rows into text chunks (simulating a vector store)
- Asks: *"What is the average Mathematics score of Computer Science students?"*
- RAG retrieves text chunks and **guesses** the answer
- The correct SQL answer (`AVG()` + `JOIN` + `WHERE`) is then shown to highlight the gap

RAG cannot:
- Perform exact aggregations (AVG, SUM, COUNT)
- Filter by precise values reliably
- Join across multiple tables

#### 🔄 Text-to-SQL Pipeline (4 Steps)

```
Natural Language Question
          │
          ▼
Step 1: LangChain create_sql_query_chain
        LLM reads schema + question → generates SQL
          │
          ▼
Step 2: clean_generated_sql()
        Strips markdown fences (```sql), "SQLQuery:" prefixes, extra whitespace
          │
          ▼
Step 3: SQLAlchemy executes SQL on SQLite DB
        Returns rows + column names
          │
          ▼
Step 4: LLM interprets raw results
        Converts table output → clear natural language answer (1–3 sentences)
```

#### 📋 Sample Questions Answered
- *"Who scored the highest marks in Mathematics?"*
- *"What is the average score of Computer Science students across all subjects?"*
- *"Which students have attendance below 70% in any subject?"*
- *"List all students in year 3 along with their department."*
- *"Which subject has the highest average marks overall?"*
- *"How many students are there in each department?"*

#### ⚠️ Important: Rate Limiting
A `time.sleep(15)` is placed between each question in the demo loop to respect the **Gemini free tier limit of 5 requests per minute**. Remove or reduce this if you are on a paid plan.

#### 🛠️ Key Functions
| Function | Purpose |
|----------|---------|
| `clean_generated_sql(raw)` | Strips all LLM formatting artifacts from generated SQL |
| `run_text_to_sql(question)` | Full 4-step pipeline: generate → clean → execute → interpret |
| `print_result(result)` | Pretty-prints question, generated SQL, raw DB result, and final answer |

---

## 🚀 Getting Started

### Running on Google Colab (Recommended)

1. Open any notebook in [Google Colab](https://colab.research.google.com/)
2. Go to **Tools → Secrets** and add your API keys:
   - `GOOGLE_API_KEY` (required for all projects)
   - `TAVILY_API_KEY` (required for Project 02 only)
3. Run all cells in order — `%pip install` cells handle all dependencies automatically

### Running Locally

```bash
# Clone the repository
git clone <your-repo-url>
cd <repo-folder>

# Set environment variables
export GOOGLE_API_KEY="your_google_api_key"
export TAVILY_API_KEY="your_tavily_api_key"   # Project 02 only

# Install dependencies (example for Project 01)
pip install google-genai

# Launch Jupyter
jupyter notebook
```

---

## 🧠 Learning Path

These projects are designed to be followed **in order**. Each one builds on intuitions from the previous:

```
Project 01 ──► Understand prompt engineering fundamentals
    │
    ▼
Project 02 ──► Combine multiple tools + build a router
    │
    ▼
Project 03 ──► Solve the memory/context problem in long conversations
    │
    ▼
Project 04 ──► Query structured data reliably with Text-to-SQL
```

---

## 📌 Tech Stack

| Tool | Used In | Purpose |
|------|---------|---------|
| Google Gemini 2.5 Flash | All | Core LLM for generation, routing, summarization |
| `google-genai` SDK | Project 01 | Direct Gemini API calls |
| LangChain | Projects 02–04 | LLM orchestration, tool binding, chains |
| ChromaDB | Project 02 | Vector store for RAG |
| HuggingFace Sentence Transformers | Project 02 | Text embeddings (`all-MiniLM-L6-v2`) |
| Tavily | Project 02 | Real-time web search |
| Open-Meteo | Project 02 | Free live weather API (no key needed) |
| SQLite + SQLAlchemy | Project 04 | Structured data storage and querying |

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).
