# Conversation Evaluation Benchmark

This repository presents a production-ready benchmark system designed to score every conversation turn against a dynamic set of facets covering linguistic quality, pragmatics, safety, and emotion. The architecture is built to be scalable to over 5000 facets without requiring a complete redesign, adhering to strict constraints on model size and solution type.

## Table of Contents

1.  [Project Goal](#project-goal)
2.  [Hard Constraints & Solution Adherence](#hard-constraints--solution-adherence)
3.  [Architecture Overview](#architecture-overview)
    * [Facet Management](#facet-management)
    * [Scoring Mechanism](#scoring-mechanism)
4.  [Setup & Installation](#setup--installation)
    * [Google Colab Setup](#google-colab-setup)
    * [Local Setup (Advanced)](#local-setup-advanced)
5.  [Data Preparation](#data-preparation)
6.  [Running the Benchmark](#running-the-benchmark)
7.  [Output Format](#output-format)
8.  [Brownie Points](#brownie-points)
9.  [Limitations & Future Improvements](#limitations--future-improvements)
10. [Deliverables Checklist](#deliverables-checklist)

---

### 1. Project Goal

The primary goal is to create a benchmark that can evaluate individual conversation turns across 300 distinct facets (scalable to 5000+). Each facet must be scored on a five-point integer scale.

### 2. Hard Constraints & Solution Adherence

1.  **No one-shot prompt solutions:**
    * **Adherence:** Our solution employs an iterative and categorized prompting strategy. Instead of feeding all 300 (or 5000+) facets in a single prompt, facets are first grouped into logical categories (e.g., 'Linguistic Quality', 'Safety', 'Emotion'). The LLM then processes turns by being prompted with relevant facets *per category*, reducing cognitive load and improving accuracy. This prevents overwhelming the model with a single, massive input.

2.  **Must use open-weight licenses ( $\le 16$B):**
    * **Adherence:** We utilize the `meta-llama/Meta-Llama-3-8B-Instruct` model from Meta, which is an open-weight model well within the 16B parameter limit. This model offers a good balance of performance and efficiency for this task.

3.  **Architecture must support $\ge 5000$ facets without redesign:**
    * **Adherence:** The core architectural decision for scalability is the **facet categorization** and **dynamic/iterative prompting**.
        * **Categorization:** Facets are pre-processed and assigned to broader categories (e.g., 'Grammar', 'Sentiment', 'Toxicity'). This structure is stored in `processed_facets.csv`.
        * **Iterative Prompting:** During evaluation, for a given conversation turn, the system iterates through these categories. For each category, a prompt is dynamically constructed, containing only the facets belonging to that specific category, along with their definitions and examples. This ensures that the prompt size remains manageable even as the total number of facets grows significantly, as the *number of facets per prompt* (i.e., within a category) remains relatively constant. Adding new facets simply involves adding them to `processed_facets.csv` and assigning them to an existing or new category.

### 3. Architecture Overview

#### Facet Management

* **`Facets Assignment.csv` (Input):** Contains the raw list of 300 facet names.
* **`processed_facets.csv` (Generated/Curated):** This crucial file extends the raw facets with:
    * `category`: A broader thematic grouping (e.g., "Linguistic Quality", "Safety").
    * `description`: A concise explanation of the facet.
    * `example_good`: An example of a conversation turn demonstrating a good score for the facet.
    * `example_bad`: An example of a conversation turn demonstrating a bad score for the facet.
    This structured information is vital for the LLM to understand and score each facet accurately.

#### Scoring Mechanism

The `run_benchmark.py` script orchestrates the evaluation:

1.  **Load Data:** Reads `processed_facets.csv` and `conversations.json`.
2.  **Initialize LLM:** Loads the chosen open-weight model (`Meta-Llama-3-8B-Instruct`) using `transformers` and `torch` for GPU-accelerated inference. `bitsandbytes` is used for 4-bit quantization to optimize memory usage on Colab's T4 GPUs.
3.  **Iterate Conversations & Turns:** Loops through each conversation and each individual turn within it.
4.  **Categorized Prompting:**
    * For each turn, it iterates through all unique facet categories defined in `processed_facets.csv`.
    * For each category, a specific prompt is constructed. This prompt includes:
        * The current conversation turn text.
        * Only the facets belonging to the current category.
        * The `description`, `example_good`, and `example_bad` for each facet to provide context to the LLM.
        * Clear instructions for the 1-5 scoring scale and the desired JSON output format.
5.  **LLM Inference:** The LLM processes the turn against the selected facets for that category, generating scores in JSON format.
6.  **Output Parsing & Aggregation:** The JSON output from the LLM is parsed, and scores for all facets across all categories are aggregated for the current turn.
7.  **Results Storage:** All scores are saved into `conversation_scores.json`.

### 4. Setup & Installation

#### Google Colab Setup

This is the recommended environment due to free GPU access and pre-installed dependencies.

1.  **Open Google Colab:** Go to [colab.research.google.com](https://colab.research.google.com/).
2.  **New Notebook:** Create a new notebook (`File > New notebook`).
3.  **Change Runtime Type:** Ensure GPU is enabled. Go to `Runtime > Change runtime type` and select `T4 GPU` (or a more powerful GPU if available, like A100 if on Colab Pro/Plus).
4.  **Clone Repository:**
    ```bash
    !git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
    %cd your-repo-name
    ```
    (Replace `your-username` and `your-repo-name` with your actual GitHub details.)
5.  **Install Dependencies:**
    ```bash
    !pip install -r requirements.txt
    ```
6.  **Generate `processed_facets.csv`:** Run the manual/assisted categorization as described in [Data Preparation](#data-preparation). You'll save this file to your cloned repository.
7.  **Upload `conversations.json`:** Place your sample conversation data in the `conversations.json` file within the cloned directory.
8.  **Run Benchmark:** Execute the `run_benchmark.py` script.

    ```bash
    !python run_benchmark.py
    ```

#### Local Setup (Advanced)

* **Prerequisites:** Python 3.9+, pip, NVIDIA GPU with CUDA drivers (for GPU acceleration).
* **Clone Repository:** `git clone https://github.com/your-username/your-repo-name.git`
* **Navigate:** `cd your-repo-name`
* **Install Dependencies:** `pip install -r requirements.txt`
* **Data Preparation:** Follow steps in [Data Preparation](#data-preparation) to create `processed_facets.csv` and `conversations.json`.
* **Run Benchmark:** `python run_benchmark.py`

### 5. Data Preparation

* **`Facets Assignment.csv`**: This is the initial input. Assuming it's a single column of facet names.
* **Generating `processed_facets.csv`**:
    This is a critical manual/semi-automated step.
    1.  Load `Facets Assignment.csv` into a Pandas DataFrame.
    2.  Add the following columns: `category`, `description`, `example_good`, `example_bad`.
    3.  **Manual/Assisted Categorization:** Go through the 300 facet names. For each, assign it to one of your defined categories (e.g., 'Linguistic Quality', 'Safety', 'Pragmatics', 'Emotion', 'Style'). It's highly recommended to have a pre-defined set of about 5-10 broad categories.
    4.  **Description & Examples:** For each facet, write a concise description. Then, provide a short example of a conversation snippet that would score high (e.g., '5') and one that would score low (e.g., '1') for that specific facet. This contextual information significantly aids the LLM in understanding the scoring criteria.
    5.  Save the resulting DataFrame as `processed_facets.csv` in your repository. This file will be loaded by `run_benchmark.py`.

    *Self-correction:* For 300 facets, this is time-consuming. You can partially automate this using a smaller, faster LLM (e.g., Gemma 2B via `ollama` or even a quick prompt to a public API) to *draft* descriptions and examples, then manually review and refine them. The key is to have this structured data available.

### 6. Running the Benchmark

Once the setup is complete and `processed_facets.csv` and `conversations.json` are prepared:

```bash
python run_benchmark.py
