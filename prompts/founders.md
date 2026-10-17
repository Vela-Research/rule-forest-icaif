# Founder prompts

Every prompt the founder experiments used, copied verbatim from the code that ran them. The LLM
only ever sees the anonymised founder summaries in `data/founders/profiles/`. Labels appear in
one prompt only, the few-shot prompting baseline, which shows 2 successful and 2 failed example
founders.

## 1. Writing the questions

The model is GPT-4o-mini at temperature 0. The 497 question-generation founders are split into 25 sets of up to
20 by `split_into_sets` (reproduced in `experiments/questions.py`), with 10 successful and 10
failed founders where possible, though the prompt never says which is which. The first set uses the
first template and every later set the second, each asking for 10 questions; 240 came back in all.
Exact duplicates are then removed (132 left). Then, for each pair of questions whose
all-MiniLM-L6-v2 sentence embeddings have cosine similarity above 0.8, the later one is dropped
(62 left). The lists are in `data/founders/question_chain/`.

Copied from the research code that ran the experiments (files `prompts.py` and `anonymised_generation.py`), which is not part of this release.

```python
def get_anonymised_question_prompt(num_questions: int, founder_summaries: str) -> tuple[str, str]:
    system_message = (
        "You are a VC analyst specializing in evaluating founders. "
        "Your task is to generate high-quality, objective YES/NO questions based on provided founder summaries. "
        "These questions should evaluate factors such as education, work history, leadership, and relevant experience."
    )

    user_prompt = f"""Generate {num_questions} YES/NO questions that can be answered using the anonymised founder summaries below.
Each question should be objective and specific, relating to observable traits in the summaries (e.g., education, roles, founding experience).
IMPORTANT FORMATTING REQUIREMENTS:
1. Return ONLY the questions, one per line
2. Do NOT include any introductory text or explanations
3. Do NOT include any numbering or bullet points
4. Each line should contain exactly one question that can be answered with Yes or No

Example format (DO NOT USE THIS QUESTION):
Has the founder previously held a senior leadership role?

Founder Summaries:

{founder_summaries}"""
    
    return system_message, user_prompt


def get_anonymised_updated_question_prompt(num_questions: int, founder_summaries_str: str) -> tuple[str, str]:
    system_msg = (
        "You are a VC analyst specializing in evaluating startup founders. "
        "Your task is to generate objective YES/NO questions based on the anonymised founder summaries below. "
        "These questions should focus on traits like education, leadership, experience, and founding history. "
        "Each question must be answerable with 'Yes' or 'No' based solely on the summaries provided."
    )

    user_prompt = f"""We are now evaluating a new batch of anonymised founders. Please generate {num_questions} new objective YES/NO questions inspired by this batch.
    Each question must be specific, verifiable from the data, and helpful for distinguishing successful from unsuccessful founders.

    IMPORTANT FORMATTING REQUIREMENTS:
    1. Return ONLY the questions, one per line.
    2. Do NOT include any introductory text, explanations, or conversational filler.
    3. Do NOT include any numbering or bullet points.
    4. Each line must contain exactly one question that can be answered with 'Yes' or 'No'.

    Example format (DO NOT USE THESE QUESTIONS):
    Has the founder held a leadership role in a technology company?
    Did the founder complete postgraduate education?

    New Founder Summaries:\n\n{founder_summaries_str}"""

    return system_msg, user_prompt
```

## 2. Answering one question for each founder

The model is GPT-4o-mini at temperature 0, with one question and 20 founders per call. A reply that cannot be parsed
leaves those cells blank, and every analysis counts a blank as NO.

Copied from the research code (file `testing.py`), which is not part of this release.

```python
def test_question_anonymised(llm_client, question, founders_df):
    """Test a single question against anonymised founder summaries."""
    system_message = (
        "You are a VC analyst evaluating founders. "
        "Your task is to decide whether the provided question applies to each founder based on their anonymised summary. "
        "Return your answer as 'Yes' or 'No' for each founder in JSON format."
    )

    # Remove the 'startup_success' column just in case
    founders_df = founders_df.drop(columns=['startup_success'], errors='ignore')

    founder_summaries = "\n\n".join(
        founders_df.apply(lambda row: f"Founder {row['founder_uuid']}:\n{row['summary']}", axis=1)
    )

    user_prompt = f"""
    Given the following anonymised founder summaries, answer the question concisely as 'Yes' or 'No'.

    **Question:** {question}

    **Founder Summaries:**
    {founder_summaries}

    **Expected Output Format (JSON List):**
    ```json
    [
        {{"Founder ID": "uuid1", "Answer": "Yes"}},
        {{"Founder ID": "uuid2", "Answer": "No"}}
    ]
    """

    try:
        response = llm_client.send_prompt(system_message, user_prompt)

        # Print and log the raw response for debugging
        print(f"Raw response:\n{response}\n")
        logging.info(f"Raw response: {response}")

        cleaned_response = response.replace("```json", "").replace("```", "").strip()
        predictions = json.loads(cleaned_response)

        if isinstance(predictions, list):
            logging.info("Successfully parsed JSON response.")
            return predictions
        else:
            logging.error("Response was not a list.")
            return []

    except json.JSONDecodeError:
        logging.error(f"Failed to parse response as JSON. Raw response: {response}")
        print(f"Failed to parse response as JSON. Raw response:\n{response}")
        return []
```

## 3. Direct prompting baselines (Table 2)

The LLM predicts success directly, 20 founders per call, temperature 0. Zero-shot uses the first
prompt. Few-shot adds 2 successful and 2 failed founders drawn at random from the
question-generation founders for each call. Table 2 reports o3 few-shot and GPT-4o-mini
zero-shot; `results/table2.json` lists all six runs.

Copied from the research code (file `vanilla_testing.py`), which is not part of this release.

```python
def test_vanilla_predictions_anonymised(llm_client, founders_df):
    """
    Generate vanilla GPT predictions for founder success based on anonymised summaries.

    Returns a list of dicts:
    [
        {"Founder ID": "<uuid1>", "Prediction": "Successful"},
        {"Founder ID": "<uuid2>", "Prediction": "Unsuccessful"},
        ...
    ]
    """

    # 1. Build the system message
    system_message = (
        "You are an expert in venture capital asked to predict, as best you can, "
        "which founders will ultimately build companies that exit or IPO at a value "
        "greater than $500 M.  Current funding information is unavailable to you; "
        "assume each company is still early-stage with no meaningful difference in "
        "capital raised.  Base your judgement **only** on the information provided "
        "in each founder’s summary (background, prior exits, domain knowledge, etc.)."
    )

    system_message += (
        "\n\nRules:\n"
        "1. Do NOT use the internet, external databases, or real-world knowledge about these companies.\n"
        "2. Do NOT use – or attempt to infer – the amount of funding a company has raised.\n"
        "3. Use ONLY the anonymised information provided below.\n"
    )

    # 2. Format summaries
    summaries = []
    for _, row in founders_df.iterrows():
        summaries.append(f"Founder {row['founder_uuid']}: {row['summary']}")
    founder_summaries = "\n\n".join(summaries)

    # 3. Create user prompt
    user_prompt = f"""
    Based on the summaries below, classify each founder as "Successful" or "Unsuccessful".

    **Founder Summaries:**
    {founder_summaries}

    **Return ONLY your answer in JSON array form like:**
    ```json
    [
        {{"Founder ID": "uuid1", "Prediction": "Successful"}},
        {{"Founder ID": "uuid2", "Prediction": "Unsuccessful"}}
    ]
    ```

    **Reminder:** In this dataset, only ~2% of founders are actually Successful. Your predictions should reflect this rarity. If you’re unsure, lean toward classifying as "Unsuccessful".
    """

    # 4. Query LLM and parse response
    try:
        raw = llm_client.send_prompt(system_message, user_prompt)
        logging.debug(f"LLM raw response: {raw}")
        cleaned = raw.replace("```json", "").replace("```", "").strip()
        preds = json.loads(cleaned)
        if isinstance(preds, list):
            return preds
        else:
            logging.error("Unexpected JSON structure from LLM")
            return []
    except json.JSONDecodeError:
        logging.error(f"JSON decode failed on LLM response: {raw}")
        return []
    except Exception as e:
        logging.error(f"Error calling LLM: {e}")
        return []


def test_vanilla_predictions_few_shot_anonymised(llm_client, founders_df, examples_df, n_success=2, n_unsuccess=2, seed=None):
    """
    Generate GPT predictions using anonymised summaries and few-shot examples.

    Parameters:
        llm_client: LLM client to send prompt.
        founders_df: Unlabeled test founders (with 'founder_uuid' and summary fields).
        examples_df: Labeled example founders (must include 'startup_success').
        n_success: Number of successful examples to sample.
        n_unsuccess: Number of unsuccessful examples to sample.
        seed: Optional random seed.

    Returns:
        A list of dicts: [{"Founder ID": "...", "Prediction": "Successful"}, ...]
    """
    # Sample few-shot examples
    rng = examples_df.sample(frac=1, random_state=seed) if seed else examples_df.sample(frac=1)
    success_examples = rng[rng["startup_success"] == 1].head(n_success)
    unsuccess_examples = rng[rng["startup_success"] == 0].head(n_unsuccess)

    # Format few-shot examples (anonymised)
    def format_example(row):
        return row.get("summary", "N/A")

    success_block = "\n\n".join(format_example(row) for _, row in success_examples.iterrows())
    unsuccess_block = "\n\n".join(format_example(row) for _, row in unsuccess_examples.iterrows())

    # Format test summaries
    test_summaries = []
    for _, row in founders_df.iterrows():
        test_summaries.append(f"Founder {row['founder_uuid']}: {row.get('summary', 'N/A')}")
    test_block = "\n\n".join(test_summaries)

    # Prompt construction
    system_message = (
        "You are an expert in venture capital asked to predict, as best you can, "
        "which founders will ultimately build companies that exit or IPO at a value "
        "greater than $500 M. You are provided with anonymised summaries only—no funding data or external knowledge."
        "\n\nRules:\n"
        "1. Do NOT use the internet, external databases, or real-world knowledge about these companies.\n"
        "2. Do NOT use – or attempt to infer – the amount of funding a company has raised.\n"
        "3. Use ONLY the anonymised information provided below.\n"
    )

    user_prompt = f"""
    These are successful founders:

    {success_block}

    These are unsuccessful founders:

    {unsuccess_block}

    Now, classify the following new founders:

    {test_block}

    Return ONLY your answer in valid JSON array format, with NO explanations, NO comments, and NO text before or after.
    Your output should look exactly like this:
    [
    {{"Founder ID": "uuid1", "Prediction": "Successful"}},
    {{"Founder ID": "uuid2", "Prediction": "Unsuccessful"}}
    ]

    **Reminder:** In this dataset, only ~2% of founders are actually Successful. Your predictions should reflect this rarity. If you’re unsure, lean toward classifying as "Unsuccessful".
    """

    # Send prompt and parse response
    try:
        raw = llm_client.send_prompt(system_message, user_prompt)
        logging.debug(f"LLM raw response: {raw}")
        cleaned = raw.replace("```json", "").replace("```", "").strip()
        preds = json.loads(cleaned)
        if isinstance(preds, list):
            return preds
        else:
            logging.error("Unexpected JSON structure from LLM")
            return []
    except json.JSONDecodeError:
        logging.error(f"JSON decode failed on LLM response: {raw}")
        return []
    except Exception as e:
        logging.error(f"Error calling LLM: {e}")
        return []
```
