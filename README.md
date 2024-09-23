# **CHESS_internship**

## **Project Overview**
This repository contains an implementation of the CHESS method for text-to-SQL generation, alongside additional improvements to enhance performance and usability.

---
## **Installation**

To get started, clone this repository and install the required dependencies by running the following commands:

```bash
git clone https://github.com/ConvergenceAI/CHESS_internship
cd CHESS_internship
pip install -r requirements.txt
```

**Data**

This project works with the BIRD dev set data, which you can download using the following links:
- [BIRD Dev Set](https://bird-bench.oss-cn-beijing.aliyuncs.com/dev.zip)
- [Subsampled Dev Set (CHESS Paper)](https://github.com/ShayanTalaei/CHESS/blob/main/data/dev/sub_sampled_bird_dev_set.json)
  
Unzip the downloaded data into the data directory.

---

## **Directory structure**

├── data                  # Download the datasets here

├── files                 # files of the presentation of the internship and the detailed error analysis    

├── notebooks             # Jupyter notebooks for testing modules and analysis

├── src                   # Source code of the project

│   ├── embedding         # Embedding management and model config

│   ├── llm               # LLM interface and model config

│   ├── pipeline          # CHESS pipeline modules

│   ├── preprocess        # Preprocessing logic for databases

│   ├── prompts           # Prompt templates for LLM calls

│   ├── main.py           # Main pipeline file to run the full process

│   └── pipeline_configs  # Config files for each module (LLMs, embeddings, repetitions, etc.)

├── utils                 # Utility functions for database processing

└── README.md             # This README file

---

## **How to use**

1- create a .env file containing : 

**HF_TOKEN**  # Huggingface token

**OPENAI_API_KEY** # OpenAI API key 

**DB_ROOT_PATH** # The root directory for databases (dev_databases directory downloaded from the BIRD dev set)

**PROMPT_ROOT_PATH** # path to src/prompts

**OUTPUT_DIR_PATH** # directory where do you want to store the run results 

**STATS_PATH** # the path to the result directory that you want to make stats for it

**RUN_PATH** # The json data that you want to run the pipeline on it

2- **Data Preprocessing:** Preprocess the BIRD dataset by running the preprocessing notebook:
```bash
jupyter notebook notebooks/preprocess.ipynb
```

3- **Running the Pipeline:** After preprocessing, and configuring the pipeline run the entire pipeline on the data by executing the following:

```bash
python src/main.py
```

---

## **Added Value**

Here are the key improvements and added functionalities beyond the standard CHESS method:

- **Entity Retrieval**: Latency of the entity retrieval module has been greatly reduced while maintaining high accuracy.
- **Schema Selection Fusion**: Integrated table and column selection for optimized query performance.
- **Detailed Error Analysis**: In-depth analysis of errors including vague questions, incorrect predicted SQL, and golden SQL issues.
- **Cost and Latency Optimization**: Improved cost-effectiveness and reduced latency in execution.

  
  



