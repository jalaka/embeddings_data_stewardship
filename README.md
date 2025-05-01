# Project: Domain-Specific Classifier for Predicting Webpage Relevance

This project involves fine-tuning a sentence transformer model (BAAI/bge-large-en-v1.5) for relevance categorization and comparing its performance against the base model.

## Dependencies

The following Python libraries are required to run the notebooks:

*   `FlagEmbedding[finetune]`
*   `deepspeed`
*   `flash-attn`
*   `torch`
*   `sentence-transformers`
*   `scikit-learn`
*   `pandas`
*   `numpy`
*   `matplotlib`
*   `pathvalidate`
*   `dbrepo` (for fetching data from the TU Wien repository)
*   `requests` (for uploading the model)
*   `ipython`
*   `tqdm`
*   `boto3` (if using AWS S3 for Common Crawl data, optional)
*   `findspark` & `pyspark` (for data acquisition notebooks)
*   `warcio` (for data acquisition notebooks)

You can install most dependencies using pip:

```bash
pip install -U FlagEmbedding[finetune] deepspeed flash-attn --no-build-isolation torch sentence-transformers scikit-learn pandas numpy matplotlib pathvalidate dbrepo requests ipython tqdm boto3 findspark pyspark warcio
```

Note:
*   `flash-attn` might require specific build environments or CUDA versions.
*   `deepspeed` is used for distributed training.
*   `pyspark` requires a local Spark installation. Set the `SPARK_HOME` environment variable or adjust the `findspark.init()` path in the notebooks.
*   **Credentials**: You will need credentials for the `dbrepo` client (TU Wien repository) and potentially an API token if you intend to use the model upload functionality in `01_model_training.ipynb`. These need to be inserted directly into the code where indicated (e.g., `RestClient(..., password="...")`, `headers = {"Authorization": "..."}`).

## Execution Order

The notebooks should be executed in the following order:

1.  **`01_model_training.ipynb`**:
    *   Fetches training data from the dbrepo (**Requires dbrepo credentials**).
    *   Prepares the data in the required format (`fine_data.jsonl`).
    *   Fine-tunes the `BAAI/bge-large-en-v1.5` model using the prepared data.
    *   Saves the fine-tuned model locally (in `./test_encoder_only_base_bge-large-en-v1.5`).
    *   (Optional) Contains code to upload the trained model to a repository (**Requires API token**).

2.  **`02_embedding_calculation_multiple.ipynb`**:
    *   Loads test data (either from dbrepo (**Requires dbrepo credentials**) or a local `savedata.json`).
    *   Loads both the fine-tuned model and the base `BAAI/bge-large-en-v1.5` model.
    *   Calculates embeddings for the test data content and predefined categories ("cyber", "krimisi", "innotech") using both models.
    *   Saves the calculated embeddings and associated metadata into pickle files inside the `model_data/` directory (one file per model).

3.  **`03_model_comparison.ipynb`**:
    *   Loads the embedding data saved by the previous notebook from the `model_data/` directory.
    *   Calculates cosine similarity between content embeddings and category embeddings for each document and model.
    *   Computes evaluation metrics (Accuracy, F1, Precision, Recall, AP, AUC-ROC) based on the `is_relevant` flag and the calculated similarities.
    *   Displays statistical summaries and comparison plots for the models.

## Data Acquisition (Optional)

The `DATA_aquisition_preprocessing/` folder contains notebooks used for fetching and preprocessing data directly from the Common Crawl dataset. This is **not required** if you use the `dbrepo` client in notebooks `01` and `02` to fetch the pre-processed data. However, if you wish to understand or replicate the data acquisition process from scratch, you can explore these notebooks:

1.  **`DATA_aquisition_preprocessing/1_download_url_index.ipynb`**: Downloads the Common Crawl index files relevant to specific domains (e.g., `.at`). Requires Spark.
2.  **`DATA_aquisition_preprocessing/2_filter_index.ipynb`**: Filters the downloaded index files based on specific criteria or URLs. Requires Spark.
3.  **`DATA_aquisition_preprocessing/3_get_metadata.ipynb`**: Fetches metadata associated with the filtered URLs. Requires Spark.
4.  **`DATA_aquisition_preprocessing/4_filter_metadata.ipynb`**: Filters the metadata based on specific criteria (e.g., language, keywords). Requires Spark.
5.  **`DATA_aquisition_preprocessing/5_download_wat_wet.ipynb`**: Downloads the actual web content (WET) and metadata/link information (WAT) files from Common Crawl for the final set of URLs and extracts the relevant records. Requires Spark and `warcio`. Saves the final data to `savedata.json`.

**Note:** Running the data acquisition notebooks requires significant disk space, bandwidth, and processing time. They rely on Apache Spark for distributed processing.

## Results

The comparison between the custom fine-tuned model and the base BAAI/bge-large-en-v1.5 model yielded the following results:

### Evaluation Metrics
![Evaluation Metrics](https://github.com/user-attachments/assets/ecdeb49b-3ee1-41a9-a522-95b50347bfc7)

### AUC-ROC Scores

![AUC-ROC Scores](https://github.com/user-attachments/assets/5336027c-69b9-44a6-8ad9-51e4fcffecca)

The results indicate that the base `BAAI/bge-large-en-v1.5` model generally performs better on this specific relevance classification task according to the AUC-ROC score, although the custom model shows slightly different characteristics in terms of precision/recall trade-offs based on the chosen thresholds. This is probably the case as the resulting model wasn't finteuned very much, because of contrained GPU resources (Google Colab).

## Used datasources

[![DOI](https://test.researchdata.tuwien.ac.at/badge/DOI/10.82556/x0q6-dm10.svg)](https://handle.stage.datacite.org/10.82556/x0q6-dm10)
[Training Data](https://test.dbrepo.tuwien.ac.at/database/e31b2788-e1da-4a4a-9892-d2b5a1216df6/info)