---
tags:
- sentence-transformers
- sentence-similarity
- feature-extraction
- generated_from_trainer
- dataset_size:28
- loss:MultipleNegativesRankingLoss
base_model: sentence-transformers/all-MiniLM-L6-v2
widget:
- source_sentence: Where is the campus located?
  sentences:
  - Students gain hands-on experience through capstone projects, industry collaborations,
    and events such as company site visits, helping them build practical skills that
    are directly applicable to the workforce.
  - Our campus is strategically situated at the NBC Tower and Gleacher Center in downtown
    Chicago’s Streeterville neighborhood
  - The curriculum emphasizes a combination of computational and analytical skills,
    domain knowledge, communication, and ethics, preparing students for broad application
    of data science in various sectors.
- source_sentence: What is the structure of the capstone experience in the MS in Applied
    Data Science program?
  sentences:
  - The MS in Applied Data Science program offers partial tuition scholarships to
    top applicants. These scholarships do not require a separate application but it
    is recommended that candidates submit their applications ahead of the early deadline
    to maximize their chances of securing a scholarship.
  - The capstone is a project-based course where students work in teams to solve real-world
    problems for clients from industry, academia, and social impact organizations,
    applying data science in a practical setting.
  - The MS in Applied Data Science program is STEM/OPT eligible
- source_sentence: What is the purpose of the capstone projects in the MS in Applied
    Data Science program?
  sentences:
  - To earn the MS-ADS degree students must successfully complete 12 courses (6 core,
    4 elective, 2 Capstone) and our tailored Career Seminar
  - Capstone projects allow students to address real-world challenges across industries,
    showcasing practical and impactful data science solutions. Outstanding projects
    are recognized with awards such as 'Best in Show'.
  - Introduction to Statistical Concepts (Foundational, optional), R for Data Science
    (Foundational, optional), Python for Data Science (Foundational, optional), Advanced
    Linear Algebra for Machine Learning (Foundational, optional)
- source_sentence: How long will it take for me to receive a decision on my application?
  sentences:
  - Located within Hyde Park campus
  - Most courses are from 6-9pm Monday through Thursday with some offered on Fridays
    and Saturdays. This allows you to work in an internship and/or job during the
    program
  - In-Person application decisions are released approximately 1 to 2 months after
    each respected deadline. Online application decisions are released on a rolling
    basis
- source_sentence: Where can I mail my official transcripts?
  sentences:
  - Ethics is integrated into the curriculum alongside technical skills to ensure
    graduates are equipped to consider the social impact of data science solutions
    and apply responsible practices in real-world contexts.
  - The MS in Applied Data Science program requires two letters of recommendation
  - 'The University of Chicago

    Attention: MS in Applied Data Science Admissions

    455 N Cityfront Plaza Dr., Suite 950

    Chicago, Illinois 60611'
pipeline_tag: sentence-similarity
library_name: sentence-transformers
metrics:
- pearson_cosine
- spearman_cosine
co2_eq_emissions:
  emissions: 0.12810612663829163
  energy_consumed: 0.00034810723388881034
  source: codecarbon
  training_type: fine-tuning
  on_cloud: false
  cpu_model: Intel(R) Core(TM) i5-5350U CPU @ 1.80GHz
  ram_total_size: 8.0
  hours_used: 0.035
model-index:
- name: SentenceTransformer based on sentence-transformers/all-MiniLM-L6-v2
  results:
  - task:
      type: semantic-similarity
      name: Semantic Similarity
    dataset:
      name: Unknown
      type: unknown
    metrics:
    - type: pearson_cosine
      value: .nan
      name: Pearson Cosine
    - type: spearman_cosine
      value: .nan
      name: Spearman Cosine
---

# SentenceTransformer based on sentence-transformers/all-MiniLM-L6-v2

This is a [sentence-transformers](https://www.SBERT.net) model finetuned from [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2). It maps sentences & paragraphs to a 384-dimensional dense vector space and can be used for semantic textual similarity, semantic search, paraphrase mining, text classification, clustering, and more.

## Model Details

### Model Description
- **Model Type:** Sentence Transformer
- **Base model:** [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) <!-- at revision c9745ed1d9f207416be6d2e6f8de32d1f16199bf -->
- **Maximum Sequence Length:** 256 tokens
- **Output Dimensionality:** 384 dimensions
- **Similarity Function:** Cosine Similarity
<!-- - **Training Dataset:** Unknown -->
<!-- - **Language:** Unknown -->
<!-- - **License:** Unknown -->

### Model Sources

- **Documentation:** [Sentence Transformers Documentation](https://sbert.net)
- **Repository:** [Sentence Transformers on GitHub](https://github.com/UKPLab/sentence-transformers)
- **Hugging Face:** [Sentence Transformers on Hugging Face](https://huggingface.co/models?library=sentence-transformers)

### Full Model Architecture

```
SentenceTransformer(
  (0): Transformer({'max_seq_length': 256, 'do_lower_case': False}) with Transformer model: BertModel 
  (1): Pooling({'word_embedding_dimension': 384, 'pooling_mode_cls_token': False, 'pooling_mode_mean_tokens': True, 'pooling_mode_max_tokens': False, 'pooling_mode_mean_sqrt_len_tokens': False, 'pooling_mode_weightedmean_tokens': False, 'pooling_mode_lasttoken': False, 'include_prompt': True})
  (2): Normalize()
)
```

## Usage

### Direct Usage (Sentence Transformers)

First install the Sentence Transformers library:

```bash
pip install -U sentence-transformers
```

Then you can load this model and run inference.
```python
from sentence_transformers import SentenceTransformer

# Download from the 🤗 Hub
model = SentenceTransformer("sentence_transformers_model_id")
# Run inference
sentences = [
    'Where can I mail my official transcripts?',
    'The University of Chicago\nAttention: MS in Applied Data Science Admissions\n455 N Cityfront Plaza Dr., Suite 950\nChicago, Illinois 60611',
    'The MS in Applied Data Science program requires two letters of recommendation',
]
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 384]

# Get the similarity scores for the embeddings
similarities = model.similarity(embeddings, embeddings)
print(similarities.shape)
# [3, 3]
```

<!--
### Direct Usage (Transformers)

<details><summary>Click to see the direct usage in Transformers</summary>

</details>
-->

<!--
### Downstream Usage (Sentence Transformers)

You can finetune this model on your own dataset.

<details><summary>Click to expand</summary>

</details>
-->

<!--
### Out-of-Scope Use

*List how the model may foreseeably be misused and address what users ought not to do with the model.*
-->

## Evaluation

### Metrics

#### Semantic Similarity

* Evaluated with [<code>EmbeddingSimilarityEvaluator</code>](https://sbert.net/docs/package_reference/sentence_transformer/evaluation.html#sentence_transformers.evaluation.EmbeddingSimilarityEvaluator)

| Metric              | Value   |
|:--------------------|:--------|
| pearson_cosine      | nan     |
| **spearman_cosine** | **nan** |

<!--
## Bias, Risks and Limitations

*What are the known or foreseeable issues stemming from this model? You could also flag here known failure cases or weaknesses of the model.*
-->

<!--
### Recommendations

*What are recommendations with respect to the foreseeable issues? For example, filtering explicit content.*
-->

## Training Details

### Training Dataset

#### Unnamed Dataset

* Size: 28 training samples
* Columns: <code>sentence_0</code>, <code>sentence_1</code>, and <code>label</code>
* Approximate statistics based on the first 28 samples:
  |         | sentence_0                                                                        | sentence_1                                                                        | label                                                         |
  |:--------|:----------------------------------------------------------------------------------|:----------------------------------------------------------------------------------|:--------------------------------------------------------------|
  | type    | string                                                                            | string                                                                            | float                                                         |
  | details | <ul><li>min: 8 tokens</li><li>mean: 14.18 tokens</li><li>max: 23 tokens</li></ul> | <ul><li>min: 7 tokens</li><li>mean: 32.07 tokens</li><li>max: 67 tokens</li></ul> | <ul><li>min: 1.0</li><li>mean: 1.0</li><li>max: 1.0</li></ul> |
* Samples:
  | sentence_0                                             | sentence_1                                                                                                                                                    | label            |
  |:-------------------------------------------------------|:--------------------------------------------------------------------------------------------------------------------------------------------------------------|:-----------------|
  | <code>Is there an application fee?</code>              | <code>There is a $90 non-refundable application fee</code>                                                                                                    | <code>1.0</code> |
  | <code>Where can I mail my official transcripts?</code> | <code>The University of Chicago<br>Attention: MS in Applied Data Science Admissions<br>455 N Cityfront Plaza Dr., Suite 950<br>Chicago, Illinois 60611</code> | <code>1.0</code> |
  | <code>Is there an application fee waiver?</code>       | <code>For questions regarding an application fee waiver, please refer to the Physical Sciences Division fee waiver policy.</code>                             | <code>1.0</code> |
* Loss: [<code>MultipleNegativesRankingLoss</code>](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#multiplenegativesrankingloss) with these parameters:
  ```json
  {
      "scale": 20.0,
      "similarity_fct": "cos_sim"
  }
  ```

### Training Hyperparameters
#### Non-Default Hyperparameters

- `eval_strategy`: steps
- `per_device_train_batch_size`: 4
- `per_device_eval_batch_size`: 4
- `num_train_epochs`: 4
- `multi_dataset_batch_sampler`: round_robin

#### All Hyperparameters
<details><summary>Click to expand</summary>

- `overwrite_output_dir`: False
- `do_predict`: False
- `eval_strategy`: steps
- `prediction_loss_only`: True
- `per_device_train_batch_size`: 4
- `per_device_eval_batch_size`: 4
- `per_gpu_train_batch_size`: None
- `per_gpu_eval_batch_size`: None
- `gradient_accumulation_steps`: 1
- `eval_accumulation_steps`: None
- `torch_empty_cache_steps`: None
- `learning_rate`: 5e-05
- `weight_decay`: 0.0
- `adam_beta1`: 0.9
- `adam_beta2`: 0.999
- `adam_epsilon`: 1e-08
- `max_grad_norm`: 1
- `num_train_epochs`: 4
- `max_steps`: -1
- `lr_scheduler_type`: linear
- `lr_scheduler_kwargs`: {}
- `warmup_ratio`: 0.0
- `warmup_steps`: 0
- `log_level`: passive
- `log_level_replica`: warning
- `log_on_each_node`: True
- `logging_nan_inf_filter`: True
- `save_safetensors`: True
- `save_on_each_node`: False
- `save_only_model`: False
- `restore_callback_states_from_checkpoint`: False
- `no_cuda`: False
- `use_cpu`: False
- `use_mps_device`: False
- `seed`: 42
- `data_seed`: None
- `jit_mode_eval`: False
- `use_ipex`: False
- `bf16`: False
- `fp16`: False
- `fp16_opt_level`: O1
- `half_precision_backend`: auto
- `bf16_full_eval`: False
- `fp16_full_eval`: False
- `tf32`: None
- `local_rank`: 0
- `ddp_backend`: None
- `tpu_num_cores`: None
- `tpu_metrics_debug`: False
- `debug`: []
- `dataloader_drop_last`: False
- `dataloader_num_workers`: 0
- `dataloader_prefetch_factor`: None
- `past_index`: -1
- `disable_tqdm`: False
- `remove_unused_columns`: True
- `label_names`: None
- `load_best_model_at_end`: False
- `ignore_data_skip`: False
- `fsdp`: []
- `fsdp_min_num_params`: 0
- `fsdp_config`: {'min_num_params': 0, 'xla': False, 'xla_fsdp_v2': False, 'xla_fsdp_grad_ckpt': False}
- `tp_size`: 0
- `fsdp_transformer_layer_cls_to_wrap`: None
- `accelerator_config`: {'split_batches': False, 'dispatch_batches': None, 'even_batches': True, 'use_seedable_sampler': True, 'non_blocking': False, 'gradient_accumulation_kwargs': None}
- `deepspeed`: None
- `label_smoothing_factor`: 0.0
- `optim`: adamw_torch
- `optim_args`: None
- `adafactor`: False
- `group_by_length`: False
- `length_column_name`: length
- `ddp_find_unused_parameters`: None
- `ddp_bucket_cap_mb`: None
- `ddp_broadcast_buffers`: False
- `dataloader_pin_memory`: True
- `dataloader_persistent_workers`: False
- `skip_memory_metrics`: True
- `use_legacy_prediction_loop`: False
- `push_to_hub`: False
- `resume_from_checkpoint`: None
- `hub_model_id`: None
- `hub_strategy`: every_save
- `hub_private_repo`: None
- `hub_always_push`: False
- `gradient_checkpointing`: False
- `gradient_checkpointing_kwargs`: None
- `include_inputs_for_metrics`: False
- `include_for_metrics`: []
- `eval_do_concat_batches`: True
- `fp16_backend`: auto
- `push_to_hub_model_id`: None
- `push_to_hub_organization`: None
- `mp_parameters`: 
- `auto_find_batch_size`: False
- `full_determinism`: False
- `torchdynamo`: None
- `ray_scope`: last
- `ddp_timeout`: 1800
- `torch_compile`: False
- `torch_compile_backend`: None
- `torch_compile_mode`: None
- `include_tokens_per_second`: False
- `include_num_input_tokens_seen`: False
- `neftune_noise_alpha`: None
- `optim_target_modules`: None
- `batch_eval_metrics`: False
- `eval_on_start`: False
- `use_liger_kernel`: False
- `eval_use_gather_object`: False
- `average_tokens_across_devices`: False
- `prompts`: None
- `batch_sampler`: batch_sampler
- `multi_dataset_batch_sampler`: round_robin

</details>

### Training Logs
| Epoch  | Step | spearman_cosine |
|:------:|:----:|:---------------:|
| 0.7143 | 5    | nan             |
| 1.0    | 7    | nan             |
| 1.4286 | 10   | nan             |
| 2.0    | 14   | nan             |
| 2.1429 | 15   | nan             |
| 2.8571 | 20   | nan             |
| 3.0    | 21   | nan             |
| 3.5714 | 25   | nan             |
| 4.0    | 28   | nan             |


### Environmental Impact
Carbon emissions were measured using [CodeCarbon](https://github.com/mlco2/codecarbon).
- **Energy Consumed**: 0.000 kWh
- **Carbon Emitted**: 0.000 kg of CO2
- **Hours Used**: 0.035 hours

### Training Hardware
- **On Cloud**: No
- **GPU Model**: No GPU used
- **CPU Model**: Intel(R) Core(TM) i5-5350U CPU @ 1.80GHz
- **RAM Size**: 8.00 GB

### Framework Versions
- Python: 3.11.5
- Sentence Transformers: 4.1.0
- Transformers: 4.51.3
- PyTorch: 2.1.2
- Accelerate: 1.6.0
- Datasets: 2.16.1
- Tokenizers: 0.21.1

## Citation

### BibTeX

#### Sentence Transformers
```bibtex
@inproceedings{reimers-2019-sentence-bert,
    title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2019",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/1908.10084",
}
```

#### MultipleNegativesRankingLoss
```bibtex
@misc{henderson2017efficient,
    title={Efficient Natural Language Response Suggestion for Smart Reply},
    author={Matthew Henderson and Rami Al-Rfou and Brian Strope and Yun-hsuan Sung and Laszlo Lukacs and Ruiqi Guo and Sanjiv Kumar and Balint Miklos and Ray Kurzweil},
    year={2017},
    eprint={1705.00652},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```

<!--
## Glossary

*Clearly define terms in order to be accessible across audiences.*
-->

<!--
## Model Card Authors

*Lists the people who create the model card, providing recognition and accountability for the detailed work that goes into its construction.*
-->

<!--
## Model Card Contact

*Provides a way for people who have updates to the Model Card, suggestions, or questions, to contact the Model Card authors.*
-->