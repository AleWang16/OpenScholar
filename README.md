# Important Information

DISCLAIMER: all commands displayed in this section are for bash, and thus it is highly recommended to use Ubuntu for best results.

This forked repository contains two new scripts that contain the fundamental functions for chunking, embedding, indexing, and 
searching.  Two files are relevant for these functions are ```run_chunk_and_embed.py``` and ```build_indices_and_search.py```.  Below is a simplified file tree of the relevant files:

```
OpenScholar
|   README.md
|
└─── retriever
|   |   run_chunk_and_embed.py  
|   |
|   └─── src
|       |   build_indices_and_search.py
|       |
|
└─── environments
|   |   build_indices_and_search.yml
|   |   run_chunk_and_embed.txt
|   |
|   
└─── papers
|   |   # insert your Grobid-preproceseed papers in .json format here 
|   |
|
└─── processed_papers
|   |
|   └─── embeddings
|   |   | 
|   |   └─── index
|   |   |   |
|   |   |   └─── {shard number}
|   |   |   |   |   # faiss indices in .faiss format
|   |   |   # directory for passage chunk embeddings in .pkl format
|   |   |
|   |
|   └─── normalized_jsonl
|   |   |   # papers processed in .jsonl ready to be fed to the pipeline
|   |   | 
|   |
|   └─── passages
|   |   |   # passage vector database shards saved in .pkl format
|   |   
|
└─── queries
|   |   # input queries in .jsonl format. One query per file.
|   |
|
└─── outputs
|   |
|   └─── {shard number}
|       |   # retrieved passages based on query
|   
```
Before running the scripts, create relevant conda environments, which are listed in the ```OpenScholar/environments``` folder.

For ```run_chunk_and_embed.py```:

```
conda create --name chunking_and_embedding python=3.10.19
conda activate chunking_and_embedding
pip install -r run_chunk_and_embed.txt
```
For ```build_indices_and_search.py```:

```
conda env create -f build_indices_and_search.yml   
conda activate build_indices_and_search
```

```run_chunk_and_embed.py``` is responsible for taking a Grobid-processed paper in json format from the ```OpenScholar/papers``` directory, converting it to a jsonl file with formatting that is compatible with the chunking, embedding, and sharding functions of this repository.  This is important because Grobid formats the processed papers with nested json objects, and it needs to be flattened before being passed through the chunking and embedding pipeline.  To run, please run the following command from the root directory, inserting relevant paths and files into the placeholders, as denoted by the ```{}``` braces:

```
PYTHONPATH=. python retriever/run_chunk_and_embed.py   --input_path /{path to repo}/OpenScholar/papers/{paper_in_json_format}.json   --work_dir /{path to repo}/OpenScholar/processed_papers   --text_key text   --id_key id   --num_shards 1   --chunk_size 256   --min_chunk_size 0   --model_name_or_path akariasai/pes2o_contriever   --passage_maxlength 256   --per_gpu_batch_size 256
```
The outputs of running this script will be in the the ```OpenScholar/processed_papers``` directory, which itself includes three subdirectories.  The ```processed_papers/embeddings``` directory contains the chunked passages in .pkl format.  ```processed_papers/normalized_jsonl``` contains the processed papers.  Finally, ```processed_papers/passages``` contains the sharded chunked passages in .pkl format. 

Once the passages are chunked and sharded, you can run ```build_indices_and_search.py``` from the root directory with the following commands, changing the path and file placeholders with your own, as denoted by the ```{}``` braces: 

```
cd retriever

PYTHONPATH=. python src/build_indices_and_search.py \
  --config-name default \
  datastore.domain=workdir \
  evaluation.domain=queries \
  tasks.eval.task_name=lm-eval \
  tasks.eval.search=true \
  datastore.embedding.num_shards=1 \
  datastore.embedding.shard_ids=[0] \
  datastore.index.index_shard_ids=[0] \
  datastore.embedding.embedding_dir=/{path to repo}/OpenScholar/processed_papers/embeddings \
  datastore.embedding.passages_dir=/{path to repo}/OpenScholar/processed_papers/passages \
  evaluation.data.eval_data=/{path to repo}/OpenScholar/queries/{your_query}.jsonl \
  evaluation.search.n_docs=10 \
  evaluation.eval_output_dir=/{path to repo}/OpenScholar/outputs \
  evaluation.results_only_log_file=/{path to repo}/OpenScholar/outputs/retrieval.log
```

As input, the command takes a file containing the query in .jsonl format from the ```OpenScholar/queries``` directory, where the query should be formatted in the form:
```
{"query": "this is my query"}
```

according to .jsonl formatting.  It should also be given an output directory to store the top-k passages that best match the query.  For our case, these will be in the ```OpenScholar/outputs``` directory.  The script will create subdirectories titled by the shard number (numbers that are zero-indexed), which will contain the top-k passages for the given shard in .jsonl format.  Since we used one shard in the above command, the filetree will look like this:

```
OpenScholar
|
└─── outputs
|   |
|   └─── 0
|       | test_retrieved_results.jsonl
|       |
```

Where the ```0``` directory corresponds to the ```0```th shard.  Likewise the indices created for the will be stored in the ```OpenScholar/embeddings/index``` directories, where each index is also organized into subdirectories based on the shard number, as shown below (example for one shard):
```
OpenScholar
|
└─── embeddings
|   |
|   └─── index
|       |
|       └─── 0
|           |   index_meta.faiss
|           |   index.faiss
```

Note that this a work in progress and there are many thing that may be added and modified.  Thus, many of the things here may be subject to change. 

# OpenScholar 

This repository contains the code bases of OpenScholar. 

[**Blog**](https://allenai.org/blog/openscholar) | [**Demo**](https://open-scholar.allen.ai/) |
[**Paper**](https://arxiv.org/abs/2411.14199) | [**Model checkpoints and data**](https://huggingface.co/collections/OpenScholar/openscholar-v1-67376a89f6a80f448da411a6) | [**ScholarQABench**](https://github.com/AkariAsai/ScholarQABench/) | [**Expert Evaluation**](https://github.com/AkariAsai/OpenScholar_ExpertEval) | [**Public demo code**](https://github.com/allenai/open-scholar-demo) | [**Public demo data**](https://huggingface.co/datasets/allenai/openscilm_queries)
 
### Table of contents
1. [Overview of OpenScholar](#overview-of-openscholar)
2. [Repository Organizations](#repository-organizations)
3. [Installation](#installation)
4. [Run OpenScholar](#run-openscholar-inference)
5. [Train OpenScholar-8B](#training)
6. [Run Retriever](#run-retriever)
6. [Contact and Citation](#contact-and-citation)


## Overview of OpenScholar
Scientific progress hinges on our ability to find, synthesize, and build on relevant knowledge from the scientific literature. However, the exponential growth of this literature—with millions of papers now published each year—has made it increasingly difficult for scientists to find the information they need or even stay abreast of the latest findings in a single subfield.

To help scientists effectively navigate and synthesize scientific literature, we introduce **OpenScholar**, a retrieval-augmented language model (LM) designed to answer user queries by first searching for relevant papers in the literature and then generating responses grounded in those sources. Try [open-scholar.allen.ai/](https://open-scholar.allen.ai/) and check [our paper](https://openscholar.allen.ai/paper) for more detail.


![Overview of OpenScholar](imgs/open_scholar.png)


## Repository Organizations
This repository contains codes to run OpenScholar inference. 

- [`src/`](src): Main source codes for OpenScholar. 
- [`training/`](training): Our training code to train Llama 3.1 8B using our processed data. We modified earlier version of `torchtune` for training. 
- [`retriever/`](retriever): Code base to run retrieval offline & host retrieval servers for online retrieval.  

For automatic and human evaluations, please check the following repositories. 
- To run evaluations on **ScholarQABench**, please check the [ScholarQABench](https://github.com/AkariAsai/ScholarQABench/) repository. 
- For our human evaluation interfaces as well as the results, please check the [OpenScholar_ExpertEval](https://github.com/AkariAsai/OpenScholar_ExpertEval) repository. 

## Installation 
To run OpenScholar inference, please ensure that all necessary libraries are installed. 

[test environment command]

```python
conda create -n os_env python=3.10.0
conda activate os_env
pip install -r requirements.txt
python -m spacy download en_core_web_sm
``` 

Also please set the following API keys:

```sh
export S2_API_KEY=YOUR_S2_API_KEY
```
See instructions to acquire API keys at [Semantic Scholar API Page](https://www.semanticscholar.org/product/api). 

If you want to also want to use web search engine, then sign up for you.com web API and set the key.
```sh
export YOUR_API_KEY=YOUR_YOU_COM_API_KEY
```

For information related to OpenScholar training and retriever components, refer to the [`training/`](training/) and [`retrieval/`](retrieval) directories, respectively.

## Run OpenScholar inference

By default, OpenScholar takes retrieval results from off-line retrieval results after running the retrieval scripts in [retrieval/](retireval), followed by additional retrieval from Semantic Scholar Paper API and web search API results. See the script [src/use_search_apis.py](src/use_search_apis.py) to retrieve related passages offline using external APIs. 

We released our retrieval results at [google drive](https://drive.google.com/drive/folders/1lOloYPOveKesD-37lD4Dlju96tc0XIm9?usp=sharing).  

### Use Open LMs (e.g., `Llama-3.1_OpenScholar-8B`) locally 
- Run a Standard RAG pipeline using top 10 

```sh
python run.py \
    --input_file YOUR_INPUT_FILE \
    --model_name OpenScholar/Llama-3.1_OpenScholar-8B \
    --use_contexts \
    --output_file OUTPUT_FILE_PATH \
    --top_n 10 --llama3 --zero_shot
```

- Run a Retriever+ Reranker Pipeline

```sh
python run.py \
    --input_file YOUR_INPUT_FILE \
    --model_name OpenScholar/Llama-3.1_OpenScholar-8B \
    --use_contexts \
    --ranking_ce \
    --reranker OpenScholar/OpenScholar_Reranker \
    --output_file OUTPUT_FILE_PATH \
    --top_n 10 --llama3 --zero_shot
```

- Run Open Retriever Self-reflective Generation pipeline


```sh
python run.py \
    --input_file YOUR_INPUT_FILE \
    --model_name  OpenScholar/Llama-3.1_OpenScholar-8B \
    --use_contexts --output_file OUTPUT_FILE_NAME \
    --top_n 10 --llama3 --use_contexts \
    --ranking_ce --reranker OpenScholar/OpenScholar_Reranker \ 
    --posthoc --feedack --ss_retriever \
    --use_abstract --norm_cite --zero_shot --max_per_paper 3 \
```


#### Use propriety LMs e.g., OpenAI GPT4o 

You can also combine the OpenScholar pipeline with propriety LLMs, by specifying  `model_name`, `api` and `api_key_fp`. 

```sh
python run.py \
    --input_file YOUR_INPUT_FILE \
    --model_name "gpt-4o" \
    --api "openai" \
    --api_key_fp PATH_TO_YOUR_OPEN_AI_KEY \ 
    --use_contexts \
    --output_file OUTPUT_FILE_PATH \
    --top_n 10 --llama3 --zero_shot
```

## Details of configurations 
Below, we provide the detailed of configurations. 

- `top_n`: The number of passages to be fed into the underlying LM. By default, we use `10` for multi-paper tasks. 
- `feedback`: Set true if you want to use the self-feedback loop during generation.
- `posthoc_at`: Set true if you want to run posthoc citation attributions 
- `zero_shot`: Set true if you want to run inference in a zero-shot manner. 
- `ranking_ce`: Use a reranking model to rerank `top_n` passages; If not set true, we take the `top_n` passages from the `ctxs` in the provided input file. 
- `reranker`: Specify the path to the reranker model file (local or HF hub). If you use our OpenScholar reranker, set `OpenScholar/OpenScholar_Reranker`
- `min_citation`: You can set the minimum number of citations. If any `int` is given, we exclude papers whose citations is below `min_citation`. By default, we set it to `None` and all papers are considered regardless of their citation counts. 
- `ss_retriever`: Use semantic scholar API during the feedback generation loop to enhance the feedback results. 
- `use_abstract`: Consider abstract to enhance the reranking results. 
- `max_per_paper`: set the maximum number of passages from the same paper used during inference time. 
- `task_name`: specify the task names when you run the single paper tasks. For SciFact, PubmedQA and QASA, the corresponding task names are `claim_full`, `boolean_question_full` and `single_qa`, respectively. 

## Training

### Embedding model training
- We trained our embedding model using [peS2o](https://huggingface.co/datasets/allenai/peS2o/tree/main/data/v2), which is used to form the OpenScholar Datastore.
- We used the [Contriever](https://github.com/facebookresearch/contriever) code base to continue pre-training [contriever (base)](https://huggingface.co/facebook/contriever), using [the training script](https://github.com/facebookresearch/contriever/blob/main/example_scripts/contriever.sh). 

### Reranker model training 
- We trained our reranker model using [FlagEmbeddings](https://github.com/FlagOpen/FlagEmbedding/tree/lm-cocktail/examples/reranker).
- We formatted our reranker training data, following [the original repo instructions](https://github.com/FlagOpen/FlagEmbedding/blob/lm-cocktail/examples/reranker/README.md), and then trained [BGE reranker large](https://huggingface.co/BAAI/bge-reranker-large) with torch tune.

```
torchrun --nproc_per_node {number of gpus} \
-m FlagEmbedding.reranker.run \
--output_dir {path to save model} \
--model_name_or_path BAAI/bge-reranker-base \
--train_data PATH_TO_TRAIN_DATA \
--learning_rate 6e-5 \
--fp16 \
--num_train_epochs 5 \
--per_device_train_batch_size {batch size; set 1 for toy data} \
--gradient_accumulation_steps 4 \
--dataloader_drop_last True \
--train_group_size 16 \
--max_len 512 \
--weight_decay 0.01 \
--logging_steps 10 
```

### Generator LM training 
- We trained our [OpenScholar-8B](https://huggingface.co/OpenScholar/OpenScholar_Llama-3.1-8B) using our [OpenScholar/OS_Train_Data]([https://huggingface.co/OpenScholar/OpenScholar_Train_Data](https://huggingface.co/datasets/OpenScholar/OS_Train_Data)) data, which consists of 13k instruction-tuning data. We use our modified version of [torchtune](https://github.com/pytorch/torchtune) to train our 8B model using 8*A100. 

See mode detailed instructions for setting up the training in [train/](train)

## Run Retriever
Both our peS2o v2 and v3 datastore (chunked text + index) are available: 
- [OpenScholar/OpenScholar-DataStore-V2](https://huggingface.co/datasets/OpenScholar/OpenScholar-DataStore-V2)
- [OpenScholar/OpenScholar-DataStore-V3](https://huggingface.co/datasets/OpenScholar/OpenScholar-DataStore-V3)

See instructions under [retriever](retriever) to run the peS2o index locally. Note that due to the massive-scale of index (200+M embeddings based on 45 million papers), the peS2o retriever requires a lot of CPU memory. In our main experiments, we retrieved initial passages offline. 

**We are planning to release our efficient sparse-dense retriever API endpoint used for the OpenScholar Demo publicly via Semantic Scholar API to accelerate research for LLMs for scientific literature synthesis. Stay tune!!d!**


## Contact and Citation
If you have any questions, please contact `akari@cs.washington`. Note that I am currently applying for academic jobs so I may be slow to respond. 
If you have any questions related to demo, please file your request from [google form](https://docs.google.com/forms/d/e/1FAIpQLSfqPUKxxXlV16Bs8ZGcasXMP35WKQU6eeQhYViPQ9_Cmeq5Kw/viewform).

```
@article{openscholar,
  title={{OpenScholar}: Synthesizing Scientific Literature with Retrieval-Augmented Language Models},
  author={Asai, Akari and He*, Jacqueline and Shao*, Rulin and Shi, Weijia and Singh, Amanpreet and Chang, Joseph Chee  and Lo,  Kyle and Soldaini, Luca and Feldman, Tian, Sergey and Mike, D’arcy and Wadden, David and Latzke, Matt and Minyang and Ji, Pan and Liu, Shengyan and Tong, Hao and Wu, Bohao and Xiong, Yanyu and Zettlemoyer, Luke and Weld, Dan and Neubig, Graham and Downey, Doug and Yih, Wen-tau and Koh, Pang Wei and Hajishirzi, Hannaneh},
  journal={Arxiv},
  year={2024},
}
```
