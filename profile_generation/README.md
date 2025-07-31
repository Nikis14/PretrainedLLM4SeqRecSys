# **Profile Generation Launch Guide**

## **Launching Profile Generation**
The process is initiated using `encode.py`.

### **Parameters for a Pre-existing Dataset**
1. **`--dataset`**  
2. **`--dataset-path`** - Path to the dataset. The dataset should follow this structure:

   ```
   data
   ├── <dataset_name>
   │   ├── raw
   |   |   |─── users.csv - user IDs
   |   |   |─── items.csv - item IDs with metadata
   |   |   |─── interactions.csv - user IDs, item IDs, interaction data
   │   ├── processed
   |   |   |─── <processed_data> - results of dataset preprocessing
   ```

3. **`--llm`** - Choice of LLM. The following LLMs are currently supported:
   - `gigachat`
   - `openai`
   - `gemma2-9b`

4. **`--descriptions-path`** - Path to the folder where user profiles will be saved. All profiles will be stored in the `descriptions_all` file.

5. **`--long-gen-strategy`** - Profile aggregation strategy:
   - `agg_after` - merge profiles for samples after computing all samples.
   - `agg_with` - merge profiles for samples sequentially during generation.

6. **`--prompts-type`** - Prompt types:
   - `long` - profiles will be in free format.
   - `short` - profiles will be in structured format.

7. **`--max_output_tokens`** - Maximum number of generated tokens.

8. **`--user-ids-path`** - Path to a JSON file with user IDs for which profiles will be generated.

### **Launching with a New Dataset**

1. Preprocess the dataset.
2. Create a dataset handler similar to those in `src/datasets`. A special class can also be created for items with unique metadata.
3. Add prompts to `src/prompts`.
4. Add the new dataset to `encode.py`.
5. Run `encode.py`.

### **Example Launch Command**

```bash
python .\encode.py --dataset='beauty' --dataset-path='../data/amazon_beauty' --llm='gigachat' --descriptions-path='./output' --long-gen-strategy="agg_after" --max_output_tokens=1000
```

## **Launching Embedding Generation**
Run `encode_descriptions.py`, specifying:

1. **`--embedder`** - Choice of embedder, currently supported:
   - `gigachat`
   - `openai`
   - `e5`

2. **`--descriptions-path`** - Path to a JSON file formatted as `user_id: description`.
3. **`--embeddings-path`** - Path to save the JSON file formatted as `user_id: embedding`.

```bash
python -m encode_descriptions   --embedder "e5"    --descriptions-path "data/kion_en/descriptions.json"     --embeddings-path "data/kion_en/embeddings/embeddings.json"
