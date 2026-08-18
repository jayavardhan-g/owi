'''
Part 2: are we lost in the middle?

Goal:
    - visualize the attention from the query to gold document based on the distance between them
    - use attention as a metric to rank documents for a query 
'''
import gc
import os
os.environ["TRANSFORMERS_OFFLINE"] = "1"
import argparse
import json 
import time
import pandas as pd
from tqdm import tqdm
import torch
import random
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from utils import load_model_tokenizer, PromptUtils, get_queries_and_items

# -------------------------
# Do NOT change
# -------------------------
def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed) 
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def query_to_docs_attention(attentions, query_span, doc_spans):
    """
    attentions: tuple(num_layers) of [1, heads, N, N]
    query_span: (start, end)
    doc_spans: list of (start, end)
    """
    doc_scores = torch.zeros(len(doc_spans), device=attentions[0].device)
    query_start, query_end = query_span
    num_layers = len(attentions)
    for layer in range(num_layers):
        avg_attn = attentions[layer][0].mean(dim=0)
        query_attn = avg_attn[query_start:query_end, :]
        for i, (doc_start, doc_end) in enumerate(doc_spans):
            doc_scores[i] += query_attn[:, doc_start:doc_end].mean()
    doc_scores /= num_layers
    return doc_scores


def analyze_gold_attention(results, save_path="plot2/gold_attention_plot.png"):
    os.makedirs("plot2", exist_ok=True)
    positions = [r["gold_position"] for r in results]
    scores = [r["gold_score"] for r in results]

    num_bins = 20
    bin_size = 100 // num_bins
    bin_positions = []
    bin_scores = []
    for b in range(num_bins):
        low = b * bin_size
        high = (b + 1) * bin_size
        bin_data = [s for p, s in zip(positions, scores) if low <= p < high]
        if bin_data:
            bin_positions.append((low + high) / 2)
            bin_scores.append(np.mean(bin_data))

    plt.figure(figsize=(10, 6))
    plt.plot(bin_positions, bin_scores, marker='o', linewidth=2)
    plt.xlabel("Position of Gold Tool in Prompt")
    plt.ylabel("Average Attention Score")
    plt.title("Attention to Gold Tool vs. Position in Prompt")
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to {save_path}")


def get_query_span(input_ids):
    input_ids_list = input_ids.tolist()
    i = len(input_ids_list) - 1
    while i >= 0:
        if input_ids_list[i] == 2929:
            position_of_query = i
            break
        i -= 1
    actual_query_st_pos = position_of_query + 2
    for i in range(actual_query_st_pos, len(input_ids_list)):
        if input_ids_list[i] == 34192:
            break
    return (actual_query_st_pos, i)


parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=64)
parser.add_argument('--model', type=str, default="meta-llama/Llama-3.2-1B-Instruct")
parser.add_argument('--top_heads', type=int, default=20)
parser.add_argument("--debug", action="store_true", help="Enable debug mode")
args = parser.parse_args()


if __name__ == '__main__':
    seed_all(seed=args.seed)
    model_name = args.model
    device = "cuda:0"
    
    tokenizer, model = load_model_tokenizer(model_name=model_name, device=device, dtype=torch.float16)
    num_heads = model.config.num_attention_heads
    num_layers = model.config.num_hidden_layers
    d = getattr(model.config, "head_dim", model.config.hidden_size // model.config.num_attention_heads)
    num_key_value_groups = num_heads//model.config.num_key_value_heads
    softmax_scaling=d**-0.5
    train_queries, test_queries, tools = get_queries_and_items()
 
    print("---- debug print start ----")
    print(f"seed: {args.seed}, model: {model_name}")
    print("model.config._attn_implementation: ", model.config._attn_implementation)

    dict_head_freq = {}
    df_data = []
    avg_latency = []
    count = 0
    start_time = time.time()
    results = []
    correct_at_1 = 0
    correct_at_5 = 0
    total = 0

    for qix in tqdm(range(len(test_queries))):
        sample = test_queries[qix]
        qid = sample["qid"]
        question = sample["text"]
        gold_tool_name = sample["gold_tool_name"]

        # --------------------
        # Do Not change the shuffling here
        # --------------------
        num_dbs = len(tools)
        shuffled_keys = list(tools.keys())
        random.shuffle(shuffled_keys)

        putils = PromptUtils(
            tokenizer=tokenizer, 
            doc_ids=shuffled_keys, 
            dict_all_docs=tools,
            )
        item_spans = putils.doc_spans
        doc_lengths = putils.doc_lengths
        map_docname_id = putils.dict_doc_name_id
        map_id_docname = {v:k for k, v in map_docname_id.items()}
        db_lengths_pt = torch.tensor(doc_lengths, device=device)
        
        gold_tool_id = map_docname_id[gold_tool_name]

        prompt = putils.create_prompt(query=question)
        inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)

        if args.debug and qix < 5:
            ip_ids = inputs.input_ids[0].cpu()
            print("-------"*5)
            print(prompt)
            print("-------"*5)
            print("---- doc1 ----")
            print(tokenizer.decode(ip_ids[item_spans[0][0]: item_spans[0][1]]))
            print("---- lastdoc ----")
            print(tokenizer.decode(ip_ids[item_spans[-1][0]: item_spans[-1][1]]))
            print("-------"*5)

        with torch.no_grad():
            attentions = model(**inputs).attentions
            '''
                attentions - tuple of length = # layers
                attentions[0].shape - [1, h, N, N] : first layer's attention matrix for h heads
            '''
        
        input_ids = inputs.input_ids[0]
        query_span = get_query_span(input_ids)
        doc_scores = query_to_docs_attention(attentions, query_span, item_spans)

        ranked_docs = torch.argsort(doc_scores, descending=True)
        gold_rank = (ranked_docs == gold_tool_id).nonzero(as_tuple=True)[0].item()
        gold_score = doc_scores[gold_tool_id].item()
        
        results.append({
            "qid": qid,
            "gold_position": gold_tool_id,
            "gold_score": gold_score,
            "gold_rank": gold_rank
        })

        if gold_rank < 1:
            correct_at_1 += 1
        if gold_rank < 5:
            correct_at_5 += 1
        total += 1

        del attentions, inputs, doc_scores, input_ids
        gc.collect()
        torch.cuda.empty_cache()

    print(f"\nFinal Recall@1: {correct_at_1/total:.4f}")
    print(f"Final Recall@5: {correct_at_5/total:.4f}")

    analyze_gold_attention(results)
