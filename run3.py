'''
Part 3: Retrieval Heads

Goal:
    - Select a subset of attention heads using training data
    - Use ONLY those heads to rank tools at test time
'''

import os
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import argparse
import time
import random
import numpy as np
import torch
from tqdm import tqdm

from utils import load_model_tokenizer, PromptUtils, get_queries_and_items

from code3 import select_retrieval_heads

import gc
# -------------------------
# Do NOT change
# -------------------------
def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def query_to_docs_attention_heads(attentions, query_span, doc_spans, selected_heads):
    # TODO 2: Head-based scoring
    """
    Compute document scores using ONLY selected_heads.

    Inputs:
        attentions: tuple of (num_layers) each [1, heads, N, N]
        query_span: (start, end)
        doc_spans: list of (start, end)
        selected_heads: list of (layer, head)

    Output:
        doc_scores: tensor of shape [num_docs]
    """

    doc_scores = torch.zeros(len(doc_spans), device=attentions[0].device)
    query_start, query_end = query_span
 
    for layer_idx, head_idx in selected_heads:
        head_attn = attentions[layer_idx][0, head_idx]  # [N, N]
        query_attn = head_attn[query_start:query_end, :]  # [query_len, N]
 
        for doc_i, (doc_start, doc_end) in enumerate(doc_spans):
            doc_scores[doc_i] += query_attn[:, doc_start:doc_end].mean()
 
    doc_scores /= len(selected_heads)
    return doc_scores


def get_query_span(input_ids, tokenizer):
    # TODO 3: Query span
    """
    Identify the token span corresponding to the query.
    Note: you are free to add/remove args in this function
    """
    input_ids_list = input_ids.tolist()
    # Search backward for token 2929 ("Query")
    i = len(input_ids_list) - 1
    while i >= 0:
        if input_ids_list[i] == 2929:
            position_of_query = i
            break
        i -= 1
    actual_query_st_pos = position_of_query + 2
    # Search forward for token 34192 ("tool_id")
    for i in range(actual_query_st_pos, len(input_ids_list)):
        if input_ids_list[i] == 34192:
            break
    return (actual_query_st_pos, i)

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=64)
parser.add_argument('--model', type=str, default="meta-llama/Llama-3.2-1B-Instruct")
parser.add_argument('--max_heads', type=int, default=20)
parser.add_argument('--train_samples', type=int, default=200)
parser.add_argument("--debug", action="store_true")
args = parser.parse_args()


if __name__ == '__main__':

    seed_all(args.seed)
    model_name = args.model
    device = "cuda:0"    
    tokenizer, model = load_model_tokenizer(model_name=model_name, device=device, dtype=torch.float16)

    train_queries, test_queries, tools = get_queries_and_items()
    print("\n[Phase 1] Selecting retrieval heads...")

    selected_heads = select_retrieval_heads(
        train_queries=train_queries[:args.train_samples],
        model=model,
        tokenizer=tokenizer,
        tools=tools,
        device=device,
        max_heads=args.max_heads
    )

    print(f"Selected {len(selected_heads)} heads")
    print(selected_heads)
    
    for lh in selected_heads:
        print(f"Layer{lh[0]}, Head {lh[1]}")

    print("\n[Phase 2] Evaluating on test set...")
    correct_at_1 = 0
    correct_at_5 = 0
    total = 0

    for qix in tqdm(range(len(test_queries))):

        sample = test_queries[qix]
        question = sample["text"]
        gold_tool_name = sample["gold_tool_name"]

        # --------------------
        # Do Not change the shuffling here
        # --------------------
        shuffled_keys = list(tools.keys())
        random.shuffle(shuffled_keys)

        putils = PromptUtils(
            tokenizer=tokenizer,
            doc_ids=shuffled_keys,
            dict_all_docs=tools,
        )

        item_spans = putils.doc_spans
        map_docname_id = putils.dict_doc_name_id

        gold_tool_id = map_docname_id[gold_tool_name]

        prompt = putils.create_prompt(query=question)
        inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)

        input_ids = inputs.input_ids[0]

        with torch.no_grad():
            attentions = model(**inputs).attentions

        query_span = get_query_span(input_ids, tokenizer)

        doc_scores = query_to_docs_attention_heads(
            attentions,
            query_span,
            item_spans,
            selected_heads
        )


        # TODO: ranking the docs
        ranked_docs = torch.argsort(doc_scores, descending=True)
        gold_rank = 1000 
        for i, tool_id in enumerate(ranked_docs):
            if tool_id == gold_tool_id:
                gold_rank = i
                break
        # TODO: measure the recall@1, recall@5

        if gold_rank<1:
            correct_at_1 += 1
        if gold_rank<5:
            correct_at_5 += 1
        total += 1

        del attentions, inputs, input_ids, doc_scores
        gc.collect()
        torch.cuda.empty_cache()       


    recall_at_1 = correct_at_1 / total
    print(f"\nRecall@1 (selected heads): {recall_at_1:.4f}")
    recall_at_5 = correct_at_5 / total
    print(f"Recall@5 (selected heads): {recall_at_5:.4f}")