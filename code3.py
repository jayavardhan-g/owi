import torch
from tqdm import tqdm
from utils import PromptUtils
import random 
import gc

def select_retrieval_heads(train_queries, model, tokenizer, tools, device, max_heads=20):
    # TODO 3: Head selection
    """
    Identify a subset of attention heads that are most useful for retrieving the correct tool.

    Requirements:
    - Use the same prompt structure as Part-2
    - Use attention patterns(query -> tool) to score heads
    - Aggregate signals across training queries
    - Return "max_heads" heads as (layer, head)

    Notes:
    - You must construct prompts and extract attentions inside this function
    - Avoid hardcoding specific queries or tools
    """

    num_layers = model.config.num_hidden_layers
    num_heads = model.config.num_attention_heads

    # accumulate scores per head
    head_scores = torch.zeros(num_layers, num_heads, device=device)

    for qix in tqdm(range(len(train_queries))):
        sample = train_queries[qix]
        question = sample["text"]
        gold_tool_name = sample["gold_tool_name"]
 
        tool_ids = list(tools.keys())
        random.shuffle(tool_ids)
 
        putils = PromptUtils(
            tokenizer=tokenizer,
            doc_ids=tool_ids,
            dict_all_docs=tools,
        )
        item_spans = putils.doc_spans
        doc_lengths = putils.doc_lengths
        map_docname_id = putils.dict_doc_name_id
        map_id_docname = {v: k for k, v in map_docname_id.items()}
        db_lengths_pt = torch.tensor(doc_lengths, device=device)
 
        gold_tool_id = map_docname_id[gold_tool_name]
 
        prompt = putils.create_prompt(query=question)
        inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)
        input_ids = inputs.input_ids[0]
 
        with torch.no_grad():
            attentions = model(**inputs).attentions
 
        # Get query span (same logic in run2.py)
        input_ids_list = input_ids.tolist()
        i = len(input_ids_list) - 1
        while i >= 0:
            if input_ids_list[i] == 2929: 
                position_of_query = i
                break
            i -= 1
        query_start_position = position_of_query + 2
        for i in range(query_start_position, len(input_ids_list)):
            if input_ids_list[i] == 34192: 
                break
        query_start = query_start_position
        query_end = i
 
        for layer_idx in range(num_layers):
            attn = attentions[layer_idx][0]  
            for head_idx in range(num_heads):
                head_attn = attn[head_idx]  
                query_attn = head_attn[query_start:query_end, :]  
 
                doc_scores = torch.zeros(len(item_spans), device=device)
                for doc_i, (ds, de) in enumerate(item_spans):
                    doc_scores[doc_i] = query_attn[:, ds:de].mean()
 
            # Gold tool hit count
                ranked = torch.argsort(doc_scores, descending=True)
                if ranked[0].item() == gold_tool_id:
                    head_scores[layer_idx, head_idx] += 1

                    
            # Reciprocal rank sum
                # gold_tool_rank = 1000
                # for i, tool_id in enumerate(ranked):
                #     if tool_id == gold_tool_id:
                #         gold_tool_rank = i
                #         break

                # head_scores[layer_idx, head_idx] += 1.0 / (gold_tool_rank + 1)

            # Raw attention scores

                # gold_start, gold_end = item_spans[gold_tool_id]
                # head_scores[layer_idx, head_idx] += query_attn[:, gold_start:gold_end].mean()

        del attentions, inputs, input_ids
        gc.collect()
        torch.cuda.empty_cache()

    # --- Select top-K heads by hit count ---
    flat_scores = head_scores.flatten()
    top20= torch.argsort(flat_scores, descending=True)[:max_heads]
 
    selected_heads = []
    for idx in top20:
        layer = idx.item() // num_heads
        head = idx.item() % num_heads
        selected_heads.append((layer, head))

        print(f"  Head (layer={layer}, head={head}): Gold tool Hit count = {int(head_scores[layer, head].item())}")
        # print(f"  Head (layer={layer}, head={head}): Reciprocal Ranked Scores= {(head_scores[layer, head].item())}")
        # print(f"  Head (layer={layer}, head={head}): Raw attention Scores= {(head_scores[layer, head].item())}")
 
    assert len(selected_heads) == max_heads
    return selected_heads