import os
from collections import Counter
from transformers import T5TokenizerFast
from load_data import load_prompting_data   

PREFIX = "translate English to SQL: "

def compute_stats_before(nl_list, sql_list):
    nl_lengths = [len(nl.split()) for nl in nl_list]
    sql_lengths = [len(sql.split()) for sql in sql_list]

    nl_vocab = Counter()
    sql_vocab = Counter()
    for nl in nl_list:
        nl_vocab.update(nl.split())
    for sql in sql_list:
        sql_vocab.update(sql.split())

    return {
        "num_examples": len(nl_list),
        "mean_nl_length": sum(nl_lengths) / len(nl_lengths),
        "mean_sql_length": sum(sql_lengths) / len(sql_lengths),
        "nl_vocab_size": len(nl_vocab),
        "sql_vocab_size": len(sql_vocab)
    }


def compute_stats_after(nl_list, sql_list, add_prefix=False):
    tokenizer = T5TokenizerFast.from_pretrained("google-t5/t5-small")

    nl_token_lens = []
    sql_token_lens = []

    for nl in nl_list:
        if add_prefix:
            nl = PREFIX + nl

        ids = tokenizer.encode(
            nl,
            truncation=True,
            max_length=512
        )
        nl_token_lens.append(len(ids))

    for sql in sql_list:
        ids = tokenizer.encode(
            sql,
            truncation=True,
            max_length=512
        )
        sql_token_lens.append(len(ids))

    return {
        "mean_nl_tokens": sum(nl_token_lens) / len(nl_token_lens),
        "mean_sql_tokens": sum(sql_token_lens) / len(sql_token_lens),
        "t5_vocab_size": tokenizer.vocab_size
    }


if __name__ == "__main__":
    data_folder = "data"
    train_x, train_y, dev_x, dev_y, _ = load_prompting_data(data_folder)

    print("=== BEFORE PREPROCESSING ===")
    print("Train:", compute_stats_before(train_x, train_y))
    print("Dev:", compute_stats_before(dev_x, dev_y))

    print("\n=== AFTER PREPROCESSING (NO PREFIX) ===")
    print("Train:", compute_stats_after(train_x, train_y, add_prefix=False))
    print("Dev:", compute_stats_after(dev_x, dev_y, add_prefix=False))

    print("\n=== AFTER PREPROCESSING (WITH PREFIX) ===")
    print("Train:", compute_stats_after(train_x, train_y, add_prefix=True))
    print("Dev:", compute_stats_after(dev_x, dev_y, add_prefix=True))

