"""
Extract a HuggingFace-format RoBERTa encoder from the spaCy `en_core_web_trf`
model package (spacy-curated-transformers serialisation).

This is needed because the sandbox this experiment runs in has no access to the
HuggingFace model hub; the spaCy model wheel (GitHub release) bundles the full
RoBERTa-base weights and its byte-level BPE vocabulary, which we convert to the
standard HF layout so that `transformers` can load them offline.

NOTE: the encoder shipped with en_core_web_trf is RoBERTa-base further trained
inside spaCy's OntoNotes pipeline; it is therefore *not* bit-identical to the
original `roberta-base` checkpoint. We refer to it as RoBERTa-base (spaCy).
"""
import io, json, os, argparse
import msgpack, torch
from transformers import RobertaConfig, RobertaModel

def load_blob(path):
    with open(path, "rb") as f:
        return msgpack.unpackb(f.read(), raw=False, strict_map_key=False)

def curated_to_hf(sd, n_layers=12, hidden=768):
    new = {}
    p = "curated_encoder."
    new["embeddings.word_embeddings.weight"] = sd[p + "embeddings.inner.word_embeddings.weight"]
    new["embeddings.token_type_embeddings.weight"] = sd[p + "embeddings.inner.token_type_embeddings.weight"]
    new["embeddings.position_embeddings.weight"] = sd[p + "embeddings.inner.position_embeddings.weight"]
    new["embeddings.LayerNorm.weight"] = sd[p + "embeddings.inner.layer_norm.weight"]
    new["embeddings.LayerNorm.bias"] = sd[p + "embeddings.inner.layer_norm.bias"]
    for i in range(n_layers):
        c = f"{p}layers.{i}."
        h = f"encoder.layer.{i}."
        w = sd[c + "mha.input.weight"]; b = sd[c + "mha.input.bias"]
        q, k, v = w[:hidden], w[hidden:2 * hidden], w[2 * hidden:]
        qb, kb, vb = b[:hidden], b[hidden:2 * hidden], b[2 * hidden:]
        new[h + "attention.self.query.weight"] = q
        new[h + "attention.self.query.bias"] = qb
        new[h + "attention.self.key.weight"] = k
        new[h + "attention.self.key.bias"] = kb
        new[h + "attention.self.value.weight"] = v
        new[h + "attention.self.value.bias"] = vb
        new[h + "attention.output.dense.weight"] = sd[c + "mha.output.weight"]
        new[h + "attention.output.dense.bias"] = sd[c + "mha.output.bias"]
        new[h + "attention.output.LayerNorm.weight"] = sd[c + "attn_output_layernorm.weight"]
        new[h + "attention.output.LayerNorm.bias"] = sd[c + "attn_output_layernorm.bias"]
        new[h + "intermediate.dense.weight"] = sd[c + "ffn.intermediate.weight"]
        new[h + "intermediate.dense.bias"] = sd[c + "ffn.intermediate.bias"]
        new[h + "output.dense.weight"] = sd[c + "ffn.output.weight"]
        new[h + "output.dense.bias"] = sd[c + "ffn.output.bias"]
        new[h + "output.LayerNorm.weight"] = sd[c + "ffn_output_layernorm.weight"]
        new[h + "output.LayerNorm.bias"] = sd[c + "ffn_output_layernorm.bias"]
    return new

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spacy_model", default="/home/claude/models/en_core_web_trf/en_core_web_trf-3.7.3/transformer/model")
    ap.add_argument("--out", default="/home/claude/models/roberta-base-spacy")
    args = ap.parse_args()

    d = load_blob(args.spacy_model)
    shim = d["shims"][6][0]
    shim = msgpack.unpackb(shim, raw=False, strict_map_key=False)
    sd = torch.load(io.BytesIO(shim["state"]), map_location="cpu", weights_only=True)
    hf_sd = curated_to_hf(sd)

    cfg = RobertaConfig(vocab_size=50265, hidden_size=768, num_hidden_layers=12,
                        num_attention_heads=12, intermediate_size=3072,
                        max_position_embeddings=514, type_vocab_size=1,
                        layer_norm_eps=1e-5, pad_token_id=1, bos_token_id=0, eos_token_id=2)
    model = RobertaModel(cfg, add_pooling_layer=False)
    missing, unexpected = model.load_state_dict(hf_sd, strict=False)
    missing = [m for m in missing if "position_ids" not in m]
    assert not missing and not unexpected, (missing, unexpected)

    os.makedirs(args.out, exist_ok=True)
    model.save_pretrained(args.out)

    bp = msgpack.unpackb(d["attrs"][3]["byte_bpe_processor"], raw=False, strict_map_key=False)
    with open(os.path.join(args.out, "vocab.json"), "w") as f:
        json.dump(bp["vocab"], f)
    with open(os.path.join(args.out, "merges.txt"), "w") as f:
        f.write("#version: 0.2\n")
        for a, b in bp["merges"]:
            f.write(f"{a} {b}\n")

    from tokenizers import Tokenizer, models, pre_tokenizers, processors, decoders
    from transformers import PreTrainedTokenizerFast
    merges = [tuple(m) for m in bp["merges"]]
    tk = Tokenizer(models.BPE(bp["vocab"], merges, unk_token=None))
    tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True, use_regex=True)
    tk.decoder = decoders.ByteLevel()
    tk.post_processor = processors.TemplateProcessing(
        single="<s> $A </s>", pair="<s> $A </s> </s> $B </s>",
        special_tokens=[("<s>", 0), ("</s>", 2)])
    tok = PreTrainedTokenizerFast(tokenizer_object=tk, bos_token="<s>", eos_token="</s>",
                                  sep_token="</s>", cls_token="<s>", unk_token="<unk>",
                                  pad_token="<pad>", mask_token="<mask>", model_max_length=512)
    tok.save_pretrained(args.out)
    print("saved ->", args.out)

    # sanity check: contextual polysemy behaviour
    import torch.nn.functional as F
    model.eval()
    sents = ["I sat on the river bank and watched the water .",
             "She deposited the cheque at the bank yesterday .",
             "The bank raised interest rates on savings accounts ."]
    vecs = []
    for s in sents:
        enc = tok(s, return_tensors="pt", return_offsets_mapping=True)
        idx = [i for i, (a, b) in enumerate(enc["offset_mapping"][0].tolist())
               if s[a:b].strip().lower() == "bank"][0]
        with torch.no_grad():
            out = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"]).last_hidden_state
        vecs.append(out[0, idx])
    print("cos(river-bank, cheque-bank) =", round(float(F.cosine_similarity(vecs[0], vecs[1], dim=0)), 3))
    print("cos(cheque-bank, rates-bank) =", round(float(F.cosine_similarity(vecs[1], vecs[2], dim=0)), 3))

if __name__ == "__main__":
    main()
