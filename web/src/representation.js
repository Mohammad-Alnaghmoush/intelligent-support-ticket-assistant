/* Single-query XLM-R tokenization with Python's 256-token special-token budget. */
export function queryInputs(tokenizer, Tensor, text, maximum = 256) {
  const content = tokenizer.encode("query: " + text, {
    add_special_tokens: false,
  });
  // Transformers.js 3's generic truncation slices the postprocessed sequence.
  // Reserve both special tokens before truncating, matching the Python tokenizer.
  const special = tokenizer.encode("", { add_special_tokens: true });
  if (special.length !== 2)
    throw new Error("Unexpected special tokens for the pinned encoder.");
  const ids = [special[0], ...content.slice(0, maximum - 2), special[1]];
  const shape = [1, ids.length];
  return {
    input_ids: new Tensor("int64", BigInt64Array.from(ids, BigInt), shape),
    attention_mask: new Tensor(
      "int64",
      new BigInt64Array(ids.length).fill(1n),
      shape,
    ),
    token_type_ids: new Tensor("int64", new BigInt64Array(ids.length), shape),
  };
}
