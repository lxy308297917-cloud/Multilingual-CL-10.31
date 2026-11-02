"""One Base tokenizer/template for every model; explicit article-only truncation."""
def render(tokenizer,row,max_input_tokens):
 original_article_tokens=None;truncated=False
 if 'messages' in row:messages=row['messages']
 else:
  article=tokenizer.encode(row['source'],add_special_tokens=False);original_article_tokens=len(article);template=row['article_prompt_template'];fixed=tokenizer.apply_chat_template([{'role':'user','content':template.format(article='')}],tokenize=True,add_generation_prompt=True,enable_thinking=False)
  budget=max_input_tokens-len(fixed)
  if budget<1:raise ValueError('No room for article in context')
  truncated=len(article)>budget;article=article[:budget]
  while True:
   text=tokenizer.decode(article,skip_special_tokens=False);messages=[{'role':'user','content':template.format(article=text)}];tokens=tokenizer.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,enable_thinking=False)
   if len(tokens)<=max_input_tokens:break
   article=article[:-(len(tokens)-max_input_tokens)]
   truncated=True
 prompt=tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False);tokens=tokenizer.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,enable_thinking=False)
 if len(tokens)>max_input_tokens:raise ValueError('Overlong non-article prompt')
 return dict(prompt=prompt,prompt_token_ids=tokens,input_tokens=len(tokens),article_tokens_before_truncation=original_article_tokens,article_truncated=truncated)
