"""Official XQuAD/SQuAD v1.1 English normalization; max over all gold answers."""
from transformers.data.metrics.squad_metrics import compute_exact,compute_f1
def process_results(doc,results):
 prediction=results[0];answers=doc['answers']['text'];assert answers
 return dict(exact_match=max(compute_exact(a,prediction) for a in answers),f1=max(compute_f1(a,prediction) for a in answers))
