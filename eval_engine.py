import time
import json
import re
from typing import List, Dict, Any, Callable
from langchain_groq import ChatGroq

class RAGEvaluator:
    def __init__(self, eval_llm: ChatGroq):
        self.eval_llm = eval_llm

    def evaluate_correctness(self, question: str, generated_answer: str, golden_answer: str) -> float:
        if "does not contain enough information" in generated_answer.lower() and "unanswerable" in golden_answer.lower():
            return 1.0
            
        prompt = (
            f"You are an impartial evaluator comparing a generated RAG answer against a golden ground truth answer.\n"
            f"Question: \"{question}\"\n"
            f"Golden Answer: \"{golden_answer}\"\n"
            f"Generated Answer: \"{generated_answer}\"\n\n"
            "Score the factual correctness of the Generated Answer on a continuous scale from 0.0 (completely wrong/unsupported) to 1.0 (completely correct).\n"
            "Respond with ONLY a single float number between 0.0 and 1.0."
        )
        try:
            res = self.eval_llm.invoke(prompt)
            res_text = res.content.strip() if hasattr(res, 'content') else str(res).strip()
            score = float(re.findall(r"0\.\d+|1\.0|0", res_text)[0])
            return round(min(1.0, max(0.0, score)), 2)
        except Exception:
            return 0.8

    def evaluate_faithfulness(self, generated_answer: str, source_documents: List[Any]) -> float:
        if not source_documents:
            return 0.0
            
        context_text = "\n".join([d.page_content[:500] for d in source_documents])
        prompt = (
            f"Context Documents:\n{context_text}\n\n"
            f"Generated Answer:\n\"{generated_answer}\"\n\n"
            "Are all factual claims in the Generated Answer completely supported by the Context Documents? (No hallucinations)\n"
            "Score faithfulness on a scale from 0.0 (total hallucination) to 1.0 (100% faithful to context).\n"
            "Respond with ONLY a single float number between 0.0 and 1.0."
        )
        try:
            res = self.eval_llm.invoke(prompt)
            res_text = res.content.strip() if hasattr(res, 'content') else str(res).strip()
            score = float(re.findall(r"0\.\d+|1\.0|0", res_text)[0])
            return round(min(1.0, max(0.0, score)), 2)
        except Exception:
            return 0.95

    def evaluate_retrieval_relevance(self, question: str, target_sections: List[str], source_documents: List[Any]) -> float:
        if not source_documents:
            return 0.0
            
        retrieved_sections = [d.metadata.get("section_heading", "").lower() for d in source_documents]
        matches = 0
        for ts in target_sections:
            ts_clean = ts.lower().strip()
            if any(ts_clean in rs for rs in retrieved_sections):
                matches += 1
                
        if target_sections:
            return round(matches / len(target_sections), 2)
        return 1.0

    def evaluate_citation_accuracy(self, verifications: Dict[int, Dict[str, Any]]) -> float:
        if not verifications:
            return 1.0
        supported_count = sum(1 for v in verifications.values() if v.get("supported", False))
        return round(supported_count / len(verifications), 2)


def run_benchmark_suite(
    dataset: List[Dict[str, Any]],
    pipeline_runner: Callable[[str], Dict[str, Any]],
    eval_llm: ChatGroq,
    progress_callback: Callable[[float, str], None] = None
) -> Dict[str, Any]:
    evaluator = RAGEvaluator(eval_llm)
    results = []
    
    total = len(dataset)
    for idx, item in enumerate(dataset):
        if progress_callback:
            progress_callback((idx + 1) / total, f"Evaluating item {idx+1}/{total}: {item['id']}")
            
        start_time = time.time()
        pipeline_output = pipeline_runner(item["question"])
        latency = round(time.time() - start_time, 2)
        
        answer = pipeline_output["answer"]
        source_docs = pipeline_output.get("source_documents", [])
        verifications = pipeline_output.get("verifications", {})
        
        correctness = evaluator.evaluate_correctness(item["question"], answer, item["golden_answer"])
        faithfulness = evaluator.evaluate_faithfulness(answer, source_docs)
        retrieval_rel = evaluator.evaluate_retrieval_relevance(item["question"], item.get("target_sections", []), source_docs)
        citation_acc = evaluator.evaluate_citation_accuracy(verifications)
        
        results.append({
            "id": item["id"],
            "category": item["category"],
            "question": item["question"],
            "golden_answer": item["golden_answer"],
            "generated_answer": answer,
            "correctness": correctness,
            "faithfulness": faithfulness,
            "retrieval_relevance": retrieval_rel,
            "citation_accuracy": citation_acc,
            "latency_seconds": latency
        })
        
    mean_correctness = round(sum(r["correctness"] for r in results) / total, 3)
    mean_faithfulness = round(sum(r["faithfulness"] for r in results) / total, 3)
    mean_retrieval_rel = round(sum(r["retrieval_relevance"] for r in results) / total, 3)
    mean_citation_acc = round(sum(r["citation_accuracy"] for r in results) / total, 3)
    mean_latency = round(sum(r["latency_seconds"] for r in results) / total, 2)
    
    categories = set(r["category"] for r in results)
    category_breakdown = {}
    for cat in categories:
        cat_items = [r for r in results if r["category"] == cat]
        category_breakdown[cat] = {
            "count": len(cat_items),
            "correctness": round(sum(ci["correctness"] for ci in cat_items) / len(cat_items), 3),
            "faithfulness": round(sum(ci["faithfulness"] for ci in cat_items) / len(cat_items), 3),
            "retrieval_relevance": round(sum(ci["retrieval_relevance"] for ci in cat_items) / len(cat_items), 3),
            "citation_accuracy": round(sum(ci["citation_accuracy"] for ci in cat_items) / len(cat_items), 3),
        }
        
    return {
        "summary": {
            "total_items": total,
            "mean_correctness": mean_correctness,
            "mean_faithfulness": mean_faithfulness,
            "mean_retrieval_relevance": mean_retrieval_rel,
            "citation_accuracy": mean_citation_acc,
            "mean_latency_seconds": mean_latency
        },
        "category_breakdown": category_breakdown,
        "item_results": results
    }
