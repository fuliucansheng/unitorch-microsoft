# -*- coding: utf-8 -*-
"""
QwenVL Inference with LogProbs Support
Extended version with top-K logprobs output for token-level analysis.
"""
import argparse
import os
import re
import yaml
import pandas as pd
import time
import json
from typing import List, Dict, Any, Optional, Tuple
from tqdm import tqdm

import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor
from vllm import LLM, SamplingParams
from PIL import Image
import requests
from io import BytesIO

os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'


def load_prompts(prompt_file: str) -> Dict[str, Any]:
    """Load prompts from YAML file."""
    with open(prompt_file, 'r', encoding='utf-8') as f:
        prompts = yaml.safe_load(f)
    return prompts


def estimate_prompt_tokens(text: str, processor: Any) -> int:
    """Estimate the number of tokens in a text prompt."""
    try:
        tokens = processor.tokenizer.encode(text, add_special_tokens=True)
        return len(tokens)
    except Exception:
        return len(text) // 4


def resize_image_if_needed(image_url: str, max_image_size: int = 448) -> Optional[str]:
    """Load and resize image if needed."""
    try:
        if image_url.startswith('http://') or image_url.startswith('https://'):
            response = requests.get(image_url, timeout=5)
            img = Image.open(BytesIO(response.content))
        else:
            img = Image.open(image_url)

        width, height = img.size

        if max(width, height) <= max_image_size:
            return image_url

        scale = max_image_size / max(width, height)
        new_width = int(width * scale)
        new_height = int(height * scale)

        img_resized = img.resize((new_width, new_height), Image.LANCZOS)

        import tempfile
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
        img_resized.save(temp_file.name, 'JPEG', quality=95)
        temp_file.close()

        return temp_file.name

    except Exception:
        return image_url


def prepare_inputs_for_vllm(messages: List[Dict], processor: Any) -> Optional[Dict[str, Any]]:
    """Prepare inputs for vLLM inference."""
    try:
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        image_inputs, video_inputs, video_kwargs = process_vision_info(
            messages,
            image_patch_size=processor.image_processor.patch_size,
            return_video_kwargs=True,
            return_video_metadata=True
        )

        mm_data = {}
        if image_inputs is not None:
            mm_data['image'] = image_inputs
        if video_inputs is not None:
            mm_data['video'] = video_inputs

        return {
            'prompt': text,
            'multi_modal_data': mm_data,
            'mm_processor_kwargs': video_kwargs
        }
    except Exception as e:
        print(f"Warning: Failed to prepare inputs - {str(e)}")
        return None


def create_messages_from_row(row: pd.Series, prompt_template: str, max_doc_length: int = 8000,
                            max_image_size: Optional[int] = None) -> List[Dict]:
    """Create message list from a data row."""
    doc_text = str(row['doc']) if pd.notna(row['doc']) else ''
    if len(doc_text) > max_doc_length:
        doc_text = doc_text[:max_doc_length] + '... [truncated]'

    text_prompt = prompt_template.format(
        FinalUrl=row['FinalUrl'],
        ImgUrl=row['ImgUrl'],
        doc=doc_text
    )

    image_url = row['ImgUrl']
    if max_image_size is not None:
        resized_url = resize_image_if_needed(image_url, max_image_size)
        if resized_url:
            image_url = resized_url

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image_url,
                },
                {"type": "text", "text": text_prompt},
            ],
        }
    ]

    return messages


def parse_result(output_text: str) -> Dict[str, str]:
    """Parse model output to extract Think and Result tags."""
    result = {
        'think': '',
        'result': ''
    }

    think_match = re.search(r'<Think>(.*?)</Think>', output_text, re.DOTALL | re.IGNORECASE)
    if think_match:
        result['think'] = think_match.group(1).strip()

    result_match = re.search(r'<Result>(.*?)</Result>', output_text, re.DOTALL | re.IGNORECASE)
    if result_match:
        result['result'] = result_match.group(1).strip()
    else:
        if not think_match:
            result['result'] = output_text.strip()

    return result


def extract_logprobs_from_output(output, top_k: int = 5) -> Dict[str, Any]:
    """
    Extract logprobs information from vLLM output.

    Args:
        output: vLLM RequestOutput object
        top_k: Number of top logprobs to extract per token

    Returns:
        Dictionary with logprobs information
    """
    logprobs_data = {
        'tokens': [],
        'token_logprobs': [],
        'top_logprobs': []
    }

    if not hasattr(output, 'outputs') or len(output.outputs) == 0:
        return logprobs_data

    completion_output = output.outputs[0]

    # Check if logprobs are available
    if not hasattr(completion_output, 'logprobs') or completion_output.logprobs is None:
        return logprobs_data

    # Extract logprobs for each token
    for token_logprobs_dict in completion_output.logprobs:
        if token_logprobs_dict is None:
            continue

        # Get the most likely token and its logprob
        if len(token_logprobs_dict) > 0:
            # token_logprobs_dict is a dict: {token_id: Logprob object}
            # Sort by logprob (descending)
            sorted_tokens = sorted(
                token_logprobs_dict.items(),
                key=lambda x: x[1].logprob if hasattr(x[1], 'logprob') else x[1],
                reverse=True
            )[:top_k]

            # Most likely token (first in sorted list)
            if sorted_tokens:
                top_token_id, top_logprob_obj = sorted_tokens[0]

                # Extract token string
                if hasattr(top_logprob_obj, 'decoded_token'):
                    token_str = top_logprob_obj.decoded_token
                else:
                    token_str = str(top_token_id)

                # Extract logprob value
                if hasattr(top_logprob_obj, 'logprob'):
                    token_logprob = top_logprob_obj.logprob
                else:
                    token_logprob = float(top_logprob_obj)

                logprobs_data['tokens'].append(token_str)
                logprobs_data['token_logprobs'].append(token_logprob)

                # Extract top-k alternatives
                top_k_dict = {}
                for token_id, logprob_obj in sorted_tokens:
                    if hasattr(logprob_obj, 'decoded_token'):
                        alt_token = logprob_obj.decoded_token
                    else:
                        alt_token = str(token_id)

                    if hasattr(logprob_obj, 'logprob'):
                        alt_logprob = logprob_obj.logprob
                    else:
                        alt_logprob = float(logprob_obj)

                    top_k_dict[alt_token] = alt_logprob

                logprobs_data['top_logprobs'].append(top_k_dict)

    return logprobs_data


def calculate_score_from_first_token(logprobs_data: Dict[str, Any]) -> Optional[float]:
    """
    Calculate Good vs Bad score from the first token's logprobs.
    Score = P(Good) - P(Bad)

    Args:
        logprobs_data: Extracted logprobs data

    Returns:
        Score (Good - Bad), or None if not available
    """
    if not logprobs_data['top_logprobs'] or len(logprobs_data['top_logprobs']) == 0:
        return None

    first_token_logprobs = logprobs_data['top_logprobs'][0]

    # Get logprobs for 'Good' and 'Bad' (case-insensitive)
    good_logprob = None
    fair_logprob = None
    bad_logprob = None

    for token, logprob in first_token_logprobs.items():
        token_lower = token.strip().lower()
        if token_lower == 'good':
            good_logprob = logprob
        elif token_lower == 'fair':
            fair_logprob = logprob
        elif token_lower == 'bad':
            bad_logprob = logprob

    #If both 'Good' and 'Fair' found, use the higher of the two as the "good" logprob
    if good_logprob is not None and fair_logprob is not None:
        good_logprob = max(good_logprob, fair_logprob)
    if good_logprob is None and fair_logprob is not None:
        good_logprob = fair_logprob

    # If both found, calculate score
    if good_logprob is not None and bad_logprob is not None:
        return good_logprob - bad_logprob

    # If only one found, use -100 as the default for the missing one
    if good_logprob is not None:
        return good_logprob - (-100.0)
    if bad_logprob is not None:
        return (-100.0) - bad_logprob

    return None


def evaluate_predictions(df: pd.DataFrame) -> Dict[str, Any]:
    """Evaluate predictions with binary classification."""
    df['Label_normalized'] = df['Label'].str.lower().str.strip()
    df['Prediction_normalized'] = df['Prediction'].str.lower().str.strip()

    df['Label_binary'] = df['Label_normalized'].replace('fair', 'good')
    df['Prediction_binary'] = df['Prediction_normalized'].replace('fair', 'good')

    correct = (df['Label_binary'] == df['Prediction_binary']).sum()
    total = len(df)
    accuracy = correct / total if total > 0 else 0.0

    binary_classes = ['good', 'bad']

    confusion_matrix = {}
    for true_label in binary_classes:
        confusion_matrix[true_label] = {}
        for pred_label in binary_classes:
            count = ((df['Label_binary'] == true_label) &
                    (df['Prediction_binary'] == pred_label)).sum()
            confusion_matrix[true_label][pred_label] = int(count)

    class_metrics = {}

    for label in binary_classes:
        tp = ((df['Label_binary'] == label) & (df['Prediction_binary'] == label)).sum()
        fp = ((df['Label_binary'] != label) & (df['Prediction_binary'] == label)).sum()
        fn = ((df['Label_binary'] == label) & (df['Prediction_binary'] != label)).sum()
        tn = ((df['Label_binary'] != label) & (df['Prediction_binary'] != label)).sum()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        label_total = (df['Label_binary'] == label).sum()

        class_metrics[label] = {
            'total_in_ground_truth': int(label_total),
            'true_positives': int(tp),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1)
        }

    metrics = {
        'overall_accuracy': float(accuracy),
        'total_samples': int(total),
        'correct_predictions': int(correct),
        'confusion_matrix': confusion_matrix,
        'class_metrics': class_metrics,
        'note': 'Binary classification: fair labels are treated as good'
    }

    return metrics


def main():
    parser = argparse.ArgumentParser(
        description='QwenVL Inference with LogProbs Support'
    )
    parser.add_argument('--input_file', type=str, default='../data/Internal100WithSD.tsv')
    parser.add_argument('--output_folder', type=str, default='../output/test/')
    parser.add_argument('--model_path', type=str, default='/data/xiaoyukou/ckpts/Qwen3-VL-2B-Instruct')
    parser.add_argument('--prompt_file', type=str, default='prompts.yaml')
    parser.add_argument('--prompt_name', type=str, default='relevance')
    parser.add_argument('--batch_size', type=int, default=40)
    parser.add_argument('--max_tokens', type=int, default=2048)
    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--top_p', type=float, default=0.8)
    parser.add_argument('--top_k', type=int, default=20)
    parser.add_argument('--repetition_penalty', type=float, default=1.0)
    parser.add_argument('--presence_penalty', type=float, default=1.5)
    parser.add_argument('--seed', type=int, default=3407)
    parser.add_argument('--tensor_parallel_size', type=int, default=None)
    parser.add_argument('--enable_expert_parallel', action='store_true')
    parser.add_argument('--max_model_len', type=int, default=32000)
    parser.add_argument('--gpu_memory_utilization', type=float, default=0.95)
    parser.add_argument('--max_doc_length', type=int, default=8000)
    parser.add_argument('--max_image_size', type=int, default=None)
    parser.add_argument('--output_file_name', type=str, default='')
    parser.add_argument('--logprobs', type=int, default=5,
                        help='Number of top logprobs to return for each token (default: 5)')

    args = parser.parse_args()

    os.makedirs(args.output_folder, exist_ok=True)

    if args.output_file_name:
        base_name = args.output_file_name
    else:
        input_basename = os.path.splitext(os.path.basename(args.input_file))[0]
        model_name = os.path.basename(args.model_path.rstrip('/'))
        base_name = f"{input_basename}_{args.prompt_name}_{model_name}"

    output_file = os.path.join(args.output_folder, f"{base_name}_predictions.tsv")
    eval_file = os.path.join(args.output_folder, f"{base_name}_evaluation.json")
    logprobs_file = os.path.join(args.output_folder, f"{base_name}_logprobs.jsonl")

    print(f"Loading prompts from {args.prompt_file}...")
    prompts = load_prompts(args.prompt_file)

    if args.prompt_name not in prompts['prompts']:
        raise ValueError(f"Prompt '{args.prompt_name}' not found in {args.prompt_file}")

    prompt_config = prompts['prompts'][args.prompt_name]
    prompt_template = prompt_config['user']

    print(f"Loading input data from {args.input_file}...")
    df = pd.read_csv(args.input_file, sep='\t')
    print(f"Loaded {len(df)} samples")

    required_columns = ['FinalUrl', 'ImgUrl', 'Label', 'doc']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    original_count = len(df)
    df = df[df['Label'].str.lower().str.strip() != "can't load"].reset_index(drop=True)
    filtered_count = original_count - len(df)
    if filtered_count > 0:
        print(f"Filtered out {filtered_count} samples with 'can't load' label")

    empty_doc_count = df['doc'].isna().sum() + (df['doc'].astype(str).str.strip() == '').sum()
    empty_doc_ratio = empty_doc_count / len(df) if len(df) > 0 else 0.0
    print(f"Empty doc entries: {empty_doc_count}/{len(df)} ({empty_doc_ratio:.2%})")
    print(f"Processing {len(df)} valid samples")

    print(f"Loading model from {args.model_path}...")
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)

    tensor_parallel_size = args.tensor_parallel_size or torch.cuda.device_count()

    llm_kwargs = {
        'model': args.model_path,
        'mm_encoder_tp_mode': 'data',
        'tensor_parallel_size': tensor_parallel_size,
        'seed': args.seed,
        'max_model_len': args.max_model_len,
        'gpu_memory_utilization': args.gpu_memory_utilization,
        'disable_log_stats': True
    }

    if args.enable_expert_parallel:
        llm_kwargs['enable_expert_parallel'] = True
        print("Expert parallelism enabled (MoE mode)")

    print(f"Initializing LLM with max_model_len={args.max_model_len}, gpu_memory_utilization={args.gpu_memory_utilization}")
    llm = LLM(**llm_kwargs)

    # Enable logprobs in sampling parameters
    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        top_p=args.top_p,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
        presence_penalty=args.presence_penalty,
        seed=args.seed,
        stop_token_ids=[],
        logprobs=args.logprobs,  # Enable logprobs output
    )

    print(f"Sampling parameters: temperature={args.temperature}, top_p={args.top_p}, "
          f"top_k={args.top_k}, max_tokens={args.max_tokens}, seed={args.seed}, logprobs={args.logprobs}")

    print(f"Processing {len(df)} samples with batch size {args.batch_size}...")

    successful_indices = []
    all_predictions = []
    all_thoughts = []
    all_raw_outputs = []
    all_scores = []
    failed_indices = []

    total_start_time = time.time()
    total_inference_time = 0.0

    # Open logprobs file for writing
    logprobs_fh = open(logprobs_file, 'w', encoding='utf-8')

    for batch_start in tqdm(range(0, len(df), args.batch_size)):
        batch_end = min(batch_start + args.batch_size, len(df))
        batch_df = df.iloc[batch_start:batch_end]

        batch_messages = []
        batch_original_indices = []

        for idx, (df_idx, row) in enumerate(batch_df.iterrows()):
            messages = create_messages_from_row(
                row,
                prompt_template,
                max_doc_length=args.max_doc_length,
                max_image_size=args.max_image_size
            )
            batch_messages.append(messages)
            batch_original_indices.append(df_idx)

        batch_inputs = []
        valid_indices = []
        for idx, (msg, original_idx) in enumerate(zip(batch_messages, batch_original_indices)):
            prepared_input = prepare_inputs_for_vllm(msg, processor)
            if prepared_input is not None:
                batch_inputs.append(prepared_input)
                valid_indices.append(original_idx)
            else:
                failed_indices.append(original_idx)

        if batch_inputs:
            batch_start_time = time.time()
            outputs = llm.generate(batch_inputs, sampling_params=sampling_params)
            batch_end_time = time.time()
            batch_inference_time = batch_end_time - batch_start_time
            total_inference_time += batch_inference_time

            for output, original_idx in zip(outputs, valid_indices):
                raw_text = output.outputs[0].text
                parsed = parse_result(raw_text)

                # Extract logprobs
                logprobs_data = extract_logprobs_from_output(output, top_k=args.logprobs)

                # Calculate score from first token
                score = calculate_score_from_first_token(logprobs_data)

                # Save logprobs to JSONL file
                logprobs_entry = {
                    'index': int(original_idx),
                    'prediction': parsed['result'],
                    'score': score,
                    'logprobs': logprobs_data
                }
                logprobs_fh.write(json.dumps(logprobs_entry, ensure_ascii=False) + '\n')

                successful_indices.append(original_idx)
                all_raw_outputs.append(raw_text)
                all_thoughts.append(parsed['think'])
                all_predictions.append(parsed['result'])
                all_scores.append(score if score is not None else float('nan'))

    logprobs_fh.close()

    total_end_time = time.time()
    total_elapsed_time = total_end_time - total_start_time

    result_df = df.loc[successful_indices].copy()
    result_df['Prediction'] = all_predictions
    result_df['Think'] = all_thoughts
    result_df['RawOutput'] = all_raw_outputs
    result_df['Score'] = all_scores

    output_columns = ['FinalUrl', 'ImgUrl', 'Label', 'Prediction', 'Score', 'RawOutput']
    result_df_output = result_df[output_columns]

    print(f"Saving predictions to {output_file}...")
    result_df_output.to_csv(output_file, sep='\t', index=False)

    print("Evaluating predictions...")
    successful_samples = len(result_df)
    if successful_samples > 0:
        metrics = evaluate_predictions(result_df)
    else:
        metrics = {
            'overall_accuracy': 0.0,
            'total_samples': 0,
            'correct_predictions': 0,
            'confusion_matrix': {},
            'class_metrics': {}
        }

    avg_inference_time = total_inference_time / successful_samples if successful_samples > 0 else 0.0

    metrics['latency_stats'] = {
        'total_elapsed_time_seconds': round(total_elapsed_time, 2),
        'total_inference_time_seconds': round(total_inference_time, 2),
        'average_inference_time_per_sample_seconds': round(avg_inference_time, 4),
        'samples_per_second': round(successful_samples / total_inference_time, 2) if total_inference_time > 0 else 0.0
    }

    metrics['sample_stats'] = {
        'original_total_samples': original_count,
        'cant_load_samples': filtered_count,
        'valid_samples': len(df),
        'successful_samples': successful_samples,
        'failed_during_processing': len(failed_indices),
        'success_rate': round(successful_samples / len(df), 4) if len(df) > 0 else 0.0,
        'empty_doc_count': int(empty_doc_count),
        'empty_doc_ratio': round(empty_doc_ratio, 4)
    }

    if failed_indices:
        metrics['failed_indices'] = failed_indices

    print(f"Saving evaluation results to {eval_file}...")
    with open(eval_file, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)
    print(f"Total Samples in Input: {len(df)}")
    print(f"Successfully Processed: {successful_samples} | Failed to Load: {len(failed_indices)}")
    if len(failed_indices) > 0:
        print(f"Failed Indices: {failed_indices[:10]}{'...' if len(failed_indices) > 10 else ''}")
    print(f"Success Rate: {metrics['sample_stats']['success_rate']:.2%}")
    print(f"Empty Doc Entries: {metrics['sample_stats']['empty_doc_count']} ({metrics['sample_stats']['empty_doc_ratio']:.2%})")

    print(f"\nLatency Statistics:")
    print(f"  Total Elapsed Time: {metrics['latency_stats']['total_elapsed_time_seconds']:.2f}s")
    print(f"  Total Inference Time: {metrics['latency_stats']['total_inference_time_seconds']:.2f}s")
    print(f"  Average Time per Sample: {metrics['latency_stats']['average_inference_time_per_sample_seconds']:.4f}s")
    print(f"  Throughput: {metrics['latency_stats']['samples_per_second']:.2f} samples/s")

    print(f"\nOverall Accuracy: {metrics['overall_accuracy']:.4f} ({metrics['correct_predictions']}/{metrics['total_samples']})")

    if 'confusion_matrix' in metrics and metrics['confusion_matrix']:
        print(f"\nConfusion Matrix:")
        print(f"{'':>10} ", end='')
        labels = sorted(metrics['confusion_matrix'].keys())
        for label in labels:
            print(f"{label:>8}", end='')
        print()
        for true_label in labels:
            print(f"{true_label:>10} ", end='')
            for pred_label in labels:
                count = metrics['confusion_matrix'][true_label].get(pred_label, 0)
                print(f"{count:>8}", end='')
            print()

    print(f"\nPer-Class Metrics:")
    print(f"{'Class':>8} {'Total':>8} {'TP':>8} {'FP':>8} {'FN':>8} {'Precision':>12} {'Recall':>12} {'F1':>12}")
    print("-" * 80)
    for label in sorted(metrics['class_metrics'].keys()):
        cm = metrics['class_metrics'][label]
        print(f"{label.upper():>8} {cm['total_in_ground_truth']:>8} {cm['true_positives']:>8} "
              f"{cm['false_positives']:>8} {cm['false_negatives']:>8} "
              f"{cm['precision']:>12.4f} {cm['recall']:>12.4f} {cm['f1_score']:>12.4f}")

    print(f"\nLogProbs saved to: {logprobs_file}")
    print("=" * 80)


if __name__ == '__main__':
    main()
