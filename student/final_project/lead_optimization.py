#!/usr/bin/env python3
"""
Prompting-Based Lead Optimization with Open LLMs -- Evaluation Framework

Systematically evaluates prompting strategies (zero-shot, few-shot, chain-of-thought)
on general-purpose and task-specific LLMs for molecular lead optimization.
Measures validity, property control, synthesizability, and diversity.
"""

import os
os.environ['DISABLE_VERSION_CHECK'] = '1'

# Fix multi-GPU issue: set multiprocessing start method
import multiprocessing
try:
    multiprocessing.set_start_method("spawn", force=True)
except RuntimeError:
    pass  # Already set

import torch
import argparse
import json
import numpy as np
import random
from typing import List, Dict, Optional
from pathlib import Path
from collections import defaultdict
import time

# vLLM imports
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from transformers import Seq2SeqTrainingArguments

# LLaMA-Factory template system
from llamafactory.data import get_template_and_fix_tokenizer
from llamafactory.hparams import get_infer_args
from llamafactory.model import load_tokenizer

# RDKit for molecular operations
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.DataStructs import FingerprintSimilarity
import rdkit.RDLogger as rkl
rkl.logger().setLevel(rkl.ERROR)

# TDC Oracle for property scoring
from tdc import Oracle


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

# SFT model instruction template (GeLLMO, LlaSMol, DrugAssist, etc.)
INSTRUCTION_TEMPLATE_SFT = """You are an expert medicinal chemist specializing in molecular optimization. You understand how structural modifications affect key molecular properties including drug-likeness, lipophilicity, synthetic accessibility, and target inhibition activities.

Your task is to modify the given molecule to adjust the specified molecular properties while keeping structural changes as minimal as possible. The modified molecule should maintain a structural similarity of at least 0.6 with the original molecule.

Input molecule: <SMILES> {input_smiles} </SMILES>
Requested modifications: {property_description}

Please provide the optimized molecule in SMILES format wrapped in <SMILES> </SMILES> tags, without any other text."""

# General-purpose model instruction template (zero-shot baseline)
INSTRUCTION_TEMPLATE_GENERAL = """You are an expert medicinal chemist specializing in molecular optimization. You understand how structural modifications affect key molecular properties including drug-likeness, lipophilicity, synthetic accessibility, and target inhibition activities.

Your task is to modify the given molecule to adjust the specified molecular properties while keeping structural changes as minimal as possible. The modified molecule should maintain a structural similarity of at least 0.6 with the original molecule.

Input molecule: {input_smiles}
Requested modifications: {property_description}

Response only with SMILES, no other text."""

# Chain-of-thought suffix appended to the general template
COT_SUFFIX = "\n\nPlease think step-by-step about what structural modifications would improve the target property, then output the final SMILES."

# Few-shot demonstration examples (successful optimizations from ZINC)
FEW_SHOT_EXAMPLES = [
    {
        "input": "O=C(O)c1ccc(NC(=O)c2ccc(Br)cc2)cc1",
        "output": "O=C(O)c1ccc(NC(=O)c2ccc(F)cc2)cc1",
    },
    {
        "input": "CC(C)Oc1ccc(CC(=O)N2CCOCC2)cc1",
        "output": "CCOc1ccc(CC(=O)N2CCOCC2)cc1",
    },
    {
        "input": "O=C(Nc1ccc(Cl)cc1)c1cc2ccccc2o1",
        "output": "O=C(Nc1ccccc1)c1cc2ccccc2o1",
    },
]

# Property description mapping
PROPERTY_DESCRIPTIONS = {
    "qed": "increase drug-likeness (QED)",
    "logp": "increase lipophilicity (LogP)",
    "jnk3": "increase JNK3 inhibition probability",
    "gsk3b": "increase GSK3\u03b2 inhibition probability",
    "drd2": "increase DRD2 inhibition probability",
    "sa": "decrease synthetic accessibility score (lower is better)",
}


class MoleculeTracker:
    """Track Oracle calls and candidate evaluation for a single source molecule."""

    def __init__(self, source_smiles: str, target_oracle_calls: int = 500):
        self.source_smiles = source_smiles
        self.target_oracle_calls = target_oracle_calls
        self.oracle_calls_made = 0
        self.candidates_evaluated = []   # all evaluated candidates
        self.valid_candidates = []       # candidates passing similarity filter
        self.generation_attempts = 0     # total generation attempts
        self.invalid_smiles_count = 0    # count of unparseable SMILES
        self.low_similarity_count = 0    # count of low-similarity outputs

    def is_complete(self) -> bool:
        """Check whether target Oracle calls have been reached."""
        return self.oracle_calls_made >= self.target_oracle_calls

    def remaining_calls(self) -> int:
        """Return the number of Oracle calls still needed."""
        return max(0, self.target_oracle_calls - self.oracle_calls_made)

    def add_candidate(self, candidate_data: Dict):
        """Record an evaluated candidate molecule."""
        self.candidates_evaluated.append(candidate_data)
        if candidate_data.get('passed_similarity', False):
            self.valid_candidates.append(candidate_data)
            self.oracle_calls_made += 1


class LeadOptimizationEvaluator:
    """Evaluate prompting strategies for molecular lead optimization.

    Supports zero-shot, few-shot, and chain-of-thought prompting on
    general-purpose LLMs, as well as zero-shot evaluation of SFT baselines.
    Reports Validity, Property Control, Synthesizability, and Diversity.
    """

    def __init__(self, args):
        self.args = args
        self.base_seed = args.seed
        self.base_temperature = args.temperature

        # Set initial random seed
        self._set_random_seed(self.base_seed)

        # Improvement-based success thresholds
        self.property_success_thresholds = {
            "qed":   {"threshold": 0.1, "direction": "increase"},
            "logp":  {"threshold": 1.0, "direction": "increase"},
            "jnk3":  {"threshold": 0.1, "direction": "increase"},
            "gsk3b": {"threshold": 0.1, "direction": "increase"},
            "drd2":  {"threshold": 0.5, "direction": "increase"},
            "sa":    {"threshold": 0.5, "direction": "decrease"},
        }

        # Absolute-value success thresholds
        self.property_absolute_thresholds = {
            "qed":   {"threshold": 0.9, "direction": "increase"},
            "logp":  {"threshold": 2.0, "direction": "increase"},
            "drd2":  {"threshold": 0.8, "direction": "increase"},
            "jnk3":  {"threshold": 0.4, "direction": "increase"},
            "sa":    {"threshold": 2.5, "direction": "decrease"},
            "gsk3b": {"threshold": 0.4, "direction": "increase"},
        }

        # Parse task properties (e.g. "qed" or "qed+logp+sa")
        self.task_properties = args.task.split('+')
        print(f"Task properties: {self.task_properties}")

        # Initialize TDC Oracles
        self.oracles = {}
        self.total_oracle_calls = 0  # global Oracle call counter

        for prop in self.task_properties:
            prop_upper = prop.upper()
            print(f"Initializing TDC Oracle for {prop_upper}...")
            self.oracles[prop] = Oracle(prop_upper)
            print(f"[OK] TDC Oracle {prop_upper} ready")

        # Initialize SA Oracle for synthesizability metric (if not already a task property)
        if "sa" not in self.oracles:
            print("Initializing TDC Oracle for SA (synthesizability metric)...")
            self.oracles["sa"] = Oracle("SA")
            print("[OK] TDC Oracle SA ready")

        # Load test data
        self._load_test_data()

        # Initialize molecule trackers
        self.molecule_trackers = {}
        for idx, mol_data in enumerate(self.test_data):
            # Support multiple data formats
            source_smiles = mol_data.get('source_smiles',
                                         mol_data.get('smiles',
                                                      mol_data.get('text', '')))
            if not source_smiles:
                print(f"Warning: No SMILES found for molecule {idx}: {mol_data}")
                continue

            self.molecule_trackers[idx] = MoleculeTracker(
                source_smiles,
                target_oracle_calls=args.target_oracle_calls,
            )

        # Initialize model and tokenizer
        self._initialize_model()

        # Batch processing configuration
        self.batch_size_per_round = args.batch_size_per_round
        self.current_round = 0

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------

    def _set_random_seed(self, seed: int):
        """Set random seed for reproducibility."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

    def _load_test_data(self):
        """Load test molecules from a JSON file."""
        print(f"Loading test data from {self.args.test_data_path}...")

        with open(self.args.test_data_path, 'r') as f:
            data = json.load(f)

        # Handle different data layouts
        if isinstance(data, list):
            self.test_data = data[:self.args.num_molecules]
        elif isinstance(data, dict) and 'data' in data:
            self.test_data = data['data'][:self.args.num_molecules]
        else:
            raise ValueError(f"Unexpected data format in {self.args.test_data_path}")

        print(f"[OK] Loaded {len(self.test_data)} test molecules")

    def _initialize_model(self):
        """Initialize the LLM, tokenizer, and chat template via LLaMA-Factory."""
        print(f"Loading tokenizer and template from {self.args.model_path}...")

        # Use get_infer_args to properly initialize parameters
        self.model_args, self.data_args, _, self.generating_args = get_infer_args(
            dict(
                model_name_or_path=self.args.model_path,
                adapter_name_or_path=self.args.adapter_path,
                template=self.args.template,
                cutoff_len=self.args.max_model_len,
                temperature=self.base_temperature,
                top_p=self.args.top_p,
                max_new_tokens=self.args.max_new_tokens,
                trust_remote_code=True,
            )
        )

        # Initialize tokenizer and template
        Seq2SeqTrainingArguments(output_dir="dummy_dir")  # required side-effect
        tokenizer_module = load_tokenizer(self.model_args)
        self.tokenizer = tokenizer_module["tokenizer"]

        # Apply chat template
        self.template = get_template_and_fix_tokenizer(self.tokenizer, self.data_args)

        if hasattr(self.template, 'mm_plugin') and self.template.mm_plugin:
            self.template.mm_plugin.expand_mm_tokens = False

        print("[OK] Tokenizer and template loaded")

        # Sampling parameters with stop tokens
        self.sampling_params = SamplingParams(
            temperature=self.base_temperature,
            top_p=self.args.top_p,
            max_tokens=self.args.max_new_tokens,
            stop_token_ids=self.template.get_stop_token_ids(self.tokenizer),
        )

        # LoRA adapter (optional)
        self.lora_request = None
        if self.model_args.adapter_name_or_path is not None:
            self.lora_request = LoRARequest(
                "default", 1, self.model_args.adapter_name_or_path[0]
            )
            print(f"[OK] LoRA adapter configured: {self.model_args.adapter_name_or_path[0]}")

        # Initialize vLLM engine
        print(f"Initializing vLLM with {self.args.num_gpus} GPU(s)...")

        vllm_kwargs = {
            "model": self.args.model_path,
            "trust_remote_code": True,
            "dtype": "half",
            "tensor_parallel_size": self.args.num_gpus,
            "disable_log_stats": True,
            "max_model_len": self.args.max_model_len,
            "gpu_memory_utilization": 0.95,
            "enable_lora": self.lora_request is not None,
        }

        if self.lora_request is not None:
            vllm_kwargs["max_lora_rank"] = 64

        try:
            self.llm = LLM(**vllm_kwargs)
            print("[OK] vLLM model loaded successfully")
        except Exception as e:
            print(f"[FAIL] Error initializing vLLM with {self.args.num_gpus} GPU(s): {e}")
            if self.args.num_gpus > 1:
                print("Multi-GPU initialization failed. Check configuration or try single GPU.")
            raise

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    def create_prompt(self, source_smiles: str) -> str:
        """Build the optimization prompt based on prompt_type and prompting_strategy.

        prompt_type:
            'sft'     -> SFT template with <SMILES> tags (for fine-tuned models)
            'general' -> plain template (for general-purpose LLMs)

        prompting_strategy (only for prompt_type='general'):
            'zero-shot' -> instruction only
            'few-shot'  -> 3 demonstration examples + instruction
            'cot'       -> instruction + chain-of-thought suffix
        """
        # Build property description from task properties
        if len(self.task_properties) == 1:
            prop = self.task_properties[0]
            property_description = PROPERTY_DESCRIPTIONS.get(prop, f"optimize {prop}")
        else:
            descriptions = [
                PROPERTY_DESCRIPTIONS.get(p, f"optimize {p}")
                for p in self.task_properties
            ]
            property_description = " and ".join(descriptions)

        # SFT models always use the SFT template (no prompting strategy variants)
        if self.args.prompt_type == "sft":
            return INSTRUCTION_TEMPLATE_SFT.format(
                input_smiles=source_smiles,
                property_description=property_description,
            )

        # --- General-purpose model prompting strategies ---
        strategy = self.args.prompting_strategy

        if strategy == "few-shot":
            # Prepend few-shot demonstration examples
            examples_block = "Here are some examples of successful molecular optimizations:\n\n"
            for i, ex in enumerate(FEW_SHOT_EXAMPLES, 1):
                examples_block += f"Example {i}:\n"
                examples_block += f"Input: {ex['input']}\n"
                examples_block += f"Output: {ex['output']}\n\n"
            examples_block += "Now, perform the following optimization:\n\n"

            base_prompt = INSTRUCTION_TEMPLATE_GENERAL.format(
                input_smiles=source_smiles,
                property_description=property_description,
            )
            return examples_block + base_prompt

        elif strategy == "cot":
            # Append chain-of-thought instruction
            base_prompt = INSTRUCTION_TEMPLATE_GENERAL.format(
                input_smiles=source_smiles,
                property_description=property_description,
            )
            # Replace the last line (output-only instruction) with CoT instruction
            return base_prompt.rsplit("\n", 1)[0] + COT_SUFFIX

        else:
            # Default: zero-shot
            return INSTRUCTION_TEMPLATE_GENERAL.format(
                input_smiles=source_smiles,
                property_description=property_description,
            )

    # ------------------------------------------------------------------
    # SMILES extraction and error correction
    # ------------------------------------------------------------------

    def fix_common_smiles_errors(self, smiles: str) -> str:
        """Attempt to fix common SMILES syntax errors."""
        if not smiles:
            return smiles

        import re

        # 1. For reaction SMILES (>>), keep only the first reactant
        if '>>' in smiles and '.' in smiles:
            smiles = smiles.split('>>')[0]

        # 2. For multi-fragment SMILES, keep the longest fragment
        if '.' in smiles:
            parts = smiles.split('.')
            smiles = max(parts, key=len)

        # 3. Fix unbalanced parentheses
        open_parens = smiles.count('(')
        close_parens = smiles.count(')')
        if open_parens > close_parens:
            smiles += ')' * (open_parens - close_parens)
        elif close_parens > open_parens:
            excess = close_parens - open_parens
            for _ in range(excess):
                idx = smiles.rfind(')')
                if idx != -1:
                    smiles = smiles[:idx] + smiles[idx + 1:]

        # 4. Truncate overly long SMILES (likely repetition errors)
        if len(smiles) > 200:
            smiles = smiles[:200]

        # 5. Remove residual XML-like markup
        smiles = re.sub(r'[<>{}]', '', smiles)

        return smiles

    def extract_smiles_from_output(self, output: str) -> Optional[str]:
        """Robust SMILES extractor that handles diverse model output formats."""
        if not output:
            return None

        output = output.strip()
        import re

        # Truncate excessively long outputs
        if len(output) > 1000:
            lines = output.split('\n')
            if len(lines) > 10:
                output = '\n'.join(lines[:10])
            else:
                output = output[:1000]

        # Remove excessively repeated patterns
        output = re.sub(r'(>>[^>]{1,50})\1{3,}', r'\1', output)

        # Ordered extraction strategies
        extraction_strategies = [
            # 1. Standard <SMILES></SMILES> tags
            lambda text: re.search(r'<SMILES>\s*([^<\s]+)\s*</SMILES>', text, re.IGNORECASE),
            # 2. Incomplete <SMILES> tag (no closing tag)
            lambda text: re.search(r'<SMILES>\s*([^<\n]+)', text, re.IGNORECASE),
            # 3. Bracket format [SMILES]content
            lambda text: re.search(r'\[SMILES\]\s*([^\[\n]+)', text, re.IGNORECASE),
            # 4. Double-quoted string
            lambda text: re.search(r'"([^"]+)"', text),
            # 5. Single-quoted string
            lambda text: re.search(r"'([^']+)'", text),
            # 6. Back-ticked string
            lambda text: re.search(r'`([^`]+)`', text),
            # 7. Content after a colon
            lambda text: re.search(r':\s*([^\s\n]+)', text),
        ]

        for strategy in extraction_strategies:
            match = strategy(output)
            if match:
                candidate = match.group(1).strip()
                candidate = re.sub(r'</SMI.*$', '', candidate)
                candidate = candidate.strip('"\'` \t.,!?')
                candidate = self.fix_common_smiles_errors(candidate)
                if candidate:
                    try:
                        mol = Chem.MolFromSmiles(candidate)
                        if mol is not None:
                            return Chem.MolToSmiles(mol)
                    except Exception:
                        pass

        # 8. Line-by-line scan
        lines = output.split('\n')
        for line in lines:
            line = line.strip()
            if not line:
                continue
            if any(w in line.lower() for w in ['error', 'failed', 'warning', 'info']):
                continue

            cleaned = line.strip('"\'` \t.,!?')
            if cleaned and ' ' not in cleaned and len(cleaned) > 3:
                fixed = self.fix_common_smiles_errors(cleaned)
                if fixed:
                    try:
                        mol = Chem.MolFromSmiles(fixed)
                        if mol is not None:
                            return Chem.MolToSmiles(mol)
                    except Exception:
                        pass

            # Check descriptive lines that might embed a SMILES after a colon
            if any(w in line.lower() for w in ['molecule', 'smiles', 'output', 'result', 'optimized']):
                if ':' in line:
                    parts = line.split(':', 1)
                    if len(parts) > 1:
                        candidate = parts[1].strip().strip('"\'` ')
                        fixed = self.fix_common_smiles_errors(candidate)
                        if fixed:
                            try:
                                mol = Chem.MolFromSmiles(fixed)
                                if mol is not None:
                                    return Chem.MolToSmiles(mol)
                            except Exception:
                                pass

        # 9. Fallback: regex scan for SMILES-like substrings
        potential_patterns = [
            r'\b[A-Za-z0-9\[\]()=@#+\-\\.\\\\/:]{6,}\b',
            r'\b[CNOSPFClBrI][A-Za-z0-9\[\]()=@#+\-\\.\\\\/:]{4,}\b',
            r'\b[cno][A-Za-z0-9\[\]()=@#+\-\\.\\\\/:]{4,}\b',
        ]

        for pattern in potential_patterns:
            matches = re.findall(pattern, output)
            for candidate in matches:
                if len(candidate) > 200:
                    continue
                if candidate.count('>>') > 2:
                    continue
                if candidate.count('.') > 10:
                    continue

                fixed = self.fix_common_smiles_errors(candidate)
                if fixed:
                    try:
                        mol = Chem.MolFromSmiles(fixed)
                        if mol is not None:
                            return Chem.MolToSmiles(mol)
                    except Exception:
                        pass

        return None

    # ------------------------------------------------------------------
    # Molecular evaluation helpers
    # ------------------------------------------------------------------

    def calculate_similarity(self, smiles1: str, smiles2: str) -> float:
        """Compute Tanimoto similarity between two molecules using Morgan fingerprints."""
        try:
            mol1 = Chem.MolFromSmiles(smiles1)
            mol2 = Chem.MolFromSmiles(smiles2)

            if mol1 is None or mol2 is None:
                return 0.0

            fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, 2, nBits=2048)
            fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, 2, nBits=2048)

            return FingerprintSimilarity(fp1, fp2)
        except Exception:
            return 0.0

    def is_absolute_success(self, properties: Dict[str, float], property_name: str) -> bool:
        """Check whether a single property meets the absolute-value threshold."""
        if property_name not in self.property_absolute_thresholds:
            return False
        if property_name not in properties or properties[property_name] is None:
            return False

        config = self.property_absolute_thresholds[property_name]
        threshold = config["threshold"]
        direction = config["direction"]
        value = properties[property_name]

        if direction == "increase":
            return value >= threshold
        else:  # decrease (e.g. SA)
            return value <= threshold

    def evaluate_all_properties(self, smiles: str) -> Dict[str, float]:
        """Score the molecule on all task properties via TDC Oracles."""
        properties = {}
        for prop in self.task_properties:
            try:
                score = self.oracles[prop](smiles)
                properties[prop] = float(score)
                self.total_oracle_calls += 1
            except Exception as e:
                print(f"Error evaluating {prop} for {smiles}: {e}")
                properties[prop] = None
        return properties

    def calculate_sa_score(self, smiles: str) -> Optional[float]:
        """Compute the SA score for a single molecule (for the Syn metric)."""
        try:
            score = self.oracles["sa"](smiles)
            return float(score)
        except Exception:
            return None

    def calculate_pairwise_diversity(self, smiles_list: List[str]) -> float:
        """Compute internal diversity: 1 - mean(pairwise Tanimoto).

        If fewer than 2 valid molecules, returns 0.0.
        """
        if len(smiles_list) < 2:
            return 0.0

        fps = []
        for smi in smiles_list:
            try:
                mol = Chem.MolFromSmiles(smi)
                if mol is not None:
                    fps.append(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048))
            except Exception:
                continue

        if len(fps) < 2:
            return 0.0

        similarities = []
        for i in range(len(fps)):
            for j in range(i + 1, len(fps)):
                similarities.append(FingerprintSimilarity(fps[i], fps[j]))

        if not similarities:
            return 0.0

        return 1.0 - np.mean(similarities)

    # ------------------------------------------------------------------
    # Generation loop
    # ------------------------------------------------------------------

    def run_single_round(self) -> Dict[int, List[str]]:
        """Run one round of batch generation for all active molecules."""
        self.current_round += 1

        # Adaptive temperature and seed per round
        round_temperature = self.base_temperature + (self.current_round - 1) * 0.05
        round_temperature = min(round_temperature, 1.5)  # cap at 1.5
        round_seed = self.base_seed + self.current_round * 1000

        # Update sampling parameters
        self.sampling_params = SamplingParams(
            temperature=round_temperature,
            top_p=self.args.top_p,
            max_tokens=self.args.max_new_tokens,
            stop_token_ids=self.template.get_stop_token_ids(self.tokenizer),
        )

        print(f"\n{'=' * 60}")
        print(f"Round {self.current_round}: Temperature={round_temperature:.2f}, Seed={round_seed}")
        print(f"{'=' * 60}")

        self._set_random_seed(round_seed)

        # Collect molecules that still need more Oracle calls
        active_molecules = [
            (idx, tracker)
            for idx, tracker in self.molecule_trackers.items()
            if not tracker.is_complete()
        ]

        if not active_molecules:
            print("All molecules completed!")
            return {}

        print(f"Active molecules: {len(active_molecules)}")

        # Build batch conversations
        all_conversations = []
        batch_indices = []

        for idx, tracker in active_molecules:
            source_smiles = tracker.source_smiles
            num_to_generate = self.batch_size_per_round

            for _ in range(num_to_generate):
                prompt = self.create_prompt(source_smiles)
                conversations = [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": ""},  # placeholder for generation
                ]
                all_conversations.append(conversations)
                batch_indices.append(idx)

        print(f"Generating {len(all_conversations)} candidates...")

        # Encode conversations to token IDs
        all_inputs = []
        valid_batch_indices = []

        for i, conversations in enumerate(all_conversations):
            try:
                prompt_ids, _ = self.template.encode_oneturn(self.tokenizer, conversations)

                if prompt_ids is not None and len(prompt_ids) > 0:
                    if len(prompt_ids) <= self.args.max_model_len:
                        all_inputs.append(prompt_ids)
                        valid_batch_indices.append(batch_indices[i])
                    else:
                        print(f"Warning: Input too long ({len(prompt_ids)} > {self.args.max_model_len})")

            except Exception as e:
                print(f"Error encoding conversation: {e}")
                continue

        if not all_inputs:
            raise RuntimeError(
                f"Template encoding failed for all {len(all_conversations)} conversations. "
                "Check the template or tokenizer configuration."
            )

        if len(all_inputs) < len(all_conversations):
            print(f"Warning: Only {len(all_inputs)}/{len(all_conversations)} conversations encoded successfully")

        # Batch generation via vLLM
        try:
            outputs = self.llm.generate(
                prompts=None,
                sampling_params=self.sampling_params,
                prompt_token_ids=all_inputs,
                lora_request=self.lora_request,
            )
        except Exception as e:
            if self.lora_request and "unsupported LoRA weight" in str(e):
                print(f"Warning: LoRA adapter failed ({str(e)[:100]}...)")
                print("Falling back to base model without LoRA...")
                outputs = self.llm.generate(
                    prompts=None,
                    sampling_params=self.sampling_params,
                    prompt_token_ids=all_inputs,
                )
            else:
                raise

        # Collect generated text per molecule
        round_results = defaultdict(list)

        for i, output in enumerate(outputs):
            if i >= len(valid_batch_indices):
                break
            mol_idx = valid_batch_indices[i]
            if output.outputs:
                generated_text = output.outputs[0].text.strip()
                round_results[mol_idx].append(generated_text)

        return round_results

    def process_round_results(self, round_results: Dict[int, List[str]]):
        """Evaluate and record candidates generated in a single round."""
        print("\nProcessing round results...")

        for mol_idx, generated_texts in round_results.items():
            tracker = self.molecule_trackers[mol_idx]
            source_smiles = tracker.source_smiles

            # Evaluate source molecule properties once
            if len(tracker.candidates_evaluated) == 0:
                source_properties = self.evaluate_all_properties(source_smiles)
                tracker.source_properties = source_properties

            for generated_text in generated_texts:
                tracker.generation_attempts += 1

                # Extract SMILES from model output
                candidate_smiles = self.extract_smiles_from_output(generated_text)

                if candidate_smiles is None:
                    tracker.invalid_smiles_count += 1
                    continue

                # Skip if identical to source
                if candidate_smiles == source_smiles:
                    continue

                # Similarity filter
                similarity = self.calculate_similarity(source_smiles, candidate_smiles)
                if similarity < self.args.similarity_threshold:
                    tracker.low_similarity_count += 1
                    continue

                # Passed validity and similarity checks -> Oracle evaluation
                candidate_properties = self.evaluate_all_properties(candidate_smiles)

                candidate_data = {
                    'smiles': candidate_smiles,
                    'similarity': similarity,
                    'properties': candidate_properties,
                    'passed_similarity': True,
                    'round': self.current_round,
                }

                tracker.add_candidate(candidate_data)

                if tracker.is_complete():
                    break

        # Print progress
        completed = sum(1 for t in self.molecule_trackers.values() if t.is_complete())
        total = len(self.molecule_trackers)
        print(f"Progress: {completed}/{total} molecules completed")

        if completed < total:
            remaining = [idx for idx, t in self.molecule_trackers.items() if not t.is_complete()]
            if len(remaining) <= 5:
                for idx in remaining:
                    t = self.molecule_trackers[idx]
                    print(f"  Molecule {idx}: {t.oracle_calls_made}/{t.target_oracle_calls} "
                          f"(remaining: {t.remaining_calls()})")

    # ------------------------------------------------------------------
    # Main evaluation loop
    # ------------------------------------------------------------------

    def run_batch_evaluation(self) -> Dict:
        """Run the full evaluation loop until all molecules are complete."""
        print("\n" + "=" * 60)
        print(f"Starting batch evaluation for {len(self.test_data)} molecules")
        print(f"Prompting strategy: {self.args.prompting_strategy}")
        print(f"Target Oracle calls per molecule: {self.args.target_oracle_calls}")
        print(f"Batch size per round: {self.args.batch_size_per_round}")
        print("=" * 60)

        start_time = time.time()

        while not all(t.is_complete() for t in self.molecule_trackers.values()):
            round_results = self.run_single_round()

            if not round_results:
                break

            self.process_round_results(round_results)

            if self.current_round >= self.args.max_rounds:
                print(f"Warning: Reached maximum rounds ({self.args.max_rounds}), stopping...")
                break

        # Compile per-molecule results
        final_results = self._compile_final_results()

        # Compute aggregate metrics
        metrics = self._calculate_metrics(final_results)

        end_time = time.time()

        return {
            'results': final_results,
            'metrics': metrics,
            'config': {
                'prompting_strategy': self.args.prompting_strategy,
                'prompt_type': self.args.prompt_type,
                'similarity_threshold': self.args.similarity_threshold,
                'temperature': self.args.temperature,
                'task': self.args.task,
            },
            'runtime': end_time - start_time,
            'total_rounds': self.current_round,
            'total_oracle_calls': self.total_oracle_calls,
        }

    # ------------------------------------------------------------------
    # Results compilation
    # ------------------------------------------------------------------

    def _compile_final_results(self) -> List[Dict]:
        """Aggregate per-molecule results including best candidate selection."""
        final_results = []

        for mol_idx, tracker in self.molecule_trackers.items():
            source_smiles = tracker.source_smiles
            source_properties = getattr(tracker, 'source_properties', {})

            # Select best candidate based on the primary task property
            best_candidate = None
            best_improvement = 0.0
            best_relative_improvement = 0.0

            if tracker.valid_candidates and source_properties:
                prop = self.task_properties[0]

                if prop == 'sa':
                    # For SA, lower is better
                    best_candidate = min(
                        tracker.valid_candidates[:self.args.target_oracle_calls],
                        key=lambda c: (c['properties'].get(prop, float('inf'))
                                       if c['properties'].get(prop) is not None
                                       else float('inf')),
                    )
                else:
                    best_candidate = max(
                        tracker.valid_candidates[:self.args.target_oracle_calls],
                        key=lambda c: (c['properties'].get(prop, float('-inf'))
                                       if c['properties'].get(prop) is not None
                                       else float('-inf')),
                    )

                # Compute improvement for the best candidate
                if (best_candidate
                        and prop in source_properties
                        and best_candidate['properties'].get(prop) is not None):
                    source_val = source_properties[prop]
                    target_val = best_candidate['properties'][prop]

                    if prop == 'sa':
                        best_improvement = source_val - target_val
                    else:
                        best_improvement = target_val - source_val

                    # Relative improvement: sgn(w_j) * (F_j(m') - F_j(m)) / |F_j(m)|
                    if source_val != 0:
                        sign = -1 if prop == 'sa' else +1
                        best_relative_improvement = sign * (target_val - source_val) / abs(source_val)
            else:
                # No valid candidates: use source molecule as fallback
                if source_properties:
                    best_candidate = {
                        'smiles': source_smiles,
                        'similarity': 1.0,
                        'properties': source_properties,
                        'passed_similarity': True,
                        'round': 0,
                    }

            # Improvement-based success
            improvement_success = False
            if self.task_properties[0] in self.property_success_thresholds:
                threshold = self.property_success_thresholds[self.task_properties[0]]['threshold']
                improvement_success = best_improvement >= threshold

            # Absolute-value success
            absolute_success = False
            if best_candidate and 'properties' in best_candidate:
                absolute_success = self.is_absolute_success(
                    best_candidate['properties'],
                    self.task_properties[0],
                )

            # Average similarity across valid candidates
            avg_similarity = 0.0
            if tracker.valid_candidates:
                sims = [c.get('similarity', 0.0) for c in tracker.valid_candidates]
                avg_similarity = np.mean(sims) if sims else 0.0

            final_results.append({
                'mol_idx': mol_idx,
                'source_smiles': source_smiles,
                'source_properties': source_properties,
                'oracle_calls_made': tracker.oracle_calls_made,
                'valid_candidates': len(tracker.valid_candidates),
                'total_attempts': tracker.generation_attempts,
                'invalid_smiles': tracker.invalid_smiles_count,
                'low_similarity': tracker.low_similarity_count,
                'best_candidate': best_candidate,
                'best_improvement': best_improvement,
                'best_relative_improvement': best_relative_improvement,
                'avg_similarity': avg_similarity,
                'success': improvement_success,
                'absolute_success': absolute_success,
            })

        return final_results

    # ------------------------------------------------------------------
    # Metrics (aligned with report Section 3.4)
    # ------------------------------------------------------------------

    def _safe_mean(self, values: List) -> float:
        """Compute mean treating None values as 0."""
        valid = [v if v is not None else 0.0 for v in values]
        return np.mean(valid) if valid else 0.0

    def _calculate_metrics(self, results: List[Dict]) -> Dict:
        """Compute evaluation metrics including Val, PC, Syn, and Div.

        Validity (Val):  fraction of generated outputs that are valid SMILES
        Property Control (PC): fraction of valid outputs with improved property
                               AND Tanimoto >= threshold
        Synthesizability (Syn): mean SA score across valid candidates
        Diversity (Div): internal diversity = 1 - mean(pairwise Tanimoto)
        """
        if not results:
            return {}

        improvement_successes = [r for r in results if r['success']]
        absolute_successes = [r for r in results if r['absolute_success']]

        # --- Validity (Val) ---
        # Valid = parseable by RDKit, regardless of similarity
        total_attempts = sum(r['total_attempts'] for r in results)
        total_invalid = sum(r['invalid_smiles'] for r in results)
        validity = (total_attempts - total_invalid) / total_attempts if total_attempts > 0 else 0.0

        # --- Property Control (PC) ---
        # Fraction of molecules where the best candidate improved the target property
        property_improved_count = 0
        for r in results:
            if r['best_candidate'] and r['best_improvement'] > 0:
                property_improved_count += 1
        property_control = property_improved_count / len(results) if results else 0.0

        # --- Synthesizability (Syn) ---
        # Mean SA score of all valid candidate molecules
        sa_scores = []
        for r in results:
            if r['best_candidate'] and r['best_candidate'].get('smiles'):
                sa = self.calculate_sa_score(r['best_candidate']['smiles'])
                if sa is not None:
                    sa_scores.append(sa)
        mean_sa = np.mean(sa_scores) if sa_scores else float('nan')

        # --- Diversity (Div) ---
        # Internal diversity across all valid generated molecules
        all_valid_smiles = []
        for r in results:
            mol_idx = r['mol_idx']
            tracker = self.molecule_trackers.get(mol_idx)
            if tracker:
                for c in tracker.valid_candidates:
                    if c.get('smiles'):
                        all_valid_smiles.append(c['smiles'])
        diversity = self.calculate_pairwise_diversity(all_valid_smiles)

        metrics = {
            'total_molecules': len(results),

            # Report-aligned metrics (Section 3.4)
            'validity': validity,
            'property_control': property_control,
            'synthesizability': mean_sa,
            'diversity': diversity,

            # Legacy success metrics
            'successful_molecules': len(improvement_successes),
            'success_rate': len(improvement_successes) / len(results) if results else 0,
            'absolute_successful_molecules': len(absolute_successes),
            'absolute_success_rate': len(absolute_successes) / len(results) if results else 0,

            # Efficiency metrics
            'avg_oracle_calls': np.mean([r['oracle_calls_made'] for r in results]),
            'total_oracle_calls': self.total_oracle_calls,
            'total_rounds': self.current_round,
            'avg_similarity': np.mean([r.get('avg_similarity', 0.0) for r in results]),
            'avg_improvement': np.mean([r['best_improvement'] for r in results]),
            'avg_relative_improvement': np.mean([r['best_relative_improvement'] for r in results]),
            'oracle_efficiency': self.total_oracle_calls / len(results) if results else 0,
            'generation_success_rate': np.mean([
                min(r['valid_candidates'] / max(r['total_attempts'], 1), 1.0)
                for r in results
            ]),
        }

        if improvement_successes:
            metrics['avg_improvement_successful'] = np.mean(
                [r['best_improvement'] for r in improvement_successes]
            )

        # Per-property breakdown
        if self.task_properties:
            prop = self.task_properties[0]
            metrics[f'{prop}_avg_improvement'] = np.mean(
                [r['best_improvement'] for r in results]
            )
            if improvement_successes:
                metrics[f'{prop}_avg_improvement_successful'] = np.mean(
                    [r['best_improvement'] for r in improvement_successes]
                )

        return metrics


# ======================================================================
# CLI entry point
# ======================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Prompting-Based Lead Optimization Evaluator'
    )

    # Model arguments
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the LLM (HuggingFace model or local directory)')
    parser.add_argument('--adapter_path', type=str, default=None,
                        help='Path to LoRA adapter (optional)')
    parser.add_argument('--template', type=str, required=True,
                        help='Chat template name (e.g. llama3, qwen2, drugassist)')
    parser.add_argument('--prompt_type', type=str, default='sft',
                        choices=['sft', 'general'],
                        help='sft: SFT models with <SMILES> tags; '
                             'general: general-purpose LLMs')
    parser.add_argument('--prompting_strategy', type=str, default='zero-shot',
                        choices=['zero-shot', 'few-shot', 'cot'],
                        help='Prompting strategy for general-purpose models: '
                             'zero-shot, few-shot (3 examples), or cot (chain-of-thought)')

    # Task arguments
    parser.add_argument('--task', type=str, required=True,
                        help='Target property to optimize (e.g. "qed", "logp", "jnk3", '
                             '"gsk3b", "drd2") or multi-property "qed+logp"')
    parser.add_argument('--test_data_path', type=str, required=True,
                        help='Path to JSON file with test molecules')
    parser.add_argument('--num_molecules', type=int, default=200,
                        help='Number of test molecules to evaluate')
    parser.add_argument('--target_oracle_calls', type=int, default=500,
                        help='Target number of Oracle calls per molecule')
    parser.add_argument('--batch_size_per_round', type=int, default=500,
                        help='Candidates to generate per molecule per round')

    # Evaluation arguments
    parser.add_argument('--similarity_threshold', type=float, default=0.6,
                        help='Minimum Tanimoto similarity to source molecule (default: 0.6)')
    parser.add_argument('--output_dir', type=str, default='./lead_opt_results',
                        help='Directory to save result JSON files')

    # Generation arguments
    parser.add_argument('--temperature', type=float, default=0.7,
                        help='Sampling temperature (use 0.0 for greedy decoding)')
    parser.add_argument('--top_p', type=float, default=0.9,
                        help='Nucleus sampling top-p')
    parser.add_argument('--max_new_tokens', type=int, default=256,
                        help='Maximum tokens to generate per candidate')
    parser.add_argument('--max_model_len', type=int, default=2048,
                        help='Maximum model context length')

    # System arguments
    parser.add_argument('--num_gpus', type=int, default=1,
                        help='Number of GPUs for tensor parallelism')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--max_rounds', type=int, default=10,
                        help='Maximum generation rounds before stopping')

    args = parser.parse_args()

    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Run evaluation
    evaluator = LeadOptimizationEvaluator(args)
    results = evaluator.run_batch_evaluation()

    # Save results
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_file = Path(args.output_dir) / f"{args.task}_results_{timestamp}.json"

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)
    metrics = results['metrics']
    print(f"Task: {args.task}")
    print(f"Prompting strategy: {args.prompting_strategy}")
    print(f"Total molecules: {metrics['total_molecules']}")

    # Report-aligned metrics
    print(f"\n--- Report Metrics (Section 3.4) ---")
    print(f"Validity (Val):          {metrics['validity']:.4f}")
    print(f"Property Control (PC):   {metrics['property_control']:.4f}")
    print(f"Synthesizability (Syn):  {metrics['synthesizability']:.2f}")
    print(f"Diversity (Div):         {metrics['diversity']:.4f}")

    # Success rates
    print(f"\n--- Success Rates ---")
    print(f"Improvement success: {metrics['successful_molecules']}/{metrics['total_molecules']} "
          f"({metrics['success_rate']:.2%})")
    print(f"Absolute success:    {metrics['absolute_successful_molecules']}/{metrics['total_molecules']} "
          f"({metrics['absolute_success_rate']:.2%})")

    # Efficiency
    print(f"\n--- Efficiency ---")
    print(f"Avg Oracle calls/molecule: {metrics['avg_oracle_calls']:.1f}")
    print(f"Total Oracle calls: {metrics['total_oracle_calls']}")
    print(f"Total rounds: {metrics['total_rounds']}")
    print(f"Results saved to: {output_file}")


if __name__ == "__main__":
    main()
