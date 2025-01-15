from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Union, Iterator
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import yaml
import json
from pathlib import Path
import logging
from functools import lru_cache
import gc
from transformers.generation import TextStreamer

@dataclass
class LLMConfig:
    model_name: str = "gpt2"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    max_length: int = 1024
    batch_size: int = 4
    cache_size: int = 1000
    temperature: float = 0.7
    top_p: float = 0.9
    streaming: bool = False
    timeout: int = 30

class LLMInterface:
    def __init__(self, config_path: Optional[str] = None):
        self._setup_logging()
        self.config = self._load_config(config_path)
        self._init_model()
        
    def _setup_logging(self) -> None:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
    def _load_config(self, config_path: Optional[str]) -> LLMConfig:
        try:
            if config_path and Path(config_path).exists():
                self.logger.info(f"Loading config from {config_path}")
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)['model_config']
                    return LLMConfig(**config)
            else:
                self.logger.info("Using default configuration")
                return LLMConfig()
        except Exception as e:
            self.logger.warning(f"Error loading config: {e}, using default values")
            return LLMConfig()

    def _init_model(self) -> None:
        try:
            self.logger.info(f"Initializing model {self.config.model_name}")
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_name,
                trust_remote_code=True
            )
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                torch_dtype=torch.float16 if self.config.device == "cuda" else torch.float32,
                trust_remote_code=True
            )
            
            if torch.cuda.is_available() and self.config.device == "cuda":
                self.model = self.model.to("cuda")
                self.logger.info("Model moved to CUDA")
            else:
                self.logger.info("Model running on CPU")
                
        except Exception as e:
            self.logger.error(f"Error initializing model: {e}")
            raise

    def _validate_input(self, prompt: str) -> None:
        """Valide les entrées avant la génération"""
        if not prompt or not prompt.strip():
            self.logger.error("Empty prompt received")
            raise ValueError("Le prompt ne peut pas être vide")

    def generate_response(self, prompt: str, context: str = "") -> str:
        """Génère une réponse à partir d'un prompt et d'un contexte optionnel"""
        try:
            # Validation des entrées
            self._validate_input(prompt)
            
            # Préparation du prompt
            full_prompt = f"Context: {context}\nQuestion: {prompt}\nAnswer:"
            self.logger.debug(f"Generated prompt: {full_prompt}")
            
            # Tokenization
            inputs = self.tokenizer(
                full_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=self.config.max_length
            )
            self.logger.debug(f"Input shape: {inputs['input_ids'].shape}")
            
            if self.config.device == "cuda":
                inputs = inputs.to("cuda")

            # Calcul de la longueur maximale
            input_length = inputs['input_ids'].shape[1]
            max_new_tokens = min(self.config.max_length - input_length, 512)

            # Génération
            self.logger.debug("Starting generation...")
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                num_return_sequences=1,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
            self.logger.debug(f"Output shape: {outputs.shape}")

            # Décodage et nettoyage
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = response.replace(full_prompt, "").strip()
            
            self.logger.debug(f"Final response: {response}")
            return response

        except Exception as e:
            self.logger.error(f"Error generating response: {e}")
            raise

    def __del__(self):
        try:
            if hasattr(self, 'model'):
                del self.model
            if hasattr(self, 'tokenizer'):
                del self.tokenizer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
            self.logger.info("Resources cleaned up")
        except:
            pass