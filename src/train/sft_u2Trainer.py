import os
import torch
from transformers import Trainer
from transformers.utils import logging, SAFE_WEIGHTS_NAME, WEIGHTS_NAME
from typing import Optional, Any

logger = logging.get_logger(__name__)
TRAINING_ARGS_NAME = "training_args.bin"

class u2Trainer(Trainer):
    def _strip_tensor_subclasses(self, obj: Any) -> Any:
        """
        Ensure we never hand a Tensor subclass (e.g. MONAI MetaTensor) to
        `Trainer`'s distributed gather helpers, since they may `clone()` and
        trigger subclass-specific deep-copy / pickling code paths.
        """
        if isinstance(obj, torch.Tensor) and type(obj) is not torch.Tensor:
            try:
                return obj.as_subclass(torch.Tensor)
            except Exception:
                # Fallbacks for older torch / odd subclasses.
                if hasattr(obj, "as_tensor"):
                    try:
                        base = obj.as_tensor()
                        if isinstance(base, torch.Tensor) and type(base) is torch.Tensor:
                            return base
                    except Exception:
                        pass
                # Last-resort: materialize a plain tensor via NumPy roundtrip.
                return torch.tensor(obj.detach().cpu().numpy(), device=obj.device, dtype=obj.dtype)

        if isinstance(obj, dict):
            return {k: self._strip_tensor_subclasses(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self._strip_tensor_subclasses(v) for v in obj]
        if isinstance(obj, tuple):
            return tuple(self._strip_tensor_subclasses(v) for v in obj)
        return obj

    def _nested_gather(self, tensors, name: Optional[str] = None):
        tensors = self._strip_tensor_subclasses(tensors)
        return super()._nested_gather(tensors, name=name)

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        # If we are executing this function, we are the process zero, so we don't check for that.
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving model checkpoint to {output_dir}")

        if state_dict is None:
            state_dict = self.model.state_dict()

        logger.info("Trainer.model is not a `PreTrainedModel`, only saving its state dict.")
        torch.save(state_dict, os.path.join(output_dir, WEIGHTS_NAME))

        if self.args.lora_enable:
            state_dict_with_lora = self.model.state_dict()
            torch.save(state_dict_with_lora, os.path.join(self.args.output_dir, 'model_with_lora.bin'))
            
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))
