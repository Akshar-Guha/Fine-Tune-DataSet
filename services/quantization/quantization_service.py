"""Quantization Service - Support AWQ, GPTQ, and GGUF quantization."""
from typing import Dict, Any, Optional
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class QuantizationService:
    """Handle model quantization for deployment."""
    
    async def quantize_awq(
        self,
        model_path: str,
        output_path: str,
        bits: int = 4,
        group_size: int = 128
    ) -> Dict[str, Any]:
        """Quantize model using AutoAWQ.
        
        Args:
            model_path: Path to model
            output_path: Output directory
            bits: Number of bits (4, 8)
            group_size: Group size for quantization
            
        Returns:
            Quantization results
        """
        try:
            from awq import AutoAWQForCausalLM
            from transformers import AutoTokenizer
            
            print(f"Loading model from {model_path}...")
            model = AutoAWQForCausalLM.from_pretrained(model_path)
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            
            # Prepare calibration data (small sample)
            calib_data = [
                "The quick brown fox jumps over the lazy dog.",
                "Machine learning is transforming the world.",
                "Natural language processing enables computers to understand text.",
            ] * 32  # Repeat to get ~100 samples
            
            print(f"Quantizing to {bits}-bit AWQ...")
            quant_config = {
                "zero_point": True,
                "q_group_size": group_size,
                "w_bit": bits,
                "version": "GEMM"
            }
            
            model.quantize(tokenizer, quant_config=quant_config, calib_data=calib_data)
            
            # Save quantized model
            Path(output_path).mkdir(parents=True, exist_ok=True)
            model.save_quantized(output_path)
            tokenizer.save_pretrained(output_path)
            
            # Get model size
            model_size_mb = sum(
                f.stat().st_size for f in Path(output_path).rglob("*") if f.is_file()
            ) / (1024**2)
            
            print(f"✓ AWQ quantization complete: {output_path}")
            
            return {
                "method": "awq",
                "bits": bits,
                "output_path": output_path,
                "size_mb": model_size_mb,
                "success": True
            }
            
        except ImportError:
            raise ImportError(
                "AutoAWQ not installed. Install with: pip install autoawq"
            )
        except Exception as e:
            return {
                "method": "awq",
                "success": False,
                "error": str(e)
            }
    
    async def quantize_gptq(
        self,
        model_path: str,
        output_path: str,
        bits: int = 4,
        group_size: int = 128
    ) -> Dict[str, Any]:
        """Quantize model using AutoGPTQ.
        
        Args:
            model_path: Path to model
            output_path: Output directory
            bits: Number of bits (4, 8)
            group_size: Group size for quantization
            
        Returns:
            Quantization results
        """
        try:
            from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
            from transformers import AutoTokenizer
            
            print(f"Loading model from {model_path}...")
            
            quantize_config = BaseQuantizeConfig(
                bits=bits,
                group_size=group_size,
                desc_act=False
            )
            
            model = AutoGPTQForCausalLM.from_pretrained(
                model_path,
                quantize_config=quantize_config
            )
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            
            # Prepare calibration data
            calib_data = [
                tokenizer(
                    "The quick brown fox jumps over the lazy dog. " * 10,
                    return_tensors="pt"
                )
            ] * 16
            
            print(f"Quantizing to {bits}-bit GPTQ...")
            model.quantize(calib_data)
            
            # Save quantized model
            Path(output_path).mkdir(parents=True, exist_ok=True)
            model.save_quantized(output_path)
            tokenizer.save_pretrained(output_path)
            
            model_size_mb = sum(
                f.stat().st_size for f in Path(output_path).rglob("*") if f.is_file()
            ) / (1024**2)
            
            print(f"✓ GPTQ quantization complete: {output_path}")
            
            return {
                "method": "gptq",
                "bits": bits,
                "output_path": output_path,
                "size_mb": model_size_mb,
                "success": True
            }
            
        except ImportError:
            raise ImportError(
                "AutoGPTQ not installed. Install with: pip install auto-gptq"
            )
        except Exception as e:
            return {
                "method": "gptq",
                "success": False,
                "error": str(e)
            }
    
    async def export_gguf(
        self,
        model_path: str,
        output_path: str,
        quantization_type: str = "q4_k_m"
    ) -> Dict[str, Any]:
        """Export model to GGUF format for llama.cpp.
        
        Args:
            model_path: Path to model
            output_path: Output directory
            quantization_type: GGUF quantization type (q4_k_m, q5_k_m, q8_0)
            
        Returns:
            Export results
        """
        try:
            import subprocess
            import sys
            
            Path(output_path).mkdir(parents=True, exist_ok=True)
            
            # First convert to f16 GGUF
            print("Converting to GGUF format...")
            convert_cmd = [
                sys.executable,
                "-m", "llama_cpp.convert",
                model_path,
                "--outfile", str(Path(output_path) / "model_f16.gguf"),
                "--outtype", "f16"
            ]
            
            result = subprocess.run(convert_cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                raise RuntimeError(f"GGUF conversion failed: {result.stderr}")
            
            # Then quantize
            print(f"Quantizing to {quantization_type}...")
            quant_cmd = [
                "llama-quantize",
                str(Path(output_path) / "model_f16.gguf"),
                str(Path(output_path) / f"model_{quantization_type}.gguf"),
                quantization_type
            ]
            
            result = subprocess.run(quant_cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                print(f"Warning: Quantization failed, using f16 model: {result.stderr}")
                final_path = str(Path(output_path) / "model_f16.gguf")
            else:
                final_path = str(Path(output_path) / f"model_{quantization_type}.gguf")
            
            model_size_mb = Path(final_path).stat().st_size / (1024**2)
            
            print(f"✓ GGUF export complete: {final_path}")
            
            return {
                "method": "gguf",
                "quantization_type": quantization_type,
                "output_path": final_path,
                "size_mb": model_size_mb,
                "success": True
            }
            
        except Exception as e:
            return {
                "method": "gguf",
                "success": False,
                "error": str(e)
            }
