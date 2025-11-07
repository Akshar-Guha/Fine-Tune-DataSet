"""Flash Attention 2 utilities."""


def patch_model_with_flash_attention(model):
    """Patch model to use Flash Attention 2.
    
    Args:
        model: Transformer model
        
    Returns:
        Patched model
    """
    # Flash Attention 2 is auto-enabled in transformers >= 4.34
    # when use_flash_attention_2=True in from_pretrained
    return model
