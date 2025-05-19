import torch
from torch.nn import CrossEntropyLoss
import numpy as np


MODEL_ZOO = {
    'llama2-7b': 'meta-llama/Llama-2-7b-chat-hf',
    'llama2-13b': 'meta-llama/Llama-2-13b-chat-hf',
    'llama3-8b': 'meta-llama/Meta-Llama-3-8B-Instruct',
    'gemma-2b': 'google/gemma-1.1-2b-it',
    'gemma-7b': 'google/gemma-1.1-7b-it', 
    'mistral-7b': 'mistralai/Mistral-7B-Instruct-v0.2',
}


def compute_fce_loss(logits, targets, text_slice):
    """
    Compute the FCE loss by shifting indices by 1.
    Returns a NumPy array of loss values.
    """
    loss = CrossEntropyLoss(reduction='none')(
        logits[0, text_slice.start-1:text_slice.stop-1, :],
        targets
    )
    return loss.detach().cpu().numpy()

def compute_bce_loss(logits, targets, text_slice):
    """
    Compute the BCE loss without shifting indices.
    Returns a NumPy array of loss values.
    """
    loss = CrossEntropyLoss(reduction='none')(
        logits[0, text_slice, :],
        targets
    )
    return loss.detach().cpu().numpy()

def detect_single_sample(model, tokenizer, summary_model, summary_tokenizer, sample, max_length = 1000, device='cuda'):
    """
    Process a sample by generating a summary-based prompt, tokenizing (with clipping),
    obtaining model outputs, and computing loss-based features (FCE and BCE).
    Returns a list of loss features computed over 10 segments.
    """
    # Prompt templates for text completion.
    COMPLETION_PROMPT_ONLY = "Complete the following text: "
    COMPLETION_PROMPT = "Given the summary:\n{prompt}\n Complete the following text: "
    
    model_device = next(model.parameters()).device
    
    # Generate the summary-based prompt.
    if summary_model in MODEL_ZOO:
        summary_input = f"Write a title for this text: {sample}\nJust output the title:"
        summary_ids = summary_tokenizer(summary_input, return_tensors='pt',
                                        max_length=max_length, truncation=True).input_ids.to(model_device)
        summary_ids = summary_ids[:, 1:]  # Remove start token.
        gen_ids = generate(summary_model, summary_tokenizer, summary_ids, summary_ids.shape[1], 64)
        summary_text = summary_tokenizer.decode(gen_ids, skip_special_tokens=True).strip().split('\n')[0]
        prompt_text = COMPLETION_PROMPT.format(prompt=summary_text)
    else:
        prompt_text = COMPLETION_PROMPT_ONLY

    # Tokenize the prompt and sample with token-level clipping.
    prompt_ids = tokenizer(prompt_text, return_tensors='pt').input_ids.to(model_device)
    text_ids = tokenizer(sample, return_tensors='pt', max_length=max_length, truncation=True).input_ids.to(model_device)
    combined_ids = torch.cat([prompt_ids, text_ids], dim=1)
    text_slice = slice(prompt_ids.shape[1], combined_ids.shape[1])

    outputs = model(input_ids=combined_ids)
    logits = outputs.logits
    targets = combined_ids[0][text_slice]

    # Compute loss features from FCE and BCE losses.
    fce_loss = compute_fce_loss(logits, targets, text_slice)
    bce_loss = compute_bce_loss(logits, targets, text_slice)
    features = []
    for p in range(1, 10):
        split = len(fce_loss) * p // 10
        features.extend([
            np.mean(fce_loss[split:]), np.max(fce_loss[split:]), 
            np.min(fce_loss[split:]), np.std(fce_loss[split:]),
            np.mean(bce_loss[split:]), np.max(bce_loss[split:]), 
            np.min(bce_loss[split:]), np.std(bce_loss[split:])
        ])
    return features

def generate(model, tokenizer, input_ids, trigger_length, target_length):
    """
    Generate additional tokens using the model's generation API.
    
    Parameters:
      model: the language model for generation.
      tokenizer: associated tokenizer.
      input_ids: input token IDs (either 1D or 2D).
      trigger_length: the length of the prompt (number of tokens to skip in the output).
      target_length: the number of new tokens to generate.
      
    Returns:
      Generated tokens (as a 2D tensor) after removing the trigger tokens.
    """
    config = model.generation_config
    config.max_new_tokens = target_length
    # If input_ids is 1D, add a batch dimension; otherwise, assume it's already 2D.
    if input_ids.dim() == 1:
        input_ids = input_ids.to(model.device).unsqueeze(0)
    else:
        input_ids = input_ids.to(model.device)
    # Create an attention mask of the same shape.
    attn_masks = torch.ones(input_ids.shape, device=input_ids.device)
    # Generate new tokens.
    out = model.generate(
        input_ids, 
        attention_mask=attn_masks,
        generation_config=config,
        pad_token_id=tokenizer.pad_token_id
    )[0]
    # Return output tokens after the prompt (slice along dimension 1).
    return out[trigger_length:]