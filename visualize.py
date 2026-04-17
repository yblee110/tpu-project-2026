"""
Visualize the generation process step by step
Shows diffusion denoising in a cryptographic terminal theme.
"""

import sys
import os
import time
import random
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn.functional as F

import diffusion
import gpt

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def display_crypto_state_terminal(tokens, block_mask, block_start, is_gpt=False):
    clear_screen()
    
    crypto_chars = ['@', '#', '$', '%', '&', '*', '?', '!', '~', '+', '=', 'A', 'X', 'Z', 'Q']
    
    # ANSI Colors
    GRAY = "\033[90m"
    BLUE = "\033[94;1m"
    RED = "\033[91;1m"
    RESET = "\033[0m"
    
    # 1. Completed text (past blocks)
    completed_text = diffusion.decode(tokens[:block_start])
    
    # 2. Current block
    current_text_parts = []
    for i, token in enumerate(tokens[block_start:]):
        if i < len(block_mask) and block_mask[i]:
            current_text_parts.append(f"{BLUE}{random.choice(crypto_chars)}{RESET}")
        else:
            char = diffusion.decode([token])
            current_text_parts.append(f"{RED}{char}{RESET}")
            
    current_text = "".join(current_text_parts)
    
    if is_gpt:
        print(f"\033[97;1m🤖 GPT: Autoregressive Generation...\033[0m\n")
    else:
        print(f"\033[97;1m🕵️‍♂️ Sherlock Holmes: Cryptographic Decoupling...\033[0m\n")
        
    print(f"{GRAY}{completed_text}{RESET}{current_text}")
    print("\n" + "-"*50)
    time.sleep(0.05 if not is_gpt else 0.02)

def generate_and_visualize_diffusion(model, num_blocks=5, prompt_len=16, temp=0.8, confidence_threshold=0.95, top_k=2):
    device = next(model.parameters()).device
    block_size = diffusion.block_size
    mask_token_id = diffusion.mask_token_id

    all_tokens = diffusion.data[:prompt_len].tolist()
    
    for block_idx in range(num_blocks):
        block_start = len(all_tokens)
        max_new_tokens = 240
        block_len = min(block_size - prompt_len, max_new_tokens)

        x = torch.full((1, block_size), mask_token_id, dtype=torch.long, device=device)
        x[0, :prompt_len] = torch.tensor(all_tokens[-prompt_len:], device=device)

        masked = torch.zeros(1, block_size, dtype=torch.bool, device=device)
        masked[0, prompt_len : prompt_len + block_len] = True

        temp_tokens = list(all_tokens) + [mask_token_id] * block_len

        while masked.any():
            logits, _ = model(x)
            probs = F.softmax(logits / temp, dim=-1)
            top_k_probs, top_k_indices = torch.topk(probs, k=top_k, dim=-1)
            confidences = top_k_probs.sum(dim=-1)

            decode_mask = (confidences >= confidence_threshold) & masked
            if not decode_mask.any():
                masked_confidences = torch.where(
                    masked, confidences, torch.tensor(-float("inf"), device=device)
                )
                decode_mask.view(-1)[masked_confidences.argmax()] = True

            top_k_probs_norm = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)
            sampled_k = torch.multinomial(top_k_probs_norm.view(-1, top_k), 1).view(1, block_size)
            sampled_tokens = torch.gather(top_k_indices, -1, sampled_k.unsqueeze(-1)).squeeze(-1)

            x = torch.where(decode_mask, sampled_tokens, x)
            masked = masked & ~decode_mask

            current_block_tokens = x[0, prompt_len : prompt_len + block_len].cpu().numpy()
            current_mask = masked[0, prompt_len : prompt_len + block_len].cpu().numpy()
            
            temp_tokens[block_start:] = current_block_tokens.tolist()
            display_crypto_state_terminal(temp_tokens, current_mask, block_start)

        all_tokens.extend(x[0, prompt_len : prompt_len + block_len].tolist())

    clear_screen()
    print("\n✨ Diffusion decoding complete! (Final Output):\n")
    print(diffusion.decode(all_tokens))
    print("\n" + "="*50 + "\n")
    time.sleep(2)

def generate_and_visualize_gpt(model, max_new_tokens, prompt_len=16, temp=0.8):
    device = next(model.parameters()).device
    block_size = gpt.block_size

    all_tokens = gpt.data[:prompt_len].tolist()
    x = torch.tensor(all_tokens, dtype=torch.long, device=device).unsqueeze(0)

    for _ in range(max_new_tokens):
        cur_context = x[:, -block_size:]
        logits, _ = model(cur_context)
        logits = logits[:, -1, :]
        probs = F.softmax(logits / temp, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        x = torch.cat((x, next_token), dim=1)
        
        all_tokens.append(next_token.item())
        
        # Display
        display_crypto_state_terminal(all_tokens, [False], len(all_tokens)-1, is_gpt=True)

    clear_screen()
    print("\n✨ GPT generation complete! (Final Output):\n")
    print(gpt.decode(all_tokens))
    print("\n" + "="*50 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize diffusion and/or GPT generation in terminal"
    )
    parser.add_argument(
        "--compare", action="store_true", help="Show diffusion then GPT sequentially"
    )
    parser.add_argument(
        "--blocks",
        type=int,
        default=5,
        help="Number of blocks to generate (default: 5)",
    )
    parser.add_argument(
        "--prompt-len",
        type=int,
        default=16,
        help="Length of initial prompt (default: 16)",
    )

    args = parser.parse_args()

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using device: {device}\n")

    # Load and generate diffusion
    diffusion_path = os.path.join(os.path.dirname(__file__), "weights", "diffusion.pt")
    if not os.path.exists(diffusion_path):
        print(f"Error: Weights not found at {diffusion_path}. Run 'uv run diffusion.py --train' first or download weights.")
        return

    print(f"Loading diffusion model from {diffusion_path}...")
    diffusion_model = diffusion.Model().to(device)
    diffusion_model.load_state_dict(torch.load(diffusion_path, map_location=device, weights_only=True))
    diffusion_model.eval()

    generate_and_visualize_diffusion(
        diffusion_model, args.blocks, args.prompt_len
    )

    if args.compare:
        max_new_tokens = args.blocks * (diffusion.block_size - args.prompt_len)

        gpt_path = os.path.join(os.path.dirname(__file__), "weights", "gpt.pt")
        if not os.path.exists(gpt_path):
            print(f"Error: Weights not found at {gpt_path}.")
            return
            
        print(f"Loading GPT model from {gpt_path}...")
        gpt_model = gpt.Model().to(device)
        gpt_model.load_state_dict(torch.load(gpt_path, map_location=device, weights_only=True))
        gpt_model.eval()

        generate_and_visualize_gpt(gpt_model, max_new_tokens, args.prompt_len)

if __name__ == "__main__":
    main()
