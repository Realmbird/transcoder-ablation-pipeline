"""
Sanity check for unlearned models - verify they still work properly
"""
import torch

def sanity_check_model(model, tokenizer, model_name="Model"):
    """
    Quick sanity checks to verify model still generates reasonable text.

    Tests:
    1. General knowledge (capital of France)
    2. Basic reasoning (2+2)
    3. Language generation quality
    4. Checks output is in English, not gibberish
    """
    print(f"\n{'='*60}")
    print(f"Sanity Check: {model_name}")
    print(f"{'='*60}")

    model.eval()

    test_prompts = [
        "The capital of France is",
        "2 + 2 =",
        "The sun rises in the",
        "Water freezes at",
        "The first president of the United States was",
    ]

    expected_keywords = [
        ["Paris", "paris"],
        ["4", "four"],
        ["east", "East"],
        ["0", "zero", "32"],
        ["Washington", "George"],
    ]

    results = []

    with torch.no_grad():
        for prompt, keywords in zip(test_prompts, expected_keywords):
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

            outputs = model.generate(
                **inputs,
                max_new_tokens=10,
                temperature=0.1,  # Low temperature for deterministic output
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = generated_text[len(prompt):].strip()

            # Check if any expected keyword appears
            passed = any(kw.lower() in response.lower() for kw in keywords)

            # Check it's not gibberish (no Chinese characters, reasonable length)
            is_english = all(ord(c) < 128 or c.isspace() for c in response[:50])
            has_content = len(response.split()) > 0

            test_passed = passed and is_english and has_content

            status = "✓" if test_passed else "✗"
            print(f"{status} Q: {prompt}")
            print(f"   A: {response[:100]}")

            results.append({
                'prompt': prompt,
                'response': response,
                'passed': test_passed,
                'is_english': is_english,
                'has_content': has_content,
            })

    # Summary
    passed_count = sum(r['passed'] for r in results)
    total_count = len(results)

    print(f"\n{'='*60}")
    print(f"Results: {passed_count}/{total_count} tests passed")

    if passed_count == total_count:
        print("✓ Model looks healthy - generating reasonable English responses")
    elif passed_count >= total_count * 0.6:
        print("⚠ Model partially working - some responses degraded")
    else:
        print("✗ Model severely damaged - most responses are wrong/gibberish")
    print(f"{'='*60}\n")

    return {
        'passed_count': passed_count,
        'total_count': total_count,
        'pass_rate': passed_count / total_count,
        'results': results,
    }


def compare_models(original_model, original_tokenizer, unlearned_model, unlearned_tokenizer):
    """Compare original vs unlearned model side-by-side"""
    print("\n" + "="*80)
    print("SIDE-BY-SIDE COMPARISON: Original vs Unlearned")
    print("="*80)

    original_results = sanity_check_model(original_model, original_tokenizer, "Original Model")
    unlearned_results = sanity_check_model(unlearned_model, unlearned_tokenizer, "Unlearned Model")

    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Original:  {original_results['passed_count']}/{original_results['total_count']} ({original_results['pass_rate']:.1%})")
    print(f"Unlearned: {unlearned_results['passed_count']}/{unlearned_results['total_count']} ({unlearned_results['pass_rate']:.1%})")

    if unlearned_results['pass_rate'] >= 0.8:
        print("\n✓ Unlearning preserved general capabilities well!")
    elif unlearned_results['pass_rate'] >= 0.6:
        print("\n⚠ Unlearning caused some degradation in general capabilities")
    else:
        print("\n✗ Unlearning severely damaged the model!")
    print("="*80 + "\n")


if __name__ == "__main__":
    print("Sanity check utility for unlearned models")
