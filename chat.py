import torch
import re
import os
import warnings
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
import numpy as np
import readline

# ========== CONFIG ==========
BASE_DIR = "./DeepSeek-R1-Distill-Qwen-1.5B-finetuned/"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
COHERENCE_THRESHOLD = 0.4
USE_COHERENCE = True
SHOW_DEBUG = False
# ============================

def find_latest_checkpoint(base_dir):
    """Find the latest checkpoint in the directory"""
    if not os.path.exists(base_dir):
        raise FileNotFoundError(f"Directory not found: {base_dir}")
    
    # Look for checkpoint directories
    checkpoint_dirs = []
    for item in os.listdir(base_dir):
        if item.startswith("checkpoint-") and os.path.isdir(os.path.join(base_dir, item)):
            try:
                # Extract checkpoint number
                checkpoint_num = int(item.split("-")[1])
                checkpoint_dirs.append((checkpoint_num, os.path.join(base_dir, item)))
            except (IndexError, ValueError):
                continue
    
    if not checkpoint_dirs:
        raise FileNotFoundError(f"No checkpoint directories found in {base_dir}")
    
    # Sort by checkpoint number and return the latest
    checkpoint_dirs.sort(key=lambda x: x[0], reverse=True)
    latest_checkpoint = checkpoint_dirs[0][1]
    
    print(f"🔍 Found {len(checkpoint_dirs)} checkpoints")
    print(f"📂 Using latest: {latest_checkpoint}")
    
    return latest_checkpoint

def load_model_and_tokenizer():
    """Load model and tokenizer with proper configuration"""
    print("🔄 Loading model and tokenizer...")
    
    # Find the latest checkpoint
    checkpoint_path = find_latest_checkpoint(BASE_DIR)
    
    # Suppress initial loading warnings
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
        model = AutoModelForCausalLM.from_pretrained(checkpoint_path)
        
        # Fix pad token issue
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        
        model.to(DEVICE).eval()
        
    return tokenizer, model

def is_deflecting_question(response, user_input):
    """Check if response deflects back to user instead of answering"""
    deflection_patterns = [
        r'what do you (do|think|feel)',
        r'how (do|does) you',
        r"that's (what|how|why) you",
        r'you (should|need to|have to)',
        r"what about you",
        r"how about you",
        r"what's your (thought|opinion|view)",
    ]
    
    response_lower = response.lower()
    for pattern in deflection_patterns:
        if re.search(pattern, response_lower):
            return True
    return False

def is_response_problematic(response, user_input):
    """Comprehensive check for problematic response patterns"""
    if not response or len(response.strip()) < 15:
        return True, "too_short"
    
    response_lower = response.lower()
    
    # Check for deflection
    if is_deflecting_question(response, user_input):
        return True, "deflection"
    
    # Check for circular/meta responses
    problematic_patterns = [
        r"that's (the|what|why|how) (beauty|part|thing|point)",
        r"it'?s (not|just) (like|that|what)",
        r"that'?s (not|just) (something|what|how)",
        r"you'?re (right|thinking|asking)[\w\s]{0,20}but",
        r"it (just )?means that",
        r"that'?s (why|what|how) (it'?s|you|we)",
    ]
    
    for pattern in problematic_patterns:
        if re.search(pattern, response_lower):
            return True, "circular"
    
    # Check for incomplete thoughts
    incomplete_patterns = [
        r":\s*it'?s not",
        r"—.*?(it'?s|that'?s|you)",
        r"\.\s*that'?s",
    ]
    
    for pattern in incomplete_patterns:
        if re.search(pattern, response):
            return True, "incomplete"
    
    # Check word repetition within short spans
    words = response.split()
    for i in range(len(words) - 2):
        if words[i].lower() == words[i+2].lower():
            return True, "repetitive"
    
    return False, "valid"

def clean_and_truncate_response(response):
    """Clean response and truncate at natural stopping points"""
    if not response.strip():
        return ""
    
    # Remove special tokens
    response = re.sub(r'<\|[^|]*\|>', '', response)
    response = re.sub(r'</?[^>]+>', '', response)
    
    # Remove markdown artifacts
    response = re.sub(r'\*+', '', response)
    response = re.sub(r'#+\s*', '', response)
    response = re.sub(r'^\s*[-*+•]\s+', '', response, flags=re.MULTILINE)
    
    # Basic cleanup
    response = ''.join(c for c in response if c.isprintable() or c.isspace())
    response = re.sub(r'\s+([.,!?;:])', r'\1', response)
    response = re.sub(r'\s{2,}', ' ', response)
    response = response.strip()
    
    # Find good truncation points
    sentences = re.split(r'[.!?]+', response)
    good_sentences = []
    
    for sentence in sentences:
        sentence = sentence.strip()
        if len(sentence) < 8:
            continue
        
        sentence_lower = sentence.lower()
        skip_patterns = [
            r"that'?s (the|what|why)",
            r"it (just )?means",
            r"you'?re (right|thinking)",
            r"what (do|does) you",
        ]
        
        should_skip = False
        for pattern in skip_patterns:
            if re.search(pattern, sentence_lower):
                should_skip = True
                break
        
        if not should_skip:
            good_sentences.append(sentence)
    
    if good_sentences:
        if len(good_sentences[0]) > 50:
            result = good_sentences[0]
        elif len(good_sentences) > 1:
            result = good_sentences[0] + '. ' + good_sentences[1]
        else:
            result = good_sentences[0]
        
        if not result.endswith(('.', '!', '?')):
            result += '.'
        
        return result
    
    return ""

def create_proper_input(user_input, tokenizer):
    """Create properly formatted input with attention mask"""
    formats = [
        f"<|user|>\n{user_input}\n<|assistant|>\n",
        f"User: {user_input}\nAssistant:",
        user_input + tokenizer.eos_token
    ]
    
    results = []
    for fmt in formats:
        encoded = tokenizer(
            fmt, 
            return_tensors="pt", 
            truncation=True, 
            max_length=400,
            padding=True,
            return_attention_mask=True
        )
        results.append((encoded, fmt))
    
    return results

def get_model_generation_params(model, tokenizer):
    """Get generation parameters that work with this specific model"""
    # Check if model supports sampling parameters using proper tokenized input
    supports_temperature = supports_top_k = supports_top_p = False
    
    try:
        # Create a proper dummy input with attention mask
        dummy_text = "Hello"
        dummy_encoded = tokenizer(
            dummy_text,
            return_tensors="pt",
            return_attention_mask=True,
            padding=True
        )
        dummy_input = {k: v.to(DEVICE) for k, v in dummy_encoded.items()}
        
        with torch.no_grad():
            # Suppress warnings during testing
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                
                # Try temperature first
                try:
                    model.generate(
                        input_ids=dummy_input["input_ids"],
                        attention_mask=dummy_input["attention_mask"],
                        max_new_tokens=1, 
                        temperature=0.7, 
                        do_sample=True,
                        pad_token_id=tokenizer.pad_token_id
                    )
                    supports_temperature = True
                except:
                    supports_temperature = False
                
                # Try top_k
                try:
                    model.generate(
                        input_ids=dummy_input["input_ids"],
                        attention_mask=dummy_input["attention_mask"],
                        max_new_tokens=1, 
                        top_k=50, 
                        do_sample=True,
                        pad_token_id=tokenizer.pad_token_id
                    )
                    supports_top_k = True
                except:
                    supports_top_k = False
                    
                # Try top_p
                try:
                    model.generate(
                        input_ids=dummy_input["input_ids"],
                        attention_mask=dummy_input["attention_mask"],
                        max_new_tokens=1, 
                        top_p=0.9, 
                        do_sample=True,
                        pad_token_id=tokenizer.pad_token_id
                    )
                    supports_top_p = True
                except:
                    supports_top_p = False
    except:
        supports_temperature = supports_top_k = supports_top_p = False
    
    # Create configurations based on what's supported
    configs = []
    
    # Configuration 1: Conservative
    config1 = {
        "max_new_tokens": 35,
        "repetition_penalty": 1.8,
        "no_repeat_ngram_size": 3,
        "do_sample": True,
    }
    if supports_temperature:
        config1["temperature"] = 0.6
    if supports_top_k:
        config1["top_k"] = 25
    if supports_top_p:
        config1["top_p"] = 0.8
    configs.append(config1)
    
    # Configuration 2: Moderate
    config2 = {
        "max_new_tokens": 50,
        "repetition_penalty": 1.6,
        "no_repeat_ngram_size": 4,
        "do_sample": True,
    }
    if supports_temperature:
        config2["temperature"] = 0.7
    if supports_top_k:
        config2["top_k"] = 30
    if supports_top_p:
        config2["top_p"] = 0.85
    configs.append(config2)
    
    # Configuration 3: Greedy (always works)
    config3 = {
        "max_new_tokens": 30,
        "do_sample": False,
        "repetition_penalty": 1.5,
    }
    configs.append(config3)
    
    return configs, supports_temperature

def generate_response(user_input, tokenizer, model, coherence_model=None, gen_configs=None):
    """Generate response with multiple attempts and validation"""
    
    input_formats = create_proper_input(user_input, tokenizer)
    all_candidates = []
    
    # Try each input format with each generation config
    for input_data, format_name in input_formats:
        input_data = {k: v.to(DEVICE) for k, v in input_data.items()}
        
        for i, config in enumerate(gen_configs):
            try:
                with torch.no_grad():
                    outputs = model.generate(
                        input_ids=input_data["input_ids"],
                        attention_mask=input_data["attention_mask"],
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                        **config
                    )
                
                # Extract only new tokens
                new_tokens = outputs[0][input_data["input_ids"].shape[1]:]
                response = tokenizer.decode(new_tokens, skip_special_tokens=True)
                
                if response.strip():
                    cleaned = clean_and_truncate_response(response)
                    if cleaned:
                        all_candidates.append((cleaned, f"{format_name[:6]}-{i+1}"))
                
            except Exception as e:
                if SHOW_DEBUG:
                    print(f"   🔧 Config {i+1} failed: {str(e)[:50]}")
                continue
    
    # Evaluate candidates
    valid_candidates = []
    for response, method in all_candidates:
        is_problematic, reason = is_response_problematic(response, user_input)
        
        if not is_problematic:
            coherence_score = 0
            if coherence_model:
                try:
                    embeddings = coherence_model.encode([user_input, response])
                    coherence_score = np.dot(embeddings[0], embeddings[1]) / (
                        np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
                    )
                except:
                    coherence_score = 0
            
            valid_candidates.append((response, method, coherence_score))
    
    # Return best candidate
    if valid_candidates:
        if coherence_model:
            valid_candidates.sort(key=lambda x: x[2], reverse=True)
            if valid_candidates[0][2] >= COHERENCE_THRESHOLD:
                return valid_candidates[0][0], valid_candidates[0][1]
        else:
            return valid_candidates[0][0], valid_candidates[0][1]
    
    # Thoughtful fallbacks
    fallback_responses = [
        "I find that question challenging to approach directly.",
        "That's a complex topic that deserves more careful consideration than I can provide.",
        "I'm not sure I have the right framework to answer that well.",
        "That touches on some deep questions I'm still thinking through.",
        "I'd need to reflect more on that before giving you a meaningful response."
    ]
    
    return np.random.choice(fallback_responses), "fallback"

def show_help():
    """Display help information"""
    print("\n📋 Available Commands:")
    print("  /help or /h        - Show this help")
    print("  /debug on/off      - Toggle debug information")
    print("  /coherence on/off  - Toggle coherence checking")
    print("  /stats             - Show session statistics")
    print("  /config            - Show current configuration")
    print("  /clear             - Clear screen")
    print("  /reset             - Reset conversation stats")
    print("  exit/quit/bye      - Exit the program")
    print("\n💡 Tips:")
    print("  • Try rephrasing if you get fallback responses")
    print("  • Shorter inputs often work better")
    print("  • Enable debug mode to see generation details")

def show_stats(conversation_count, fallback_count, method_counts):
    """Show session statistics"""
    success_rate = ((conversation_count - fallback_count) / conversation_count * 100) if conversation_count > 0 else 0
    
    print(f"\n📊 Session Statistics:")
    print(f"  Total responses: {conversation_count}")
    print(f"  Success rate: {success_rate:.1f}%")
    print(f"  Fallbacks: {fallback_count}")
    
    if method_counts:
        print(f"  Generation methods used:")
        for method, count in sorted(method_counts.items()):
            print(f"    {method}: {count}")

def show_config(device, use_coherence, show_debug, coherence_threshold):
    """Show current configuration"""
    print(f"\n⚙️  Current Configuration:")
    print(f"  Device: {device}")
    print(f"  Coherence checking: {'ON' if use_coherence else 'OFF'}")
    print(f"  Coherence threshold: {coherence_threshold}")
    print(f"  Debug mode: {'ON' if show_debug else 'OFF'}")

def main():
    global USE_COHERENCE, SHOW_DEBUG, COHERENCE_THRESHOLD
    
    print("🧠 RhizomeML Chat - Enhanced Version")
    print("🔧 Fixed generation parameters and added commands")
    print("📝 Type '/help' for commands or 'exit' to quit.\n")
    
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tokenizer, model = load_model_and_tokenizer()
        
        # Get supported generation parameters
        gen_configs, supports_temp = get_model_generation_params(model, tokenizer)
        
        if USE_COHERENCE:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                coherence_model = SentenceTransformer("all-MiniLM-L6-v2")
        else:
            coherence_model = None
        
        print(f"✅ Model loaded successfully")
        print(f"🔧 Device: {DEVICE}")
        print(f"📏 Vocab size: {len(tokenizer)}")
        print(f"🎯 Coherence checking: {'ON' if USE_COHERENCE else 'OFF'}")
        print(f"🌡️  Temperature support: {'YES' if supports_temp else 'NO'}")
        
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return
    
    conversation_count = 0
    fallback_count = 0
    method_counts = {}
    
    while True:
        try:
            user_input = input("\nYou: ").strip()
            
            # Handle commands
            if user_input.lower().startswith('/'):
                cmd = user_input.lower()
                
                if cmd in ['/help', '/h']:
                    show_help()
                    continue
                elif cmd.startswith('/debug'):
                    if 'on' in cmd:
                        SHOW_DEBUG = True
                        print("🔧 Debug mode: ON")
                    elif 'off' in cmd:
                        SHOW_DEBUG = False
                        print("🔧 Debug mode: OFF")
                    else:
                        print(f"🔧 Debug mode: {'ON' if SHOW_DEBUG else 'OFF'}")
                    continue
                elif cmd.startswith('/coherence'):
                    if 'on' in cmd:
                        USE_COHERENCE = True
                        if coherence_model is None:
                            coherence_model = SentenceTransformer("all-MiniLM-L6-v2")
                        print("🎯 Coherence checking: ON")
                    elif 'off' in cmd:
                        USE_COHERENCE = False
                        print("🎯 Coherence checking: OFF")
                    else:
                        print(f"🎯 Coherence checking: {'ON' if USE_COHERENCE else 'OFF'}")
                    continue
                elif cmd == '/stats':
                    show_stats(conversation_count, fallback_count, method_counts)
                    continue
                elif cmd == '/config':
                    show_config(DEVICE, USE_COHERENCE, SHOW_DEBUG, COHERENCE_THRESHOLD)
                    continue
                elif cmd == '/clear':
                    os.system('clear' if os.name == 'posix' else 'cls')
                    continue
                elif cmd == '/reset':
                    conversation_count = 0
                    fallback_count = 0
                    method_counts = {}
                    print("📊 Statistics reset")
                    continue
                else:
                    print("❓ Unknown command. Type '/help' for available commands.")
                    continue
            
            if user_input.lower() in ['exit', 'quit', 'bye']:
                show_stats(conversation_count, fallback_count, method_counts)
                print("👋 Goodbye!")
                break
            
            if not user_input:
                continue
            
            conversation_count += 1
            response, method = generate_response(
                user_input, 
                tokenizer, 
                model, 
                coherence_model if USE_COHERENCE else None,
                gen_configs
            )
            
            if method == "fallback":
                fallback_count += 1
            
            # Track method usage
            method_counts[method] = method_counts.get(method, 0) + 1
            
            # Method indicators
            method_emoji = {
                "fallback": "🆘",
                "<|user": "🤖",
                "User:": "💬",
            }
            
            emoji = "❓"
            for key, emj in method_emoji.items():
                if key in method:
                    emoji = emj
                    break
            
            print(f"Bot {emoji}: {response}")
            
            # Show debugging info
            if SHOW_DEBUG or (method == "fallback" and conversation_count > 1):
                if method == "fallback":
                    print(f"   💭 Using fallback - model couldn't generate coherent response")
                else:
                    print(f"   🔧 Method: {method}")
            
        except KeyboardInterrupt:
            print("\n👋 Exiting.")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            if SHOW_DEBUG:
                import traceback
                traceback.print_exc()

if __name__ == "__main__":
    main()
