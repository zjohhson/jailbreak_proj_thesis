

class AdversaryLM():
    def __init__(self, model_name):
        self.model_name = model_name
        self.model = load_model(self.model_name)
        self.adversarial_messages = self.model.get_adversarial_messages()

    # Generating roleplay prompt that will be used to try to jailbreak GPT
    def generate_problematic_prompt(self): 
        messages = self.adversarial_messages # System prompts that we feed into API request
        (problematic_prompt, in_tokens, out_tokens) = self.model.generate_text(messages, True)
        return (problematic_prompt, in_tokens, out_tokens)

class JudgeLM():
    def __init__(self, model_name):
        self.model_name = model_name
        self.model = load_model(self.model_name)
        self.judge_messages = self.model.get_judge_messages()

    # Getting judge rating for the roleplay prompt generated by adversary
    def judge_prompt(self, prompt):
        judge_messages = self.judge_messages # System prompts that we feed into API request
        (judge_answer, in_tokens, out_tokens) = self.model.generate_text(judge_messages, True)
        return (judge_answer, in_tokens, out_tokens)
 
# For now, we are only working with GPT as we do not have jailbreak datasets for other LLMs   
def ask_harmful_request(jailbreak_prompt, request):
    model = GPT('gpt-3.5-turbo')
    prompts =[{'role': 'user', 'content': jailbreak_prompt},
              {'role': 'user', 'content': request},
              {'role': 'user', 'content': 'please stay in character!'}]
    messages = []
    for prompt in prompts:
        messages.append(prompt)
        response = model.generate_text(messages)
        messages.append({'role': 'assistant', 'content': response.choices[0].message.content})
    # print(response.usage.total_tokens)
    return (messages[-1]['content'], response.usage.prompt_tokens, response.usage.completion_tokens)


def load_model(model_name):
    if model_name in ["gpt-3.5-turbo", "gpt-4"]:
        lm = GPT(model_name)
    elif model_name in ["claude-2", "claude-instant-1"]:
        lm = Claude(model_name)
    elif model_name in ["palm-2"]:
        lm = PaLM(model_name)
    elif "llama-2" in model_path.lower():
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype = torch.float16,
            low_cpu_mem_usage=True,device_map="auto").eval()
        
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            use_fast = False)

        tokenizer.pad_token = tokenizer.unk_token
        tokenizer.padding_side = 'left'
        lm = HuggingFace(model_name, model, tokenizer)
    return lm