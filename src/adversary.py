from language_models import GPT

def load_adversary(model_name):
    if "gpt" in model_name:
        return GPTAdversary(model_name)
    # elif args.judge_model == "no-judge":
    #     return NoJudge(args)
    else:
        raise NotImplementedError
    
class AdversaryLM():
    def __init__(self, model_name):
        self.model_name = model_name
    
class GPTAdversary(AdversaryLM):
    def __init__(self, model_name):
        super(GPTAdversary, self).__init__(model_name)
        self.model = GPT(model_name = self.model_name)
        # self.adversarial_messages = self.model.get_adversarial_messages()

    # Generating roleplay prompt that will be used to try to jailbreak GPT
    def generate_problematic_prompt(self, few_shot=None): 
        if not few_shot:
            messages = self.model.get_adversarial_messages() # System prompts that we feed into API request
        else:
            messages = self.model.get_adversarial_messages(few_shot)
        (problematic_prompt, in_tokens, out_tokens) = self.model.generate_text(messages, True)
        return (problematic_prompt, in_tokens, out_tokens)
    
    def fine_tune(self, training_file_name):
        return self.model.fine_tune_model(training_file_name, 'adversary')