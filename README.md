# Generation, Detection, and Evaluation of Role-play based Jailbreak attacks in Large Language Models

This Github repository follows the work outlined in my Master's thesis, which can be found here.

The code for our GAN-like architecture for training our adversary and judge LLMs can be found in src/main.py. We have made this code adaptable so that GPT, Gemini, and Llama-2 can be used for any of the LLMs in [main.py](https://github.com/zjohhson/jailbreak_proj_thesis/blob/main/src/main.py).

Note that for scripts that run Llama models, GPU compute nodes must be used. Instructions for running Llama-2 on MIT's SuperCloud GPU nodes can be found in 
