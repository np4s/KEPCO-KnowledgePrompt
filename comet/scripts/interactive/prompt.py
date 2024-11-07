import src.interactive.functions as interactive
import src.data.config as cfg
import src.data.data as data
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import os
import sys
import argparse
import torch

sys.path.append(os.getcwd())


if __name__ == "__main__":

    tokenizer = AutoTokenizer.from_pretrained("facebook/blenderbot_small-90M")
    model = AutoModelForSeq2SeqLM.from_pretrained(
        "facebook/blenderbot_small-90M")

    model_file = "pretrained_models/atomic_pretrained_model.pickle"
    
    opt, state_dict = interactive.load_model_file(model_file)
    data_loader, text_encoder = interactive.load_data("atomic", opt)

    n_ctx = data_loader.max_event + data_loader.max_effect
    n_vocab = len(text_encoder.encoder) + n_ctx
    model = interactive.make_model(opt, n_vocab, n_ctx, state_dict)

    cfg.device = "cpu"

    sampling_algorithm = "beam-3"
    sampler = interactive.set_sampler(opt, sampling_algorithm, data_loader)

    def expand_input(prompt):

        wants = interactive.get_atomic_sequence(
            prompt, model, sampler, data_loader, text_encoder, "xWant")
        needs = interactive.get_atomic_sequence(
            prompt, model, sampler, data_loader, text_encoder, "xNeed")

        for want in wants["beams"]:
            prompt = prompt + f" want to {want}"
        for need in needs["beams"]:
            prompt = prompt + f" need to {need}"

        return prompt

    while True:
        prompt = input("Human: ")
        if prompt == "exit":
            break
        prompt = expand_input(prompt)
        input = tokenizer([prompt], return_tensors="pt")
        output = model.generate(**input)
        print("Bot:", tokenizer.batch_decode(output, skip_special_tokens=True)[0])
