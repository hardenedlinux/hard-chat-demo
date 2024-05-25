"""-*-  indent-tabs-mode:nil; coding: utf-8 -*-.

Copyright (C) 2024
    HardenedLinux community
This is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.
If not, see <http://www.gnu.org/licenses/>.

"""

import os
from typing import Iterator
import gradio as gr
from model import LLAMA_wrapper
from config import config

llama_wrapper = LLAMA_wrapper(config)

DESCRIPTION = """
# Hard-Chat

Hard-Chat is a chatbot that can talk about anything you want. It's a demo for [HardenedLinux AI Infra best practices](https://github.com/hardenedlinux/ai-infra/).

You must abide by the Llama use policy (https://ai.meta.com/llama/use-policy/).
"""


def clear_and_save_textbox(message: str) -> tuple[str, str]:
    """Clear the textbox and save the input."""
    return "", message


def display_input(message: str, history: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """Display the input in the chat history."""
    history.append((message, ""))
    return history


def delete_prev_fn(history: list[tuple[str, str]]) -> tuple[list[tuple[str, str]], str]:
    """Delete the previous message in the chat history."""
    try:
        message, _ = history.pop()
    except IndexError:
        message = ""
    return history, message or ""


def generate(message: str,
             history_with_input: list[tuple[str, str]],
             system_prompt: str,
             max_new_tokens: int,
             temperature: float,
             top_p: float,
             top_k: int
             ) -> Iterator[list[tuple[str, str]]]:
    """Generate a response."""
    if max_new_tokens > config["max_new_tokens"]:
        raise ValueError

    history = history_with_input[:-1]
    generator = llama_wrapper.run(
        message, history, system_prompt, max_new_tokens, temperature, top_p, top_k
    )
    try:
        first_response = next(generator)
        yield history + [(message, first_response)]
    except StopIteration:
        yield history + [(message, "")]
    for response in generator:
        yield history + [(message, response)]


def process_example(message: str) -> tuple[str, list[tuple[str, str]]]:
    """Process an example."""
    generator = generate(message, [], config["prompt"], 1024, 1, 0.95, 50)
    for x in generator:
        pass
    return "", x


def check_input_token_length(message: str,
                             chat_history: list[tuple[str, str]],
                             system_prompt: str
                             ) -> None:
    """Check the input token length."""
    input_token_length = llama_wrapper.get_input_token_length(
        message, chat_history, system_prompt
    )
    if input_token_length > config["max_input"]:
        errmsg = f"The accumulated input is too long ({input_token_length} >" \
                 f"{config['max_input']})." \
                 f"Clear your chat history and try again."
        raise gr.Error(errmsg)


with gr.Blocks(css="style.css") as demo:
    gr.Markdown(DESCRIPTION)

    with gr.Group():
        chatbot = gr.Chatbot(label="Hard-Chat")
        with gr.Row():
            textbox = gr.Textbox(container=False,
                                 show_label=False,
                                 placeholder="Type a message...",
                                 scale=10)
            submit_button = gr.Button(value="Submit", variant="primary", scale=1, min_width=0)
    with gr.Row():
        retry_button = gr.Button("🔄  Retry", variant="secondary")
        undo_button = gr.Button("↩️ Undo", variant="secondary")
        clear_button = gr.Button("🗑️  Clear", variant="secondary")

    saved_input = gr.State()

    with gr.Accordion(label="Advanced options", open=False):
        system_prompt = gr.Textbox(
             label="System prompt", value=config["prompt"], lines=6,
        )
        max_new_tokens = gr.Slider(
            label="Max new tokens",
            minimum=1,
            maximum=config["max_new_tokens"],
            step=1,
            value=config["max_new_tokens"],
        )
        temperature = gr.Slider(
            label="Temperature",
            minimum=0.1,
            maximum=4.0,
            step=0.1,
            value=1.0
        )
        top_p = gr.Slider(
            label="Top-p (nucleus sampling)",
            minimum=0.05,
            maximum=1.0,
            step=0.05,
            value=0.95
        )
        top_k = gr.Slider(
            label="Top-k",
            minimum=1,
            maximum=1000,
            step=1,
            value=50
        )

    gr.Examples(
        examples=[
            "Hello there! How are you doing?",
            "Can you explain briefly to me what is the Python programming language?",
            "Explain the plot of Cinderella in a sentence.",
            "How many hours does it take a man to eat a Helicopter?",
            "Write a 100-word article on 'Benefits of Open-Source in AI research'",
        ],
        inputs=textbox,
        outputs=[textbox, chatbot],
        fn=process_example,
        cache_examples=True,
    )

    textbox.submit(
        fn=clear_and_save_textbox,
        inputs=textbox,
        outputs=[textbox, saved_input],
        api_name=False,
        queue=False,
    ).then(
        fn=display_input,
        inputs=[saved_input, chatbot],
        outputs=chatbot,
        api_name=False,
        queue=False,
    ).then(
        fn=check_input_token_length,
        inputs=[saved_input, chatbot, system_prompt],
        api_name=False,
        queue=False,
    ).success(
        fn=generate,
        inputs=[
            saved_input,
            chatbot,
            system_prompt,
            max_new_tokens,
            temperature,
            top_p,
            top_k,
        ],
        outputs=chatbot,
        api_name=False,
    )

    button_event_preprocess = (
        submit_button.click(
            fn=clear_and_save_textbox,
            inputs=textbox,
            outputs=[textbox, saved_input],
            api_name=False,
            queue=False,
        )
        .then(
            fn=display_input,
            inputs=[saved_input, chatbot],
            outputs=chatbot,
            api_name=False,
            queue=False,
        )
        .then(
            fn=check_input_token_length,
            inputs=[saved_input, chatbot, system_prompt],
            api_name=False,
            queue=False,
        )
        .success(
            fn=generate,
            inputs=[
                saved_input,
                chatbot,
                system_prompt,
                max_new_tokens,
                temperature,
                top_p,
                top_k,
            ],
            outputs=chatbot,
            api_name=False,
        )
    )

    retry_button.click(
        fn=delete_prev_fn,
        inputs=chatbot,
        outputs=[chatbot, saved_input],
        api_name=False,
        queue=False,
    ).then(
        fn=display_input,
        inputs=[saved_input, chatbot],
        outputs=chatbot,
        api_name=False,
        queue=False,
    ).then(
        fn=generate,
        inputs=[
            saved_input,
            chatbot,
            system_prompt,
            max_new_tokens,
            temperature,
            top_p,
            top_k,
        ],
        outputs=chatbot,
        api_name=False,
    )

    undo_button.click(
        fn=delete_prev_fn,
        inputs=chatbot,
        outputs=[chatbot, saved_input],
        api_name=False,
        queue=False,
    ).then(
        fn=lambda x: x,
        inputs=[saved_input],
        outputs=textbox,
        api_name=False,
        queue=False,
    )

    clear_button.click(
        fn=lambda: ([], ""),
        outputs=[chatbot, saved_input],
        queue=False,
        api_name=False,
    )


def get_server_name() -> str:
    """
    Get the server name.

    First try to get it from /run/alexon/env/host.
    If it doesn't exist, return getenv('DOMAIN_NAME').
    Otherwie return 0.0.0.0.
    """
    try:
        with open("/run/alexon/env/host") as f:
            return f.read().strip()
    except FileNotFoundError:
        return os.getenv("DOMAIN_NAME", "0.0.0.0")


demo.queue(max_size=20).launch(server_name=get_server_name())
