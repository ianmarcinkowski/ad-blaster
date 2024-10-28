""" AdBlaster """

import json
from ollama import Client as OllamaClient
from ad_blaster.util import ns_to_ms


class AdBlaster:

    def __init__(self, llm_hostname):
        self.llm_hostname = llm_hostname
        self.ollama = OllamaClient(host=llm_hostname)

    def ask(self, image_content):
        system_prompt = """
        You are assisting the user to describe the contents of the television programs they are watching.
        Please categorize the contents on screen with common types of programs that a user may see on screen.  For example:
          - Sports
          - Drama
          - Comedy
          - Advertisement
          - Blank screen
          - Sci-fi
        """

        prompt = """
        Please describe and categorize the images on the television screen that I am watching right now.
        """
        response = self.ollama.chat(
            model="x/llama3.2-vision",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt, "images": [image_content]},
            ],
        )
        duration = ns_to_ms(response["total_duration"])
        print(f"Ask duration: {duration}")
        print(json.dumps(response["message"], indent=2))
