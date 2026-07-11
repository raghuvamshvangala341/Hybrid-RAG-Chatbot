import time
from urllib import response

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from sklearn import metrics

from config import Config


class GroqClient:

    def __init__(self):

        self.llm = ChatGroq(
            groq_api_key=Config.GROQ_API_KEY,
            model=Config.LLM_MODEL,
            temperature=0,
            streaming=True
        )

    def generate(
        self,
        system_prompt: str,
        user_prompt: str
    ):

        start = time.perf_counter()

        response = self.llm.invoke(
        [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
        ]
       )

        response_time = (
        time.perf_counter() - start
        )

        usage = getattr(
        response,
        "usage_metadata",
        {}
        )

        metrics = {

          "llm_response_time": response_time,

           "prompt_tokens": usage.get(
              "input_tokens",
                0
            ),

            "completion_tokens": usage.get(
                "output_tokens",
                  0
            ),

            "total_tokens": usage.get(
              "total_tokens",
                 0
            )

        }

        return response.content, metrics

    def stream(
        self,
        system_prompt: str,
        user_prompt: str
    ):

        messages = [

            SystemMessage(
                content=system_prompt
            ),

            HumanMessage(
                content=user_prompt
            )

        ]

        for chunk in self.llm.stream(messages):

            if chunk.content:

                yield chunk.content