# transformers_chat_client.py
import asyncio, torch, time
from typing import List, Dict, Optional
from autogen_core.models import (
    ChatCompletionClient, CreateResult, RequestUsage,
    UserMessage, AssistantMessage, SystemMessage
)
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

_TEMPLATE = "<|im_start|>{role}\n{content}<|im_end|>\n"
_ASSISTANT_PREFIX = "<|im_start|>assistant\n"

def _format_prompt(msgs: List[object]) -> str:
    parts = []
    for m in msgs:
        role = ("system" if isinstance(m, SystemMessage)
                else "assistant" if isinstance(m, AssistantMessage)
                else "user")
        parts.append(_TEMPLATE.format(role=role, content=m.content))
    # Ask the model to continue as assistant
    if not isinstance(msgs[-1], AssistantMessage):
        parts.append(_ASSISTANT_PREFIX)
    return "".join(parts)

class TransformersChatCompletionClient(ChatCompletionClient):
    """
    In-process HF Transformers backend (MPS-accelerated).
    Implements all abstract methods required by autogen-core>=0.2.0.
    """

    def __init__(self,
                 model_id: str = "Qwen/Qwen2-1.5B-Instruct",
                 max_new_tokens: int = 512,
                 temperature: float = 0.7):
        #super().__init__(model=model_id)
        self._model_id = model_id

        self._tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        self._model = (AutoModelForCausalLM
                       .from_pretrained(model_id,
                                        device_map={"": "mps"},
                                        torch_dtype=torch.float16)
                       .to(memory_format=torch.channels_last)
                       .eval())
        self._pipe = pipeline("text-generation",
                              model=self._model,
                              tokenizer=self._tok,
                              model_kwargs=dict(
                                  max_new_tokens=max_new_tokens,
                                  temperature=temperature,
                                  do_sample=temperature > 0,
                                  pad_token_id=self._tok.eos_token_id,
                              ))
        # simple bookkeeping
        self._total_prompt = 0
        self._total_completion = 0
        self._context_len = getattr(self._model.config, "max_position_embeddings", 32768)
        self.model_info = self.model_info()

    # ----------  Core async generation  ----------
    async def create(self, messages: List[object], **_) -> CreateResult:
        prompt = _format_prompt(messages)
        loop = asyncio.get_running_loop()
        t0 = time.time()
        generated = await loop.run_in_executor(
            None,
            lambda: self._pipe(prompt, return_full_text=False)[0]["generated_text"]
        )
        latency = time.time() - t0
        prompt_toks      = len(self._tok.encode(prompt))
        completion_toks  = len(self._tok.encode(generated))
        self._total_prompt     += prompt_toks
        self._total_completion += completion_toks

        return CreateResult(
            finish_reason="stop",
            content=generated.strip(),
            usage=RequestUsage(prompt_tokens=prompt_toks,
                               completion_tokens=completion_toks),
            cached=False,
            logprobs=None,
            thought=f"(elapsed {latency:.2f}s)"
        )

    # ----------  *Required* abstract stubs ----------

    async def create_stream(self, *_, **__) -> None:
        """Streaming not yet implemented for this client."""
        raise NotImplementedError("Streaming not implemented. Use create().")

    def count_tokens(self, messages: List[object]) -> int:
        return len(self._tok.encode(_format_prompt(messages)))

    # Per-request usage is already returned by create(); this helper exposes it again
    def actual_usage(self, messages: List[object], reply: str) -> RequestUsage:
        return RequestUsage(self.count_tokens(messages),
                            len(self._tok.encode(reply)))

    # Running totals since instantiation
    def total_usage(self) -> RequestUsage:
        return RequestUsage(self._total_prompt, self._total_completion)

    def remaining_tokens(self) -> Optional[int]:
        # simplistic: assume single-turn prompt
        return self._context_len - self._total_prompt

    def capabilities(self) -> Dict[str, bool]:
        return {"streaming": False, "json_mode": False}

    def model_info(self) -> Dict[str, object]:
        return {
            "model_id": self._model_id,
            "context_length": self._context_len,
            "dtype": str(self._model.dtype),
             "vision": False,
            "function_calling": False,
         "json_output" :True,
        "family":  "qwen",
        "structured_output": False,
        "multiple_system_messages": False,
        }

    # ----------  House-keeping ----------
    async def close(self) -> None:
        # nothing explicit to free: Python GC will release GPU memory
        pass

async def main():
    llm = TransformersChatCompletionClient()
    resp = await llm.create(
        [UserMessage(content="What is the capital of France?", source="user")]
    )
    print(repr(resp.content))      # ► Paris
    await llm.close()
    return resp


if __name__ == "__main__":
    import asyncio
    print(asyncio.run(main()))

