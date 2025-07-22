from autogen_core.memory import Memory, MemoryContent, MemoryMimeType, MemoryQueryResult, UpdateContextResult
from autogen_core.models import SystemMessage
from vectorstore import PromptChromaStore


class ChromaPromptMemory(Memory):
    """
    Read-only Chroma memory. Inserts each (task, prompt) snippet **once**:
    duplicates already present in model_context are skipped.
    """

    def __init__(self, store: PromptChromaStore, k: int = 3):
        super().__init__()
        self.store, self.k = store, k

    def _extract_text(self, content_item: str | MemoryContent) -> str:
        """Extract searchable text from content."""
        if isinstance(content_item, str):
            return content_item

        content = content_item.content
        mime_type = content_item.mime_type

        if mime_type in [MemoryMimeType.TEXT, MemoryMimeType.MARKDOWN]:
            return str(content)
        elif mime_type == MemoryMimeType.JSON:
            if isinstance(content, dict):
                # Store original JSON string representation
                return str(content).lower()
            raise ValueError("JSON content must be a dict")
        # elif isinstance(content, Image):
        #   raise ValueError("Image content cannot be converted to text")
        else:
            raise ValueError(f"Unsupported content type: {mime_type}")

    async def query(self, query_text):
        task_desc = query_text  # refined task line
        rows = self.store.search(task_desc, self.k)
        memory_contents = []
        for i, record in enumerate(rows):
            memory_contents.append(
                MemoryContent(mime_type=MemoryMimeType.TEXT,
                              content=f"### Example {i + 1}\nTask: {record['task']}\nPrompt:\n{record['prompt']}",

                              ))

        return MemoryQueryResult(results=memory_contents)

    async def update_context(self, model_context):
        messages = await model_context.get_messages()
        if not messages:
            return UpdateContextResult(memories=MemoryQueryResult(results=[]))
        last_message = messages[-1]
        query_text = last_message.content if isinstance(last_message.content, str) else str(last_message)
        # all memory text
        model_context_messages_string = ' '.join([x.content for x in messages])
        # Query memory and get results
        query_results = await self.query(query_text)

        memory_strings = []
        for record in query_results.results:
            # record already in agent memory
            if record.content not in model_context_messages_string:
                memory_strings.append(record.content)
        if len(memory_strings) > 0:
            memory_context = "\nSome Examples:\n" + "\n".join(memory_strings)
            # may add 'you can reuse the examples given to you previous'
            # Add to context
            await model_context.add_message(SystemMessage(content=memory_context))
        return UpdateContextResult(memories=query_results)

    async def close(self):
        self.store.close()

    async def clear(self):
        self.store.clear()

    async def add(self):
        pass
        '''
   
        try:
            # Extract text from content
            text = self._extract_text(content)
            tasks, prompts = self.parse(text)

            # Use metadata directly from content
            metadata_dict = content.metadata or {}
            metadata_dict["mime_type"] = str(content.mime_type)

            # Add to ChromaDB
            self.store.add_prompts( task_descriptions =tasks, prompts=prompts )

        except Exception as e:
            logger.error(f"Failed to add content to ChromaDB: {e}")
            raise
        '''

    def parse(self, text):
        raise NotImplementedError
