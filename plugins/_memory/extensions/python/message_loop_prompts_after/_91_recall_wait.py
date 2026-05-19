import asyncio

from helpers.extension import Extension
from agent import LoopData
from plugins._memory.extensions.python.message_loop_prompts_after._50_recall_memories import (
    DATA_NAME_TASK as DATA_NAME_TASK_MEMORIES,
    DATA_NAME_ITER as DATA_NAME_ITER_MEMORIES,
    SEARCH_TIMEOUT,
)
from helpers import errors, plugins

class RecallWait(Extension):
    async def execute(self, loop_data: LoopData = LoopData(), **kwargs):

        if not self.agent:
            return

        set = plugins.get_plugin_config("_memory", self.agent)
        if not set:
            return None

        task = self.agent.get_data(DATA_NAME_TASK_MEMORIES)
        iter = self.agent.get_data(DATA_NAME_ITER_MEMORIES) or 0

        if task and not task.done():

            # if memory recall is set to delayed mode, do not await on the iteration it was called
            if set["memory_recall_delayed"]:
                if iter == loop_data.iteration:
                    # insert info about delayed memory to extras
                    delay_text = self.agent.read_prompt("memory.recall_delay_msg.md")
                    loop_data.extras_temporary["memory_recall_delayed"] = delay_text
                    return

            # otherwise await the task. Memory recall is helpful but must never
            # abort the monologue: embedding backends can timeout or transiently
            # fail while FAISS is preparing the query vector. Convert those
            # failures to visible warnings and continue without recalled memory.
            try:
                await task
            except TimeoutError:
                self.agent.context.log.log(
                    type="warning",
                    heading="Memory recall timed out",
                    content=(
                        f"FAISS memory recall exceeded {SEARCH_TIMEOUT}s while "
                        "embedding/searching; continuing without recalled memories."
                    ),
                )
                self.agent.set_data(DATA_NAME_TASK_MEMORIES, None)
            except asyncio.CancelledError:
                # Preserve real cancellation semantics for shutdown/session abort.
                raise
            except Exception as e:
                self.agent.context.log.log(
                    type="warning",
                    heading="Memory recall failed",
                    content=errors.format_error(e),
                )
                self.agent.set_data(DATA_NAME_TASK_MEMORIES, None)
