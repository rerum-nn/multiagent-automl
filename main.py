from auto_eda.ds_star import DSStar
import os
from langchain_core.language_models.chat_models import BaseChatModel
from utility.chat_model import GatewayChatModel

from dotenv import load_dotenv


load_dotenv()


def build_llm_from_env() -> BaseChatModel:
    """
    Build a provider-agnostic LangChain chat model from env vars.

    """
    return GatewayChatModel(
        model=os.getenv("LLM_MODEL"),
        base_url=os.getenv("LLM_BASE_URL"),
        api_key=os.getenv("LLM_API_KEY"),
    )


def main():
    llm = build_llm_from_env()
    ds_star = DSStar(llm)

    query = "Generate a Python script that counts the number of lines in a file."
    workspace = "./workspace"
    max_rounds = 20
    max_debug_steps = 5
    finalyze = True

    success, plans, final_code_filename, final_output = ds_star.run(
        query, workspace, max_rounds, max_debug_steps, finalyze
    )

    if success:
        print("Task solved successfully!")
        print(f"Final code: {final_code_filename}")
        print(f"Final output: {final_output}")
        print(f"Plans:", '\n'.join([f'{i+1}. {plan}' for i, plan in enumerate(plans)]))
    else:
        print("Task failed to solve!")
        print(f"Plans:", '\n'.join([f'{i+1}. {plan}' for i, plan in enumerate(plans)]))


if __name__ == "__main__":
    main()

