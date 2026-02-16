from ds_star import DSStar
from openai import OpenAI
import os
import httpx
import certifi

client = OpenAI(base_url=os.getenv("OPENAI_BASE_URL"), api_key=os.getenv("OPENAI_API_KEY"))
model_name = "gpt-4o"
ds_star = DSStar(client, model_name)

def main():
    query = "Generate a Python script that counts the number of lines in a file."
    workspace = "./workspace"
    max_rounds = 20
    max_debug_steps = 5
    finalyze = True

    success, plans, final_code_filename, final_output = ds_star.run(query, workspace, max_rounds, max_debug_steps, finalyze)

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

