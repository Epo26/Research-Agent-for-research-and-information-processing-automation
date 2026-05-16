from langchain_google_genai import ChatGoogleGenerativeAI
from .config import CONFIG
from dotenv import load_dotenv

load_dotenv()

llm_cheap = ChatGoogleGenerativeAI(
    model=CONFIG["llm"]["cheap_model"],
    temperature=CONFIG["llm"]["cheap_temperature"],
)

llm_smart = ChatGoogleGenerativeAI(
    model=CONFIG["llm"]["smart_model"],
    temperature=CONFIG["llm"]["smart_temperature"],
)

llm_judge = ChatGoogleGenerativeAI(
    model=CONFIG["llm"]["judge_model"],
    temperature=CONFIG["llm"]["judge_temperature"],
)