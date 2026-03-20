from langchain_groq import ChatGroq
from .config import CONFIG
from dotenv import load_dotenv

load_dotenv()

llm_cheap = ChatGroq(model=CONFIG["llm"]["cheap_model"], temperature=CONFIG["llm"]["cheap_temperature"])
llm_smart = ChatGroq(model=CONFIG["llm"]["smart_model"], temperature=CONFIG["llm"]["smart_temperature"])
llm_judge = ChatGroq(model=CONFIG["llm"]["judge_model"], temperature=CONFIG["llm"]["judge_temperature"])