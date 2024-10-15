OPENAI_MODELS = [
    "o1-mini-2024-09-12",
    "o1-preview-2024-09-12",
    "gpt-4o-2024-08-06",
    "gpt-4o-2024-05-13",
    "gpt-4o-mini-2024-07-18",
    "gpt-4-turbo-2024-04-09",
    "gpt-3.5-turbo-0125",
]

# for m in genai.list_models():
#   if 'generateContent' in m.supported_generation_methods:
#     print(m.name)
GOOGLE_MODELS = [
    "gemini-1.5-pro-latest",
    "gemini-1.0-pro-latest",
]

# https://docs.anthropic.com/en/docs/about-claude/models
ANTHROPIC_MODELS = [
    "claude-3-5-sonnet-20240620",
    "claude-3-opus-20240229",
    "claude-3-sonnet-20240229",
    "claude-3-haiku-20240307",
]

# https://docs.mistral.ai/getting-started/models/
MISTRAL_MODELS = [
    "mistral-large-latest",
    "mistral-large-2407",
    "mistral-small-latest",
    "open-mixtral-8x7b",
    "open-mistral-7b",
    "open-mixtral-8x22b",
]

ALL_MODELS = OPENAI_MODELS + GOOGLE_MODELS + ANTHROPIC_MODELS + MISTRAL_MODELS


from google.generativeai.types import HarmCategory, HarmBlockThreshold
GOOGLE_SAFETY_SETTINGS = {
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
}


ALLOWED_ENCLOSURES = ["`", "\"", "'", "*", " ", "\n", "."]
ALLOWED_DELIMITERS = [",", " ", ", ", "|", " | ", "\n"]