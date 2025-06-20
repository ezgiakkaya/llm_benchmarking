import os
from dotenv import load_dotenv


load_dotenv()


print("Testing environment variables:")
print("-" * 50)


env_vars = {
    'PINECONE_API_KEY': os.getenv('PINECONE_API_KEY'),
    'PINECONE_ENVIRONMENT': os.getenv('PINECONE_ENVIRONMENT'),
    'PINECONE_INDEX_NAME': os.getenv('PINECONE_INDEX_NAME'),
    'GROQ_API_KEY': os.getenv('GROQ_API_KEY')
}

for var_name, value in env_vars.items():
    print(f"{var_name}: {'✓ Set' if value else '✗ Not Set'}")


print("\nValues (masked):")
print("-" * 50)
for var_name, value in env_vars.items():
    if value:
        masked_value = value[:4] + '*' * (len(value) - 4) if len(value) > 4 else '****'
        print(f"{var_name}: {masked_value}")
    else:
        print(f"{var_name}: None") 