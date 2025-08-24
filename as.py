import json
from convokit import Corpus, download

# We need a better tokenizer
def count_tokens(text):
    """Approximates the number of tokens by counting words."""
    return len(text.split())

def format_case_facts(conversation):
    """
    Creates a richer, formatted string with the facts of the case
    using all available relevant metadata.
    """
    meta = conversation.meta
    
    # Extract all relevant metadata with fallbacks. However, a lot of these return "N/A", so we gotta fix that down the line...
    title = meta.get('title', 'N/A')
    petitioner = meta.get('petitioner', 'N/A')
    respondent = meta.get('respondent', 'N/A')
    year = meta.get('year', 'N/A')
    docket = meta.get('docket_no', 'N/A')
    citation = meta.get('citation', 'N/A')
    win_side_code = meta.get('win_side')
    win_side = 'Petitioner' if win_side_code == 1 else 'Respondent' if win_side_code == 2 else 'N/A'


    facts = (
        f"Case Information Summary:\n"
        f"- Case Title: {title}\n"
        f"- Docket Number: {docket}\n"
        f"- Year of Argument: {year}\n"
        f"- Citation: {citation}\n\n"
        f"Parties:\n"
        f"- Petitioner (appealing party): {petitioner}\n"
        f"- Respondent: {respondent}\n\n"
        f"Known Outcome (for context): The case was ultimately decided in favor of the {win_side}."
    )
    return facts

def format_context_for_prompt(context_messages):
    """
    Converts the list of context messages into a single formatted string.
    """
    # The first message is the system prompt with case facts, which we'll handle separately
    case_facts = context_messages[0]['content']
    
    #  conversational turns
    dialogue_history = []
    for msg in context_messages[1:]: # Skip system msg
        speaker_role = "Justice" if msg['role'] == 'user' else "Advocate"
        dialogue_history.append(f"{speaker_role}: {msg['content']}")
        
    return f"{case_facts}\n\nOral Argument History:\n" + "\n".join(dialogue_history)


def create_targeted_examples(corpus, max_tokens=1500):
    """
    Generates training examples where the target is a Justice's question.
    """
    targeted_examples = []

    
    for conversation in corpus.iter_conversations():
        # Get the formatted case facts for context
        case_facts = format_case_facts(conversation)
        
        # The first message is always the system prompt with the facts of the case
        base_context = [{"role": "system", "content": case_facts}]
        sorted_utterances = sorted(conversation.iter_utterances(), key=lambda utt: utt.timestamp or 0)
        current_context = list(base_context)
        for utt in sorted_utterances:
            speaker_type = utt.speaker.meta.get('type')
            if speaker_type == 'J': # speaker is a justice
                final_context = list(current_context) # Make a copy
                total_tokens = sum(count_tokens(msg['content']) for msg in final_context)

                # Trim from oldest messages (skipping the system prompt) if over the limit
                while total_tokens > max_tokens and len(final_context) > 1:
                    removed_message = final_context.pop(1) 
                    total_tokens -= count_tokens(removed_message['content'])
                developer_message = (
                    "You are a Supreme Court Justice who will ask a question. "
                    "You are provided with previous conversations from the Oral Argument. "
                    "Continue the line of questioning or raise another constitutional issue related to the case at hand."
                )
                
                # user content into single string
                user_prompt_content = format_context_for_prompt(final_context)

                messages = [
                    {
                        "content": f"reasoning language: English\n\n{developer_message}",
                        "role": "system",
                    },
                    {
                        "content": user_prompt_content,
                        "role": "user",
                    },
                    {
                        "content": utt.text,
                        "role": "assistant",
                    }
                ]
                
                new_example = {
                    "reasoning_language": "English",
                    "developer": developer_message,
                    "user": user_prompt_content,
                    "analysis": "",
                    "final": utt.text, # the Justice's question is the target
                    "messages": messages
                }
                targeted_examples.append(new_example)
            role = ''
            if speaker_type == 'J':
                role = 'user'
            elif speaker_type == 'A':
                role = 'assistant'
            
            if role:
                current_context.append({"role": role, "content": utt.text})

    return targeted_examples
print("📥 Downloading and loading the Supreme Court corpus...")
corpus = Corpus(filename=download("supreme-corpus"))

print("⚙️ Transforming data into targeted examples with enriched context...")
new_dataset = create_targeted_examples(corpus, max_tokens=2000)
print(f"Generated {len(new_dataset)} examples.")

output_filename = 'supreme_court_targeted_examples_enriched.json'
print(f"💾 Saving examples to {output_filename}...")
with open(output_filename, 'w') as f:
    json.dump(new_dataset, f, indent=2)

print("✅ Success! The targeted dataset has been created.")
