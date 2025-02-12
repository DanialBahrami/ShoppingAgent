import gradio as gr
from main_v1 import load_text_index, load_image_index, create_conversational_agent

# Load vector stores and create agent
text_store = load_text_index()
image_store = load_image_index()
agent = create_conversational_agent(text_store, image_store)

# Function to interact with the AI assistant
def chatbot_response(user_input, history):
    """
    - Takes user input and conversation history.
    - Passes input to the LangChain agent.
    - Returns the agent's response and updates the conversation history.
    """
    response = agent.run(user_input)
    history.append((user_input, response))  # Append user message and AI response to chat history
    return history, history  # Return updated history for display

# Gradio UI
with gr.Blocks() as chat_ui:
    gr.Markdown("<h1 style='text-align: center;'>🛍️ AI Shopping Assistant</h1>")
    
    chatbot = gr.Chatbot()  # Chat UI component
    user_input = gr.Textbox(placeholder="Ask me anything about fashion, products, or shopping...")
    
    with gr.Row():
        submit_btn = gr.Button("Send")
        clear_btn = gr.Button("Clear Chat")

    # Submit Button: Calls chatbot_response function
    submit_btn.click(chatbot_response, inputs=[user_input, chatbot], outputs=[chatbot, chatbot])
    
    # Clear Button: Resets the chat history
    clear_btn.click(lambda: [], None, chatbot)

# Launch the UI
if __name__ == "__main__":
    chat_ui.launch(share=True)  # share=True generates a public Gradio link
