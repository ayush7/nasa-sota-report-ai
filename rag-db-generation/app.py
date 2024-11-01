import gradio_app
from gradio_app import * 

# Create Gradio interface
iface = gr.Interface(
    fn=process_question,
    inputs=gr.Textbox(lines=2, placeholder="Enter your question here..."),
    outputs=[
        gr.Textbox(label="Answer"),
        gr.Textbox(label="Sources")
    ],
    title="NASA Report Q&A System",
    description="Ask questions about the NASA reports and get answers with sources."
)

if __name__ == "__main__":
    iface.launch(share=True)