"""Local Gradio demo. Nothing is sent to an external model endpoint."""
import argparse
import gradio as gr
from src.pipeline import TicketAssistant

def build_app(assistant):
    def run(text):
        try:
            out=assistant(text)
            return out["intent"] or "human_review",out["confidence"],out["category"] or "unknown",out["priority"],out["reply"],out
        except ValueError as error:
            raise gr.Error(str(error))
    return gr.Interface(fn=run,
        inputs=gr.Textbox(lines=4,label="Customer message"),
        outputs=[gr.Textbox(label="Intent"),gr.Number(label="Confidence"),gr.Textbox(label="Category"),
            gr.Textbox(label="Priority"),gr.Textbox(lines=8,label="Draft reply"),gr.JSON(label="Full output and sources")],
        title="Intelligent Support Ticket Assistant",
        description="DemoMart course project. Enter English or Arabic text. Every response is a draft for a support agent.",
        examples=[["Where is my refund?"],["تم خصم المبلغ مرتين من بطاقتي!!"],["How do I delete my account?"],["ابي اكلم موظف الحين لو سمحت"]],flagging_mode="never")

if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--llm",action="store_true");args=parser.parse_args()
    build_app(TicketAssistant(use_llm=args.llm)).launch(server_name="127.0.0.1",share=False)
