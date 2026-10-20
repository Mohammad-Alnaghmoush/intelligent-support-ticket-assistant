"""Dark Gradio interface around the unchanged support-ticket pipeline."""
import argparse
import html
from pathlib import Path
import gradio as gr
from src.pipeline import TicketAssistant

CSS = (Path(__file__).parent / 'ui/theme.css').read_text(encoding='utf-8')

def evidence_html(items, historical=False):
    cards=[]
    for item in items:
        title=item['ticket_id'] if historical else item['section']
        source=item['intent'] if historical else item['id']
        cards.append(f'<article class="evidence-card"><strong>{html.escape(title)}</strong><code>{html.escape(source)}</code><p dir="auto">{html.escape(item["text"])}</p></article>')
    return ''.join(cards)

def build_app(assistant):
    def run(text):
        try:
            out=assistant(text)
        except ValueError as error:
            raise gr.Error(str(error)) from error
        human=out['route']=='human_review'
        state='review' if human else 'ready'
        title='Support-agent review needed' if human else 'Reply draft ready'
        note='This message needs clarification or a support agent’s review.' if human else 'Review the cited evidence before sending this reply to a customer.'
        route=f'<div class="route {state}"><strong>{title}</strong><span>{note}</span></div>'
        sources=''.join(f'<span class="source-pill">{html.escape(source)}</span>' for source in out['citations']) or '<span class="muted">No source selected for this review response.</span>'
        return out['intent'] or 'human_review',out['confidence'],out['category'] or 'unknown',out['priority'],out['reply'],out,route,sources,evidence_html(out['kb_sections']),evidence_html(out['similar_tickets'],True)

    theme=gr.themes.Base(primary_hue='blue',neutral_hue='slate').set(
        body_background_fill='#090e18',body_background_fill_dark='#090e18',
        body_text_color='#ecf1f9',body_text_color_dark='#ecf1f9',
        block_background_fill='#101827',block_background_fill_dark='#101827',
        block_border_color='#263348',block_border_color_dark='#263348',
        input_background_fill='#0b1322',input_background_fill_dark='#0b1322',
        input_border_color='#31415a',input_border_color_dark='#31415a',
        button_primary_background_fill='#94bfff',button_primary_background_fill_dark='#94bfff',
        button_primary_text_color='#101e32',button_primary_text_color_dark='#101e32')
    with gr.Blocks(theme=theme,css=CSS,title='Intelligent Support Ticket Assistant') as demo:
        gr.HTML('<header class="app-header"><div class="eyebrow">APPLIED NLP · ENGLISH + ARABIC</div><h1>Intelligent Support<br><span>Ticket Assistant</span></h1><p>Understand the request. Find the evidence. Prepare a reply for review.</p><div class="feature-pills"><span>27 support intents</span><span>Cited reply drafts</span><span>Confidence-based review</span></div></header>')
        with gr.Row(equal_height=False):
            with gr.Column(scale=5,elem_classes='panel'):
                gr.HTML('<div class="section-heading"><span>01</span><div><h2>Customer message</h2><p>What does the customer need help with?</p></div></div>')
                ticket=gr.Textbox(label='Customer message',show_label=False,lines=6,max_lines=12,placeholder='Type an English or Arabic support ticket…',elem_id='ticket-input')
                gr.Examples(examples=[['أرغب في حذف حسابي نهائياً'],['How long does a refund to my Visa card take?'],['تم خصم المبلغ مرتين من بطاقتي!!']],inputs=ticket,label='Try an example')
                analyze=gr.Button('Analyze ticket →',variant='primary',elem_id='analyze-button')
                clear=gr.Button('Clear message and results',variant='secondary',size='sm')
                gr.HTML('<p class="note">DemoMart is fictional course data. All replies are drafts for a support agent. The program performs no account or order actions.</p>')
            with gr.Column(scale=7,elem_classes='panel'):
                gr.HTML('<div class="section-heading"><span>02</span><div><h2>Analysis & reply</h2><p>Classification, evidence and a suggested reply.</p></div></div>')
                route=gr.HTML('<div class="route idle"><strong>Ready when you are</strong><span>Choose an example or enter a customer message to begin.</span></div>')
                with gr.Row():
                    intent=gr.Textbox(label='Intent',interactive=False)
                    confidence=gr.Number(label='Confidence',interactive=False,precision=5)
                with gr.Row():
                    category=gr.Textbox(label='Category',interactive=False)
                    priority=gr.Textbox(label='Priority',interactive=False)
                reply=gr.Textbox(label='Draft reply',lines=7,max_lines=16,interactive=False,show_copy_button=True,elem_id='reply-output')
                sources=gr.HTML('')
                with gr.Accordion('Knowledge-base evidence',open=False):evidence=gr.HTML('')
                with gr.Accordion('Similar resolved cases',open=False):history=gr.HTML('')
                with gr.Accordion('Full structured output',open=False):raw=gr.JSON(label='Full output and sources')
        outputs=[intent,confidence,category,priority,reply,raw,route,sources,evidence,history]
        analyze.click(run,inputs=ticket,outputs=outputs,api_name='analyze_ticket',concurrency_limit=1)
        ticket.submit(run,inputs=ticket,outputs=outputs,api_name=False,concurrency_limit=1)
        clear.click(lambda:('',None,None,None,None,None,None,'<div class="route idle"><strong>Ready when you are</strong><span>Choose an example or enter a customer message to begin.</span></div>','','',''),outputs=[ticket,*outputs],queue=False,api_name=False)
        gr.HTML('<footer class="app-footer"><a href="https://github.com/Mohammad-Alnaghmoush/intelligent-support-ticket-assistant" target="_blank" rel="noopener noreferrer">Code & evaluation ↗</a><a href="https://sdaiaacademy.github.io/applied-nlp/" target="_blank" rel="noopener noreferrer">SDAIA Academy · Applied NLP ↗</a></footer>')
    return demo

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--llm',action='store_true')
    parser.add_argument('--host',default='127.0.0.1')
    parser.add_argument('--port',type=int,default=7860)
    args=parser.parse_args()
    build_app(TicketAssistant(use_llm=args.llm)).queue(max_size=20).launch(server_name=args.host,server_port=args.port,share=False)
