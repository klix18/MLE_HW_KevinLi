import gradio as gr
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.chat_models import ChatOllama

# Initialize the model
model = ChatOllama(model="llama2")

# 1. Capital Finder with Few-Shot Prompting
def create_capital_chain():
    """Create the capital finding chain with few-shot prompting"""
    system_prompt = """You are a helpful AI assistant that provides ONLY the capital name as a one-word answer.

Here are some examples of good responses:

Question: What is the capital of France?
Answer: Paris.

Question: What is the capital of Japan?
Answer: Tokyo.

Question: What is the capital of Brazil?
Answer: Brasília.

Question: What is the capital of India?
Answer: New Delhi.

IMPORTANT: Respond with ONLY the capital name, nothing else. No extra words, no explanations."""

    human_template = "What is the capital of {topic}?"
    
    chat_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", human_template)
    ])
    
    return (
        {"topic": RunnablePassthrough()}
        | chat_prompt
        | model
        | StrOutputParser()
    )

# 2. Country Detection Chain
def create_country_detection_chain():
    """Create the country detection chain"""
    country_detection_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful assistant that extracts country names from user input.
        
        Examples:
        Input: "what about china" → Output: "china"
        Input: "tell me about france" → Output: "france"
        Input: "what's the capital of india" → Output: "india"
        Input: "how about japan" → Output: "japan"
        Input: "pizza" → Output: "NOT_A_COUNTRY"
        Input: "hello" → Output: "NOT_A_COUNTRY"
        Input: "123" → Output: "NOT_A_COUNTRY"
        
        Extract ONLY the country name if present, or return "NOT_A_COUNTRY" if no country is mentioned.
        Return just the country name, nothing else."""),
        ("human", "Input: {user_input}\nOutput:")
    ])
    
    return (
        {"user_input": RunnablePassthrough()}
        | country_detection_prompt
        | model
        | StrOutputParser()
    )

# Initialize chains
capital_chain = create_capital_chain()
country_detection_chain = create_country_detection_chain()

# Gradio Interface Function
def find_capital(user_input):
    """Find capital of a country with smart detection"""
    try:
        if not user_input.strip():
            return "❌ Please enter a country name or phrase."
        
        # First, detect if input contains a country
        extracted_country = country_detection_chain.invoke(user_input).strip().lower()
        
        if extracted_country == "not_a_country":
            return "❌ Sorry, I only know the capitals of countries. Please enter a valid country name."
        
        # Get the capital
        capital = capital_chain.invoke(extracted_country)
        return f"🏛️ The capital of **{extracted_country.title()}** is: **{capital}**"
    
    except Exception as e:
        return f"❌ Error: {str(e)}"

# Create Gradio Interface
def create_interface():
    """Create the main Gradio interface"""
    
    with gr.Blocks(
        title="Country Capital Finder",
        theme=gr.themes.Soft(),
        css="""
        .gradio-container {
            max-width: 100% !important;
            margin: 0 auto !important;
            min-height: 100vh !important;
            display: flex !important;
            flex-direction: column !important;
            justify-content: center !important;
            align-items: center !important;
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%) !important;
            padding: 20px !important;
        }
        
        .main-container {
            width: 100% !important;
            max-width: 600px !important;
            background: rgba(255, 255, 255, 0.95) !important;
            border-radius: 20px !important;
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1) !important;
            padding: 40px !important;
            position: relative !important;
            overflow: hidden !important;
        }
        
        .main-container::before {
            content: '' !important;
            position: absolute !important;
            top: -50px !important;
            right: -50px !important;
            width: 200px !important;
            height: 200px !important;
            background: radial-gradient(circle, rgba(59, 130, 246, 0.3) 0%, rgba(59, 130, 246, 0.1) 50%, transparent 100%) !important;
            border-radius: 50% !important;
            z-index: 0 !important;
        }
        
        .main-container::after {
            content: '' !important;
            position: absolute !important;
            bottom: -30px !important;
            left: -30px !important;
            width: 150px !important;
            height: 150px !important;
            background: radial-gradient(circle, rgba(99, 102, 241, 0.2) 0%, rgba(99, 102, 241, 0.05) 50%, transparent 100%) !important;
            border-radius: 50% !important;
            z-index: 0 !important;
        }
        
        .header-section {
            text-align: center !important;
            margin-bottom: 40px !important;
            position: relative !important;
            z-index: 1 !important;
        }
        
        .main-title {
            background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%) !important;
            -webkit-background-clip: text !important;
            -webkit-text-fill-color: transparent !important;
            background-clip: text !important;
            font-size: 2.5rem !important;
            font-weight: 700 !important;
            margin-bottom: 10px !important;
        }
        
        .subtitle {
            color: #6b7280 !important;
            font-size: 1.1rem !important;
            margin-bottom: 20px !important;
        }
        
        .response-section {
            background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%) !important;
            border: 2px solid #e2e8f0 !important;
            border-radius: 15px !important;
            padding: 30px !important;
            margin-bottom: 40px !important;
            min-height: 120px !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            position: relative !important;
            z-index: 1 !important;
        }
        
        .response-section::before {
            content: '' !important;
            position: absolute !important;
            top: 10px !important;
            right: 10px !important;
            width: 60px !important;
            height: 60px !important;
            background: radial-gradient(circle, rgba(59, 130, 246, 0.1) 0%, transparent 70%) !important;
            border-radius: 50% !important;
        }
        
        .input-section {
            position: relative !important;
            z-index: 1 !important;
        }
        
        .input-label {
            color: #374151 !important;
            font-weight: 600 !important;
            margin-bottom: 15px !important;
            font-size: 1.1rem !important;
            text-align: center !important;
        }
        
        .textbox {
            border: 2px solid #d1d5db !important;
            border-radius: 12px !important;
            background: white !important;
            transition: all 0.3s ease !important;
            width: 100% !important;
            margin-bottom: 20px !important;
        }
        
        .textbox:focus {
            border-color: #3b82f6 !important;
            box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1) !important;
        }
        
        .submit-btn {
            background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%) !important;
            border: none !important;
            border-radius: 12px !important;
            color: white !important;
            font-weight: 600 !important;
            padding: 15px 40px !important;
            transition: all 0.3s ease !important;
            box-shadow: 0 4px 6px rgba(59, 130, 246, 0.2) !important;
            width: 100% !important;
            font-size: 1.1rem !important;
        }
        
        .submit-btn:hover {
            transform: translateY(-2px) !important;
            box-shadow: 0 6px 12px rgba(59, 130, 246, 0.3) !important;
        }
        
        .examples-section {
            background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%) !important;
            border-radius: 12px !important;
            padding: 20px !important;
            margin-top: 30px !important;
            border: 1px solid #e2e8f0 !important;
        }
        
        .examples-title {
            color: #374151 !important;
            font-weight: 600 !important;
            margin-bottom: 15px !important;
            font-size: 1rem !important;
            text-align: center !important;
        }
        
        .examples-list {
            color: #6b7280 !important;
            font-size: 0.9rem !important;
            line-height: 1.6 !important;
            text-align: center !important;
            list-style: none !important;
            padding: 0 !important;
        }
        
        .examples-list li {
            margin-bottom: 8px !important;
            display: inline-block !important;
            margin-right: 15px !important;
        }
        
        .examples-list li:last-child {
            margin-right: 0 !important;
        }
        
        .footer {
            text-align: center !important;
            margin-top: 30px !important;
            color: #9ca3af !important;
            font-size: 0.9rem !important;
        }
        
        /* Hide default Gradio elements */
        .gradio-container > div:not(.main-container) {
            display: none !important;
        }
        
        /* Responsive design */
        @media (max-width: 768px) {
            .main-container {
                max-width: 95% !important;
                padding: 30px 20px !important;
            }
            
            .main-title {
                font-size: 2rem !important;
            }
            
            .examples-list li {
                display: block !important;
                margin-right: 0 !important;
            }
        }
        """
    ) as demo:
        
        with gr.Column(elem_classes=["main-container"]):
            
            # Header Section
            with gr.Column(elem_classes=["header-section"]):
                gr.HTML("""
                <div>
                    <h1 class="main-title">🏛️ Country Capital Finder</h1>
                    <p class="subtitle">Powered by Ollama + LangChain + Gradio</p>
                </div>
                """)
            
            # Response Section (Top/Middle)
            with gr.Column(elem_classes=["response-section"]):
                capital_output = gr.Markdown(
                    value="💡 Enter a country name below to find its capital...",
                    label="",
                    elem_classes=["response-content"]
                )
            
            # Input Section (Bottom/Middle)
            with gr.Column(elem_classes=["input-section"]):
                gr.HTML('<div class="input-label">Enter Country Name</div>')
                
                capital_input = gr.Textbox(
                    label="",
                    placeholder="e.g., France, what about china, tell me about japan...",
                    lines=2,
                    elem_classes=["textbox"]
                )
                
                capital_btn = gr.Button(
                    "🔍 Find Capital", 
                    variant="primary", 
                    size="lg",
                    elem_classes=["submit-btn"]
                )
                
                # Examples Section
                with gr.Column(elem_classes=["examples-section"]):
                    gr.HTML('<div class="examples-title">💡 Try these examples:</div>')
                    gr.HTML("""
                    <ul class="examples-list">
                        <li>"France"</li>
                        <li>"what about china"</li>
                        <li>"tell me about japan"</li>
                        <li>"how about germany"</li>
                        <li>"what's the capital of india"</li>
                    </ul>
                    """)
            
            # Footer
            gr.HTML("""
            <div class="footer">
                <p>Built with ❤️ using Ollama, LangChain, and Gradio</p>
            </div>
            """)
        
        # Connect the button
        capital_btn.click(
            fn=find_capital,
            inputs=capital_input,
            outputs=capital_output
        )
        
        # Also allow Enter key to submit
        capital_input.submit(
            fn=find_capital,
            inputs=capital_input,
            outputs=capital_output
        )
    
    return demo

# Launch the interface
if __name__ == "__main__":
    demo = create_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        show_error=True,
        debug=True
    ) 