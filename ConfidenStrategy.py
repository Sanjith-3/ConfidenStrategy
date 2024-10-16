import os
import json
import hashlib
import requests
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from io import StringIO, BytesIO
from dotenv import load_dotenv
from cryptography.fernet import Fernet
from langchain.vectorstores import FAISS
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from litellm import completion
from datetime import datetime, timedelta
import base64
import os
import base64
import tempfile
from deepgram import DeepgramClient,SpeakOptions

# Load environment variables
load_dotenv()

# Configuration
ALPHA_VANTAGE_API_KEY = os.getenv('ALPHA_VANTAGE_API_KEY', 'API_KEY')
GROQ_API_KEY = os.getenv('GROQ_API_KEY', 'API_KEY')
FINNHUB_API_KEY = os.getenv('FINNHUB_API_KEY', 'API_KEY')
DEEPGRAM_API_KEY = os.getenv('DEEPGRAM_API_KEY','API_KEY')

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'chat_visible' not in st.session_state:
    st.session_state.chat_visible = False
if 'strategy_context' not in st.session_state:
    st.session_state.strategy_context = None
if 'company_data' not in st.session_state:
    st.session_state.company_data = None
if 'tts_state' not in st.session_state:
    st.session_state.tts_state = {'playing': False, 'position': 0}
if 'strategy_sections' not in st.session_state:
    st.session_state.strategy_sections = None

# Initialize encryption
encryption_key = Fernet.generate_key()
cipher_suite = Fernet(encryption_key)

# Utility functions
def encrypt_data(data):
    return cipher_suite.encrypt(json.dumps(data).encode()).decode()

def decrypt_data(encrypted_data):
    return json.loads(cipher_suite.decrypt(encrypted_data.encode()).decode())

def hash_data(data):
    return hashlib.sha256(data.encode()).hexdigest()

# Initialize embeddings and vector store
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                   model_kwargs={'device': 'cpu'})
vectorstore = FAISS.from_texts(["Initial document"], embeddings)

# Cache decorator for API calls
def cache_api_call(ttl_seconds=3600):
    def decorator(func):
        def wrapper(*args, **kwargs):
            cache_key = f"{func.__name__}_{str(args)}_{str(kwargs)}"
            if cache_key in st.session_state:
                cached_result, cache_time = st.session_state[cache_key]
                if datetime.now() - cache_time < timedelta(seconds=ttl_seconds):
                    return cached_result
            result = func(*args, **kwargs)
            st.session_state[cache_key] = (result, datetime.now())
            return result
        return wrapper
    return decorator

# Alpha Vantage API functions
@cache_api_call(ttl_seconds=3600)
def get_company_overview(symbol):
    url = f'https://www.alphavantage.co/query?function=OVERVIEW&symbol={symbol}&apikey={ALPHA_VANTAGE_API_KEY}'
    try:
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            if 'Symbol' in data:
                return data
    except Exception as e:
        st.error(f"Error fetching company overview: {str(e)}")
    return None

@cache_api_call(ttl_seconds=3600)
def get_stock_time_series(symbol):
    url = f'https://www.alphavantage.co/query?function=TIME_SERIES_MONTHLY&symbol={symbol}&apikey={ALPHA_VANTAGE_API_KEY}'
    try:
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            if 'Monthly Time Series' in data:
                monthly_data = data['Monthly Time Series']
                df = pd.DataFrame(monthly_data).T
                df.index = pd.to_datetime(df.index)
                df = df.sort_index()
                return df['4. close'].astype(float).tail(12)
    except Exception as e:
        st.error(f"Error fetching stock time series: {str(e)}")
    return pd.Series()

@cache_api_call(ttl_seconds=86400)  # Cache for 24 hours
def get_sector_performance():
    url = f'https://www.alphavantage.co/query?function=SECTOR&apikey={ALPHA_VANTAGE_API_KEY}'
    try:
        response = requests.get(url)
        if response.status_code == 200:
            return response.json()
    except Exception as e:
        st.error(f"Error fetching sector performance: {str(e)}")
    return None

# Updated function to get company competitors
@cache_api_call(ttl_seconds=86400)  # Cache for 24 hours
def get_company_competitors(symbol):
    url = f'https://finnhub.io/api/v1/stock/peers?symbol={symbol}&token={FINNHUB_API_KEY}'
    try:
        response = requests.get(url)
        if response.status_code == 200:
            competitors = response.json()
            return competitors[:4]  # Return top 4 competitors
    except Exception as e:
        st.error(f"Error fetching company competitors: {str(e)}")
    return []

# Chat-related functions
def get_chatbot_response(user_input, strategy_context):
    try:
        # Prepare context from strategy
        context = f"""
        Based on the following strategy context:
        Company: {strategy_context['company_name']}
        Industry: {strategy_context['industry']}
        Focus Areas: {', '.join(strategy_context['strategic_focus'])}
        Time Horizon: {strategy_context['time_horizon']} months
        
        Key Metrics:
        - Current Market Share: {strategy_context['metrics']['market_share']['current']}%
        - Target Market Share: {strategy_context['metrics']['market_share']['target']}%
        - Additional Revenue: ${strategy_context['metrics']['revenue_impact']['additional_revenue']}B
        
        User Query: {user_input}
        """
        
        conversation = [
            {"role": "system", "content": "You are a strategic business consultant AI assistant. Use the provided strategy context to give informed answers."},
            {"role": "user", "content": context}
        ]
        
        response = completion(
            model="groq/llama-3.1-8b-instant",
            api_key=GROQ_API_KEY,
            messages=conversation
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"I apologize, but I'm having trouble processing your request. Error: {str(e)}"

# Update the render_chat_interface function
def render_chat_interface():
    st.sidebar.subheader("Strategy Assistant")
    
    for i, message in enumerate(st.session_state.chat_history):
        if message['role'] == 'user':
            st.sidebar.chat_message("user").write(message['content'])
        else:
            response_container = st.sidebar.chat_message("assistant")
            response_container.write(message['content'])
            col1, col2, col3 = response_container.columns(3)
            with col1:
                if st.button("🔊 Speak", key=f"speak_{i}"):
                    speak_text_deepgram(message['content'])
            with col2:
                if st.button("⏹️ Stop", key=f"stop_{i}"):
                    stop_speaking()
            with col3:
                if st.button("🔁 Restart", key=f"restart_{i}"):
                    speak_text_deepgram(message['content'], restart=True)
    
    user_input = st.sidebar.chat_input("Type your question...")
    if user_input:
        if st.session_state.strategy_context:
            response = get_chatbot_response(user_input, st.session_state.strategy_context)
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            st.session_state.chat_history.append({"role": "assistant", "content": response})
        else:
            st.sidebar.warning("Please generate a strategy first to use the chat assistant.")

# Analysis functions
def calculate_market_metrics(company_data, competitor_data):
    try:
        total_market = sum([float(data.get('MarketCapitalization', 0)) for data in [company_data] + competitor_data])
        company_market_share = (float(company_data.get('MarketCapitalization', 0)) / total_market) * 100
        
        revenue_ttm = float(company_data.get('RevenueTTM', 0))
        profit_margin = float(company_data.get('ProfitMargin', 0))
        
        metrics = {
            "market_share": {
                "current": round(company_market_share, 2),
                "target": round(min(company_market_share * 1.5, 100), 2),
                "growth_rate": round((company_market_share * 0.5) / 4, 2)
            },
            "revenue_impact": {
                "current_revenue": round(revenue_ttm / 1e9, 2),
                "additional_revenue": round(revenue_ttm * 0.15 / 1e9, 2),
                "roi": round(profit_margin * 100, 2)
            },
            "implementation": {
                "beta_customers": max(10, int(float(company_data.get('MarketCapitalization', 0)) / 1e9)),
                "industries_covered": min(15, max(3, int(float(company_data.get('MarketCapitalization', 0)) / 5e9))),
                "compliance_countries": min(50, max(5, int(float(company_data.get('MarketCapitalization', 0)) / 2e9)))
            }
        }
        return metrics
    except Exception as e:
        st.error(f"Error calculating market metrics: {str(e)}")
        return None

# Updated visualization functions
def create_market_share_graph(company_name, company_data, competitor_data):
    try:
        market_caps = [float(data.get('MarketCapitalization', 0)) for data in [company_data] + competitor_data]
        company_names = [company_name] + [data.get('Name', f'Competitor {i+1}') for i, data in enumerate(competitor_data)]
        
        total_market = sum(market_caps)
        current_shares = [round((mc / total_market) * 100, 2) for mc in market_caps]
        
        quarters = ['Q2 2024', 'Q3 2024', 'Q4 2024', 'Q1 2025']
        growth_factors = {
            0: [1, 1.1, 1.2, 1.3],  # Company growth
            1: [1, 1.05, 1.1, 1.15],  # Top competitor
            2: [1, 1.03, 1.06, 1.09],  # Second competitor
            3: [1, 1.02, 1.04, 1.06],  # Third competitor
        }
        
        fig = go.Figure()
        for i, (share, name) in enumerate(zip(current_shares, company_names)):
            growth = growth_factors.get(i, [1, 1.01, 1.02, 1.03])
            projections = [share * factor for factor in growth]
            fig.add_trace(go.Bar(name=name, x=quarters, y=projections))
        
        fig.update_layout(
            barmode='group',
            title='Market Share Projection',
            xaxis_title='Quarter',
            yaxis_title='Market Share (%)',
            height=400
        )
        return fig
    except Exception as e:
        st.error(f"Error creating market share graph: {str(e)}")
        return None

def create_resource_allocation_pie(company_data):
    try:
        # Calculate resource allocation based on company financials
        r_and_d = float(company_data.get('R&DExpenses', 0))
        revenue = float(company_data.get('RevenueTTM', 0))
        assets = float(company_data.get('TotalAssets', 0))
        
        r_and_d_percentage = min(50, max(20, (r_and_d / revenue) * 100)) if revenue > 0 else 30
        go_to_market = min(40, max(20, (assets / revenue) * 10)) if revenue > 0 else 25
        partnerships = min(30, max(10, (float(company_data.get('OperatingMarginTTM', 10)) * 2)))
        security_compliance = min(30, max(10, 100 - r_and_d_percentage - go_to_market - partnerships))
        
        labels = ['R&D', 'Go-to-Market', 'Partnerships', 'Security & Compliance']
        values = [r_and_d_percentage, go_to_market, partnerships, security_compliance]
        
        fig = go.Figure(data=[go.Pie(labels=labels, values=values)])
        fig.update_layout(title='Resource Allocation', height=400)
        return fig
    except Exception as e:
        st.error(f"Error creating resource allocation pie chart: {str(e)}")
        return None

def create_revenue_projection(company_data):
    try:
        current_revenue = float(company_data.get('RevenueTTM', 0))
        growth_rate = float(company_data.get('QuarterlyRevenueGrowthYOY', 10)) / 100
        profit_margin = float(company_data.get('ProfitMargin', 0))
        
        months = ['Jun 2024', 'Sep 2024', 'Dec 2024', 'Mar 2025']
        baseline = [current_revenue * (1 + growth_rate * i/4) for i in range(4)]
        
        # Adjust the strategy impact based on company's current performance and profit margin
        strategy_impact = growth_rate * (1 + profit_margin) * 1.5 if growth_rate > 0.05 else growth_rate * (1 + profit_margin) * 2
        with_strategy = [current_revenue * (1 + (growth_rate + strategy_impact) * i/4) for i in range(4)]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=months, y=baseline, name='Baseline Revenue', line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=months, y=with_strategy, name='With AI Strategy', line=dict(color='green')))
        
        fig.update_layout(
            title='Revenue Impact Projection',
            xaxis_title='Month',
            yaxis_title='Revenue (Billions $)',
            height=400,
            yaxis=dict(tickformat='$,.0f')
        )
        return fig
    except Exception as e:
        st.error(f"Error creating revenue projection: {str(e)}")
        return None

def get_llm_response(prompt, company_data):
    try:
        enhanced_prompt = f"""
        Based on the following company data:
        - Company: {company_data.get('Name')}
        - Industry: {company_data.get('Industry')}
        - Sector: {company_data.get('Sector')}
        - Market Cap: ${float(company_data.get('MarketCapitalization', 0))/1e9:.2f}B
        - P/E Ratio: {company_data.get('PERatio')}
        - Profit Margin: {float(company_data.get('ProfitMargin', 0)) * 100:.2f}%
        - Revenue Growth YOY: {company_data.get('QuarterlyRevenueGrowthYOY')}%

        {prompt}
        """
        
        conversation = [
            {"role": "system", "content": "You are a strategic business analyst specializing in technology companies."},
            {"role": "user", "content": enhanced_prompt}
        ]
        
        response = completion(
            model="groq/llama-3.1-8b-instant",
            api_key=GROQ_API_KEY,
            messages=conversation
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Error getting LLM response: {str(e)}")
        return "Unable to generate analysis at this time."

def create_strategy(company_symbol, query):
    try:
        company_data = get_company_overview(company_symbol)
        if not company_data:
            return None, None, None, None
        
        competitor_symbols = get_company_competitors(company_symbol)
        competitor_data = [get_company_overview(symbol) for symbol in competitor_symbols if symbol != company_symbol]
        competitor_data = [data for data in competitor_data if data is not None]
        
        metrics = calculate_market_metrics(company_data, competitor_data)
        
        strategy_sections = {
            "executive_summary": get_llm_response(
                f"Generate an executive summary for AI strategy based on: {query}", company_data
            ),
            "market_analysis": get_llm_response(
                f"Provide a market analysis for position in AI market", company_data
            ),
            "strategic_initiatives": get_llm_response(
                f"List key strategic initiatives in AI market", company_data
            ),
            "implementation_plan": get_llm_response(
                f"Create an implementation plan for AI strategy", company_data
            ),
            "risk_analysis": get_llm_response(
                f"Analyze potential risks for AI strategy", company_data
            ),
        }
        
        return strategy_sections, metrics, company_data, competitor_data
    except Exception as e:
        st.error(f"Error creating strategy: {str(e)}")
        return None, None, None, None



def generate_audio_deepgram(text):
    try:
        deepgram = DeepgramClient(api_key=DEEPGRAM_API_KEY)
        options = SpeakOptions(
            model="aura-asteria-en",
            encoding="linear16",
            container="wav"
        )
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as fp:
            response = deepgram.speak.v("1").save(fp.name, {"text": text}, options)
            st.success(f"Audio file generated: {fp.name}")
            return fp.name
    except Exception as e:
        st.error(f"Error generating audio: {str(e)}")
        return None

def get_base64_encoded_audio(file_path):
    with open(file_path, "rb") as audio_file:
        return base64.b64encode(audio_file.read()).decode('utf-8')

def speak_text_deepgram(text, restart=False):
    if restart:
        st.session_state.tts_state['position'] = 0
    
    MAX_CHARS = 1999
    remaining_text = text[st.session_state.tts_state['position']:]
    
    while remaining_text:
        chunk = remaining_text[:MAX_CHARS]
        audio_file = generate_audio_deepgram(chunk)
        
        if audio_file:
            st.session_state.tts_state['playing'] = True
            st.session_state.tts_state['audio_file'] = audio_file
            
            # Encode the audio file to base64
            audio_base64 = get_base64_encoded_audio(audio_file)
            
            # Use HTML5 audio player
            st.markdown(
                f'<audio autoplay="true" src="data:audio/wav;base64,{audio_base64}">',
                unsafe_allow_html=True
            )
            
            # Also provide a download link for the audio
            st.download_button(
                label="Download Audio",
                data=open(audio_file, "rb"),
                file_name="tts_audio.wav",
                mime="audio/wav"
            )
            
            # Update the position for the next chunk
            st.session_state.tts_state['position'] += len(chunk)
            remaining_text = remaining_text[MAX_CHARS:]
        else:
            st.error("Failed to generate audio file.")
            break

def stop_speaking():
    st.session_state.tts_state['playing'] = False
    if 'audio_file' in st.session_state.tts_state:
        try:
            os.remove(st.session_state.tts_state['audio_file'])
            st.success(f"Removed audio file: {st.session_state.tts_state['audio_file']}")
        except Exception as e:
            st.error(f"Error removing audio file: {str(e)}")
        del st.session_state.tts_state['audio_file']

# Initialize TTS state in session state if not already present
if 'tts_state' not in st.session_state:
    st.session_state.tts_state = {'playing': False, 'position': 0}



def main():
    st.set_page_config(page_title="ConfidenStrategy", page_icon="📊", layout="wide")
    
    st.title("ConfidenStrategy: AI Strategy Analysis Platform")
    
    # Sidebar
    st.sidebar.header("Strategy Assistant")
    if st.sidebar.checkbox("Enable Chat Interface", value=st.session_state.chat_visible):
        st.session_state.chat_visible = True
        render_chat_interface()
    else:
        st.session_state.chat_visible = False
    
    company_symbol = st.text_input("Enter company stock symbol (e.g., AAPL):")
    
    # Initialize competitor_data
    competitor_data = []
    
    col1, col2, col3 = st.columns(3)
    with col1:
        industry = st.selectbox("Select industry focus:",
                            ["Technology", "Finance", "Healthcare", "Retail", "Manufacturing", "Energy", "Telecommunications"])
    with col2:
        strategic_focus = st.multiselect("Select strategic focus areas:",
                                    ["AI Integration", "Market Expansion", "Product Development", 
                                     "Competitive Positioning", "Customer Experience", "Digital Transformation",
                                     "Sustainability", "Innovation", "Operational Efficiency", "Talent Acquisition",
                                     "Global Expansion", "Supply Chain Optimization", "Brand Development"],
                                    default=["AI Integration"])
    with col3:
        time_horizon = st.slider("Strategy time horizon (months):", 6, 36, 18)

    competition_level = st.select_slider("Competition level in target market:",
                                        options=["Low", "Medium", "High", "Very High"])

    if company_symbol:
        query_template = f"""
        Develop a {time_horizon}-month strategy for the company in the {industry} industry, 
        focusing on {', '.join(strategic_focus)}. Consider:
        1. Current market with {competition_level} competition level
        2. Technological capabilities and potential
        3. Resource allocation and ROI
        4. Implementation timeline and milestones
        """

        strategy_query = st.text_area("Refine your strategic query:", value=query_template, height=100)

        if st.button("Generate Comprehensive Strategy"):
            with st.spinner("Analyzing market data and generating strategy..."):
                strategy_sections, metrics, company_data, competitor_data = create_strategy(company_symbol, strategy_query)
                
                if company_data:
                    # Save strategy context for chat
                    st.session_state.strategy_context = {
                        'company_name': company_data.get('Name', company_symbol),
                        'industry': industry,
                        'strategic_focus': strategic_focus,
                        'time_horizon': time_horizon,
                        'metrics': metrics
                    }
                    
                    # Save company data and strategy sections for future use
                    st.session_state.company_data = company_data
                    st.session_state.strategy_sections = strategy_sections
                    st.session_state.competitor_data = competitor_data  # Save competitor_data in session state

    # Display strategy if it exists
    if 'strategy_sections' in st.session_state and st.session_state.strategy_sections:
        display_strategy(st.session_state.strategy_sections, 
                         st.session_state.company_data, 
                         st.session_state.strategy_context, 
                         st.session_state.get('competitor_data', []))  # Use .get() to safely access competitor_data

    # Additional Analysis Options
    st.sidebar.header("Additional Analysis Options")
    
    if st.sidebar.checkbox("Show Detailed Metrics"):
        strategy_context = st.session_state.get('strategy_context')
        if strategy_context:
            metrics = strategy_context.get('metrics')
            if metrics:
                st.sidebar.json(metrics)
            else:
                st.sidebar.warning("No metrics available. Generate a strategy to see detailed metrics.")
        else:
            st.sidebar.warning("Generate a strategy to see detailed metrics")
    
    if st.sidebar.checkbox("Show Sector Performance"):
        sector_data = get_sector_performance()
        if sector_data:
            st.sidebar.subheader("Sector Performance")
            for sector, performance in sector_data.get("Rank A: Real-Time Performance", {}).items():
                st.sidebar.markdown(f"{sector}: {performance}")
        else:
            st.sidebar.warning("Unable to fetch sector performance data")
    
    # File upload for additional context
    uploaded_file = st.sidebar.file_uploader("Upload additional company data", type=["txt", "pdf"])
    if uploaded_file is not None:
        try:
            # Read the file as bytes and then decode
            file_contents = uploaded_file.read()
            try:
                # Try UTF-8 decoding first
                stringio = StringIO(file_contents.decode("utf-8"))
            except UnicodeDecodeError:
                # If UTF-8 fails, try ISO-8859-1
                stringio = StringIO(file_contents.decode("iso-8859-1"))
            st.sidebar.success("File uploaded successfully. Analysis updated.")
        except Exception as e:
            st.sidebar.error(f"Error processing file: {str(e)}")

def display_strategy(strategy_sections, company_data, strategy_context, competitor_data):
    company_name = company_data.get('Name', 'Company')
    
    # Company Overview Section
    st.header(f"Strategic Analysis for {company_name}")
    
    overview_col1, overview_col2, overview_col3, overview_col4 = st.columns(4)
    with overview_col1:
        st.metric("Market Cap", f"${float(company_data.get('MarketCapitalization', 0))/1e9:.2f}B")
    with overview_col2:
        st.metric("P/E Ratio", company_data.get('PERatio', 'N/A'))
    with overview_col3:
        st.metric("Profit Margin", f"{float(company_data.get('ProfitMargin', 0)) * 100:.2f}%")
    with overview_col4:
        st.metric("YoY Growth", f"{company_data.get('QuarterlyRevenueGrowthYOY', 'N/A')}%")
    
    st.subheader("Executive Summary")
    exec_summary = st.empty()
    exec_summary.write(strategy_sections["executive_summary"])
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("🔊 Speak", key="speak_exec_summary"):
            speak_text_deepgram(strategy_sections["executive_summary"])
    with col2:
        if st.button("⏹️ Stop", key="stop_exec_summary"):
            stop_speaking()
    with col3:
        if st.button("🔁 Restart", key="restart_exec_summary"):
            speak_text_deepgram(strategy_sections["executive_summary"], restart=True)

    
    # Market Analysis with Visualizations
    st.subheader("Market Analysis")
    market_analysis = st.empty()
    market_analysis.write(strategy_sections["market_analysis"])
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("🔊 Speak", key="speak_market_analysis"):
            speak_text_deepgram(strategy_sections["market_analysis"])
    with col2:
        if st.button("⏹️ Stop", key="stop_market_analysis"):
            stop_speaking()
    with col3:
        if st.button("🔁 Restart", key="restart_market_analysis"):
            speak_text_deepgram(strategy_sections["market_analysis"], restart=True)
    
    visualization_col1, visualization_col2 = st.columns(2)
    with visualization_col1:
        market_share_fig = create_market_share_graph(company_name, company_data, competitor_data)
        if market_share_fig:
            st.plotly_chart(market_share_fig, use_container_width=True)
    with visualization_col2:
        resource_allocation_fig = create_resource_allocation_pie(company_data)
        if resource_allocation_fig:
            st.plotly_chart(resource_allocation_fig, use_container_width=True)
    
    revenue_projection_fig = create_revenue_projection(company_data)
    if revenue_projection_fig:
        st.plotly_chart(revenue_projection_fig, use_container_width=True)
    
    # Strategic Initiatives
    st.subheader("Strategic Initiatives")
    strategic_initiatives = st.empty()
    strategic_initiatives.write(strategy_sections["strategic_initiatives"])
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("🔊 Speak", key="speak_strategic_initiatives"):
            speak_text_deepgram(strategy_sections["strategic_initiatives"])
    with col2:
        if st.button("⏹️ Stop", key="stop_strategic_initiatives"):
            stop_speaking()
    with col3:
        if st.button("🔁 Restart", key="restart_strategic_initiatives"):
            speak_text_deepgram(strategy_sections["strategic_initiatives"], restart=True)
    
    # Implementation Plan
    st.subheader("Implementation Plan")
    implementation_plan = st.empty()
    implementation_plan.write(strategy_sections["implementation_plan"])
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("🔊 Speak", key="speak_implementation_plan"):
            speak_text_deepgram(strategy_sections["implementation_plan"])
    with col2:
        if st.button("⏹️ Stop", key="stop_implementation_plan"):
            stop_speaking()
    with col3:
        if st.button("🔁 Restart", key="restart_implementation_plan"):
            speak_text_deepgram(strategy_sections["implementation_plan"], restart=True)
    
    # Key Metrics
    st.subheader("Key Strategy Metrics")
    metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
    
    with metrics_col1:
        st.metric("Market Share Target", 
                 f"{strategy_context['metrics']['market_share']['target']}%",
                 f"+{strategy_context['metrics']['market_share']['target'] - strategy_context['metrics']['market_share']['current']}%")
    with metrics_col2:
        st.metric("Projected Additional Revenue", 
                 f"${strategy_context['metrics']['revenue_impact']['additional_revenue']}B",
                 f"{strategy_context['metrics']['revenue_impact']['roi']}% ROI")
    with metrics_col3:
        st.metric("Target Beta Customers", 
                 str(strategy_context['metrics']['implementation']['beta_customers']))
    
    # Risk Analysis
    st.subheader("Risk Analysis")
    risk_analysis = st.empty()
    risk_analysis.write(strategy_sections["risk_analysis"])
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("🔊 Speak", key="speak_risk_analysis"):
            speak_text_deepgram(strategy_sections["risk_analysis"])
    with col2:
        if st.button("⏹️ Stop", key="stop_risk_analysis"):
            stop_speaking()
    with col3:
        if st.button("🔁 Restart", key="restart_risk_analysis"):
            speak_text_deepgram(strategy_sections["risk_analysis"], restart=True)


def generate_audio_deepgram(text):
    try:
        deepgram = DeepgramClient(api_key=DEEPGRAM_API_KEY)
        options = SpeakOptions(
            model="aura-asteria-en",
            encoding="linear16",
            container="wav"
        )
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as fp:
            response = deepgram.speak.v("1").save(fp.name, {"text": text}, options)
            st.success(f"Audio file generated: {fp.name}")
            return fp.name
    except Exception as e:
        st.error(f"Error generating audio: {str(e)}")
        return None

def get_base64_encoded_audio(file_path):
    with open(file_path, "rb") as audio_file:
        return base64.b64encode(audio_file.read()).decode('utf-8')

def speak_text_deepgram(text, restart=False):
    if restart:
        st.session_state.tts_state['position'] = 0
    
    MAX_CHARS = 1999
    remaining_text = text[st.session_state.tts_state['position']:]
    
    while remaining_text:
        chunk = remaining_text[:MAX_CHARS]
        audio_file = generate_audio_deepgram(chunk)
        
        if audio_file:
            st.session_state.tts_state['playing'] = True
            st.session_state.tts_state['audio_file'] = audio_file
            
            # Encode the audio file to base64
            audio_base64 = get_base64_encoded_audio(audio_file)
            
            # Use HTML5 audio player
            st.markdown(
                f'<audio autoplay="true" src="data:audio/wav;base64,{audio_base64}">',
                unsafe_allow_html=True
            )
            
            # Also provide a download link for the audio
            st.download_button(
                label="Download Audio",
                data=open(audio_file, "rb"),
                file_name="tts_audio.wav",
                mime="audio/wav"
            )
            
            # Update the position for the next chunk
            st.session_state.tts_state['position'] += len(chunk)
            remaining_text = remaining_text[MAX_CHARS:]
        else:
            st.error("Failed to generate audio file.")
            break

def stop_speaking():
    st.session_state.tts_state['playing'] = False
    if 'audio_file' in st.session_state.tts_state:
        try:
            os.remove(st.session_state.tts_state['audio_file'])
            st.success(f"Removed audio file: {st.session_state.tts_state['audio_file']}")
        except Exception as e:
            st.error(f"Error removing audio file: {str(e)}")
        del st.session_state.tts_state['audio_file']

# Initialize TTS state in session state if not already present
if 'tts_state' not in st.session_state:
    st.session_state.tts_state = {'playing': False, 'position': 0}


# Update the render_chat_interface function
def render_chat_interface():
    st.sidebar.subheader("Strategy Assistant")
    
    for i, message in enumerate(st.session_state.chat_history):
        if message['role'] == 'user':
            st.sidebar.chat_message("user").write(message['content'])
        else:
            response_container = st.sidebar.chat_message("assistant")
            response_container.write(message['content'])
            col1, col2, col3 = response_container.columns(3)
            with col1:
                if st.button("🔊 Speak", key=f"speak_{i}"):
                    speak_text_deepgram(message['content'])
            with col2:
                if st.button("⏹️ Stop", key=f"stop_{i}"):
                    stop_speaking()
            with col3:
                if st.button("🔁 Restart", key=f"restart_{i}"):
                    speak_text_deepgram(message['content'], restart=True)
    
    user_input = st.sidebar.chat_input("Type your question...")
    if user_input:
        if st.session_state.strategy_context:
            response = get_chatbot_response(user_input, st.session_state.strategy_context)
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            st.session_state.chat_history.append({"role": "assistant", "content": response})
        else:
            st.sidebar.warning("Please generate a strategy first to use the chat assistant.")

if __name__ == "__main__":
    main()