# ConfidenStrategy 📊

## Overview 🌟

ConfidenStrategy is an AI-powered business strategy analysis platform that helps companies develop and visualize strategic plans using market data and AI-driven insights. Built for a GenAI hackathon, this tool leverages various APIs and machine learning models to provide comprehensive business strategy recommendations.

[![ConfidenStrategy Demo](https://img.youtube.com/vi/VIDEO_ID/0.jpg)](https://youtu.be/_nFEyKvajJ8?si=cbpK8udqOAapZ3bw)
*👆 Click to watch demo video*

## Features ✨

- 📈 **Market Analysis**: Automatically analyzes company and competitor data
- 🧠 **AI-Generated Strategies**: Produces tailored strategic recommendations
- 📊 **Interactive Visualizations**: Market share projections, resource allocation, and revenue forecasts
- 💬 **Strategy Assistant**: Chat with an AI assistant about your strategy
- 🔊 **Text-to-Speech**: Listen to strategy sections with high-quality TTS
- 📱 **Responsive Design**: Works on desktop and mobile devices

## Getting Started 🚀

### Prerequisites

- Python 3.8 or higher
- Streamlit
- Required API keys:
  - Alpha Vantage API key (for financial data)
  - Groq API key (for LLM access)
  - Finnhub API key (for competitor data)
  - Deepgram API key (for text-to-speech)

### Installation

1. Clone the repository:
   ```bash
   git clone [https://github.com/username/ConfidenStrategy.git](https://github.com/Sanjith-3/ConfidenStrategy.git)
   cd ConfidenStrategy
   ```

2. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Create a `.env` file with your API keys:
   ```bash
   ALPHA_VANTAGE_API_KEY=your_alpha_vantage_key
   GROQ_API_KEY=your_groq_key
   FINNHUB_API_KEY=your_finnhub_key
   DEEPGRAM_API_KEY=your_deepgram_key
   ```

4. Run the application:
   ```bash
   streamlit run app.py
   ```

## How It Works 🛠️

### For Technical Users

ConfidenStrategy uses a multi-step pipeline to generate business strategies:

1. **Data Collection**: Fetches company and competitor data from Alpha Vantage and Finnhub APIs
2. **Metric Calculation**: Computes essential business metrics like market share and revenue projections
3. **Strategy Generation**: Uses Groq's LLM API (Llama 3.1) to create tailored strategy sections
4. **Visualization Creation**: Generates interactive charts using Plotly
5. **Assistant Integration**: Implements a context-aware chat interface for strategy questions
6. **Text-to-Speech**: Converts strategy text to audio using Deepgram's API

Key technical components:
- **Streamlit**: For the web interface
- **Plotly**: For interactive data visualizations
- **LangChain**: For embedding and vector search capabilities
- **Cryptography**: For secure data handling
- **Pandas**: For data manipulation and analysis
- **Deepgram SDK**: For high-quality text-to-speech conversion

### For Non-Technical Users 🔍

Using ConfidenStrategy is simple, even if you're not familiar with technical details:

1. **Enter a Stock Symbol**: Start by typing a company's stock symbol (e.g., "AAPL" for Apple)
2. **Select Industry & Focus Areas**: Choose the industry and strategic areas relevant to your analysis
3. **Set Time Horizon**: Use the slider to select how far into the future you want to plan
4. **Generate Strategy**: Click the "Generate Comprehensive Strategy" button
5. **Review Results**: Explore the generated strategy sections, visualizations, and key metrics
6. **Ask Questions**: Use the chat interface in the sidebar to ask questions about your strategy
7. **Listen to Analysis**: Click the speaker button to hear any section read aloud


ConfidenStrategy was developed as a team entry for a GenAI hackathon. Although we didn't advance in the competition, the project demonstrates the potential of AI in business strategy and planning. Our team built this tool to showcase how generative AI can help businesses make data-driven strategic decisions.

#### Future Improvements 🔮

- Additional data sources for more comprehensive analysis
- Support for more industries and specialized metrics
- Enhanced visualization options
- Competitor comparison features
- Strategy export capabilities (PDF, PowerPoint)
- Fine-tuned industry-specific models

## Contributors 👥

- [Team Member 1 (Main Contributor of the Entire Project)](https://github.com/AtharshKrishnamoorthy)

