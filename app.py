import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import io # NEW: Helps us save images to memory
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_google_genai import ChatGoogleGenerativeAI

# --- CONFIGURATION ---
st.set_page_config(page_title="Universal Data Analyst", page_icon="📊", layout="wide")
st.title("📊 Universal Data Analyst (Persistent Charts)")

# --- SIDEBAR ---
with st.sidebar:
    st.header("1. Upload Data")
    uploaded_file = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])
    
    st.header("2. Settings")
    # api_key = st.text_input("Google API Key", type="password")
    # Auto-fills from secrets if available, but lets user change it
    default_key = st.secrets.get("GOOGLE_API_KEY", "")
    api_key = st.text_input("Google API Key", value=default_key, type="password")
    
    
# --- LOGIC ---
if uploaded_file and api_key:
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
            
        # Stats
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Rows", df.shape[0])
        col2.metric("Total Columns", df.shape[1])
        col3.metric("Missing Values", df.isna().sum().sum())

        with st.expander("👀 View Raw Data (First 100 rows)"):
            st.dataframe(df.head(100)) # Shows more rows now
        
        # Agent Setup
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=api_key, temperature=0)
        agent = create_pandas_dataframe_agent(
            llm,
            df,
            verbose=True,
            allow_dangerous_code=True,
            handle_parsing_errors=True
        )
        
        # --- CHAT INTERFACE ---
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # 1. RENDER HISTORY (Text + Images)
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                # NEW: Check if this message has an image attached
                if "image" in message:
                    st.image(message["image"])

        # 2. HANDLE NEW INPUT
        if prompt := st.chat_input("Ask: 'Plot a bar chart of Sales by Region'"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                try:
                    with st.spinner("Analyzing..."):
                        # Clear old plots to avoid mixing
                        plt.clf()
                        
                        # Run Agent
                        response = agent.invoke(prompt)
                        answer = response["output"]
                        
                        st.markdown(answer)
                        
                        # NEW: CAPTURE THE CHART
                        # We check if the agent drew anything on the canvas
                        img_buffer = None
                        if plt.get_fignums():
                            # Save plot to a bytes buffer (like a virtual file)
                            img_buffer = io.BytesIO()
                            plt.savefig(img_buffer, format='png')
                            img_buffer.seek(0)
                            
                            # Display it now
                            st.image(img_buffer)
                            
                            # Clear the canvas for next time
                            plt.clf()
                        
                        # SAVE TO HISTORY (Text + Image Buffer)
                        msg_data = {"role": "assistant", "content": answer}
                        if img_buffer:
                            msg_data["image"] = img_buffer
                            
                        st.session_state.messages.append(msg_data)
                            
                except Exception as e:
                    st.error(f"Error: {e}")

    except Exception as e:
        st.error(f"Could not process file: {e}")

elif not uploaded_file:
    st.info("Upload a CSV to begin.")
