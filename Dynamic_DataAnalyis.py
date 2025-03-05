import gradio as gr
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
from openai import OpenAI
import os
from dotenv import load_dotenv
import numpy as np
import time

# Load environment variables and set OpenAI API key
load_dotenv()
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

def load_data(file, chunk_size=10000):
    try:
        chunks = []
        for chunk in pd.read_csv(file.name, chunksize=chunk_size):
            chunks.append(chunk)
        df = pd.concat(chunks, ignore_index=True)
        
        if df.empty:
            return "Error: The uploaded file is empty."
        
        return df
    except pd.errors.EmptyDataError:
        return "Error: The uploaded file is empty."
    except Exception as e:
        return f"Error: Unable to load the file. {str(e)}"

def create_visualization(df, visualization_type, x_column, y_column, z_column=None):
    if x_column not in df.columns or y_column not in df.columns:
        return "Error: Selected columns not found in the dataset."
    
    fig = None
    if visualization_type == "Scatter Plot":
        fig = px.scatter(df, x=x_column, y=y_column, color=z_column if z_column else None)
    elif visualization_type == "Line Plot":
        fig = px.line(df, x=x_column, y=y_column, color=z_column if z_column else None)
    elif visualization_type == "Bar Plot":
        fig = px.bar(df, x=x_column, y=y_column, color=z_column if z_column else None)
    elif visualization_type == "Box Plot":
        fig = px.box(df, x=x_column, y=y_column, color=z_column if z_column else None)
    elif visualization_type == "Violin Plot":
        fig = px.violin(df, x=x_column, y=y_column, color=z_column if z_column else None)
    else:
        return "Error: Invalid visualization type."
    
    if fig:
        fig.update_layout(
            xaxis=dict(title=x_column),
            yaxis=dict(title=y_column),
            title=f"{visualization_type}: {y_column} vs {x_column}"
        )
    
    return fig

def analyze_data(df, question, target_lang, max_retries=3, initial_delay=1):
    summary = []
    summary.append(f"Dataset shape: {df.shape}")
    summary.append("\nColumn names and data types:")
    summary.append(df.dtypes.to_string())
    summary.append("\nBasic statistics:")
    summary.append(df.describe().to_string())
    
    if 'Gender' in df.columns:
        summary.append("\nGender-specific analysis:")
        for col in df.select_dtypes(include=[np.number]).columns:
            summary.append(f"\n{col} by Gender:")
            summary.append(df.groupby('Gender')[col].describe().to_string())
    
    sample_rows = pd.concat([df.head(3), df.sample(3), df.tail(3)])
    summary.append("\nSample rows from the dataset:")
    summary.append(sample_rows.to_string())
    
    dataset_summary = "\n".join(summary)
    
    prompt = f"""Analyze the following dataset summary and answer the question in {target_lang}:

Dataset Summary:
{dataset_summary}

Question: {question}

Provide a detailed analysis based on the entire dataset summary, not just the sample rows."""

    retries = 0
    delay = initial_delay
    while retries < max_retries:
        try:
            completion = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": f"You are a data analyst assistant. Analyze the entire dataset summary provided and respond in {target_lang}."},
                    {"role": "user", "content": prompt}
                ]
            )
            return completion.choices[0].message.content
        except Exception as e:
            if "429" in str(e):
                retries += 1
                print(f"Rate limit exceeded. Retrying in {delay} seconds...")
                time.sleep(delay)
                delay *= 2
            else:
                return f"Error during OpenAI call: {str(e)}"
    return "Error: Max retries exceeded during OpenAI call (Rate Limit)."

def get_column_names(file):
    df = load_data(file)
    if isinstance(df, str):  # Error occurred during loading
        return []
    return df.columns.tolist()

with gr.Blocks() as demo:
    gr.Markdown("# Advanced Data Visualization and Analysis App")
    
    file_input = gr.File(label="Upload CSV File")
    
    with gr.Row():
        viz_type = gr.Dropdown(choices=["Scatter Plot", "Line Plot", "Bar Plot", "Box Plot", "Violin Plot"], label="Visualization Type")
        x_col = gr.Dropdown(label="X-axis Column")
        y_col = gr.Dropdown(label="Y-axis Column")
        z_col = gr.Dropdown(label="Color Column (optional)")
    
    visualize_btn = gr.Button("Visualize")
    
    output_text = gr.Textbox(label="Output Message")
    output_plot = gr.Plot()
    
    with gr.Row():
        analysis_question = gr.Textbox(label="Ask a question about the data")
        target_lang = gr.Dropdown(choices=["English", "Spanish", "French", "German", "Chinese"], label="Target Language", value="English")
    
    analyze_btn = gr.Button("Analyze")
    output_analysis = gr.Textbox(label="Analysis Result")
    
    def update_dropdowns(file):
        column_names = get_column_names(file)
        return gr.Dropdown(choices=column_names), gr.Dropdown(choices=column_names), gr.Dropdown(choices=column_names)
    
    file_input.change(
        update_dropdowns,
        inputs=[file_input],
        outputs=[x_col, y_col, z_col]
    )
    
    def process_data(file, visualization_type, x_column, y_column, z_column):
        df = load_data(file)
        if isinstance(df, str):  # Error occurred during loading
            return df, None
        return "Data loaded successfully", create_visualization(df, visualization_type, x_column, y_column, z_column)
    
    visualize_btn.click(
        process_data,
        inputs=[file_input, viz_type, x_col, y_col, z_col],
        outputs=[output_text, output_plot]
    )
    
    def process_analysis(file, question, target_lang):
        df = load_data(file)
        if isinstance(df, str):  # Error occurred during loading
            return df
        return analyze_data(df, question, target_lang)
    
    analyze_btn.click(
        process_analysis,
        inputs=[file_input, analysis_question, target_lang],
        outputs=output_analysis
    )

demo.launch()
