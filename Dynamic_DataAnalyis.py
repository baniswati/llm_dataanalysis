def load_data(file, chunk_size=10000):
    try:
        df = pd.read_csv(file.name)
        
        if df.empty:
            return "Error: The uploaded file is empty."
        
        return df
    except pd.errors.EmptyDataError:
        return "Error: The uploaded file is empty."
    except Exception as e:
        return f"Error: Unable to load the file. {str(e)}"
