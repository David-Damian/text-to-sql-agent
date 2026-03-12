import pandas as pd
import json
import ast
from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException

def process_magic_data(file_path):
    """
    Processes the magic_data_raw.csv file into a cleaner format.
    
    Parameters:
    file_path (str): The path to the CSV file.
    
    Returns:
    pd.DataFrame: A processed DataFrame.
    """
    # 1. Load the data
    df = pd.read_csv(file_path)
    
    # 2. Convert timestamps to datetime
    df['CONV_LAST_UPDATED_DTTM'] = pd.to_datetime(df['CONV_LAST_UPDATED_DTTM'])
    
    # 3. Parse JSON columns
    # Sometimes pandas reads these as strings that look like dicts/lists
    def safe_json_loads(val):
        if pd.isna(val):
            return {} if isinstance(val, str) and val.startswith('{') else []
        try:
            return json.loads(val)
        except (ValueError, TypeError):
            # Fallback to ast.literal_eval if valid JSON but maybe with single quotes or other issues
            try:
                return ast.literal_eval(val)
            except:
                return val

    df['CONVERSATION'] = df['CONVERSATION'].apply(safe_json_loads)
    df['HISTORY_MESSAGES'] = df['HISTORY_MESSAGES'].apply(safe_json_loads)
    
    # 4. Extract useful fields from the CONVERSATION object
    df['identity'] = df['CONVERSATION'].apply(lambda x: x.get('identity') if isinstance(x, dict) else None)
    df['skill_name'] = df['CONVERSATION'].apply(lambda x: x.get('skill', {}).get('name') if isinstance(x, dict) else None)
    df['status'] = df['CONVERSATION'].apply(lambda x: x.get('status') if isinstance(x, dict) else None)
    df['user_message_count'] = df['HISTORY_MESSAGES'].apply(lambda x: len([m for m in x if m.get('role') == 'user']) if isinstance(x, list) else 0)
    df['assistant_message_count'] = df['HISTORY_MESSAGES'].apply(lambda x: len([m for m in x if m.get('role') == 'assistant']) if isinstance(x, list) else 0)
    df['total_message_count'] = df['HISTORY_MESSAGES'].apply(lambda x: len(x) if isinstance(x, list) else 0)
    # 5. Explode HISTORY_MESSAGES to get one row per message (useful for NLP/Text analysis)
    # Note: This increases the number of rows significantly.
    # If the user just wants metadata, we can skip this or make it optional.
    # For now, let's provide a way to get the flat message list.
    
    return df

def explode_messages(df):
    """
    Explodes the HISTORY_MESSAGES column into individual rows.
    """
    # Filter out rows with empty history if any
    df = df[df['HISTORY_MESSAGES'].map(len) > 0].copy()
    
    df_exploded = df.explode('HISTORY_MESSAGES').reset_index(drop=True)
    
    # helper to safely get nested fields
    def get_field(item, field, nested=None):
        if not isinstance(item, dict):
            return None
        val = item.get(field)
        if nested and isinstance(val, dict):
            return val.get(nested)
        return val

    df_exploded['message_role'] = df_exploded['HISTORY_MESSAGES'].apply(lambda x: get_field(x, 'role'))
    df_exploded['message_text'] = df_exploded['HISTORY_MESSAGES'].apply(lambda x: get_field(x, 'text'))
    df_exploded['message_timestamp'] = df_exploded['HISTORY_MESSAGES'].apply(lambda x: get_field(x, 'metadata', 'timestamp'))
    
    return df_exploded.sort_values(by = ['CONVERSATION_ID', 'message_timestamp'])


def get_conv(processed_data: pd.DataFrame, conv_id: str):
    # Corrección de la validación de columnas
    required_cols = ['HISTORY_MESSAGES', 'CONVERSATION_ID', 'CONVERSATION', 'CONV_SECONDS']
    if not all(col in processed_data.columns for col in required_cols):
        print(f"[ERROR] : Tu df necesita tener las columnas: {', '.join(required_cols)}")
        return None
    
    filtered_df = processed_data[processed_data['CONVERSATION_ID'] == conv_id]
    
    if filtered_df.empty:
        print(f"[ERROR] : No se encontró la conversación con ID {conv_id}")
        return None

    # Función auxiliar para convertir string a objeto (dict/list)
    def to_object(val):
        if isinstance(val, str):
            try:
                # Intentamos JSON estándar primero
                return json.loads(val)
            except json.JSONDecodeError:
                # Si falla (ej. tiene comillas simples), usamos literal_eval
                try:
                    return ast.literal_eval(val)
                except:
                    return val
        return val

    # Convertimos las columnas problemáticas
    conversation = to_object(filtered_df['HISTORY_MESSAGES'].values[0])
    conv_meta = to_object(filtered_df['CONVERSATION'].values[0])

    # Extraemos datos del meta (ahora que sabemos que es un dict)
    user_alias = conv_meta.get('identity', 'Unknown') if isinstance(conv_meta, dict) else 'N/A'
    path = conv_meta.get('path', 'Unknown') if isinstance(conv_meta, dict) else 'N/A'

    print(f"Conversation ID: {conv_id}")
    print(f'User alias: {user_alias}')
    print(f'Conversation path: {path}')
    print(f"Conversation length: {len(conversation) if isinstance(conversation, list) else 'N/A'}")
    print(f"Conversation duration: {filtered_df['CONV_SECONDS'].values[0]} seconds")
    print(f"Next node: {filtered_df.get('NEXT_NODE_ID', pd.Series(['N/A'])).values[0]}")
    print(f"Last node: {filtered_df.get('LAST_NODE_ID', pd.Series(['N/A'])).values[0]}")
    
    return conversation


def print_conv(conversation:list):
    for msg in conversation:
        if msg['role'] == 'user':
            print(f"User: {msg['text']}")
        else:
            print(f"Assistant: {msg['text']}")

# To ensure consistent results
DetectorFactory.seed = 0

def get_language(text):
    if not text or len(text) < 3:
        return "unknown"
    try:
        return detect(text)
    except LangDetectException:
        return "unknown"