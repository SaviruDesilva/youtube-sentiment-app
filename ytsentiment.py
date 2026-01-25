import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"
import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
import matplotlib.pyplot as plt
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import re



# Page config
st.set_page_config(page_title="YouTube Sentiment Analyzer", page_icon="📊", layout="wide")

# ==================== API KEY CONFIGURATION ====================

YOUTUBE_API_KEY = "AIzaSyBgb71tS1-bhylJzZMX3wv2jsFLQwulGGA"
# ================================================================

# Load model
@st.cache_resource
def load_model():
    try:
        with tf.keras.utils.custom_object_scope(
            {"KerasLayer": hub.KerasLayer}
        ):
            model = tf.keras.models.load_model(
                "/home/saviru/youtube_sentiment_best.h5",
                compile=False
            )
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Extract video ID from URL
def extract_video_id(url):
    patterns = [
        r'(?:https?:\/\/)?(?:www\.)?youtube\.com\/watch\?v=([a-zA-Z0-9_-]+)',
        r'(?:https?:\/\/)?(?:www\.)?youtu\.be\/([a-zA-Z0-9_-]+)',
        r'(?:https?:\/\/)?(?:www\.)?youtube\.com\/embed\/([a-zA-Z0-9_-]+)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None

# Fetch YouTube comments with pagination
def get_youtube_comments(video_id, max_results=500):
    try:
        youtube = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)
        
        # Get video details
        video_response = youtube.videos().list(
            part='snippet',
            id=video_id
        ).execute()
        
        if not video_response['items']:
            return None, None
        
        video_title = video_response['items'][0]['snippet']['title']
        
        # Get comments
        comments = []
        next_page_token = None
        
        progress_text = st.empty()
        progress_bar = st.progress(0)
        
        while len(comments) < max_results:
            try:
                request = youtube.commentThreads().list(
                    part='snippet',
                    videoId=video_id,
                    maxResults=min(100, max_results - len(comments)),
                    pageToken=next_page_token,
                    textFormat='plainText'
                )
                
                response = request.execute()
                
                for item in response['items']:
                    comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
                    comments.append(comment)
                    
                    if len(comments) >= max_results:
                        break
                
                progress = len(comments) / max_results
                progress_bar.progress(progress)
                progress_text.text(f"Fetched {len(comments)} / {max_results} comments...")
                
                next_page_token = response.get('nextPageToken')
                if not next_page_token:
                    break
                    
            except Exception as e:
                st.warning(f"Stopped at {len(comments)} comments. Some videos have limited comments.")
                break
        
        progress_text.empty()
        progress_bar.empty()
        
        return video_title, comments
        
    except HttpError as e:
        if "quotaExceeded" in str(e):
            st.error("❌ API quota exceeded. Please try again tomorrow or use a different API key.")
        else:
            st.error(f"HTTP Error: {e}")
        return None, None
    except Exception as e:
        st.error(f"Error fetching comments: {e}")
        return None, None

# Predict sentiment with batching
def predict_sentiment(model, video_title, comments, batch_size=32):
    combined_texts = [f"{video_title} [SEP]comment: {comment}" for comment in comments]
    
    all_predictions = []
    progress_bar = st.progress(0)
    progress_text = st.empty()
    
    for i in range(0, len(combined_texts), batch_size):
        batch = combined_texts[i:i + batch_size]
        predictions = model.predict(batch, verbose=0)
        all_predictions.extend(predictions)
        
        # Update progress
        progress = min((i + batch_size) / len(combined_texts), 1.0)
        progress_bar.progress(progress)
        progress_text.text(f"Analyzing sentiment... {int(progress * 100)}%")
    
    progress_bar.empty()
    progress_text.empty()
    
    predicted_classes = np.argmax(all_predictions, axis=1)
    return predicted_classes

# Main app
def main():
    st.title("📊 YouTube Comment Sentiment Analyzer")
    st.markdown("Analyze the sentiment of YouTube video comments using AI")
    
    # Check if API key is configured
    if YOUTUBE_API_KEY == "YOUR_API_KEY_HERE" or not YOUTUBE_API_KEY:
        st.error("⚠️ API Key not configured! Please add your YouTube API key to the code.")
        st.info("""
        **How to add your API key:**
        1. Open the app code
        2. Find the line: `YOUTUBE_API_KEY = "YOUR_API_KEY_HERE"`
        3. Replace `YOUR_API_KEY_HERE` with your actual API key
        4. Save and restart the app
        """)
        return
    
    # Sidebar
    st.sidebar.header("⚙️ Settings")
    max_comments = st.sidebar.slider("Maximum comments to analyze", 50, 5000, 500, 50)
    
    if max_comments > 2000:
        st.sidebar.warning("⚠️ Analyzing >2000 comments may take 10+ minutes")
    
    if max_comments > 5000:
        st.sidebar.error("🚨 This will use significant API quota!")
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 💡 Tips")
    st.sidebar.markdown("""
    - Start with 100-500 comments for quick results
    - Videos with disabled comments won't work
    - API has a daily limit of ~10,000 requests
    """)
    
    # Main content
    video_url = st.text_input("🎥 Enter YouTube Video URL", placeholder="https://www.youtube.com/watch?v=...")
    
    col1, col2 = st.columns([1, 4])
    with col1:
        analyze_button = st.button("🔍 Analyze", type="primary", use_container_width=True)
    
    if analyze_button:
        if not video_url:
            st.error("⚠️ Please enter a YouTube video URL")
            return
        
        # Extract video ID
        video_id = extract_video_id(video_url)
        if not video_id:
            st.error("❌ Invalid YouTube URL. Please check and try again.")
            return
        
        # Load model
        with st.spinner("Loading AI model..."):
            model = load_model()
        
        if model is None:
            st.error("❌ Could not load the sentiment model. Make sure 'youtube_sentiment_best.h5' exists.")
            return
        
        # Fetch comments
        video_title, comments = get_youtube_comments(video_id, max_comments)
        
        if video_title is None or not comments:
            st.error("❌ Could not fetch comments. Please check the video URL and try again.")
            return
        
        st.success(f"✅ Fetched {len(comments)} comments from: **{video_title}**")
        
        # Predict sentiments
        st.info("🤖 Analyzing sentiment with AI model...")
        sentiments = predict_sentiment(model, video_title, comments)
        
        # Sentiment mapping
        sentiment_labels = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
        sentiment_counts = pd.Series(sentiments).value_counts().sort_index()
        
        # Create results dataframe
        results_df = pd.DataFrame({
            'Comment': comments,
            'Sentiment': [sentiment_labels[s] for s in sentiments]
        })
        
        # Display results
        st.markdown("---")
        st.subheader("📈 Sentiment Distribution")
        
        col1, col2, col3 = st.columns(3)
        
        total = len(sentiments)
        positive_count = sentiment_counts.get(2, 0)
        neutral_count = sentiment_counts.get(1, 0)
        negative_count = sentiment_counts.get(0, 0)
        
        with col1:
            st.metric("😊 Positive", f"{positive_count}", f"{positive_count/total*100:.1f}%")
        with col2:
            st.metric("😐 Neutral", f"{neutral_count}", f"{neutral_count/total*100:.1f}%")
        with col3:
            st.metric("😞 Negative", f"{negative_count}", f"{negative_count/total*100:.1f}%")
        
        # Pie chart
        st.markdown("### 🥧 Sentiment Pie Chart")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = ['#ff6b6b', '#95a5a6', '#4ecdc4']
        labels = ['Negative', 'Neutral', 'Positive']
        sizes = [negative_count, neutral_count, positive_count]
        
        wedges, texts, autotexts = ax.pie(
            sizes, 
            labels=labels, 
            colors=colors,
            autopct='%1.1f%%',
            startangle=90,
            textprops={'fontsize': 12, 'weight': 'bold'}
        )
        
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontsize(14)
        
        ax.axis('equal')
        plt.title(f'Sentiment Analysis Results', fontsize=14, weight='bold', pad=20)
        st.pyplot(fig)
        
        # Show comments by sentiment
        st.markdown("---")
        st.subheader("💬 Comments by Sentiment")
        
        tab1, tab2, tab3 = st.tabs(["😊 Positive", "😐 Neutral", "😞 Negative"])
        
        with tab1:
            positive_comments = results_df[results_df['Sentiment'] == 'Positive']
            if len(positive_comments) > 0:
                st.write(f"**Total Positive Comments: {len(positive_comments)}**")
                for idx, row in positive_comments.head(50).iterrows():
                    st.success(row['Comment'])
                if len(positive_comments) > 50:
                    st.info(f"Showing 50 of {len(positive_comments)} positive comments. Download CSV to see all.")
            else:
                st.info("No positive comments found")
        
        with tab2:
            neutral_comments = results_df[results_df['Sentiment'] == 'Neutral']
            if len(neutral_comments) > 0:
                st.write(f"**Total Neutral Comments: {len(neutral_comments)}**")
                for idx, row in neutral_comments.head(50).iterrows():
                    st.info(row['Comment'])
                if len(neutral_comments) > 50:
                    st.info(f"Showing 50 of {len(neutral_comments)} neutral comments. Download CSV to see all.")
            else:
                st.info("No neutral comments found")
        
        with tab3:
            negative_comments = results_df[results_df['Sentiment'] == 'Negative']
            if len(negative_comments) > 0:
                st.write(f"**Total Negative Comments: {len(negative_comments)}**")
                for idx, row in negative_comments.head(50).iterrows():
                    st.error(row['Comment'])
                if len(negative_comments) > 50:
                    st.info(f"Showing 50 of {len(negative_comments)} negative comments. Download CSV to see all.")
            else:
                st.info("No negative comments found")
        
        # Download option
        st.markdown("---")
        csv = results_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Results as CSV",
            data=csv,
            file_name=f"sentiment_analysis_{video_id}.csv",
            mime="text/csv",
            use_container_width=True
        )

if __name__ == "__main__":
    main()
