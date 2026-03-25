# Streamlit app code 
streamlit_code = """
import streamlit as st
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load pickle files
try:
    popular_df = pickle.load(open('popular.pkl', 'rb'))
    pt = pickle.load(open('pt.pkl', 'rb'))
    books = pickle.load(open('books.pkl', 'rb'))
    similarity_scores = pickle.load(open('similarity_scores.pkl', 'rb'))
except FileNotFoundError as e:
    st.error(f"Pickle file not found: {str(e)}. Ensure popular.pkl, pt.pkl, books.pkl, and similarity_scores.pkl are in the directory.")
    st.stop()
except Exception as e:
    st.error(f"Error loading pickle files: {str(e)}")
    st.stop()

# Preprocess for content-based (combine title, author, publisher for TF-IDF)
books['features'] = books['Book-Title'].fillna('') + ' ' + books['Book-Author'].fillna('') + ' ' + books['Publisher'].fillna('')
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(books['features'])

# Popular books from popular_df
popular_books = popular_df['Book-Title'].tolist()[:5]

# Function for content-based recommendations using TF-IDF cosine similarity
def get_content_recs(selected_title, top_n=6):
    try:
        selected_idx = books[books['Book-Title'] == selected_title].index[0]
        sim_scores = cosine_similarity(tfidf_matrix[selected_idx], tfidf_matrix)[0]
        sim_scores = sorted(enumerate(sim_scores), key=lambda x: x[1], reverse=True)[1:top_n+1]
        rec_indices = [i for i, score in sim_scores]
        return books.iloc[rec_indices][['Book-Title', 'Image-URL-M', 'Image-URL-S']].values.tolist()
    except IndexError:
        st.warning(f"No content-based recommendations for '{selected_title}' due to missing index.")
        return []
    except Exception as e:
        st.error(f"Error in content-based recommendations: {str(e)}")
        return []

# Function for collaborative recommendations using similarity scores
def get_collab_recs(selected_title, top_n=6):
    try:
        selected_idx = pt.index.get_loc(selected_title)
        sim_scores = list(enumerate(similarity_scores[selected_idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
        rec_indices = [i for i, score in sim_scores]
        collab_titles = pt.index[rec_indices].tolist()
        collab_data = books[books['Book-Title'].isin(collab_titles)][['Book-Title', 'Image-URL-M', 'Image-URL-S']].drop_duplicates().values.tolist()
        return collab_data
    except KeyError:
        st.warning(f"No collaborative recommendations for '{selected_title}' due to missing data.")
        return []
    except Exception as e:
        st.error(f"Error in collaborative recommendations: {str(e)}")
        return []

# Function for hybrid recommendations
def get_hybrid_recs(selected_title, alpha=0.5, top_n=6):
    try:
        content_recs = get_content_recs(selected_title, top_n=6)
        collab_recs = get_collab_recs(selected_title, top_n=6)
        scores = {}
        for title, _, _ in content_recs:
            scores[title] = scores.get(title, 0) + alpha
        for title, _, _ in collab_recs:
            scores[title] = scores.get(title, 0) + (1 - alpha)
        sorted_books = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        unique_titles = list(dict.fromkeys([t for t, _ in sorted_books]))[:top_n]
        hybrid_data = books[books['Book-Title'].isin(unique_titles)][['Book-Title', 'Image-URL-M', 'Image-URL-S']].drop_duplicates().values.tolist()
        return hybrid_data
    except Exception as e:
        st.error(f"Error in hybrid recommendations: {str(e)}")
        return []

# Display clickable book images 
def display_clickable_images(book_list, caption_prefix="Book"):
    if not book_list:
        st.write("No books to display.")
        return
    num_cols = 6
    num_rows = (len(book_list) + num_cols - 1) // num_cols
    selected_title = st.session_state.get('selected_title', None)
    for row in range(num_rows):
        cols = st.columns(num_cols)
        for col in range(num_cols):
            index = row * num_cols + col
            if index < len(book_list):
                title, url_m, url_s = book_list[index]
                with cols[col]:
                    image_url = url_m if url_m and str(url_m).strip() else (url_s if url_s and str(url_s).strip() else "https://via.placeholder.com/150?text=No+Image")
                    st.image(image_url, width=150, use_column_width=False)
                    if st.button(title, key=f"btn_{index}"):
                        st.session_state.selected_title = title
            else:
                with cols[col]:
                    st.write("")

# Display recommendations based on selected title
def display_recommendations(selected_title):
    if selected_title:
        st.header(f"Recommendations for {selected_title}")
        content_recs = get_content_recs(selected_title)
        collab_recs = get_collab_recs(selected_title)
        hybrid_recs = get_hybrid_recs(selected_title)
        
        if content_recs:
            st.subheader("Content-Based Recommendations")
            display_book_images(content_recs, "Content")
        else:
            st.write("No content recommendations available.")
        
        if collab_recs:
            st.subheader("Collaborative Recommendations")
            display_book_images(collab_recs, "Collaborative")
        else:
            st.write("No collaborative recommendations available.")
        
        alpha = st.slider("Adjust Hybrid Weight (Content vs Collab)", 0.0, 1.0, 0.5, 0.1, key="hybrid_slider")
        if hybrid_recs:
            st.subheader("Hybrid Recommendations")
            display_book_images(hybrid_recs, "Hybrid")
        else:
            st.write("No hybrid recommendations available.")

# Display book images 
def display_book_images(book_list, caption_prefix="Book"):
    if not book_list:
        st.write("No books to display.")
        return
    num_cols = 6
    num_rows = (len(book_list) + num_cols - 1) // num_cols
    for row in range(num_rows):
        cols = st.columns(num_cols)
        for col in range(num_cols):
            index = row * num_cols + col
            if index < len(book_list):
                title, url_m, url_s = book_list[index]
                with cols[col]:
                    image_url = url_m if url_m and str(url_m).strip() else (url_s if url_s and str(url_s).strip() else "https://via.placeholder.com/150?text=No+Image")
                    st.image(image_url, width=150, use_column_width=False)
                    st.markdown(
                        f"<div style='text-align:center; margin-top:5px;'><p style='font-size:16px; font-weight:bold;'>{title}</p></div>",
                        unsafe_allow_html=True
                    )
            else:
                with cols[col]:
                    st.write("")

# Streamlit App with landscape-optimized UI
st.set_page_config(layout="wide")
st.title("Book Recommender")

# Section 1: Popular Books with clickable images
st.header("Popular Books")
st.write("Click on a book image to see recommendations:")
popular_data = books[books['Book-Title'].isin(popular_books)][['Book-Title', 'Image-URL-M', 'Image-URL-S']].drop_duplicates().values.tolist()
display_clickable_images(popular_data)

# Display recommendations based on clicked image
selected_title = st.session_state.get('selected_title', None)
if selected_title:
    display_recommendations(selected_title)
"""