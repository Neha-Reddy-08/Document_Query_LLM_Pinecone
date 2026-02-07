import streamlit as st
from main import retrive_answer  # import from your module

st.title("Agriculture QA System")

user_question = st.text_input("Ask a question about agriculture in India:")

if st.button("Get Answer"):
    if user_question:
        with st.spinner("Thinking..."):
            answer = retrive_answer(user_question)
        st.success("Answer:")
        st.write(answer)
    else:
        st.warning("Please type a question!")
