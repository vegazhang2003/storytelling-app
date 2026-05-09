import streamlit as st
from transformers import pipeline
from PIL import Image
from gtts import gTTS
import tempfile

st.set_page_config(page_title="Storytelling App", page_icon="📖")

st.title("📖 Storytelling App for Kids")
st.write("Upload an image and the app will create a short story for children.")

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)


@st.cache_resource
def get_caption_model():
    caption_model = pipeline(
        "image-to-text",
        model="Salesforce/blip-image-captioning-base"
    )
    return caption_model


@st.cache_resource
def get_story_model():
    story_model = pipeline(
        "text-generation",
        model="Qwen/Qwen2.5-0.5B-Instruct"
    )
    return story_model


if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("Generate Story"):
        with st.spinner("Loading models..."):
            caption_model = get_caption_model()
            story_model = get_story_model()

        with st.spinner("Reading the image..."):
            caption_result = caption_model(image)
            caption = caption_result[0]["generated_text"]

        st.subheader("Image Caption")
        st.write(caption)

        with st.spinner("Writing the story..."):
            prompt = (
                "Write a short story for children aged 3 to 10 based on this image description: "
                + caption
                + ". The story should be 50 to 100 words. "
                + "Use simple English. Make the story warm, happy, and easy to understand. "
                + "Only write the story."
            )

            story_result = story_model(
                prompt,
                max_new_tokens=140,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.2
            )

            story = story_result[0]["generated_text"]
            story = story.replace(prompt, "").strip()
            story = story.replace("Story:", "").strip()

        st.subheader("Generated Story")
        st.write(story)

        with st.spinner("Creating audio..."):
            tts = gTTS(text=story, lang="en")

            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio:
                tts.save(temp_audio.name)
                audio_path = temp_audio.name

        st.subheader("Audio")
        st.audio(audio_path, format="audio/mp3")