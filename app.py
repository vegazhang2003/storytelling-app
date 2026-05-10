import streamlit as st
from transformers import pipeline
from PIL import Image
from gtts import gTTS
from io import BytesIO

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


def make_audio(text):
    tts = gTTS(text=text, lang="en")
    audio_file = BytesIO()
    tts.write_to_fp(audio_file)
    audio_file.seek(0)
    return audio_file


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
                + "Only write the story. Do not include instructions or extra explanation."
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
            story = story.replace("Generated Story:", "").strip()
            story = story.replace("Do not start or end with outlying information.", "").strip()

        st.subheader("Generated Story")
        st.write(story)

        with st.spinner("Creating audio..."):
            audio_file = make_audio(story)

        st.subheader("Audio")
        st.audio(audio_file, format="audio/mp3")
