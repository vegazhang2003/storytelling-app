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
    return pipeline(
        "image-to-text",
        model="Salesforce/blip-image-captioning-base"
    )


@st.cache_resource
def get_story_model():
    return pipeline(
        "text-generation",
        model="Qwen/Qwen2.5-0.5B-Instruct"
    )


def create_audio(text):
    tts = gTTS(text=text, lang="en")
    audio_fp = BytesIO()
    tts.write_to_fp(audio_fp)
    audio_fp.seek(0)
    return audio_fp.getvalue()


def clean_story(text, prompt):
    story = text.replace(prompt, "").strip()

    unwanted_phrases = [
        "Story:",
        "Generated Story:",
        "Write in present tense.",
        "Do not start or end with outlying information.",
        "Only write the story.",
    ]

    for phrase in unwanted_phrases:
        story = story.replace(phrase, "").strip()

    return story


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
                "Please write only a short children's story based on this image description: "
                + caption
                + ". The story should be 50 to 100 words. "
                + "Use simple English for children aged 3 to 10. "
                + "Make the story warm, happy, and easy to understand. "
                + "Do not include instructions, titles, or explanations. "
                + "Start the story directly."
            )

            story_result = story_model(
                prompt,
                max_new_tokens=130,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.2
            )

            full_text = story_result[0]["generated_text"]
            story = clean_story(full_text, prompt)

        st.subheader("Generated Story")
        st.write(story)

        with st.spinner("Creating audio..."):
            audio_bytes = create_audio(story)

        st.subheader("Audio")
        st.audio(audio_bytes, format="audio/mp3")
