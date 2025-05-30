import os

import altair as alt
import pandas as pd
import requests
import streamlit as st
from google.cloud import run_v2

st.set_page_config(page_title="♠️♥️Card Classifier♦️♣️")
st.markdown(
    """
<style>
    .stAppHeader {
        display: none;
    }
    .stMainBlockContainer {
        height: 300vh;
        max-width: 60rem;
        background: linear-gradient(90deg,#050,#0a0 20% 80%,#050);
        padding: 6rem 6rem 10rem;
    }
    .st-key-upload_button {
        text-align: right;
    }
    .stImage {
        padding: 20px;
        border: solid 5px white;
        border-radius: 20px;
        aspect-ratio: 4 / 5;
        align-content: center;
        &>div>div {
            color: white;
            font-size: 20px;
            font-weight: bold;
        }
    }
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource(ttl=1)  # Cache for 1 second to force frequent refresh
def get_backend_url():
    """Get the URL of the backend service."""
    try:
        parent = "projects/mlops-448220/locations/europe-west1"
        client = run_v2.ServicesClient()
        services = client.list_services(parent=parent)
        for service in services:
            if service.name.split("/")[-1] == "backend":
                return service.uri
    except Exception as e:
        st.warning(f"Could not access Cloud Run services: {str(e)}")
        
    # Fallback to environment variable or docker container name
    name = os.environ.get("BACKEND_URL", "http://api-v2:8000")
    return name


def classify_image(image_files, backend):
    """Send the image to the backend for classification."""
    predict_url = f"{backend}/classify/"
    results = []
    for file in image_files:
        response = requests.post(predict_url, files={"file": file}, timeout=10)
        if response.status_code == 200:
            results.append(response.json())
        else:
            st.error(f"Error from API: {response.status_code} - {response.text}")
            return None
    
    if not results:
        return None
        
    # Combine results
    combined_result = {
        "prediction": [r["prediction"][0] for r in results],
        "probabilities": {str(i): r["probabilities"] for i, r in enumerate(results)}
    }
    return combined_result


def main() -> None:
    """Main function of the Streamlit frontend."""
    backend = get_backend_url()
    if backend is None:
        msg = "Backend service not found"
        raise ValueError(msg)
    st.title("Card classifier")

    uploaded_file = st.file_uploader("Upload card(s)", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
    ucl, ucr = st.columns([0.85, 0.15])
    pressed = ucr.button("Upload", key="upload_button")

    if pressed and len(uploaded_file) != 0:
        bimgs = [file.read() for file in uploaded_file]
        result = classify_image(bimgs, backend=backend)
        if result is not None:
            prediction = result["prediction"]
            probabilities = result["probabilities"]

            # show the image and prediction
            for (
                img,
                prob,
            ) in probabilities.items():
                lc, rc = st.columns([0.35, 0.65], gap="small")
                img = int(img)
                lc.image(bimgs[img], caption="Card " + str(img + 1))
                rc.subheader(
                    f"Prediction: {prediction[img]}",
                )

                # make a nice bar chart
                data = prob.items()
                df = pd.DataFrame(data)
                df.rename({0: "Card", 1: "Probability"}, axis=1, inplace=True)
                # Convert probabilities to percentages
                df["Probability"] = df["Probability"] * 100
                # Sort by probability in descending order and take top 5
                df = df.sort_values("Probability", ascending=False).head(5)
                rc.altair_chart(
                    alt.Chart(df)
                    .mark_bar()
                    .encode(
                        x=alt.X("Probability", 
                               title="Probability (%)", 
                               axis=alt.Axis(format=',.1f'),
                               scale=alt.Scale(domain=[0, 100])),
                        y=alt.Y("Card", sort="-x", title="Card"),
                    )
                    .properties(height=200)
                    .configure(background="#0005")
                )
        else:
            ucl.text("Failed to get prediction!")
    elif pressed:
        ucl.text("No images uploaded!")


if __name__ == "__main__":
    main()
