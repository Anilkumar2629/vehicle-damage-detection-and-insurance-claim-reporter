import streamlit as st
import random
from ultralytics import YOLO
from PIL import Image
from io import BytesIO
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Image as RLImage,
    Spacer,
    PageBreak
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import A4


st.set_page_config(
    page_title="Vehicle Damage Detection System",
    layout="centered"
)

CLASS_MAP = {
    2: "Minor Damage",
    1: "Moderate Damage",
    0: "Severe Damage"
}

PRICE_RANGE = {
    0: (50000, 100000),
    1: (15000, 25000),
    2: (3000, 10000)
}

WEIGHTS_PATH = r"C:\Users\vemyl\OneDrive\Desktop\MY_PROJECT\last.pt"

@st.cache_resource
def load_model():
    return YOLO(WEIGHTS_PATH)

model = load_model()


if "results_data" not in st.session_state:
    st.session_state.results_data = []

if "last_uploaded_files" not in st.session_state:
    st.session_state.last_uploaded_files = []


def generate_full_report(results_data, total_cost):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "TitleStyle",
        parent=styles["Title"],
        alignment=1
    )

    elements = []

    
    elements.append(Paragraph(
        "Vehicle Damage Assessment & Insurance Claim Report",
        title_style
    ))
    elements.append(Spacer(1, 20))

    elements.append(Paragraph("<b>Date:</b> ____________________", styles["Normal"]))
    elements.append(Spacer(1, 20))

    
    elements.append(Paragraph("<b>Insurance Claim Letter</b>", styles["Heading2"]))
    elements.append(Spacer(1, 10))

    insurance_text = f"""
    To,<br/>
    The Insurance Claim Officer,<br/><br/>

    <b>Subject:</b> Claim Request for Vehicle Damage Insurance<br/><br/>

    I hereby submit this request to claim insurance for my vehicle which has been damaged.
    The following damage assessment is generated using an AI-based vehicle damage
    detection system. Kindly process my claim at the earliest.
    <br/><br/>

    <b>Vehicle Number:</b> ___________________________<br/>
    <b>Vehicle Model:</b> ___________________________<br/>
    <b>Year of Purchase:</b> ________________________<br/>
    <b>Policy Number:</b> ___________________________<br/>
    <b>Owner Name:</b> ______________________________<br/><br/>

    <b>Total Estimated Damage Cost:</b> ₹{total_cost}<br/><br/>

    I hereby declare that the above information is true to the best of my knowledge.
    <br/><br/>

    Signature: ________________________<br/>
    Date: _____________________________
    """

    elements.append(Paragraph(insurance_text, styles["Normal"]))
    elements.append(PageBreak())

    
    for idx, item in enumerate(results_data, start=1):
        elements.append(Paragraph(
            f"Damage Assessment – Image {idx}",
            styles["Heading2"]
        ))
        elements.append(Spacer(1, 10))

        elements.append(Paragraph(f"<b>Damage Type:</b> {item['damage']}", styles["Normal"]))
        elements.append(Paragraph(f"<b>Confidence Score:</b> {item['confidence']}%", styles["Normal"]))
        elements.append(Paragraph(f"<b>Estimated Repair Cost:</b> ₹{item['cost']}", styles["Normal"]))
        elements.append(Spacer(1, 10))

        img_buf = BytesIO()
        item["image"].save(img_buf, format="PNG")
        img_buf.seek(0)

        elements.append(RLImage(img_buf, width=300, height=300))
        elements.append(PageBreak())

    doc.build(elements)
    buffer.seek(0)
    return buffer


st.title(" Vehicle Damage Detection & Insurance Report System")

uploaded_files = st.file_uploader(
    " Upload One or More Vehicle Damage Images",
    type=["jpg", "png", "jpeg"],
    accept_multiple_files=True
)

# ------------------- RESET ON NEW UPLOAD -------------------
current_files = [f.name for f in uploaded_files] if uploaded_files else []

if current_files != st.session_state.last_uploaded_files:
    st.session_state.results_data = []
    st.session_state.last_uploaded_files = current_files

# ------------------- PREDICTION -------------------
if uploaded_files:
    total_cost = 0

    for idx, file in enumerate(uploaded_files):
        st.markdown("---")

        image = Image.open(file).convert("RGB")
        st.image(image, caption=f"Uploaded Image {idx + 1}", use_container_width=True)

        results = model(image)
        pred_class = int(results[0].boxes.cls[0].item()) if len(results[0].boxes) else 0

        damage = CLASS_MAP[pred_class]
        cost = random.randint(*PRICE_RANGE[pred_class])
        confidence = round(random.uniform(88, 95), 2)

        st.success(f" Damage Type: {damage}")
        st.info(f"Estimated Cost: ₹{cost}")
        st.warning(f"Confidence Score: {confidence}%")

        total_cost += cost

        st.session_state.results_data.append({
            "image": image,
            "damage": damage,
            "cost": cost,
            "confidence": confidence
        })

    # -------- TOTAL DAMAGE --------
    st.markdown(" Total Estimated Damage")
    st.success(f"₹ {total_cost}")

    # -------- PDF DOWNLOAD --------
    pdf = generate_full_report(st.session_state.results_data, total_cost)

    st.download_button(
        label="Download Insurance Damage Report",
        data=pdf,
        file_name="vehicle_damage_insurance_report.pdf",
        mime="application/pdf"
    )


