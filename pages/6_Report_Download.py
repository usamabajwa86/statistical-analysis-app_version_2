import streamlit as st
from utils.report_generator import generate_docx_report

st.title("📥 Report Download")

# Ask user for optional summary/interpretation
summary = st.text_area("Enter summary or interpretation for the report", "")

# Check if dataframe exists
df = st.session_state.get("df")
if df is None:
    st.warning("Please upload and analyze data first to generate a report.")
else:
    if st.button("Generate Report"):
        # Generate report (pass both summary and df)
        path = "analysis_report.docx"
        generate_docx_report(df, summary, path)

        # Provide download button
        with open(path, "rb") as f:
            st.download_button(
                label="Download Report",
                data=f,
                file_name="analysis_report.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            )

st.write("✔ Page loaded successfully!")
