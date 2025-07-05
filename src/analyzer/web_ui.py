
import streamlit as st
from code_suggester import CodeSuggester

st.set_page_config(page_title="PyCritic", layout="wide")

st.title("PyCritic: Python Code Analysis")

st.write("Upload a Python file to get a complete analysis of your code, including naming conventions, import issues, and a quality score.")

uploaded_file = st.file_uploader("Choose a Python file", type="py")

if uploaded_file is not None:
    code_string = uploaded_file.getvalue().decode("utf-8")

    st.subheader("Your Code")
    st.code(code_string, language="python")

    if st.button("Analyze Code"):
        with st.spinner("Analyzing your code..."):
            suggester = CodeSuggester()
            results = suggester.generate_suggestions(code_string)

            st.subheader("Analysis Results")

            # Display ML Code Scoring in columns
            if results.get("ml_results"):
                st.write("**Code Quality Score**")
                quality_scores = results["ml_results"]
                quality_emojis = {
                    "excellent": "🟢",
                    "good": "🟡",
                    "fair": "🟠",
                    "poor": "🔴",
                }

                col1, col2, col3 = st.columns(3)
                with col1:
                    quality = quality_scores.get("quality", "N/A")
                    st.metric("Quality", f'{quality_emojis.get(quality, "")} {quality.capitalize()}')
                with col2:
                    naming = quality_scores.get("naming", "N/A")
                    st.metric("Naming", f'{quality_emojis.get(naming, "")} {naming.capitalize()}')
                with col3:
                    style = quality_scores.get("style", "N/A")
                    st.metric("Style", f'{quality_emojis.get(style, "")} {style.capitalize()}')
                st.write("---")

            # Display Naming Violations in an expander
            if results.get("naming_violations", {}).get("violations"):
                with st.expander("Naming Conventions"):
                    for violation in results["naming_violations"]["violations"]:
                        st.write(f"**Line {violation['line']}:** `{violation['original']}`")
                        st.write(f"> {violation['explanation']}")
                        if violation.get('corrected'):
                            st.write(f"> **Correction:** `{violation['corrected']}`")
                        st.write("")

            # Display Import Suggestions in an expander
            if results.get("import_suggestions"):
                with st.expander("Import Analysis"):
                    st.markdown(results["import_suggestions"])

            # Display Summary
            if results.get("summary"):
                st.write("**Summary of Suggestions**")
                st.write(results["summary"])
