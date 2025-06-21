import streamlit as st

# Import functions from your existing files
from ml_static_result import code_to_trained_format, static_results_to_string, analyze_code
from static_analyzer import StaticCodeAnalyzer


def main():
    st.set_page_config(
        page_title="PyCritic - Python Code Analyzer",
        page_icon="",
        layout="wide"
    )

    st.title("PyCritic - Python Code Analyzer")
    st.markdown(
        "### Analyze your Python code for naming conventions, style, and quality")
    st.markdown("""
    <style>
        .main .block-container {
            max-width: 1600px;
            padding-left: 2rem;
            padding-right: 2rem;
        }
    </style>
""", unsafe_allow_html=True)

    # Sidebar for information
    # with st.sidebar:
    #     st.header("About PyCritic")
    #     st.markdown("""
    #     **PyCritic** combines static analysis and machine learning to evaluate your Python code on:
        
    #     - **Static Analysis**: Naming conventions, syntax issues
    #     - **ML Analysis**: Code quality, naming patterns, style
        
    #     **Quality Categories:**
    #     - 🟢 Excellent (65+)
    #     - 🟡 Good (55-64)
    #     - 🟠 Fair (45-54)
    #     - 🔴 Poor (<45)
    #     """)

    #     st.header("Sample Code")
    #     if st.button("Load Sample Code"):
    #         st.session_state.sample_code = True

    # Main content area
    col1, col2 = st.columns([1, 1])

    with col1:
        st.header("📝 Input Code")

        # Default sample code
        default_code = """import numpy as np

class ProperClass:
    def correct_method(self):
        proper_variable = 42
        MAX_vALUE = 100
        _private_var = "secret"
        return proper_variable * MAX_VALUE"""

        # Load sample code if button clicked
        if 'sample_code' in st.session_state and st.session_state.sample_code:
            code_input = default_code
            st.session_state.sample_code = False
        else:
            code_input = ""

        # Code input text area
        user_code = st.text_area(
            "Enter your Python code here:",
            value=code_input,
            height=400,
            placeholder="Type or paste your Python code here..."
        )

        # Analysis button
        analyze_button = st.button("🔍 Analyze Code", type="primary")

    with col2:
        st.header("📊 Analysis Results")

        if analyze_button and user_code.strip():
            with st.spinner("Analyzing your code..."):
                try:
                    # Use your existing analyze_code function
                    result = analyze_code(user_code)

                    if result and len(result) == 2:
                        static_result, ml_results = result
                    else:
                        static_result = "Analysis completed"
                        ml_results = None

                except Exception as e:
                    st.error(f"Analysis failed: There may be syntax error/s!!{str(e)}")
                    static_result = None
                    ml_results = None

            if static_result is not None:
                # Display results in tabs
                tab1, tab2, tab3 = st.tabs(
                    ["📋 Summary", "🔍 Static Analysis", "🤖 ML Analysis"])

                with tab1:
                    st.subheader("Analysis Summary")

                    # Create summary metrics
                    col_a, col_b, col_c = st.columns(3)

                    if ml_results:
                        # Quality score with color coding
                        quality = ml_results.get('quality', 'N/A')
                        quality_colors = {
                            'excellent': '🟢',
                            'good': '🟢',
                            'fair': '🟠',
                            'poor': '🔴'
                        }

                        with col_a:
                            st.metric(
                                "Quality",
                                f"{quality_colors.get(quality, '⚪')} {quality.title()}"
                            )

                        with col_b:
                            naming = ml_results.get('naming', 'N/A')
                            st.metric(
                                "Naming",
                                f"{quality_colors.get(naming, '⚪')} {naming.title()}"
                            )

                        with col_c:
                            style = ml_results.get('style', 'N/A')
                            st.metric(
                                "Style",
                                f"{quality_colors.get(style, '⚪')} {style.title()}"
                            )

                    # Parse static analysis for violations count
                    violation_count = static_result.count("Line ")
                    if violation_count == 0:
                        st.success("✅ No naming convention violations found!")
                    else:
                        st.warning(
                            f"⚠️ {violation_count} naming convention violation(s) found")

                with tab2:
                    st.subheader("Static Analysis Results")

                    if "Total violations found: 0" in static_result:
                        st.success("✅ No naming convention violations found!")
                    else:
                        st.text(static_result)

                with tab3:
                    st.subheader("ML Analysis Results")

                    if ml_results:
                        # Display predictions with styling
                        st.write("**Model Predictions:**")

                        prediction_data = {
                            "Aspect": ["Quality", "Naming", "Style"],
                            "Rating": [
                                ml_results.get('quality', 'N/A').title(),
                                ml_results.get('naming', 'N/A').title(),
                                ml_results.get('style', 'N/A').title()
                            ]
                        }

                        st.table(prediction_data)

                        # Display suggestions
                        if 'suggestions' in ml_results and ml_results['suggestions']:
                            st.write("**💡 Improvement Suggestions:**")
                            for suggestion in ml_results['suggestions']:
                                st.write(f"• {suggestion}")
                    else:
                        st.error(
                            "❌ ML analysis failed. Please check if the model file exists.")

        elif analyze_button and not user_code.strip():
            st.warning("⚠️ Please enter some code to analyze!")

        elif not analyze_button:
            st.info(
                "👆 Enter your Python code and click 'Analyze Code' to get started!")

    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: gray;'>"
        "Built with ❤️ using Streamlit | PyCritic v1.0"
        "</div>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
