def run(model, data, processed, cancer_col_desc):
    import streamlit as st
    import numpy as np
    import pandas as pd
    import altair as alt

    columns = processed['columns']
    scaler = processed['scaler']
    X_viz = processed['X_viz']
    y_viz = processed['y_viz']

    st.header('Breast Cancer Classifier')
    st.markdown("This app demonstrates a **binary classification** task to distinguish between _malignant_ and _benign_ tumors using 30 features from digitized images of fine needle aspirate (FNA) of breast mass. **Target labels:** Benign (0), Malignant (1). _Dataset: scikit-learn's Breast Cancer Wisconsin._")

    st.subheader('Feature Distributions')
    viz_grid = st.columns(6, gap="medium")
    for i, col in enumerate(columns[:6]):
        with viz_grid[i]:
            chart = alt.Chart(pd.DataFrame({col: X_viz[col]})).mark_bar().encode(
                alt.X(f"{col}:Q", bin=alt.Bin(maxbins=20), title=col),
                y=alt.Y('count()', title=None),
                tooltip=[col, 'count()']
            ).properties(height=250)
            st.altair_chart(chart, use_container_width=True)

    left, right = st.columns([1, 1])

    with left:
        st.subheader('Input Features')
        form = st.form('cancer_form')
        inputs = {}
        col1, col2, col3 = form.columns(3)
        for idx, col in enumerate(columns):
            avg = float(np.mean(X_viz[col]))
            desc = cancer_col_desc.get(col, col.replace('_', ' ').capitalize())
            help_txt = f"{desc[:40]} (default: {avg:.2f})"
            label_txt = col.title()
            if idx % 3 == 0:
                inputs[col] = col1.number_input(
                    label=label_txt,
                    value=avg,
                    help=help_txt
                )
            elif idx % 3 == 1:
                inputs[col] = col2.number_input(
                    label=label_txt,
                    value=avg,
                    help=help_txt
                )
            else:
                inputs[col] = col3.number_input(
                    label=label_txt,
                    value=avg,
                    help=help_txt
                )
        _, btn_col, _ = form.columns([1, 1, 1])
        submitted = btn_col.form_submit_button(
            'Classify', type="primary", use_container_width=True)

    with right:
        pred_head_col, pred_val_col = st.columns([2, 1])
        pred_head_col.subheader(
            'Predict Cancer Type')
        pred_placeholder = pred_val_col.empty()
        if 'prediction' not in st.session_state:
            st.session_state['prediction'] = None
        if submitted:
            # Scale input features before prediction
            input_arr = np.array([[inputs[c] for c in columns]])
            input_arr_scaled = scaler.transform(input_arr)
            pred = float(model.predict(input_arr_scaled)[0, 0])
            st.session_state['prediction'] = pred
        if st.session_state['prediction'] is not None:
            label = 'Malignant' if st.session_state['prediction'] >= 0.5 else 'Benign'
            pred_placeholder.success(f"{label}")

        target_counts = y_viz.value_counts().sort_index()
        pie_df = pd.DataFrame(
            {'label': ['Benign', 'Malignant'], 'value': target_counts.values})
        pie_chart = alt.Chart(pie_df).mark_arc().encode(
            theta=alt.Theta(field='value', type='quantitative'),
            color=alt.Color(field='label', type='nominal'),
            tooltip=['label', 'value']
        )
        st.altair_chart(pie_chart, use_container_width=True)


    exp_col1, exp_col2 = st.columns(2)
    with exp_col1:
        with st.expander('How the Model Works'):
            st.markdown('''**Model Overview:**
- This model has 4 layers:
    - 1 input layer (30 features)
    - 2 hidden layers (30, 60, 15 neurons)
    - 1 output layer (1 value)
- Uses ReLU and Sigmoid activations
- Loss Function: Binary Cross Entropy

**Total Parameters:** 3,721
''')
    with exp_col2:
        with st.expander('Performance & Training Insights'):
            st.markdown('''<b>Performance Metrics</b>''',
                        unsafe_allow_html=True)
            st.markdown('''
<table>
<thead>
<tr><th>Class</th><th>Precision</th><th>Recall</th><th>F1-score</th></tr>
</thead>
<tbody>
<tr><td>0</td><td>0.88</td><td>0.81</td><td>0.84</td></tr>
<tr><td>1</td><td>0.89</td><td>0.93</td><td>0.91</td></tr>
<tr><td colspan="4" style="text-align:center"><b>Accuracy: 89%</b></td></tr>
</tbody>
</table>
''', unsafe_allow_html=True)
            st.image('assets/breast_cancer_loss_curve.png',
                     caption='Loss Curve')
