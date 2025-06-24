def run():
    import streamlit as st
    import numpy as np
    import pandas as pd
    import altair as alt

    model = st.session_state['model']
    data = st.session_state['data']

    X = data['data']
    y = data['target']
    columns = X.columns.tolist()

    # Select up to 6 user-friendly columns for visualization
    viz_cols = ['mean radius', 'mean texture', 'mean perimeter',
                'mean area', 'mean smoothness', 'mean compactness']
    viz_cols = [col for col in viz_cols if col in columns][:6]

    st.header('Breast Cancer Detection')

    # Visualizations (histogram distributions with Altair)
    st.subheader('Feature Distributions')
    viz_grid = st.columns(len(viz_cols))
    for i, col in enumerate(viz_cols):
        with viz_grid[i]:
            chart = alt.Chart(pd.DataFrame({col: X[col]})).mark_bar().encode(
                alt.X(f"{col}:Q", bin=alt.Bin(maxbins=20), title=col),
                y=alt.Y('count()', title=None),
                tooltip=[col, 'count()']
            ).properties(height=250)
            st.altair_chart(chart, use_container_width=True)

    # 2-column layout
    left, right = st.columns([1, 1])

    with left:
        st.subheader('Input Features')
        form = st.form('cancer_form')
        inputs = {}
        for col in columns:
            avg = float(np.mean(X[col]))
            inputs[col] = form.number_input(
                label=col,
                value=avg,
                help=f"Average value: {avg:.2f} (default if left blank)"
            )
        submitted = form.form_submit_button('Detect')

    with right:
        st.subheader('Prediction & Target Visualization')
        if 'prediction' not in st.session_state:
            st.session_state['prediction'] = None
        if submitted:
            input_arr = np.array([[inputs[c] for c in columns]])
            pred = float(model.predict(input_arr)[0, 0])
            st.session_state['prediction'] = pred
        if st.session_state['prediction'] is not None:
            label = 'Malignant' if st.session_state['prediction'] > 0.5 else 'Benign'
            st.success(
                f"Prediction: {label} (score: {st.session_state['prediction']:.2f})")
            # Pie chart for binary target
            target_counts = pd.Series(y).value_counts().sort_index()
            pie_df = pd.DataFrame(
                {'label': ['Benign', 'Malignant'], 'value': target_counts.values})
            pie_chart = alt.Chart(pie_df).mark_arc().encode(
                theta=alt.Theta(field='value', type='quantitative'),
                color=alt.Color(field='label', type='nominal'),
                tooltip=['label', 'value']
            )
            st.altair_chart(pie_chart, use_container_width=True)

    # Expanders
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
<tr><td colspan="4" style="text-align:center"><b>Accuracy: 0.89</b></td></tr>
</tbody>
</table>
''', unsafe_allow_html=True)
            st.image('assets/breast_cancer_loss_curve.png',
                     caption='Loss Curve')
