def run():
    import streamlit as st
    import numpy as np
    import pandas as pd
    import altair as alt
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    model = st.session_state['model']
    data = st.session_state['data']

    X = data['data']
    y = data['target']
    columns = X.columns.tolist()

    # Split and scale data for visualization and prediction
    X_full = X.values if hasattr(X, 'values') else X
    y_full = y.values if hasattr(y, 'values') else y
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_full, y_full, test_size=0.2, random_state=37)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_valid_scaled = scaler.transform(X_valid)
    # For visualization and form, use unscaled X_train
    X_viz = pd.DataFrame(X_train, columns=columns)
    y_viz = pd.Series(y_train)

    # Select up to 6 user-friendly columns for visualization
    viz_cols = ['mean radius', 'mean texture', 'mean perimeter',
                'mean area', 'mean smoothness', 'mean compactness']
    viz_cols = [col for col in viz_cols if col in columns][:6]

    st.header('Breast Cancer Classifier')
    st.markdown("It is a binary classification task that aims to distinguish between malignant and benign tumors based on 30 features computed from digitized images of fine needle aspirate (FNA) of breast mass. This is a 2-class classification problem with target labels: Benign (0) and Malignant (1). The dataset is sourced from scikit-learn's built-in Breast Cancer Wisconsin dataset.")

    # Visualizations (histogram distributions with Altair)
    st.subheader('Feature Distributions')
    viz_grid = st.columns(len(viz_cols), gap="medium")
    for i, col in enumerate(viz_cols):
        with viz_grid[i]:
            chart = alt.Chart(pd.DataFrame({col: X_viz[col]})).mark_bar().encode(
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
        col1, col2, col3 = form.columns(3)
        # Short descriptions for breast cancer columns
        cancer_col_desc = {
            'mean radius': 'Mean radius of cell nuclei',
            'mean texture': 'Mean texture of cell nuclei',
            'mean perimeter': 'Mean perimeter of cell nuclei',
            'mean area': 'Mean area of cell nuclei',
            'mean smoothness': 'Mean smoothness of cell nuclei',
            'mean compactness': 'Mean compactness of cell nuclei',
            # Add more if needed
        }
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
        # Pie chart for binary target
        target_counts = y_viz.value_counts().sort_index()
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
<tr><td colspan="4" style="text-align:center"><b>Accuracy: 89%</b></td></tr>
</tbody>
</table>
''', unsafe_allow_html=True)
            st.image('assets/breast_cancer_loss_curve.png',
                     caption='Loss Curve')
