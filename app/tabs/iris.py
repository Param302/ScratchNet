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

    # Select up to 4 user-friendly columns for visualization
    viz_cols = ['sepal length (cm)', 'sepal width (cm)',
                'petal length (cm)', 'petal width (cm)']
    viz_cols = [col for col in viz_cols if col in columns][:4]

    st.header('Iris Flower Species Classification')
    st.markdown("It is a multiclass classification task that aims to predict the species of an iris flower based on four features: sepal length, sepal width, petal length, and petal width. This is a 3-class classification problem with target labels: Setosa, Versicolor, and Virginica. The dataset is sourced from scikit-learn's built-in Iris dataset.")

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
        form = st.form('iris_form')
        inputs = {}
        col1, col2 = form.columns(2)
        # Short descriptions for iris columns
        iris_col_desc = {
            'sepal length (cm)': 'Sepal length in centimeters',
            'sepal width (cm)': 'Sepal width in centimeters',
            'petal length (cm)': 'Petal length in centimeters',
            'petal width (cm)': 'Petal width in centimeters',
        }
        for idx, col in enumerate(columns):
            avg = float(np.mean(X_viz[col]))
            desc = iris_col_desc.get(col, col.replace('_', ' ').capitalize())
            help_txt = f"{desc[:40]} (default: {avg:.2f})"
            label_txt = col.title()
            if idx % 2 == 0:
                inputs[col] = col1.number_input(
                    label=label_txt,
                    value=avg,
                    help=help_txt
                )
            else:
                inputs[col] = col2.number_input(
                    label=label_txt,
                    value=avg,
                    help=help_txt
                )
        _, btn_col, _ = form.columns([1, 1, 1])
        submitted = btn_col.form_submit_button(
            'Classify', type="primary", use_container_width=True)

    with right:
        pred_head_col, pred_val_col = st.columns([2, 1])
        pred_head_col.subheader('Predict Flower Species')
        pred_placeholder = pred_val_col.empty()
        if 'prediction' not in st.session_state:
            st.session_state['prediction'] = None
        if submitted:
            # Scale input features before prediction
            input_arr = np.array([[inputs[c] for c in columns]])
            input_arr_scaled = scaler.transform(input_arr)
            pred = model.predict(input_arr_scaled)
            pred_class = int(np.argmax(pred))
            st.session_state['prediction'] = pred_class
        if st.session_state['prediction'] is not None:
            class_names = list(np.ravel(data['target_names']))
            pred_idx = int(st.session_state['prediction'])
            pred_placeholder.success(f"{class_names[pred_idx]}")
        # Pie chart for target value distribution
        pie_data = pd.DataFrame({'Species': y_viz})
        pie_counts = pie_data['Species'].value_counts(
        ).sort_index().reset_index()
        pie_counts.columns = ['Species', 'Count']
        # Add class names for labels
        class_names = list(np.ravel(data['target_names']))
        pie_counts['Label'] = pie_counts['Species'].apply(
            lambda i: class_names[int(i)])
        palette = ['#83c9ff', "#F0F089", "#C28EFA"]
        pie_counts['BaseColor'] = [palette[i %
                                           len(palette)] for i in range(len(pie_counts))]
        # Mark the predicted species
        pred_species = None
        if st.session_state['prediction'] is not None:
            pred_species = int(st.session_state['prediction'])
        pie_counts['Color'] = pie_counts.apply(
            lambda row: 'orange' if pred_species is not None and row['Species'] == pred_species else row['BaseColor'], axis=1)
        pie_chart = alt.Chart(pie_counts).mark_arc(outerRadius=100).encode(
            theta=alt.Theta(field='Count', type='quantitative'),
            color=alt.Color('Label:N', scale=alt.Scale(
                range=palette), legend=alt.Legend(title='Species')),
            tooltip=['Label', 'Count']
        )
        st.altair_chart(pie_chart, use_container_width=True)

    # Expanders
    exp_col1, exp_col2 = st.columns(2)
    with exp_col1:
        with st.expander('How the Model Works'):
            st.markdown('''**Model Overview:**
- This model has 5 layers:
    - 1 input layer (4 features)
    - 3 hidden layers (with 8, 16, and 64 neurons)
    - 1 output layer (3 classes)
- Uses ReLU and Softmax activations
- Loss Function: Sparse Categorical Cross Entropy

**Total Parameters:** 1,955
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
<tr><td>0</td><td>1.00</td><td>1.00</td><td>1.00</td></tr>
<tr><td>1</td><td>0.88</td><td>1.00</td><td>0.93</td></tr>
<tr><td>2</td><td>1.00</td><td>0.93</td><td>0.96</td></tr>
<tr><td colspan="4" style="text-align:center"><b>Accuracy: 97%</b></td></tr>
</tbody>
</table>
''', unsafe_allow_html=True)
            st.image('assets/iris_loss_curve.png', caption='Loss Curve')
