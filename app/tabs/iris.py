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

    # Select up to 4 user-friendly columns for visualization
    viz_cols = ['sepal length (cm)', 'sepal width (cm)',
                'petal length (cm)', 'petal width (cm)']
    viz_cols = [col for col in viz_cols if col in columns][:4]

    st.header('Iris Flower Species Classifier')

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
        form = st.form('iris_form')
        inputs = {}
        for col in columns:
            avg = float(np.mean(X[col]))
            inputs[col] = form.number_input(
                label=col,
                value=avg,
                help=f"Average value: {avg:.2f} (default if left blank)"
            )
        submitted = form.form_submit_button('Classify')

    with right:
        st.subheader('Prediction & Target Visualization')
        if 'prediction' not in st.session_state:
            st.session_state['prediction'] = None
        if submitted:
            input_arr = np.array([[inputs[c] for c in columns]])
            pred = model.predict(input_arr)
            pred_class = int(np.argmax(pred))
            st.session_state['prediction'] = pred_class
        if st.session_state['prediction'] is not None:
            class_names = data['target_names']
            st.success(
                f"Predicted Species: {class_names[st.session_state['prediction']]}")
            # Interactive bar chart for target distribution
            target_df = pd.DataFrame({'Species': y})
            chart_data = target_df['Species'].value_counts(
            ).sort_index().reset_index()
            chart_data.columns = ['Species', 'Count']
            chart = alt.Chart(chart_data).mark_bar().encode(
                x=alt.X('Species:N', title='Species'),
                y=alt.Y('Count:Q'),
                color=alt.condition(
                    alt.datum.Species == st.session_state['prediction'],
                    alt.value('#FF0000'), alt.value('#AAAAAA')
                ),
                tooltip=['Species', 'Count']
            ).properties(width=300, height=120)
            st.altair_chart(chart, use_container_width=True)

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
<tr><td colspan="4" style="text-align:center"><b>Accuracy: 0.97</b></td></tr>
</tbody>
</table>
''', unsafe_allow_html=True)
            st.image('assets/iris_loss_curve.png', caption='Loss Curve')
