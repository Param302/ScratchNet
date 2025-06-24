def run():
    import streamlit as st
    import numpy as np
    import pandas as pd
    import altair as alt

    # Get model and data from session state
    model = st.session_state['model']
    data = st.session_state['data']

    # California housing dataset
    X = data['data']
    y = data['target']
    columns = X.columns.tolist()

    # Select up to 6 user-friendly columns for visualization
    viz_cols = ['MedInc', 'HouseAge', 'AveRooms',
                'AveOccup', 'Latitude', 'Longitude']
    viz_cols = [col for col in viz_cols if col in columns][:6]

    st.header('California House Price Prediction')

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
        form = st.form('house_form')
        inputs = {}
        for col in columns:
            if col == 'MedHouseVal':
                continue
            avg = float(np.mean(X[col]))
            inputs[col] = form.number_input(
                label=col,
                value=avg,
                help=f"Average value: {avg:.2f} (default if left blank)"
            )
        submitted = form.form_submit_button('Predict')

    with right:
        st.subheader('Prediction & Target Visualization')
        if 'prediction' not in st.session_state:
            st.session_state['prediction'] = None
        if submitted:
            input_arr = np.array([[inputs[c] for c in columns]])
            pred = float(model.predict(input_arr)[0, 0])
            st.session_state['prediction'] = pred
        if st.session_state['prediction'] is not None:
            st.success(
                f"Predicted Median House Value: ${st.session_state['prediction']*100000:.2f}")
            # Interactive histogram for target distribution
            chart = alt.Chart(pd.DataFrame({'Target': y})).mark_bar().encode(
                alt.X('Target:Q', bin=alt.Bin(maxbins=30)),
                y='count()',
                tooltip=['Target', 'count()']
            ).properties(width=300, height=120)
            st.altair_chart(chart, use_container_width=True)

    # Expanders
    exp_col1, exp_col2 = st.columns(2)
    with exp_col1:
        with st.expander('How the Model Works'):
            st.markdown('''**Model Overview:**
- This model has 5 layers:
    - 1 input layer (8 features)
    - 3 hidden layers (10, 16, 32, 16 neurons)
    - 1 output layer (1 value)
- Uses ReLU and Linear activations
- Loss Function: MSE (Mean Squared Error)
**Total Parameters:** 1,355
''')
    with exp_col2:
        with st.expander('Performance & Training Insights'):
            st.markdown('''See README for detailed regression metrics.''')
            st.image('assets/california_housing_loss_curve.png',
                     caption='Loss Curve')
