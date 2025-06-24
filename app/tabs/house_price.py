def run():
    import streamlit as st
    import numpy as np
    import pandas as pd
    import altair as alt
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    # Get model and data from session state
    model = st.session_state['model']
    data = st.session_state['data']

    # California housing dataset
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

    # Select up to 6 user-friendly columns for visualization
    viz_cols = ['MedInc', 'HouseAge', 'AveRooms',
                'AveOccup', 'Latitude', 'Longitude']
    viz_cols = [col for col in viz_cols if col in columns][:6]

    st.header('California House Price Prediction')
    st.markdown("It is a regression task that aims to predict the median house value in a block group based on 8 features such as median income, house age, average rooms, and location. This is a regression problem with the target label: MedHouseVal (Median House Value). The dataset is sourced from scikit-learn's built-in California Housing dataset.")

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
        form = st.form('house_form')
        inputs = {}
        col1, col2, col3 = form.columns(3)
        # Short descriptions for house price columns
        house_col_desc = {
            'MedInc': 'Median income in block group',
            'HouseAge': 'Median house age in years',
            'AveRooms': 'Average rooms per household',
            'AveOccup': 'Average household occupancy',
            'Latitude': 'Block group latitude value',
            'Longitude': 'Block group longitude value',
        }
        for idx, col in enumerate([c for c in columns if c != 'MedHouseVal']):
            avg = float(np.mean(X_viz[col]))
            desc = house_col_desc.get(col, col.replace('_', ' ').capitalize())
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
            'Predict', type="primary", use_container_width=True)

    with right:
        pred_head_col, pred_val_col = st.columns([2, 1])
        pred_head_col.subheader('Predict - Median House Value')
        pred_placeholder = pred_val_col.empty()
        if 'prediction' not in st.session_state:
            st.session_state['prediction'] = None
        if submitted:
            # Scale input features before prediction
            input_arr = np.array(
                [[inputs[c] for c in columns if c != 'MedHouseVal']])
            input_arr_scaled = scaler.transform(input_arr)
            pred = float(model.predict(input_arr_scaled)[0, 0])
            st.session_state['prediction'] = pred
        if st.session_state['prediction'] is not None:
            pred_placeholder.success(
                f"${st.session_state['prediction']*100000:,.2f}$ USD")
        # Sample 500 points for scatter plot
        sample_df = pd.DataFrame({
            'MedInc': X_viz['MedInc'],
            'MedHouseVal': y_train.flatten() if hasattr(y_train, 'flatten') else y_train
        })
        sample_df = sample_df.sample(
            n=min(500, len(sample_df)), random_state=42).reset_index(drop=True)
        base = alt.Chart(sample_df).mark_circle(size=40, color='#83c9ff').encode(
            x=alt.X('MedInc', title='Median Income (MedInc)'),
            y=alt.Y('MedHouseVal', title=None),
            tooltip=['MedInc', 'MedHouseVal']
        )
        # Add predicted point as orange circle
        if st.session_state['prediction'] is not None:
            pred_point = pd.DataFrame({
                'MedInc': [inputs['MedInc']],
                'MedHouseVal': [st.session_state['prediction']]
            })
            pred_chart = alt.Chart(pred_point).mark_circle(size=200, color='red').encode(
                x='MedInc',
                y='MedHouseVal',
                tooltip=['MedInc', 'MedHouseVal']
            )
            st.altair_chart(base + pred_chart, use_container_width=True)
        else:
            st.altair_chart(base, use_container_width=True)

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
            st.image('assets/california_housing_loss_curve.png',
                     caption='Loss Curve')
